#!/usr/bin/env python3
"""Build multi-seed future oracle masks from PDM, trajectory displacement and planning harm."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


_METHOD_RE = re.compile(
    r"physical_future_latent(?P<latent>\d+)_drop_tile(?P<tile>\d+)_hidden_persistent_layer_\d+"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-roots", required=True, help="comma-separated matrix suite roots, one per seed")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--ranking", choices=("pdm_harm", "traj_disp", "planning_harm", "combined"), default="combined")
    parser.add_argument("--tile-h", type=int, default=3)
    parser.add_argument("--tile-w", type=int, default=4)
    parser.add_argument("--require-num-future-latents", type=int, default=2)
    return parser.parse_args(argv)


def _read_records(method_dir: Path) -> dict[str, dict[str, Any]]:
    path = method_dir / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing records: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
    return {str(row["scene_token"]): row for row in rows}


def _load_trajectory(method_dir: Path, token: str) -> np.ndarray | None:
    matches = sorted((method_dir / "trajectories").glob(f"{token}.rank*.npz"))
    if not matches:
        return None
    arrays = []
    for path in matches:
        with np.load(path) as payload:
            traj = np.asarray(payload["traj"], dtype=np.float32)
        if traj.ndim >= 2:
            traj = traj[0] if traj.shape[0] == 1 else traj
        arrays.append(traj)
    # A scene should have exactly one rank file in normal operation.
    return arrays[0]


def _load_target(method_dir: Path, token: str, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if token in cache:
        return cache[token]
    path = method_dir / "target_trajectories" / f"{token}.npy"
    if not path.is_file():
        cache[token] = None
        return None
    value = np.asarray(np.load(path), dtype=np.float32)
    cache[token] = value
    return value


def _trajectory_displacement(baseline: np.ndarray, masked: np.ndarray) -> float:
    if baseline.ndim == 1:
        baseline = baseline[None, :]
    if masked.ndim == 1:
        masked = masked[None, :]
    steps = min(baseline.shape[0], masked.shape[0])
    dims = min(baseline.shape[1], masked.shape[1], 3)
    if steps <= 0 or dims <= 0:
        return math.nan
    return float(np.linalg.norm(baseline[:steps, :dims] - masked[:steps, :dims], axis=-1).mean())


def _planning_harm(baseline: np.ndarray, masked: np.ndarray, target: np.ndarray | None) -> float:
    if target is None:
        return math.nan
    if target.ndim == 1:
        target = target[None, :]
    base_ade = _trajectory_displacement(target, baseline)
    masked_ade = _trajectory_displacement(target, masked)
    if not math.isfinite(base_ade) or not math.isfinite(masked_ade):
        return math.nan
    return masked_ade - base_ade


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return np.zeros_like(values)
    mean = float(values[finite].mean())
    std = float(values[finite].std())
    if std <= 1e-12:
        return np.zeros_like(values)
    result = np.zeros_like(values)
    result[finite] = (values[finite] - mean) / std
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    suite_roots = [Path(value.strip()).expanduser().resolve() for value in args.suite_roots.split(",") if value.strip()]
    if not suite_roots:
        raise ValueError("--suite-roots is empty")
    groups = args.tile_h * args.tile_w
    keep_count = max(1, min(groups, int(round(groups * float(args.keep_ratio)))))

    # (scene, latent, tile) -> metric -> list per seed
    metrics: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    tile_target_cache: dict[str, np.ndarray | None] = {}
    seeds_seen = []
    for seed_index, root in enumerate(suite_roots):
        round_dir = root / "round01"
        if not round_dir.is_dir():
            raise FileNotFoundError(f"round01 not found: {root}")
        seeds_seen.append(root.name)
        baseline_dir = round_dir / "physical_no_press"
        baseline = _read_records(baseline_dir)
        baseline_traj: dict[str, np.ndarray | None] = {}
        baseline_target: dict[str, np.ndarray | None] = {}
        for token in baseline:
            baseline_traj[token] = _load_trajectory(baseline_dir, token)
            baseline_target[token] = _load_target(baseline_dir, token, tile_target_cache)
        for method_dir in sorted(round_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            match = _METHOD_RE.match(method_dir.name)
            if match is None:
                continue
            latent = int(match.group("latent"))
            tile = int(match.group("tile"))
            if latent >= args.require_num_future_latents:
                continue
            rows = _read_records(method_dir)
            for token, row in rows.items():
                key = (token, latent, tile)
                bucket = metrics.setdefault(key, {"pdm_harm": [], "traj_disp": [], "planning_harm": []})
                bucket["pdm_harm"].append(float(baseline[token]["pdm"]) - float(row["pdm"]))
                baseline_path = baseline_traj[token]
                masked_path = _load_trajectory(method_dir, token)
                if baseline_path is not None and masked_path is not None:
                    bucket["traj_disp"].append(_trajectory_displacement(baseline_path, masked_path))
                target = baseline_target[token]
                if target is not None and baseline_path is not None and masked_path is not None:
                    bucket["planning_harm"].append(_planning_harm(baseline_path, masked_path, target))
    if not metrics:
        raise RuntimeError("no tile metrics were collected")

    # Aggregate per (scene, latent, tile).
    aggregated: dict[tuple[str, int, int], dict[str, float]] = {}
    for key, bucket in metrics.items():
        aggregated[key] = {
            name: float(np.nanmean(values)) if values and np.isfinite(values).any() else math.nan
            for name, values in bucket.items()
        }

    masks: dict[str, dict[str, list[int]]] = {}
    scene_keys = sorted({(scene, latent) for scene, latent, _ in aggregated})
    for scene, latent in scene_keys:
        tiles = sorted(tile for s, l, tile in aggregated if s == scene and l == latent)
        if not tiles:
            continue
        pdm = np.asarray([aggregated[(scene, latent, t)]["pdm_harm"] for t in tiles], dtype=np.float64)
        disp = np.asarray([aggregated[(scene, latent, t)]["traj_disp"] for t in tiles], dtype=np.float64)
        harm = np.asarray([aggregated[(scene, latent, t)]["planning_harm"] for t in tiles], dtype=np.float64)
        if args.ranking == "pdm_harm":
            score = pdm
        elif args.ranking == "traj_disp":
            score = disp
        elif args.ranking == "planning_harm":
            score = harm
        else:
            components = [_zscore(pdm), _zscore(disp), _zscore(harm)]
            score = np.nansum(np.stack(components, axis=0), axis=0)
        order = sorted(range(len(tiles)), key=lambda idx: (-np.inf if not np.isfinite(score[idx]) else score[idx]), reverse=True)
        keep_tiles = [int(tiles[idx]) for idx in order[:keep_count]]
        entry = masks.setdefault(scene, {})
        entry[f"future_latent_{latent}"] = keep_tiles

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(masks, indent=2, sort_keys=True), encoding="utf-8")

    summary_rows = []
    for (scene, latent, tile), values in aggregated.items():
        summary_rows.append({"scene_token": scene, "latent_index": latent, "tile": tile, **values})
    summary = {
        "suite_roots": [str(root) for root in suite_roots],
        "n_seeds": len(suite_roots),
        "n_scenes": len(masks),
        "tile_h": args.tile_h,
        "tile_w": args.tile_w,
        "tile_groups": groups,
        "keep_ratio": float(args.keep_ratio),
        "keep_tiles_per_latent": keep_count,
        "ranking": args.ranking,
        "mean_metrics": {
            name: float(np.nanmean([row[name] for row in summary_rows])) if summary_rows else math.nan
            for name in ("pdm_harm", "traj_disp", "planning_harm")
        },
        "n_rows": len(summary_rows),
    }
    (output.parent / f"{output.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

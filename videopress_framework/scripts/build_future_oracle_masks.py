#!/usr/bin/env python3
"""Build a per-scene oracle future-tile mask from counterfactual PDM drops."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_METHOD_RE = re.compile(
    r"physical_future_latent(?P<latent>\d+)_drop_tile(?P<tile>\d+)_hidden_persistent_layer_\d+"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Derive per-scene future tile oracle masks from a counterfactual tile suite"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
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


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.suite_root.expanduser().resolve()
    round_dir = root / "round01"
    if not round_dir.is_dir():
        raise FileNotFoundError(f"round01 not found under {root}")
    baseline = _read_records(round_dir / "physical_no_press")
    # scene_token -> latent -> tile -> method_dir row
    per_scene: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
        scene: {latent: {} for latent in range(args.require_num_future_latents)}
        for scene in baseline
    }
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
        for scene, row in rows.items():
            if scene not in per_scene:
                raise KeyError(f"method {method_dir.name} has unknown scene {scene!r}")
            per_scene[scene][latent][tile] = row

    groups = args.tile_h * args.tile_w
    keep_count = max(1, int(round(groups * float(args.keep_ratio))))
    if keep_count > groups:
        keep_count = groups
    masks: dict[str, dict[str, list[int]]] = {}
    summary_tile_harm: list[dict[str, float]] = []
    for scene, base_row in baseline.items():
        base_pdm = float(base_row["pdm"])
        entry: dict[str, list[int]] = {}
        for latent in range(args.require_num_future_latents):
            tile_rows = per_scene[scene][latent]
            missing = sorted(set(range(groups)) - set(tile_rows))
            if missing:
                raise KeyError(
                    f"scene {scene} latent {latent} missing tiles {missing}"
                )
            ranked = sorted(
                range(groups),
                key=lambda tile: base_pdm - float(tile_rows[tile]["pdm"]),
                reverse=True,
            )
            entry[f"future_latent_{latent}"] = ranked[:keep_count]
            for tile in range(groups):
                summary_tile_harm.append(
                    {
                        "scene_token": scene,
                        "latent_index": latent,
                        "tile": tile,
                        "pdm_drop": base_pdm - float(tile_rows[tile]["pdm"]),
                    }
                )
        masks[scene] = entry

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(masks, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "suite_root": str(root),
        "output_json": str(output),
        "n_scenes": len(masks),
        "tile_h": args.tile_h,
        "tile_w": args.tile_w,
        "tile_groups": groups,
        "keep_ratio": float(args.keep_ratio),
        "keep_tiles_per_latent": keep_count,
        "mean_tile_pdm_drop": sum(row["pdm_drop"] for row in summary_tile_harm)
        / max(1, len(summary_tile_harm)),
    }
    (output.parent / f"{output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

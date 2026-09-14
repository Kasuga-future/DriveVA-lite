#!/usr/bin/env python3
"""Build official NavSIM metric caches for a scene-safe manifest.

The official NavSIM caching entrypoint groups work by the original log file
name.  ``audit_navsim_split.py`` materializes one safe metadata file per
scene, so this small independent wrapper keeps the official cache primitives
(``Scene``, ``NavSimScenario`` and ``MetricCacheProcessor``) while scheduling
the already-audited windows from the manifest directly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(PROJECT_ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "third_party"))

_PROCESSOR: Any = None
_HISTORY_FRAMES = 5
_FUTURE_FRAMES = 10


def _atomic_save_buffer(output_path: Path, buf: bytes) -> None:
    """Write a local metric cache without nuPlan's aiofiles deadlock.

    The DriveVA environment can leave ``save_buffer`` waiting forever inside
    ``asyncio.run(_save_buffer_async(...))`` even though its executor thread is
    idle.  This wrapper only accepts local paths (the CLI already resolves the
    cache root to ``Path``), writes beside the destination, then atomically
    replaces it so interrupted runs cannot masquerade as valid cache entries.
    """
    output_path = Path(output_path)
    if str(output_path).startswith("s3:"):
        raise ValueError("cache_navsim_split only supports local cache paths")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(buf)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FRAMEWORK_ROOT / "outputs/navsim_split_audit/test_manifest.jsonl",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_split_test"),
    )
    parser.add_argument(
        "--map-root",
        type=Path,
        default=Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0"),
    )
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--windows-per-scene",
        type=int,
        default=None,
        help="Select temporally spread windows per semantic scene; default caches every valid window.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit it for the complete manifest.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def _init_worker(cache_root: str, map_root: str, history_frames: int, future_frames: int) -> None:
    global _PROCESSOR, _HISTORY_FRAMES, _FUTURE_FRAMES
    os.environ["NUPLAN_MAPS_ROOT"] = map_root
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    import navsim.planning.metric_caching.metric_cache as metric_cache_module

    # ``MetricCache.dump`` imports save_buffer into its own module namespace,
    # so patch that binding rather than nuPlan's source module.
    metric_cache_module.save_buffer = _atomic_save_buffer

    _HISTORY_FRAMES = history_frames
    _FUTURE_FRAMES = future_frames
    _PROCESSOR = MetricCacheProcessor(
        cache_path=cache_root,
        force_feature_computation=False,
    )


def _cache_group(item: tuple[str, list[int]]) -> dict[str, Any]:
    metadata_path, starts = item
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    frames = pickle.loads(Path(metadata_path).read_bytes())
    successes: list[str] = []
    failures: list[dict[str, str]] = []
    expected_length = _HISTORY_FRAMES + _FUTURE_FRAMES
    for start in starts:
        window = frames[start : start + expected_length]
        token = str(window[_HISTORY_FRAMES - 1].get("token", "")) if len(window) >= _HISTORY_FRAMES else ""
        try:
            if len(window) != expected_length:
                raise ValueError(f"window length {len(window)} != {expected_length}")
            scene = Scene.from_scene_dict_list(
                window,
                None,
                num_history_frames=_HISTORY_FRAMES,
                num_future_frames=_FUTURE_FRAMES,
                sensor_config=SensorConfig.build_no_sensors(),
            )
            scenario = NavSimScenario(
                scene,
                map_root=os.environ["NUPLAN_MAPS_ROOT"],
                map_version="nuplan-maps-v1.0",
            )
            cache_file = (
                Path(_PROCESSOR._cache_path)
                / scenario.log_name
                / scenario.scenario_type
                / scenario.token
                / "metric_cache.pkl"
            )
            # An interrupted legacy async write leaves a zero-byte file, and
            # MetricCacheProcessor otherwise treats mere existence as a hit.
            if cache_file.is_file() and cache_file.stat().st_size == 0:
                cache_file.unlink()
            entry = _PROCESSOR.compute_metric_cache(scenario)
            if entry is None:
                raise RuntimeError("MetricCacheProcessor returned None")
            if not Path(entry.file_name).is_file() or Path(entry.file_name).stat().st_size == 0:
                raise RuntimeError(f"MetricCacheProcessor produced an empty cache: {entry.file_name}")
            successes.append(str(Path(entry.file_name).resolve()))
        except Exception as exc:  # keep the full run resumable and report exact failures
            failures.append({"metadata_path": metadata_path, "start": str(start), "token": token, "error": repr(exc)})
    return {
        "metadata_path": metadata_path,
        "requested": len(starts),
        "successes": successes,
        "failures": failures,
    }


def _load_groups(
    manifest_path: Path,
    history_frames: int,
    future_frames: int,
    windows_per_scene: int | None = None,
) -> tuple[list[tuple[str, list[int]]], set[str]]:
    groups: list[tuple[str, list[int]]] = []
    expected_tokens: set[str] = set()
    window_length = history_frames + future_frames
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata_path = Path(row["metadata_path"]).resolve()
        frames = pickle.loads(metadata_path.read_bytes())
        starts: list[int] = []
        for start in range(0, len(frames) - window_length + 1):
            window = frames[start : start + window_length]
            if not bool(window[history_frames - 1].get("roadblock_ids")):
                continue
            frame_tokens = [str(frame.get("token")) for frame in window]
            if len(frame_tokens) != len(set(frame_tokens)):
                raise ValueError(f"duplicate frame token in {metadata_path} at start={start}")
            scene_tokens = {str(frame.get("scene_token")) for frame in window}
            if len(scene_tokens) != 1:
                raise ValueError(f"cross-scene window in {metadata_path} at start={start}")
            frame_indices = [int(frame["frame_idx"]) for frame in window]
            if any(right - left != 1 for left, right in zip(frame_indices, frame_indices[1:])):
                raise ValueError(f"non-contiguous window in {metadata_path} at start={start}")
            starts.append(start)
        if windows_per_scene is not None and len(starts) > windows_per_scene:
            indices = [
                min(len(starts) - 1, int((index + 1) * len(starts) / (windows_per_scene + 1)))
                for index in range(windows_per_scene)
            ]
            starts = [starts[index] for index in indices]
        for start in starts:
            token = str(frames[start + history_frames - 1]["token"])
            if token in expected_tokens:
                raise ValueError(f"duplicate cache token {token}")
            expected_tokens.add(token)
        if starts:
            groups.append((str(metadata_path), starts))
    return groups, expected_tokens


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.history_frames < 1 or args.future_frames < 1 or args.workers < 1:
        raise ValueError("history/future/workers must be positive")
    manifest = args.manifest.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    map_root = args.map_root.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not map_root.is_dir():
        raise FileNotFoundError(map_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    report_path = (args.report or cache_root / "split_cache_build_report.json").expanduser().resolve()

    if args.windows_per_scene is not None and args.windows_per_scene < 1:
        raise ValueError("--windows-per-scene must be positive")
    groups, expected_tokens = _load_groups(
        manifest,
        args.history_frames,
        args.future_frames,
        args.windows_per_scene,
    )
    if args.max_groups is not None:
        if args.max_groups < 1:
            raise ValueError("--max-groups must be positive")
        groups = groups[: args.max_groups]
        expected_tokens = {
            str(pickle.loads(Path(metadata_path).read_bytes())[start + args.history_frames - 1]["token"])
            for metadata_path, starts in groups
            for start in starts
        }

    started = time.time()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(cache_root), str(map_root), args.history_frames, args.future_frames),
    ) as executor:
        futures = [executor.submit(_cache_group, item) for item in groups]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            errors.extend(result["failures"])
            if index == 1 or index % 50 == 0 or index == len(futures):
                done = sum(len(item["successes"]) for item in results)
                print(f"[split-cache] groups {index}/{len(futures)}, cache entries {done}, failures {len(errors)}", flush=True)

    cache_paths = {path for item in results for path in item["successes"] if Path(path).is_file()}
    from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata

    save_cache_metadata(
        [CacheMetadataEntry(Path(path)) for path in sorted(cache_paths)],
        cache_root,
        node_id=0,
    )
    actual_tokens = {Path(path).parent.name for path in cache_paths}
    report = {
        "manifest": str(manifest),
        "cache_root": str(cache_root),
        "map_root": str(map_root),
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "windows_per_scene": args.windows_per_scene,
        "groups": len(groups),
        "expected_tokens": len(expected_tokens),
        "cache_entries": len(cache_paths),
        "missing_tokens": len(expected_tokens - actual_tokens),
        "unexpected_tokens": len(actual_tokens - expected_tokens),
        "failures": len(errors),
        "failure_examples": errors[:20],
        "elapsed_seconds": round(time.time() - started, 2),
        "ok": not errors and expected_tokens == actual_tokens,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

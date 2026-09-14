#!/usr/bin/env python3
"""Audit official NAVSIM windows before any Wan inference is started.

The audit runs the repository's official ``filter_scenes`` once without the
runtime guard to quantify the source issue, then constructs the real official
``SceneLoader`` with the framework guard installed.  It verifies that every
window used by the evaluator has one scene token/name, the expected anchor,
unique frames and contiguous frame indices, and that every corrected token has
an official metric-cache entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from scripts.run_official_navsim_press import (
    _configure_official_data_environment,
    _load_official_eval_module,
)
from videopress.adapters.scene_boundary import install_scene_boundary_guard, window_is_single_scene


def _issues(token: str, frames: list[dict[str, Any]], history: int, expected_length: int) -> list[str]:
    issues: list[str] = []
    if len(frames) != expected_length:
        issues.append(f"length={len(frames)} expected={expected_length}")
    scene_tokens = {frame.get("scene_token") for frame in frames}
    scene_names = {frame.get("scene_name") for frame in frames}
    if None in scene_tokens:
        issues.append("missing_scene_token")
    if None in scene_names:
        issues.append("missing_scene_name")
    if len(scene_tokens) != 1:
        issues.append(f"scene_token_count={len(scene_tokens)}")
    if len(scene_names) != 1:
        issues.append(f"scene_name_count={len(scene_names)}")
    if not window_is_single_scene(frames):
        issues.append("cross_scene_boundary")
    frame_tokens = [str(frame.get("token")) for frame in frames]
    if len(set(frame_tokens)) != len(frame_tokens):
        issues.append("duplicate_frame_token")
    try:
        frame_indices = [int(frame["frame_idx"]) for frame in frames]
    except (KeyError, TypeError, ValueError):
        frame_indices = []
        issues.append("missing_or_invalid_frame_idx")
    if frame_indices and any(right - left != 1 for left, right in zip(frame_indices, frame_indices[1:])):
        issues.append("noncontiguous_frame_idx")
    if len(frames) > history - 1:
        if str(frames[history - 1].get("token")) != str(token):
            issues.append("anchor_token_mismatch")
    else:
        issues.append("missing_history_anchor")
    timestamps = [frame.get("timestamp") for frame in frames]
    try:
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            issues.append("non_increasing_timestamp")
    except TypeError:
        issues.append("invalid_timestamp")
    return issues


def _example(token: str, frames: list[dict[str, Any]], issues: list[str]) -> dict[str, Any]:
    return {
        "token": str(token),
        "issues": list(issues),
        "scene_tokens": sorted(str(frame.get("scene_token")) for frame in frames),
        "scene_names": sorted(str(frame.get("scene_name")) for frame in frames),
        "frame_idx": [frame.get("frame_idx") for frame in frames],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/openscene-v1.1/meta_datas/test",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/openscene-v1.1/sensor_blobs/test",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        default=PROJECT_ROOT / "data/navsim_v1.1/metric_cache_full",
    )
    parser.add_argument(
        "--scene-filter-yaml",
        type=Path,
        default=PROJECT_ROOT / "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml",
    )
    parser.add_argument("--num-history-frames", type=int, default=5)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=FRAMEWORK_ROOT / "outputs/scene_window_audit.json",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    log_path = args.navsim_log_path.expanduser().resolve()
    sensor_path = args.sensor_blobs_path.expanduser().resolve()
    cache_path = args.metric_cache_path.expanduser().resolve()
    yaml_path = args.scene_filter_yaml.expanduser().resolve()
    for path in (log_path, sensor_path, cache_path, yaml_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # Set these before importing navsim.dataclasses, which reads them at import
    # time.  Do not inherit another project's paths from the parent shell.
    data_environment = _configure_official_data_environment(repo_root)

    official_eval = _load_official_eval_module()
    official_eval._ensure_navsim_importable(repo_root)
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    import navsim.common.dataloader as dataloader
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    overrides = official_eval._load_scene_filter_yaml(str(yaml_path), filter_only=True)
    scene_filter = SceneFilter(
        num_history_frames=int(args.num_history_frames),
        num_future_frames=int(args.num_future_frames),
        frame_interval=int(args.frame_interval),
        has_route=True,
        max_scenes=None,
        log_names=None,
    )
    for key, value in overrides.items():
        setattr(scene_filter, key, value)

    guard = install_scene_boundary_guard()
    guarded_filter = dataloader.filter_scenes
    original_filter = getattr(guarded_filter, "_driveva_lite_original_filter", guarded_filter)
    raw_windows = original_filter(log_path, scene_filter)
    expected_length = int(scene_filter.num_history_frames + scene_filter.num_future_frames)

    raw_issue_counts: dict[str, int] = {}
    raw_examples: list[dict[str, Any]] = []
    raw_invalid_count = 0
    for token, frames in raw_windows.items():
        issues = _issues(str(token), frames, int(scene_filter.num_history_frames), expected_length)
        if issues:
            raw_invalid_count += 1
        for issue in issues:
            raw_issue_counts[issue] = raw_issue_counts.get(issue, 0) + 1
        if issues and len(raw_examples) < max(0, int(args.max_examples)):
            raw_examples.append(_example(str(token), frames, issues))

    # This is the exact class used by the official evaluator.  Its constructor
    # resolves the patched module-level filter_scenes, proving the guard is not
    # merely a post-hoc report transformation.
    corrected_loader = SceneLoader(
        data_path=log_path,
        sensor_blobs_path=sensor_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=True),
        load_image_path=True,
    )
    corrected_errors: list[dict[str, Any]] = []
    for token, frames in corrected_loader.scene_frames_dicts.items():
        issues = _issues(str(token), frames, int(scene_filter.num_history_frames), expected_length)
        if issues and len(corrected_errors) < max(0, int(args.max_examples)):
            corrected_errors.append(_example(str(token), frames, issues))

    raw_guarded_tokens = {
        str(token)
        for token, frames in raw_windows.items()
        if not _issues(str(token), frames, int(scene_filter.num_history_frames), expected_length)
    }
    corrected_tokens = {str(token) for token in corrected_loader.tokens}
    cache_tokens = {str(token) for token in MetricCacheLoader(cache_path).tokens}
    missing_cache = sorted(corrected_tokens - cache_tokens)

    payload = {
        "ok": not corrected_errors
        and not missing_cache
        and raw_guarded_tokens == corrected_tokens,
        "repo_root": str(repo_root),
        "log_path": str(log_path),
        "sensor_blobs_path": str(sensor_path),
        "metric_cache_path": str(cache_path),
        "scene_filter_yaml": str(yaml_path),
        "guard": guard,
        "data_environment": data_environment,
        "scene_filter": {
            "num_history_frames": int(scene_filter.num_history_frames),
            "num_future_frames": int(scene_filter.num_future_frames),
            "frame_interval": int(scene_filter.frame_interval),
            "has_route": bool(scene_filter.has_route),
            "requested_log_names": len(scene_filter.log_names or []),
            "requested_tokens": len(scene_filter.tokens or []),
            "window_length": expected_length,
        },
        "raw_official_loader": {
            "tokens": len(raw_windows),
            "invalid_window_count": raw_invalid_count,
            "cross_scene_windows": raw_issue_counts.get("cross_scene_boundary", 0),
            "issue_counts": raw_issue_counts,
            "examples": raw_examples,
        },
        "guarded_official_loader": {
            "tokens": len(corrected_tokens),
            "errors": len(corrected_errors),
            "error_examples": corrected_errors,
            "tokens_equal_filtered_raw": raw_guarded_tokens == corrected_tokens,
        },
        "metric_cache": {
            "tokens": len(cache_tokens),
            "missing_guarded_tokens": len(missing_cache),
            "missing_examples": missing_cache[: int(args.max_examples)],
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

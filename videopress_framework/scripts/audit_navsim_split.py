#!/usr/bin/env python3
"""Audit and split the available camera-only NavSIM scene segments.

OpenScene metadata files are concatenated logs: one ``.pkl`` can contain many
NavSIM scene segments, and ``frame_idx`` resets at each segment boundary.  A
directory-level split is therefore not sufficient.  This script segments the
metadata first, writes one small metadata pickle per scene unit, and emits
manifests that reuse the original Camera directories without copying them.

The split is source-preserving:

* downloaded OpenScene ``trainval`` -> train
* original OpenScene ``test`` -> test
* metadata-only staging -> excluded

The audit checks Camera references, scene-token/name boundaries, contiguous
frames, same-scene windows, and train/test leakage by stable identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import shutil
from typing import Any, Iterable


CAMERA_KEYS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)

DATA_ROOT = Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1")
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = FRAMEWORK_ROOT / "outputs/navsim_split_audit"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    split: str | None
    metadata_dir: Path
    sensor_dir: Path | None


def _source_specs(data_root: Path) -> list[SourceSpec]:
    return [
        SourceSpec(
            name="official_test",
            split="test",
            metadata_dir=data_root / "openscene-v1.1/meta_datas/test",
            sensor_dir=data_root / "openscene-v1.1/sensor_blobs/test",
        ),
        SourceSpec(
            name="downloaded_trainval",
            split="train",
            metadata_dir=data_root / "extra_trainval_32/openscene-v1.1/meta_datas/trainval",
            sensor_dir=data_root / "extra_trainval_32/openscene-v1.1/sensor_blobs/trainval",
        ),
        # The file is reported and segmented, but cannot enter either split
        # because its Camera blobs were never downloaded.
        SourceSpec(
            name="metadata_only_staging",
            split=None,
            metadata_dir=data_root / "train32_staging/openscene-v1.1/meta_datas/trainval",
            sensor_dir=None,
        ),
    ]


def _string_set(frames: Iterable[dict[str, Any]], key: str) -> set[str | None]:
    values: set[str | None] = set()
    for frame in frames:
        value = frame.get(key)
        values.add(None if value is None else str(value))
    return values


def _json_values(values: set[str | None]) -> list[str | None]:
    return sorted(values, key=lambda value: "" if value is None else value)


def _capture_group(stem: str) -> str:
    """Remove the start/end frame suffix used by OpenScene fragment names."""

    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return stem


def _is_boundary(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return whether ``current`` starts a new NavSIM scene segment."""

    if previous.get("scene_token") != current.get("scene_token"):
        return True
    if previous.get("scene_name") != current.get("scene_name"):
        return True
    if previous.get("log_name") != current.get("log_name"):
        return True
    try:
        return int(current["frame_idx"]) != int(previous["frame_idx"]) + 1
    except (KeyError, TypeError, ValueError):
        return True


def _segment_frames(frames: list[dict[str, Any]]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Split one concatenated metadata pickle at every scene boundary."""

    if not frames:
        return []
    segments: list[tuple[int, int, list[dict[str, Any]]]] = []
    start = 0
    for index in range(1, len(frames)):
        if _is_boundary(frames[index - 1], frames[index]):
            segments.append((start, index, frames[start:index]))
            start = index
    segments.append((start, len(frames), frames[start:]))
    return segments


def _window_stats(
    frames: list[dict[str, Any]],
    history_frames: int,
    future_frames: int,
    frame_interval: int,
) -> dict[str, int]:
    """Audit the exact start positions used by NavSIM ``filter_scenes``."""

    window_length = history_frames + future_frames
    stats = {
        "candidate_windows": 0,
        "same_scene_windows": 0,
        "cross_scene_windows": 0,
        "noncontiguous_windows": 0,
        "duplicate_frame_windows": 0,
        "non_increasing_timestamp_windows": 0,
        "route_valid_windows": 0,
    }
    for start in range(0, len(frames), frame_interval):
        window = frames[start : start + window_length]
        if len(window) != window_length:
            continue
        stats["candidate_windows"] += 1

        same_scene = len(_string_set(window, "scene_token")) == 1 and len(
            _string_set(window, "scene_name")
        ) == 1
        if same_scene:
            stats["same_scene_windows"] += 1
        else:
            stats["cross_scene_windows"] += 1

        try:
            frame_indices = [int(frame["frame_idx"]) for frame in window]
            if any(right - left != 1 for left, right in zip(frame_indices, frame_indices[1:])):
                stats["noncontiguous_windows"] += 1
        except (KeyError, TypeError, ValueError):
            stats["noncontiguous_windows"] += 1

        frame_tokens = [str(frame.get("token")) for frame in window]
        if len(set(frame_tokens)) != len(frame_tokens):
            stats["duplicate_frame_windows"] += 1

        try:
            timestamps = [int(frame["timestamp"]) for frame in window]
            if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                stats["non_increasing_timestamp_windows"] += 1
        except (KeyError, TypeError, ValueError):
            stats["non_increasing_timestamp_windows"] += 1

        if bool(window[history_frames - 1].get("roadblock_ids")):
            stats["route_valid_windows"] += 1
    return stats


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _audit_segment(
    source: SourceSpec,
    metadata_path: Path,
    sensor_root: Path | None,
    metric_cache_root: Path | None,
    segment_index: int,
    frame_start: int,
    frame_end: int,
    frames: list[dict[str, Any]],
    output_dir: Path,
    history_frames: int,
    future_frames: int,
    frame_interval: int,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    scene_tokens = _string_set(frames, "scene_token")
    scene_names = _string_set(frames, "scene_name")
    log_names = _string_set(frames, "log_name")
    frame_tokens = {str(frame.get("token")) for frame in frames}
    bad: list[str] = []

    try:
        frame_indices = [int(frame["frame_idx"]) for frame in frames]
        if frame_indices and any(right - left != 1 for left, right in zip(frame_indices, frame_indices[1:])):
            bad.append("noncontiguous_frame_idx")
    except (KeyError, TypeError, ValueError):
        frame_indices = []
        bad.append("invalid_frame_idx")

    if len(scene_tokens) != 1:
        bad.append(f"scene_token_count={len(scene_tokens)}")
    if len(scene_names) != 1:
        bad.append(f"scene_name_count={len(scene_names)}")
    if len(log_names) != 1:
        bad.append(f"log_name_count={len(log_names)}")
    if len(frame_tokens) != len(frames):
        bad.append("duplicate_frame_token")

    try:
        timestamps = [int(frame["timestamp"]) for frame in frames]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            bad.append("non_increasing_timestamp")
    except (KeyError, TypeError, ValueError):
        bad.append("invalid_timestamp")

    sensor_path = sensor_root / metadata_path.stem if sensor_root is not None else None
    sensor_dir_exists = bool(sensor_path and sensor_path.is_dir())
    if source.split is not None and not sensor_dir_exists:
        bad.append("missing_sensor_dir")

    camera_refs = 0
    missing_camera_refs = 0
    camera_key_mismatch_frames = 0
    lidar_refs = 0
    existing_lidar_refs = 0
    for frame in frames:
        cams = frame.get("cams")
        if not isinstance(cams, dict) or set(cams) != set(CAMERA_KEYS):
            camera_key_mismatch_frames += 1
        else:
            for camera_key in CAMERA_KEYS:
                camera = cams.get(camera_key)
                data_path = camera.get("data_path") if isinstance(camera, dict) else None
                camera_refs += 1
                if sensor_root is not None and (
                    not data_path or not (sensor_root / str(data_path)).is_file()
                ):
                    missing_camera_refs += 1
        lidar_path = frame.get("lidar_path")
        if sensor_root is not None and lidar_path:
            lidar_refs += 1
            if (sensor_root / str(lidar_path)).is_file():
                existing_lidar_refs += 1

    if camera_key_mismatch_frames:
        bad.append(f"camera_key_mismatch_frames={camera_key_mismatch_frames}")
    if sensor_root is not None and missing_camera_refs:
        bad.append(f"missing_camera_refs={missing_camera_refs}")

    windows = _window_stats(frames, history_frames, future_frames, frame_interval)
    if windows["cross_scene_windows"]:
        bad.append(f"cross_scene_windows={windows['cross_scene_windows']}")
    if windows["noncontiguous_windows"]:
        bad.append(f"noncontiguous_windows={windows['noncontiguous_windows']}")

    scene_token = _json_values(scene_tokens)[0] if len(scene_tokens) == 1 else None
    scene_name = _json_values(scene_names)[0] if len(scene_names) == 1 else None
    log_name = _json_values(log_names)[0] if len(log_names) == 1 else None
    scene_key = f"{log_name}\t{scene_name}"
    metric_cache_path = metric_cache_root / metadata_path.stem if metric_cache_root else None
    scene_id = f"{metadata_path.stem}__segment_{segment_index:04d}"
    safe_metadata_path = None
    if source.split is not None:
        safe_metadata_path = (
            output_dir
            / "metadata"
            / source.split
            / f"{_safe_name(scene_id)}_{_safe_name(str(scene_token))}.pkl"
        )

    record = {
        "source": source.name,
        "split": source.split,
        "scene_id": scene_id,
        "scene_token": scene_token,
        "scene_name": scene_name,
        "scene_key": scene_key,
        "log_name": log_name,
        "capture_group": _capture_group(metadata_path.stem),
        "metadata_path": str(safe_metadata_path.resolve()) if safe_metadata_path else None,
        "source_metadata_path": str(metadata_path.resolve()),
        "sensor_path": str(sensor_path.resolve()) if sensor_path else None,
        "sensor_dir_exists": sensor_dir_exists,
        "metric_cache_fragment_exists": bool(metric_cache_path and metric_cache_path.exists()),
        "source_frame_start": frame_start,
        "source_frame_end_exclusive": frame_end,
        "frame_count": len(frames),
        "frame_idx_start": frame_indices[0] if frame_indices else None,
        "frame_idx_end": frame_indices[-1] if frame_indices else None,
        "frame_token_count": len(frame_tokens),
        "camera_refs": camera_refs,
        "missing_camera_refs": missing_camera_refs,
        "lidar_refs": lidar_refs,
        "existing_lidar_refs": existing_lidar_refs,
        "windows": windows,
        "bad": bad,
        "usable": source.split in {"train", "test"}
        and not bad
        and sensor_dir_exists
        and windows["candidate_windows"] > 0,
        "selected_for_manifest": False,
    }
    # Only materialize complete, contiguous scene runs.  Short runs are kept
    # in the report but never enter either manifest.
    if safe_metadata_path is not None and record["usable"]:
        safe_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_metadata_path.open("wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    runtime_sets = {
        "scene_tokens": {scene_token} if scene_token is not None else set(),
        "scene_names": {scene_name} if scene_name is not None else set(),
        "scene_keys": {scene_key} if scene_token is not None and scene_name is not None else set(),
        "log_names": {log_name} if log_name is not None else set(),
        "scene_ids": {scene_id},
        "capture_groups": {record["capture_group"]},
        "frame_tokens": frame_tokens,
    }
    return record, runtime_sets


def _audit_source(
    source: SourceSpec,
    metric_cache_root: Path | None,
    output_dir: Path,
    history_frames: int,
    future_frames: int,
    frame_interval: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, set[str]]]], dict[str, Any]]:
    rows: list[tuple[dict[str, Any], dict[str, set[str]]]] = []
    summary = {
        "split": source.split,
        "metadata_dir": str(source.metadata_dir.resolve()),
        "sensor_dir": str(source.sensor_dir.resolve()) if source.sensor_dir else None,
        "metadata_files": 0,
        "scene_units": 0,
        "usable_scene_units": 0,
        "frames": 0,
        "camera_refs": 0,
        "missing_camera_refs": 0,
        "existing_lidar_refs": 0,
        "raw_candidate_windows": 0,
        "raw_cross_scene_windows": 0,
        "raw_noncontiguous_windows": 0,
        "safe_candidate_windows": 0,
        "safe_cross_scene_windows": 0,
        "safe_noncontiguous_windows": 0,
        "route_valid_windows": 0,
        "bad_scene_units": 0,
        "short_scene_units": 0,
        "metric_cache_fragment_count": 0,
    }
    if not source.metadata_dir.is_dir():
        summary["missing"] = True
        return rows, summary

    for metadata_path in sorted(source.metadata_dir.glob("*.pkl")):
        summary["metadata_files"] += 1
        frames = pickle.loads(metadata_path.read_bytes())
        if not isinstance(frames, list):
            raise TypeError(f"{metadata_path} is {type(frames).__name__}, expected list")
        summary["frames"] += len(frames)
        raw_windows = _window_stats(frames, history_frames, future_frames, frame_interval)
        summary["raw_candidate_windows"] += raw_windows["candidate_windows"]
        summary["raw_cross_scene_windows"] += raw_windows["cross_scene_windows"]
        summary["raw_noncontiguous_windows"] += raw_windows["noncontiguous_windows"]

        segments = _segment_frames(frames)
        summary["scene_units"] += len(segments)
        if metric_cache_root is not None and (metric_cache_root / metadata_path.stem).exists():
            summary["metric_cache_fragment_count"] += 1
        for segment_index, (start, end, segment_frames) in enumerate(segments):
            row, runtime_sets = _audit_segment(
                source,
                metadata_path,
                source.sensor_dir,
                metric_cache_root,
                segment_index,
                start,
                end,
                segment_frames,
                output_dir,
                history_frames,
                future_frames,
                frame_interval,
            )
            rows.append((row, runtime_sets))
            summary["camera_refs"] += row["camera_refs"]
            summary["missing_camera_refs"] += row["missing_camera_refs"]
            summary["existing_lidar_refs"] += row["existing_lidar_refs"]
            summary["safe_candidate_windows"] += row["windows"]["candidate_windows"]
            summary["safe_cross_scene_windows"] += row["windows"]["cross_scene_windows"]
            summary["safe_noncontiguous_windows"] += row["windows"]["noncontiguous_windows"]
            summary["route_valid_windows"] += row["windows"]["route_valid_windows"]
            summary["usable_scene_units"] += int(row["usable"])
            summary["bad_scene_units"] += int(bool(row["bad"]))
            summary["short_scene_units"] += int(row["windows"]["candidate_windows"] == 0)
    return rows, summary


def _union(rows: list[tuple[dict[str, Any], dict[str, set[str]]]], key: str) -> set[str]:
    values: set[str] = set()
    for _, runtime_sets in rows:
        values.update(runtime_sets[key])
    return values


def _duplicates(rows: list[tuple[dict[str, Any], dict[str, set[str]]]], key: str) -> list[str]:
    counter: Counter[str] = Counter()
    for _, runtime_sets in rows:
        counter.update(runtime_sets[key])
    return sorted(value for value, count in counter.items() if count > 1)


def _select_unique_scene_rows(
    rows: list[tuple[dict[str, Any], dict[str, set[str]]]]
) -> tuple[list[tuple[dict[str, Any], dict[str, set[str]]]], list[tuple[dict[str, Any], dict[str, set[str]]]]]:
    """Keep one longest contiguous run for each actual scene token.

    A few source scenes contain a missing frame in the middle.  Splitting at
    that gap produces multiple contiguous runs with the same scene token.  We
    keep the longest complete run so a scene is represented once and no
    generated window crosses the missing frame.
    """

    groups: dict[str, list[tuple[dict[str, Any], dict[str, set[str]]]]] = {}
    dropped: list[tuple[dict[str, Any], dict[str, set[str]]]] = []
    selected: list[tuple[dict[str, Any], dict[str, set[str]]]] = []
    for item in rows:
        row = item[0]
        if not row["usable"]:
            dropped.append(item)
            continue
        key = str(row["scene_token"])
        groups.setdefault(key, []).append(item)

    for items in groups.values():
        best = max(
            items,
            key=lambda item: (
                int(item[0]["frame_count"]),
                int(item[0]["windows"]["candidate_windows"]),
                -int(item[0]["source_frame_start"]),
                str(item[0]["scene_id"]),
            ),
        )
        selected.append(best)
        dropped.extend(item for item in items if item is not best)
    selected.sort(key=lambda item: (str(item[0]["source_metadata_path"]), int(item[0]["source_frame_start"])))
    return selected, dropped


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    validation = report["validation"]
    text = f"""# NavSIM camera-only train/test split

The original OpenScene metadata files concatenate multiple NavSIM scene
segments. This output contains one metadata pickle per selected scene unit, plus
JSONL manifests. Camera directories are reused from NVMe and are not copied.

- train: downloaded `trainval`, {counts['train_usable_scene_tokens']} unique scene tokens
- test: original `test`, {counts['test_usable_scene_tokens']} unique scene tokens
- total usable: {counts['usable_scene_tokens']} unique scene tokens
- selected metadata runs: {counts['selected_manifest_rows']}
- metadata-only staging excluded: {counts['staging_scene_units']} scene units

Window audit: history={report['window']['history_frames']},
future={report['window']['future_frames']}, start stride={report['window']['frame_interval']}.
The materialized metadata split makes every selected file a single contiguous
scene run, so it can be passed to the existing NavSIM loader without
concatenating adjacent scenes or crossing a missing frame.

Validation result: **{'PASS' if validation['ok'] else 'FAIL'}**

- safe cross-scene windows: {validation['safe_cross_scene_windows']}
- safe non-contiguous windows: {validation['safe_noncontiguous_windows']}
- missing Camera references: {validation['missing_camera_refs']}
- train/test scene-token overlap: {validation['overlap_scene_tokens']}
- train/test `(log_name, scene_name)` overlap: {validation['overlap_scene_keys']}
- train/test capture-group overlap: {validation['overlap_capture_groups']}
- train/test frame-token overlap: {validation['overlap_frame_tokens']}

Files:

- `metadata/train/*.pkl`
- `metadata/test/*.pkl`
- `train_manifest.jsonl`
- `test_manifest.jsonl`
- `excluded_metadata_only.jsonl`
- `split_report.json`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--metric-cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--future-frames", type=int, default=10)
    parser.add_argument("--frame-interval", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.history_frames < 1 or args.future_frames < 1 or args.frame_interval < 1:
        raise ValueError("history/future/frame-interval must be positive")

    data_root = args.data_root.expanduser().resolve()
    metric_cache_root = (
        args.metric_cache_root.expanduser().resolve()
        if args.metric_cache_root
        else data_root / "metric_cache"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_metadata_dir = output_dir / "metadata"
    if generated_metadata_dir.exists():
        # This directory is generated solely by this script and is rebuilt
        # from the immutable NVMe source metadata on every run.
        shutil.rmtree(generated_metadata_dir)

    all_rows: list[tuple[dict[str, Any], dict[str, set[str]]]] = []
    source_summary: dict[str, Any] = {}
    for source in _source_specs(data_root):
        rows, summary = _audit_source(
            source,
            metric_cache_root,
            output_dir,
            args.history_frames,
            args.future_frames,
            args.frame_interval,
        )
        all_rows.extend(rows)
        source_summary[source.name] = summary

    train_rows = [item for item in all_rows if item[0].get("split") == "train"]
    test_rows = [item for item in all_rows if item[0].get("split") == "test"]
    staging_rows = [item for item in all_rows if item[0].get("split") is None]
    active_rows = train_rows + test_rows
    selected_train_rows, dropped_train_rows = _select_unique_scene_rows(train_rows)
    selected_test_rows, dropped_test_rows = _select_unique_scene_rows(test_rows)
    selected_rows = selected_train_rows + selected_test_rows

    for row, _ in selected_rows:
        row["selected_for_manifest"] = True
    for row, _ in dropped_train_rows + dropped_test_rows:
        if row.get("usable") and row.get("metadata_path"):
            Path(row["metadata_path"]).unlink(missing_ok=True)

    for source_name, source_split, selected in (
        ("downloaded_trainval", "train", selected_train_rows),
        ("official_test", "test", selected_test_rows),
    ):
        summary = source_summary.get(source_name, {})
        source_rows = [item for item in active_rows if item[0].get("split") == source_split]
        usable_rows = [item for item in source_rows if item[0].get("usable")]
        summary["usable_scene_tokens"] = len(_union(usable_rows, "scene_tokens"))
        summary["selected_scene_units"] = len(selected)
        summary["selected_scene_tokens"] = len(_union(selected, "scene_tokens"))
        summary["deduplicated_runs"] = len(usable_rows) - len(selected)
        summary["incomplete_runs"] = sum(not item[0].get("usable") for item in source_rows)

    overlap_counts: dict[str, int] = {}
    overlap_examples: dict[str, list[str]] = {}
    for key in ("scene_tokens", "scene_names", "scene_keys", "log_names", "scene_ids", "capture_groups", "frame_tokens"):
        intersection = _union(selected_train_rows, key) & _union(selected_test_rows, key)
        overlap_counts[key] = len(intersection)
        overlap_examples[key] = sorted(intersection)[:10]

    duplicate_counts = {
        key: len(_duplicates(selected_rows, key))
        for key in ("scene_tokens", "scene_keys", "frame_tokens")
    }
    validation = {
        "ok": (
            bool(selected_train_rows)
            and bool(selected_test_rows)
            and all(row[0]["usable"] for row in selected_rows)
            and sum(row[0]["windows"]["cross_scene_windows"] for row in selected_rows) == 0
            and sum(row[0]["windows"]["noncontiguous_windows"] for row in selected_rows) == 0
            and sum(row[0]["missing_camera_refs"] for row in selected_rows) == 0
            and all(overlap_counts[key] == 0 for key in ("scene_tokens", "scene_keys", "scene_ids", "capture_groups", "frame_tokens"))
            and all(value == 0 for value in duplicate_counts.values())
        ),
        "safe_cross_scene_windows": sum(row[0]["windows"]["cross_scene_windows"] for row in selected_rows),
        "safe_noncontiguous_windows": sum(row[0]["windows"]["noncontiguous_windows"] for row in selected_rows),
        "missing_camera_refs": sum(row[0]["missing_camera_refs"] for row in selected_rows),
        "overlap_scene_tokens": overlap_counts["scene_tokens"],
        "overlap_scene_names_informational": overlap_counts["scene_names"],
        "overlap_scene_keys": overlap_counts["scene_keys"],
        "overlap_log_names": overlap_counts["log_names"],
        "overlap_scene_ids": overlap_counts["scene_ids"],
        "overlap_capture_groups": overlap_counts["capture_groups"],
        "overlap_frame_tokens": overlap_counts["frame_tokens"],
        "duplicate_scene_tokens_within_split": duplicate_counts["scene_tokens"],
        "duplicate_scene_keys_within_split": duplicate_counts["scene_keys"],
        "duplicate_frame_tokens_within_split": duplicate_counts["frame_tokens"],
        "overlap_examples": overlap_examples,
    }

    report = {
        "data_root": str(data_root),
        "metric_cache_root": str(metric_cache_root),
        "window": {
            "history_frames": args.history_frames,
            "future_frames": args.future_frames,
            "frame_interval": args.frame_interval,
            "window_length": args.history_frames + args.future_frames,
        },
        "counts": {
            "train_source_scene_units": source_summary.get("downloaded_trainval", {}).get("scene_units", 0),
            "test_source_scene_units": source_summary.get("official_test", {}).get("scene_units", 0),
            "train_usable_contiguous_runs": sum(row[0]["usable"] for row in train_rows),
            "test_usable_contiguous_runs": sum(row[0]["usable"] for row in test_rows),
            "train_usable_scene_tokens": len(_union(selected_train_rows, "scene_tokens")),
            "test_usable_scene_tokens": len(_union(selected_test_rows, "scene_tokens")),
            "usable_scene_tokens": len(_union(selected_rows, "scene_tokens")),
            "selected_manifest_rows": len(selected_rows),
            "staging_scene_units": len(staging_rows),
            "all_metadata_scene_units": sum(summary.get("scene_units", 0) for summary in source_summary.values()),
            "active_metadata_files": source_summary.get("official_test", {}).get("metadata_files", 0)
            + source_summary.get("downloaded_trainval", {}).get("metadata_files", 0),
            "all_metadata_files": sum(summary.get("metadata_files", 0) for summary in source_summary.values()),
            "metadata_only_staging_files": source_summary.get("metadata_only_staging", {}).get("metadata_files", 0),
        },
        "sources": source_summary,
        "validation": validation,
        "records": [row for row, _ in all_rows],
    }

    _write_jsonl(output_dir / "train_manifest.jsonl", (row for row, _ in selected_train_rows))
    _write_jsonl(output_dir / "test_manifest.jsonl", (row for row, _ in selected_test_rows))
    _write_jsonl(output_dir / "excluded_metadata_only.jsonl", (row for row, _ in staging_rows))
    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(output_dir, report)

    print(json.dumps({"counts": report["counts"], "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

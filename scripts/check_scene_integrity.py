#!/usr/bin/env python3
"""Audit official NAVSIM VideoPress outputs for scene/rank/method mix-ups.

The official evaluator owns scene loading and metric scoring.  This audit is
deliberately independent of the evaluator implementation: it compares the
official CSV, the runner's per-scene event journal, and the joined records.
It is intended to be run after a smoke or full suite, and is safe to rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_CSV_RE = re.compile(r"^pdm_score_.*\.csv$")
_EVENT_RE = re.compile(r"press_events\.rank(\d+)\.jsonl$")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        value["_line_no"] = line_no
        rows.append(value)
    return rows


def _latest_csv(method_dir: Path) -> Path | None:
    candidates = sorted(path for path in method_dir.iterdir() if _CSV_RE.match(path.name))
    return candidates[-1] if candidates else None


def _tokens_from_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if str(row.get("token", "")) != "average"]
    tokens = [str(row.get("token", "")) for row in rows]
    return tokens, {token: row for token, row in zip(tokens, rows)}


def _duplicate_items(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _rank_from_event_path(path: Path) -> int:
    match = _EVENT_RE.search(path.name)
    if match is None:
        raise ValueError(f"unexpected event journal name: {path.name}")
    return int(match.group(1))


def _method_dirs(root: Path) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for round_dir in sorted(root.glob("round[0-9][0-9]")):
        if not round_dir.is_dir():
            continue
        for method_dir in sorted(path for path in round_dir.iterdir() if path.is_dir()):
            output.append((round_dir.name, method_dir))
    return output


def _check_score_cache_payload(path: Path, token: str) -> str | None:
    """Validate the scene identity embedded in one frozen-score payload."""

    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - reported by the caller
        return f"cannot load score cache {path}: {exc}"
    if not isinstance(payload, dict):
        return f"score cache {path} is not a dictionary payload"
    key = payload.get("key") if isinstance(payload.get("key"), dict) else {}
    key_token = key.get("scene_token")
    if key_token is not None and str(key_token) != token:
        return f"score cache {path} key scene_token={key_token!r} differs from event scene_token={token!r}"
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata_token = metadata.get("scene_token")
    if metadata_token is not None and str(metadata_token) != token:
        return f"score cache {path} metadata scene_token={metadata_token!r} differs from event scene_token={token!r}"
    return None


def _check_event_runtime(
    *,
    event_path: Path,
    item: dict[str, Any],
    token: str,
    expected_segment_length: int | None,
) -> list[str]:
    """Check scene-window identity and candidate-position metadata."""

    errors: list[str] = []
    line = item.get("_line_no")
    location = f"{event_path.name}:{line}"
    segment = item.get("segment") if isinstance(item.get("segment"), dict) else None
    if segment is None:
        errors.append(f"{location}: missing segment identity")
    else:
        required = (
            "segment_scene_token",
            "segment_scene_name",
            "segment_frame_idx_start",
            "segment_frame_idx_end",
            "segment_frame_count",
            "segment_frame_idx_contiguous",
        )
        missing = [key for key in required if segment.get(key) is None or segment.get(key) == ""]
        if missing:
            errors.append(f"{location}: segment metadata missing {missing}")
        if segment.get("segment_frame_idx_contiguous") is not True:
            errors.append(f"{location}: segment frame indices are not contiguous")
        try:
            frame_count = int(segment.get("segment_frame_count"))
            frame_start = int(segment.get("segment_frame_idx_start"))
            frame_end = int(segment.get("segment_frame_idx_end"))
            if frame_count <= 0 or frame_end - frame_start + 1 != frame_count:
                errors.append(
                    f"{location}: inconsistent segment frame range "
                    f"start={frame_start} end={frame_end} count={frame_count}"
                )
            if expected_segment_length is not None and frame_count != expected_segment_length:
                errors.append(
                    f"{location}: segment frame count={frame_count}, "
                    f"expected={expected_segment_length}"
                )
        except (TypeError, ValueError):
            errors.append(f"{location}: invalid segment frame range/count")

    runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
    last = runtime.get("last") if isinstance(runtime.get("last"), dict) else {}
    if int(runtime.get("event_count", 0) or 0) <= 0:
        # NoPress deliberately has no model hook event, but its outer segment
        # identity above is still mandatory.
        return errors

    runtime_token = last.get("scene_token")
    if runtime_token is None or str(runtime_token) != token:
        errors.append(f"{location}: runtime last scene_token={runtime_token!r} differs from {token!r}")

    scene_tokens = last.get("scene_tokens")
    if scene_tokens is not None:
        if not isinstance(scene_tokens, list) or any(str(value) != token for value in scene_tokens):
            errors.append(f"{location}: runtime scene_tokens are not all {token!r}")

    configured = last.get("configured_domain")
    resolved = last.get("resolved_domain")
    if configured is None or resolved is None:
        errors.append(f"{location}: runtime domain identity is incomplete")
    if configured is not None and resolved is not None and str(configured) != str(resolved):
        errors.append(f"{location}: domain override detected {configured!r}->{resolved!r}")
    if last.get("domain_override") is True:
        errors.append(f"{location}: domain_override=true")

    candidate_start = last.get("candidate_start")
    candidate_end = last.get("candidate_end")
    n_candidate = last.get("n_candidate")
    n_history = last.get("n_history")
    if any(value is None for value in (candidate_start, candidate_end, n_candidate, n_history)):
        errors.append(f"{location}: runtime candidate range metadata is incomplete")
    if str(resolved) == "last_history" and all(value is not None for value in (candidate_start, candidate_end, n_candidate, n_history)):
        try:
            start = int(candidate_start)
            end = int(candidate_end)
            candidates = int(n_candidate)
            history = int(n_history)
            if start != history - candidates or end != history or end - start != candidates:
                errors.append(
                    f"{location}: last_history candidate range [{start},{end}) does not match "
                    f"history={history} candidates={candidates}"
                )
        except (TypeError, ValueError):
            errors.append(f"{location}: invalid candidate range metadata")

    selection_operator = str(last.get("operator", ""))
    if selection_operator != "kv_merge" and last.get("selection_candidate_valid") is not True:
        errors.append(f"{location}: selection_candidate_valid is not true")
    if last.get("selection_candidate_valid") is False:
        errors.append(f"{location}: selected global positions leave the declared candidate domain")
    if selection_operator != "kv_merge" and last.get("selection_candidate_unique") is not True:
        errors.append(f"{location}: selection_candidate_unique is not true")
    if last.get("selection_candidate_unique") is False:
        errors.append(f"{location}: selected global positions contain duplicates")
    selected_min = last.get("selected_global_min")
    selected_max = last.get("selected_global_max")
    if selected_min is not None and selected_max is not None and candidate_start is not None and candidate_end is not None:
        try:
            if not int(candidate_start) <= int(selected_min) <= int(selected_max) < int(candidate_end):
                errors.append(
                    f"{location}: selected range [{selected_min},{selected_max}] is outside "
                    f"candidate range [{candidate_start},{candidate_end})"
                )
        except (TypeError, ValueError):
            errors.append(f"{location}: invalid selected/candidate position metadata")
    if int(runtime.get("invalid_selection_count", 0) or 0) != 0:
        errors.append(f"{location}: runtime invalid_selection_count is nonzero")
    if int(runtime.get("noncontiguous_segment_count", 0) or 0) != 0:
        errors.append(f"{location}: runtime noncontiguous_segment_count is nonzero")
    return errors


def _check_method(
    *,
    method_dir: Path,
    round_name: str,
    expected_world_size: int | None,
    expected_segment_length: int | None,
    check_score_cache: bool,
) -> dict[str, Any]:
    method = method_dir.name
    errors: list[str] = []
    warnings: list[str] = []
    csv_path = _latest_csv(method_dir)
    if csv_path is None:
        return {
            "round": round_name,
            "method": method,
            "ok": False,
            "errors": ["no official pdm_score_*.csv found"],
            "warnings": [],
            "n_csv": 0,
            "n_events": 0,
            "n_records": 0,
        }

    try:
        csv_tokens, csv_rows = _tokens_from_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        return {
            "round": round_name,
            "method": method,
            "ok": False,
            "errors": [f"cannot read official CSV: {exc}"],
            "warnings": [],
            "n_csv": 0,
            "n_events": 0,
            "n_records": 0,
        }

    csv_duplicates = _duplicate_items(csv_tokens)
    if csv_duplicates:
        errors.append(f"official CSV has duplicate scene tokens: {csv_duplicates[:5]}")

    event_rows: list[tuple[int, dict[str, Any]]] = []
    event_tokens_by_rank: dict[int, list[str]] = {}
    event_locations: dict[str, list[int]] = {}
    for event_path in sorted(method_dir.glob("press_events.rank*.jsonl")):
        rank = _rank_from_event_path(event_path)
        try:
            rows = _read_jsonl(event_path)
        except Exception as exc:  # pragma: no cover - defensive CLI reporting
            errors.append(f"cannot read {event_path.name}: {exc}")
            continue
        rank_tokens = event_tokens_by_rank.setdefault(rank, [])
        for item in rows:
            token = str(item.get("scene_token", ""))
            rank_tokens.append(token)
            event_rows.append((rank, item))
            event_locations.setdefault(token, []).append(rank)
            if item.get("valid") is not True:
                errors.append(
                    f"{event_path.name}:{item.get('_line_no')}: runtime scene evaluation is invalid: "
                    f"{item.get('error')!r}"
                )
            if not token or token == "unknown":
                errors.append(f"{event_path.name}:{item.get('_line_no')}: missing active scene token")
            if str(item.get("method", "")) != method:
                errors.append(
                    f"{event_path.name}:{item.get('_line_no')}: method field {item.get('method')!r} "
                    f"does not match directory {method!r}"
                )
            runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
            last = runtime.get("last") if isinstance(runtime.get("last"), dict) else {}
            runtime_token = last.get("scene_token")
            if runtime_token is not None and str(runtime_token) != token:
                errors.append(
                    f"{event_path.name}:{item.get('_line_no')}: runtime scene_token {runtime_token!r} "
                    f"differs from event scene_token {token!r}"
                )
            errors.extend(
                _check_event_runtime(
                    event_path=event_path,
                    item=item,
                    token=token,
                    expected_segment_length=expected_segment_length,
                )
            )
            for detail in item.get("probe_details", []) or []:
                if not isinstance(detail, dict):
                    continue
                cache_path = detail.get("score_cache")
                if cache_path and not Path(str(cache_path)).exists():
                    errors.append(
                        f"{event_path.name}:{item.get('_line_no')}: missing probe score cache {cache_path}"
                    )
                elif cache_path and check_score_cache:
                    cache_error = _check_score_cache_payload(Path(str(cache_path)), token)
                    if cache_error:
                        errors.append(f"{event_path.name}:{item.get('_line_no')}: {cache_error}")

    duplicate_event_tokens = [token for token, ranks in event_locations.items() if len(ranks) > 1]
    duplicate_event_tokens.extend(
        token
        for tokens in event_tokens_by_rank.values()
        for token in _duplicate_items(tokens)
        if token not in duplicate_event_tokens
    )
    if duplicate_event_tokens:
        errors.append(f"event journal has repeated scene tokens: {duplicate_event_tokens[:5]}")

    event_tokens = [str(item.get("scene_token", "")) for _, item in event_rows]
    missing_events = sorted(set(csv_tokens) - set(event_tokens))
    extra_events = sorted(set(event_tokens) - set(csv_tokens))
    if missing_events:
        errors.append(f"event journal missing {len(missing_events)} official scenes; first={missing_events[:3]}")
    if extra_events:
        errors.append(f"event journal contains {len(extra_events)} non-official scenes; first={extra_events[:3]}")

    if expected_world_size and expected_world_size > 0:
        expected_by_rank = {
            rank: [token for idx, token in enumerate(sorted(csv_tokens)) if idx % expected_world_size == rank]
            for rank in range(expected_world_size)
        }
        for rank, actual in event_tokens_by_rank.items():
            expected = expected_by_rank.get(rank)
            if expected is None:
                errors.append(f"event journal uses unexpected rank {rank}")
            elif actual != expected:
                # Keep this diagnostic compact; a mismatch is exactly the
                # scene-switch / shard-order issue this audit is meant to find.
                first_bad = next(
                    (idx for idx, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                    min(len(actual), len(expected)),
                )
                errors.append(
                    f"rank {rank} scene order/shard mismatch at index {first_bad}: "
                    f"actual={actual[first_bad:first_bad + 2]} expected={expected[first_bad:first_bad + 2]}"
                )
        for rank, expected in expected_by_rank.items():
            if expected and rank not in event_tokens_by_rank:
                errors.append(f"rank {rank} has no event journal but should process {len(expected)} scenes")

        for rank, item in event_rows:
            token = str(item.get("scene_token", ""))
            row = csv_rows.get(token)
            if row is None:
                continue
            csv_rank = str(row.get("rank", "")).strip()
            if csv_rank and csv_rank.isdigit() and int(csv_rank) != rank:
                errors.append(f"scene {token} CSV rank={csv_rank} but event journal rank={rank}")

    record_path = method_dir / "records.jsonl"
    record_rows: list[dict[str, Any]] = []
    if not record_path.exists():
        errors.append("records.jsonl is missing")
    else:
        try:
            record_rows = _read_jsonl(record_path)
        except Exception as exc:  # pragma: no cover - defensive CLI reporting
            errors.append(f"cannot read records.jsonl: {exc}")
    record_tokens = [str(item.get("scene_token", "")) for item in record_rows]
    record_duplicates = _duplicate_items(record_tokens)
    if record_duplicates:
        errors.append(f"records.jsonl has duplicate scene tokens: {record_duplicates[:5]}")
    if set(record_tokens) != set(csv_tokens):
        errors.append(
            f"records/official CSV token mismatch: records_only={len(set(record_tokens) - set(csv_tokens))}, "
            f"csv_only={len(set(csv_tokens) - set(record_tokens))}"
        )
    for item in record_rows:
        token = str(item.get("scene_token", ""))
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata_token = metadata.get("scene_token")
        if metadata_token is not None and str(metadata_token) != token:
            errors.append(f"record scene {token} has metadata scene_token={metadata_token!r}")

    if len(event_rows) != len(csv_tokens):
        warnings.append(f"event count {len(event_rows)} differs from official scene count {len(csv_tokens)}")

    return {
        "round": round_name,
        "method": method,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "csv": str(csv_path),
        "n_csv": len(csv_tokens),
        "n_events": len(event_rows),
        "n_records": len(record_rows),
        "rank_counts": {str(rank): len(tokens) for rank, tokens in sorted(event_tokens_by_rank.items())},
    }


def validate_suite(root: Path, *, check_score_cache: bool = True) -> dict[str, Any]:
    manifest_path = root / "suite_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    world_size = manifest.get("world_size")
    try:
        expected_world_size = int(world_size) if world_size is not None else None
    except (TypeError, ValueError):
        expected_world_size = None
    scene_filter = manifest.get("scene_filter") if isinstance(manifest.get("scene_filter"), dict) else {}
    try:
        expected_segment_length = int(scene_filter["window_length"])
    except (KeyError, TypeError, ValueError):
        expected_segment_length = None

    results = [
        _check_method(
            round_name=round_name,
            method_dir=method_dir,
            expected_world_size=expected_world_size,
            expected_segment_length=expected_segment_length,
            check_score_cache=check_score_cache,
        )
        for round_name, method_dir in _method_dirs(root)
    ]
    expected_methods = manifest.get("methods")
    if isinstance(expected_methods, list):
        found = {(item["round"], item["method"]) for item in results}
        for round_dir in sorted(root.glob("round[0-9][0-9]")):
            for method in expected_methods:
                if (round_dir.name, str(method)) not in found:
                    results.append(
                        {
                            "round": round_dir.name,
                            "method": str(method),
                            "ok": False,
                            "errors": ["method output directory or official CSV is missing"],
                            "warnings": [],
                            "n_csv": 0,
                            "n_events": 0,
                            "n_records": 0,
                        }
                    )

    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else None
    suite_errors: list[str] = []
    if coverage and int(coverage.get("missing_metric_cache_tokens", 0) or 0) != 0:
        suite_errors.append("suite manifest records missing metric-cache scenes")
    suite_errors.extend(error for item in results for error in item.get("errors", []))
    report = {
        "suite_root": str(root.resolve()),
        "ok": not suite_errors and bool(results),
        "world_size": expected_world_size,
        "coverage": coverage,
        "method_count": len(results),
        "failed_method_count": sum(not bool(item.get("ok")) for item in results),
        "suite_errors": suite_errors,
        "methods": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument(
        "--skip-score-cache",
        action="store_true",
        help="Only check event/CSV/records identities; skip loading frozen probe payloads.",
    )
    args = parser.parse_args()
    root = args.suite_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"suite root does not exist: {root}")
    report = validate_suite(root, check_score_cache=not args.skip_score_cache)
    output_path = root / "scene_integrity.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

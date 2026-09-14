#!/usr/bin/env python3
"""Read-only audit of NAVSIM split leakage and future validation eligibility.

The script never creates split manifests or modifies source data.  It reports
pairwise overlap at capture/scene/frame granularity, hashes every input
manifest, and computes balanced capture-group folds that may be used for *new*
selector development after retraining from scratch.  Historical experiment
exposure is intentionally reported separately from literal row overlap: a
disjoint manifest can still be unsuitable as a fresh confirmation set once its
outcomes have informed model or threshold decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = FRAMEWORK_ROOT / "outputs"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_tokens(rows: list[dict[str, Any]]) -> tuple[set[str], list[str]]:
    tokens: set[str] = set()
    failures: list[str] = []
    for row in rows:
        metadata_value = row.get("metadata_path")
        if not metadata_value:
            failures.append(f"{row.get('scene_id', row.get('scene_token'))}: missing metadata_path")
            continue
        path = Path(str(metadata_value))
        try:
            frames = pickle.loads(path.read_bytes())
            if not isinstance(frames, list):
                raise TypeError(f"expected list, got {type(frames).__name__}")
            tokens.update(str(frame.get("token")) for frame in frames)
        except Exception as exc:  # the report must retain every audit failure
            failures.append(f"{path}: {exc!r}")
    return tokens, failures


def _sets(rows: list[dict[str, Any]]) -> tuple[dict[str, set[str]], list[str]]:
    frames, failures = _frame_tokens(rows)
    values = {
        "capture_groups": {str(row.get("capture_group")) for row in rows},
        "scene_tokens": {str(row.get("scene_token")) for row in rows},
        "scene_keys": {str(row.get("scene_key")) for row in rows},
        "frame_tokens": frames,
    }
    return values, failures


def _balanced_capture_folds(
    rows: list[dict[str, Any]], *, folds: int, seed: int
) -> list[dict[str, Any]]:
    """Greedy bin packing by scene count with a hash-only stable tie breaker."""

    counts = Counter(str(row["capture_group"]) for row in rows)
    groups = sorted(
        counts,
        key=lambda group: (
            -counts[group],
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
        ),
    )
    bins: list[list[str]] = [[] for _ in range(folds)]
    loads = [0 for _ in range(folds)]
    for group in groups:
        fold = min(
            range(folds),
            key=lambda index: (
                loads[index],
                hashlib.sha256(f"{seed}:fold:{index}".encode("utf-8")).hexdigest(),
            ),
        )
        bins[fold].append(group)
        loads[fold] += counts[group]
    return [
        {
            "fold": index,
            "scenes": loads[index],
            "capture_groups": len(groups_in_fold),
            "capture_group_ids": sorted(groups_in_fold),
        }
        for index, groups_in_fold in enumerate(bins)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-train",
        type=Path,
        default=DEFAULT_ROOT / "selector_capture_split_20260910/selector_train_manifest.jsonl",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_ROOT / "selector_capture_split_20260910/selector_calibration_manifest.jsonl",
    )
    parser.add_argument(
        "--official-test",
        type=Path,
        default=DEFAULT_ROOT / "navsim_split_audit/test_manifest.jsonl",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    paths = {
        "selector_train_3190": args.selector_train.resolve(),
        "calibration_578": args.calibration.resolve(),
        "official_test_1920": args.official_test.resolve(),
    }
    rows = {name: _read_jsonl(path) for name, path in paths.items()}
    set_values: dict[str, dict[str, set[str]]] = {}
    metadata_failures: dict[str, list[str]] = {}
    for name, manifest_rows in rows.items():
        set_values[name], metadata_failures[name] = _sets(manifest_rows)

    pairwise: dict[str, dict[str, Any]] = {}
    names = list(paths)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            key = f"{left}__vs__{right}"
            pairwise[key] = {}
            for dimension in ("capture_groups", "scene_tokens", "scene_keys", "frame_tokens"):
                intersection = set_values[left][dimension] & set_values[right][dimension]
                pairwise[key][dimension] = {
                    "count": len(intersection),
                    "examples": sorted(intersection)[:10],
                }

    folds = _balanced_capture_folds(
        rows["selector_train_3190"], folds=args.folds, seed=args.seed
    )
    literal_overlap_ok = all(
        value["count"] == 0
        for comparison in pairwise.values()
        for value in comparison.values()
    )
    report = {
        "ok": literal_overlap_ok and not any(metadata_failures.values()),
        "scope": "read-only split/leakage audit; no GPU inference and no source mutation",
        "manifests": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "scenes": len(rows[name]),
                "capture_groups": len(set_values[name]["capture_groups"]),
                "scene_tokens": len(set_values[name]["scene_tokens"]),
                "scene_keys": len(set_values[name]["scene_keys"]),
                "frame_tokens": len(set_values[name]["frame_tokens"]),
                "metadata_failures": metadata_failures[name][:20],
            }
            for name, path in paths.items()
        },
        "pairwise_overlap": pairwise,
        "historical_exposure": {
            "selector_train_3190": {
                "exposed": True,
                "reason": "used to train both C4 selector replicas; only freshly retrained out-of-fold models may treat a held-out fold as model-disjoint development data",
                "eligible_for_global_final_confirmation": False,
            },
            "calibration_578_route_valid_577": {
                "exposed": True,
                "reason": "complete C4 triad outcomes were inspected across two checkpoints and five sampling seeds; earlier calibration probes also informed the workflow",
                "eligible_for_further_tuning": False,
                "eligible_for_global_final_confirmation": False,
            },
            "official_test_1920_navtest_7876": {
                "exposed": True,
                "reason": "repeated full test/navtest evaluations informed method, layer, persistence and threshold decisions",
                "eligible_for_further_tuning": False,
                "eligible_for_global_final_confirmation": False,
            },
            "locally_available_pristine_confirmation_scenes": 0,
        },
        "development_only_capture_folds": {
            "source": "selector_train_3190",
            "seed": args.seed,
            "algorithm": "largest capture groups first; assign to least-loaded fold; SHA-256 tie breaking",
            "warning": "These folds are valid only for newly scratch-trained out-of-fold selectors and development estimates. They are not a pristine final confirmation set.",
            "folds": folds,
        },
        "required_final_confirmation_source": (
            "newly acquired, never-inspected route groups (for example Bench2Drive), "
            "partitioned and cryptographically frozen before any model inference"
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

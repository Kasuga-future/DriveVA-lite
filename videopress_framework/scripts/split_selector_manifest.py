#!/usr/bin/env python3
"""Create deterministic capture-disjoint Selector train/calibration manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forbidden-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0,1)")

    rows = read_jsonl(args.manifest.resolve())
    forbidden = read_jsonl(args.forbidden_manifest.resolve())
    required = {"scene_token", "capture_group"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("every training row needs scene_token and capture_group")
    groups = Counter(str(row["capture_group"]) for row in rows)
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{args.seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    target = round(len(rows) * args.validation_fraction)
    validation_groups: set[str] = set()
    selected = 0
    for group in ordered:
        if selected >= target and validation_groups:
            break
        validation_groups.add(group)
        selected += groups[group]

    validation = [row for row in rows if str(row["capture_group"]) in validation_groups]
    training = [row for row in rows if str(row["capture_group"]) not in validation_groups]
    train_tokens = {str(row["scene_token"]) for row in training}
    validation_tokens = {str(row["scene_token"]) for row in validation}
    forbidden_tokens = {str(row["scene_token"]) for row in forbidden}
    train_groups = {str(row["capture_group"]) for row in training}
    forbidden_groups = {str(row.get("capture_group")) for row in forbidden}
    checks = {
        "train_validation_scene_overlap": len(train_tokens & validation_tokens),
        "train_validation_capture_overlap": len(train_groups & validation_groups),
        "train_forbidden_scene_overlap": len(train_tokens & forbidden_tokens),
        "validation_forbidden_scene_overlap": len(validation_tokens & forbidden_tokens),
        "train_forbidden_capture_overlap": len(train_groups & forbidden_groups),
        "validation_forbidden_capture_overlap": len(validation_groups & forbidden_groups),
    }
    if any(checks.values()):
        raise RuntimeError(f"split leakage detected: {checks}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "selector_train_manifest.jsonl", training)
    write_jsonl(args.output_root / "selector_calibration_manifest.jsonl", validation)
    report = {
        "source_manifest": str(args.manifest.resolve()),
        "forbidden_manifest": str(args.forbidden_manifest.resolve()),
        "seed": args.seed,
        "validation_fraction_requested": args.validation_fraction,
        "train_scenes": len(training),
        "calibration_scenes": len(validation),
        "train_capture_groups": len(train_groups),
        "calibration_capture_groups": len(validation_groups),
        "validation_fraction_actual": len(validation) / len(rows),
        "candidate_windows_train": sum(
            int(row.get("windows", {}).get("candidate_windows", 0)) for row in training
        ),
        "candidate_windows_calibration": sum(
            int(row.get("windows", {}).get("candidate_windows", 0)) for row in validation
        ),
        "checks": checks,
        "calibration_groups": sorted(validation_groups),
    }
    (args.output_root / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

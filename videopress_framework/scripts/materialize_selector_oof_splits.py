#!/usr/bin/env python3
"""Materialize capture-disjoint OOF selector-development manifests.

The source 3,190-scene manifest has already been used by historical C4 models,
so these files are *internal development* splits only.  Every new OOF selector
must start from scratch.  Fold 4 is sealed from the new development cycle until
the folds 0--3 configuration is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    FRAMEWORK_ROOT
    / "outputs/selector_capture_split_20260910/selector_train_manifest.jsonl"
)
DEFAULT_AUDIT = (
    FRAMEWORK_ROOT / "outputs/independent_validation_leakage_audit_20260914.json"
)
DEFAULT_OUTPUT = FRAMEWORK_ROOT / "outputs/selector_oof_split_20260914"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    audit_path = args.audit.resolve()
    output = args.output_dir.resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    audit = json.loads(audit_path.read_text())
    fold_specs = audit["development_only_capture_folds"]["folds"]
    if len(fold_specs) != 5:
        raise ValueError(f"expected five frozen folds, got {len(fold_specs)}")
    if len(rows) != 3190 or len({str(row["scene_token"]) for row in rows}) != 3190:
        raise ValueError("source must contain exactly 3,190 unique scene tokens")
    if sha256(source) != audit["manifests"]["selector_train_3190"]["sha256"]:
        raise ValueError("source manifest hash differs from the audited source")

    group_to_fold: dict[str, int] = {}
    for spec in fold_specs:
        fold = int(spec["fold"])
        for group in spec["capture_group_ids"]:
            if group in group_to_fold:
                raise ValueError(f"capture group occurs in multiple folds: {group}")
            group_to_fold[str(group)] = fold
    source_groups = {str(row["capture_group"]) for row in rows}
    if source_groups != set(group_to_fold):
        raise ValueError("audited capture groups do not exactly cover source manifest")

    fold_rows = {
        fold: [row for row in rows if group_to_fold[str(row["capture_group"])] == fold]
        for fold in range(5)
    }
    for spec in fold_specs:
        fold = int(spec["fold"])
        if len(fold_rows[fold]) != int(spec["scenes"]):
            raise ValueError(f"fold {fold} count differs from frozen audit")

    paths: dict[str, Path] = {}
    for fold in range(5):
        name = f"fold{fold}.jsonl"
        paths[name] = output / name
        write_jsonl_atomic(paths[name], fold_rows[fold])

    # OOF development uses one of folds 0--3 as validation, two/three other
    # development folds as training, and never exposes sealed fold 4.
    for held_out in range(4):
        train = [row for fold in range(4) if fold != held_out for row in fold_rows[fold]]
        train_name = f"oof{held_out}_train.jsonl"
        val_name = f"oof{held_out}_validation.jsonl"
        paths[train_name] = output / train_name
        paths[val_name] = output / val_name
        write_jsonl_atomic(paths[train_name], train)
        write_jsonl_atomic(paths[val_name], fold_rows[held_out])

    dev_train = [row for fold in range(4) for row in fold_rows[fold]]
    paths["frozen_dev_train_folds0_3.jsonl"] = output / "frozen_dev_train_folds0_3.jsonl"
    paths["sealed_internal_confirmation_fold4.jsonl"] = (
        output / "sealed_internal_confirmation_fold4.jsonl"
    )
    write_jsonl_atomic(paths["frozen_dev_train_folds0_3.jsonl"], dev_train)
    write_jsonl_atomic(
        paths["sealed_internal_confirmation_fold4.jsonl"], fold_rows[4]
    )

    file_rows = {}
    for name, path in sorted(paths.items()):
        materialized = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        file_rows[name] = {
            "path": str(path),
            "scenes": len(materialized),
            "capture_groups": len({str(row["capture_group"]) for row in materialized}),
            "sha256": sha256(path),
        }
    confirmation_groups = {str(row["capture_group"]) for row in fold_rows[4]}
    dev_groups = {str(row["capture_group"]) for row in dev_train}
    if confirmation_groups & dev_groups:
        raise RuntimeError("sealed confirmation overlaps dev train by capture group")
    report = {
        "ok": True,
        "scope": "internal scratch-selector development; not pristine final confirmation",
        "source": {"path": str(source), "sha256": sha256(source), "scenes": len(rows)},
        "policy": {
            "oof_development_folds": [0, 1, 2, 3],
            "sealed_internal_confirmation_fold": 4,
            "selector_initialization": "scratch_only",
            "exposed_calibration_and_navtest_forbidden_for_tuning": True,
        },
        "fold_scene_counts": {str(i): len(fold_rows[i]) for i in range(5)},
        "dev_confirmation_capture_overlap": 0,
        "files": file_rows,
        "source_capture_group_counts": dict(sorted(Counter(
            str(row["capture_group"]) for row in rows
        ).items())),
    }
    report_path = output / "split_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

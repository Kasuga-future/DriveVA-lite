#!/usr/bin/env python3
"""Build a log/scene-disjoint POC split from an existing official-test record list."""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import random
from pathlib import Path
from typing import Any


COMMANDS = {0: "left", 1: "straight", 2: "right"}


def _subset(
    logs: list[str], counts: dict[str, int], command_masks: dict[str, int], target: int
) -> list[str]:
    required_mask = (1 << len(COMMANDS)) - 1
    dp: dict[tuple[int, int], list[str]] = {(0, 0): []}
    for log in logs:
        amount = counts[log]
        for (total, mask), chosen in list(dp.items())[::-1]:
            new_total = total + amount
            state = (new_total, mask | command_masks[log])
            if new_total <= target and state not in dp:
                dp[state] = chosen + [log]
    result = dp.get((target, required_mask))
    if result is None:
        raise RuntimeError(f"No exact log-level subset with all commands for target={target}")
    return result


def _command(value: Any) -> str:
    try:
        values = list(value)
        for index, item in enumerate(values[:3]):
            if int(item) == 1:
                return COMMANDS[index]
    except Exception:
        pass
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--train-anchors", type=int, default=200)
    parser.add_argument("--val-anchors", type=int, default=50)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for path in sorted(args.metadata_root.glob("*.pkl")):
        with path.open("rb") as handle:
            rows.extend(pickle.load(handle))
    by_token = {str(row["token"]): row for row in rows}

    anchors: list[dict[str, Any]] = []
    for record in records:
        anchor_id = str(record["scene_token"])
        row = by_token.get(anchor_id)
        if row is None:
            raise RuntimeError(f"Missing metadata for anchor {anchor_id}")
        anchors.append(
            {
                "anchor_id": anchor_id,
                "scene_token": str(row["scene_token"]),
                "scene_name": str(row["scene_name"]),
                "log_name": str(row["log_name"]),
                "frame_id": int(row["frame_idx"]),
                "timestamp": int(row["timestamp"]),
                "ego_state_raw": [float(value) for value in row.get("ego_dynamic_state", [])],
                "ego_velocity_model_input": [
                    float(value) for value in list(row.get("ego_dynamic_state", []))[:2]
                ],
                "driving_command_raw": [int(value) for value in row.get("driving_command", [])],
                "command_category": _command(row.get("driving_command", [])),
            }
        )

    by_log: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for anchor in anchors:
        by_log[anchor["log_name"]].append(anchor)
    counts = {log: len(items) for log, items in by_log.items()}
    command_masks = {
        log: sum(1 << index for index, name in COMMANDS.items() if any(item["command_category"] == name for item in items))
        for log, items in by_log.items()
    }
    ordered_logs = sorted(counts)
    random.Random(args.seed).shuffle(ordered_logs)
    train_logs = _subset(ordered_logs, counts, command_masks, args.train_anchors)
    val_logs = _subset(
        [log for log in ordered_logs if log not in train_logs], counts, command_masks, args.val_anchors
    )
    train_set, val_set = set(train_logs), set(val_logs)

    split_for_log = {
        log: "mini_train" if log in train_set else "mini_val" if log in val_set else "dev_heldout"
        for log in counts
    }
    splits: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for anchor in anchors:
        item = {**anchor, "split": split_for_log[anchor["log_name"]]}
        splits[item["split"]].append(item)

    all_sets = {name: {item["scene_token"] for item in items} for name, items in splits.items()}
    all_anchor_sets = {name: {item["anchor_id"] for item in items} for name, items in splits.items()}
    all_log_sets = {name: {item["log_name"] for item in items} for name, items in splits.items()}
    overlap = {}
    names = sorted(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap[f"{left}__{right}"] = {
                "scene": len(all_sets[left] & all_sets[right]),
                "anchor": len(all_anchor_sets[left] & all_anchor_sets[right]),
                "log": len(all_log_sets[left] & all_log_sets[right]),
            }

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "anchors": len(items),
            "scenes": len({item["scene_token"] for item in items}),
            "logs": len({item["log_name"] for item in items}),
            "command_distribution": dict(collections.Counter(item["command_category"] for item in items)),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (args.output_root / f"{name}.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in items), encoding="utf-8"
        )
        if name in {"mini_train", "mini_val"}:
            logs = sorted({item["log_name"] for item in items})
            tokens = [item["anchor_id"] for item in items]
            yaml_lines = [
                "_target_: navsim.common.dataclasses.SceneFilter",
                "_convert_: 'all'",
                "num_history_frames: 5",
                "num_future_frames: 10",
                "frame_interval: 1",
                "has_route: true",
                "max_scenes: null",
                "log_names:",
                *(f"  - '{value}'" for value in logs),
                "tokens:",
                *(f"  - '{value}'" for value in tokens),
            ]
            (args.output_root / f"{name}_scene_filter.yaml").write_text(
                "\n".join(yaml_lines) + "\n", encoding="utf-8"
            )
    report = {
        "artifact_status": ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"],
        "source_records": str(args.records.resolve()),
        "source_metadata_root": str(args.metadata_root.resolve()),
        "seed": args.seed,
        "selection_unit": "complete_log",
        "source_anchor_count": len(anchors),
        "splits": {name: stats(items) for name, items in sorted(splits.items())},
        "selected_logs": {"mini_train": sorted(train_set), "mini_val": sorted(val_set)},
        "overlap_audit": overlap,
        "all_disjoint": all(not any(value.values()) for value in overlap.values()),
    }
    (args.output_root / "split_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

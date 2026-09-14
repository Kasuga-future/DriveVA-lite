#!/usr/bin/env python3
"""Instrument validation for a counterfactual-teacher run.

Answers, from a run's ``selector_metrics.jsonl`` alone:

* how big is the measured effect (the label's dynamic range),
* is the measurement reproducible (masked-forward replicate),
* do the baseline and probe code paths agree (identity-mask control),
* how does the loss-space label compare with the geometric plan displacement,
* how much of the label spread is explained by the tile index alone.

The first three are the controls that decide whether a "no signal" AUC is a
statement about the teacher or about the harness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _column(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _auc(scores: list[float], labels: list[float]) -> float:
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    if n == 0:
        return float("nan")
    ranks = [0.0] * n
    index = 0
    while index < n:
        end = index
        while end + 1 < n and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    positive = [ranks[i] for i in range(n) if pairs[i][1] >= 0.5]
    negative = [ranks[i] for i in range(n) if pairs[i][1] < 0.5]
    if not positive or not negative:
        return float("nan")
    u = sum(positive) - len(positive) * (len(positive) + 1) / 2.0
    return float(u / (len(positive) * len(negative)))


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return float(num / (den_a * den_b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        row
        for row in _load(args.metrics)
        if row.get("counterfactual_relative_delta") is not None
    ]
    if not rows:
        raise SystemExit("no counterfactual rows found")

    delta = _column(rows, "counterfactual_metrics" if False else "counterfactual_measured_delta")
    if not delta:
        delta = _column(rows, "counterfactual_relative_delta")
    labels = [1.0 if float(row["counterfactual_helpful_target"]) >= 0.5 else 0.0 for row in rows]

    report: dict = {
        "metrics_path": str(args.metrics.resolve()),
        "n": len(rows),
        "label": {
            "mean": statistics.fmean(delta),
            "std": statistics.pstdev(delta),
            "mean_abs": statistics.fmean(abs(value) for value in delta),
            "max_abs": max(abs(value) for value in delta),
            "p05": sorted(delta)[int(0.05 * len(delta))],
            "p95": sorted(delta)[min(len(delta) - 1, int(0.95 * len(delta)))],
            "helpful_rate": statistics.fmean(labels),
            "sign_balance_deviation": abs(statistics.fmean(labels) - 0.5),
        },
        "instrument_controls": {},
        "label_auc_by_tile_index": _auc(
            [float(row["counterfactual_group_index"]) for row in rows], labels
        ),
    }
    controls = report["instrument_controls"]

    identity = _column(rows, "counterfactual_control_identity_delta")
    if identity:
        controls["identity_mask_delta_max_abs"] = max(abs(value) for value in identity)
        controls["identity_mask_delta_all_zero"] = all(abs(value) == 0.0 for value in identity)
        controls["identity_mask_samples"] = len(identity)

    replicate = _column(rows, "counterfactual_control_mask_delta")
    if replicate:
        deviations = [
            abs(float(row["counterfactual_control_mask_delta"]) - float(row["counterfactual_measured_delta"]))
            for row in rows
            if row.get("counterfactual_control_mask_delta") is not None
            and row.get("counterfactual_measured_delta") is not None
        ]
        controls["replicate_mask_delta_max_abs_deviation"] = max(deviations) if deviations else None
        controls["replicate_mask_bit_identical"] = bool(deviations) and max(deviations) == 0.0

    trajectory = _column(rows, "counterfactual_traj_disp_relative")
    if trajectory and len(trajectory) == len(delta):
        report["trajectory_target"] = {
            "n": len(trajectory),
            "mean": statistics.fmean(trajectory),
            "std": statistics.pstdev(trajectory),
            "mean_abs": statistics.fmean(abs(value) for value in trajectory),
            "pearson_with_loss_delta": _pearson(delta, trajectory),
            "sign_agreement_with_loss_delta": statistics.fmean(
                1.0 if (a >= 0) == (b >= 0) else 0.0 for a, b in zip(delta, trajectory)
            ),
            "auc_signed_loss_delta_vs_disp": _auc(trajectory, labels),
        }
        # Displacement is unsigned; ask whether the *magnitude* of the loss
        # change tracks it, which is the property a robust target needs.
        abs_delta = [abs(value) for value in delta]
        report["trajectory_target"]["pearson_abs_delta_with_disp"] = _pearson(abs_delta, trajectory)

    baseline_loss = _column(rows, "counterfactual_baseline_loss_unweighted")
    if baseline_loss:
        report["baseline_loss"] = {
            "mean": statistics.fmean(baseline_loss),
            "std": statistics.pstdev(baseline_loss),
            "relative_resolution_bf16_estimate": 2 ** -8,
        }

    confidence = _column(rows, "counterfactual_confidence")
    if confidence:
        report["confidence"] = {
            "mean": statistics.fmean(confidence),
            "fraction_at_floor": statistics.fmean(
                1.0 if value <= 0.0501 else 0.0 for value in confidence
            ),
            "fraction_at_ceiling": statistics.fmean(
                1.0 if value >= 0.999 else 0.0 for value in confidence
            ),
        }

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

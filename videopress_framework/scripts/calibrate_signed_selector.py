#!/usr/bin/env python3
"""Choose a conservative keep/drop threshold from held-out counterfactual probes.

Review 2026-09-11 item 3.2 / defect E: the previous report only emitted
threshold-conditioned confusion counts.  When no threshold satisfied the safety
constraints the script fell back to ``threshold = min(values) - 1e-6`` (keep
everything), which forces ``fn = tn = 0`` and therefore
``helpful_recall = 1.0``, ``harmful_recall = 0.0`` and
``balanced_accuracy == 0.5`` *by construction*.  That identity says nothing
about the selector and was previously misread as evidence of randomness.

This version always reports, independently of any threshold:

* a threshold-free AUC (Mann-Whitney, average ranks for ties), with a bootstrap
  confidence interval and a label-permutation p-value;
* the achievable operating points over every candidate threshold;
* an explicit separation between ``recommend_no_prune`` (a *safety* decision)
  and ``discriminative_power`` (a *scientific* decision);
* the unweighted BCE, because the training loss is confidence weighted and
  therefore shrinks mechanically whenever the measured effect shrinks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _auc(scores: list[float], labels: list[float]) -> float:
    """Mann-Whitney AUC with average ranks for tied scores."""
    n = len(scores)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    index = 0
    while index < n:
        end = index
        while end + 1 < n and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average_rank
        index = end + 1
    positive = [ranks[i] for i in range(n) if labels[i] >= 0.5]
    negative = [ranks[i] for i in range(n) if labels[i] < 0.5]
    if not positive or not negative:
        return float("nan")
    u_statistic = sum(positive) - len(positive) * (len(positive) + 1) / 2.0
    return float(u_statistic / (len(positive) * len(negative)))


def _bootstrap_auc_ci(
    scores: list[float],
    labels: list[float],
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if n_resamples <= 0 or len(scores) < 4:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(scores)
    draws = []
    for _ in range(n_resamples):
        take = [rng.randrange(n) for _ in range(n)]
        value = _auc([scores[i] for i in take], [labels[i] for i in take])
        if value == value:
            draws.append(value)
    if len(draws) < 10:
        return float("nan"), float("nan")
    draws.sort()
    low = draws[max(0, int(alpha / 2.0 * len(draws)))]
    high = draws[min(len(draws) - 1, int((1.0 - alpha / 2.0) * len(draws)))]
    return float(low), float(high)


def _permutation_auc_pvalue(
    scores: list[float],
    labels: list[float],
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """Two-sided permutation test on |AUC - 0.5|."""
    observed = _auc(scores, labels)
    if observed != observed or n_permutations <= 0:
        return float("nan")
    rng = random.Random(seed)
    shuffled = list(labels)
    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        value = _auc(scores, shuffled)
        if value == value and abs(value - 0.5) >= abs(observed - 0.5) - 1e-12:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def _metrics(rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        target_keep = bool(float(row["counterfactual_helpful_target"]) >= 0.5)
        predict_keep = float(row["counterfactual_group_logit"]) >= threshold
        if target_keep and predict_keep:
            tp += 1
        elif target_keep:
            fn += 1
        elif predict_keep:
            fp += 1
        else:
            tn += 1
    helpful_recall = _safe_div(tp, tp + fn)
    helpful_precision = _safe_div(tp, tp + fp)
    # ``tn`` counts tiles we dropped that really were harmful, so
    # ``tn / (tn + fn)`` is the *precision of the drop action*, which is
    # numerically identical to the precision of the harmful class.  The old
    # key name is kept for continuity but the explicit aliases are what reports
    # should quote.
    drop_action_precision = _safe_div(tn, tn + fn)
    harmful_recall = _safe_div(tn, tn + fp)
    total = tp + fp + tn + fn
    accuracy = _safe_div(tp + tn, total)
    return {
        "raw_logit_threshold": float(threshold),
        "probability_threshold": float(1.0 / (1.0 + math.exp(-threshold))),
        "tp_keep_helpful": tp,
        "fn_drop_helpful": fn,
        "tn_drop_harmful": tn,
        "fp_keep_harmful": fp,
        "helpful_recall": helpful_recall,
        "helpful_precision": helpful_precision,
        "harmful_drop_precision": drop_action_precision,
        "harmful_drop_recall": harmful_recall,
        "drop_action_precision": drop_action_precision,
        "harmful_class_precision": drop_action_precision,
        "harmful_class_recall": harmful_recall,
        "balanced_accuracy": 0.5 * (helpful_recall + harmful_recall),
        "accuracy": accuracy,
        "confusion_matrix": {
            "keep_helpful": tp,
            "drop_helpful": fn,
            "keep_harmful": fp,
            "drop_harmful": tn,
        },
        "predicted_drop_count": tn + fn,
        "predicted_drop_ratio": _safe_div(tn + fn, total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument(
        "--timesteps",
        type=str,
        default="",
        help="optional comma-separated logged diffusion timesteps to calibrate",
    )
    parser.add_argument("--min-harmful-precision", type=float, default=0.9)
    parser.add_argument("--min-harmful-recall", type=float, default=0.1)
    parser.add_argument("--min-harmful-drops", type=int, default=10)
    parser.add_argument("--min-helpful-recall", type=float, default=0.9)
    parser.add_argument("--abstain-margin", type=float, default=0.03)
    parser.add_argument(
        "--report-threshold",
        type=float,
        default=0.0,
        help="raw-logit threshold used for the always-reported operating point",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--auc-significance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--label-regime",
        type=str,
        default="unspecified",
        choices=["unspecified", "fixed_seed", "random_noise"],
        help="noise regime the held-out labels were generated under (defect G)",
    )
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.metrics.open() if line.strip()]
    probe_rows = [
        row
        for row in all_rows
        if row.get("counterfactual_group_logit") is not None
        and row.get("counterfactual_helpful_target") is not None
    ]
    selected_timesteps = {
        int(float(value.strip()))
        for value in args.timesteps.split(",")
        if value.strip()
    }
    if selected_timesteps:
        probe_rows = [
            row
            for row in probe_rows
            if row.get("counterfactual_timestep_mean") is not None
            and int(float(row["counterfactual_timestep_mean"])) in selected_timesteps
        ]
    confidence_filter = float(args.min_confidence)
    rows = [
        row
        for row in probe_rows
        if float(row.get("counterfactual_confidence") or 0.0) >= confidence_filter
    ]
    confidence_filter_relaxed = False
    if not rows and probe_rows:
        # A tiny smoke run can have every probe below the confidence floor.  Do
        # not abort the whole wrapper: fall back to the unfiltered set and say so.
        confidence_filter_relaxed = True
        confidence_filter = 0.0
        rows = list(probe_rows)
    if not rows:
        raise ValueError("no counterfactual rows available at all")

    scores = [float(row["counterfactual_group_logit"]) for row in rows]
    labels = [
        1.0 if float(row["counterfactual_helpful_target"]) >= 0.5 else 0.0
        for row in rows
    ]

    # ---- threshold-free discriminative power (works even with no safe cut) --
    auc = _auc(scores, labels)
    auc_low, auc_high = _bootstrap_auc_ci(
        scores,
        labels,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    permutation_p = _permutation_auc_pvalue(
        scores,
        labels,
        n_permutations=args.permutations,
        seed=args.seed + 1,
    )
    ci_excludes_chance = (
        auc_low == auc_low and auc_high == auc_high and (auc_low > 0.5 or auc_high < 0.5)
    )
    significant = (
        permutation_p == permutation_p and permutation_p < args.auc_significance
    ) and ci_excludes_chance
    if auc != auc:
        verdict = "undefined"
        interpretation = "one class is missing; AUC is undefined"
    elif significant and auc > 0.5:
        verdict = "signal"
        interpretation = (
            f"AUC={auc:.3f} differs from chance (permutation p={permutation_p:.4g}, "
            f"bootstrap 95% CI [{auc_low:.3f}, {auc_high:.3f}])."
        )
    elif auc > 0.55:
        verdict = "weak_signal"
        interpretation = (
            f"AUC={auc:.3f} is above chance but not significant at n={len(rows)} "
            f"(permutation p={permutation_p:.4g}, 95% CI [{auc_low:.3f}, {auc_high:.3f}]). "
            "Treat as undetermined, not as a positive result."
        )
    else:
        verdict = "no_signal"
        interpretation = (
            f"AUC={auc:.3f} is indistinguishable from chance "
            f"(permutation p={permutation_p:.4g}, 95% CI [{auc_low:.3f}, {auc_high:.3f}]). "
            "The selector's ranking carries no measurable information about the label."
        )

    unweighted_bce_values = [
        float(row["selector_bce_unweighted"])
        for row in rows
        if row.get("selector_bce_unweighted") is not None
    ]
    weighted_bce_values = [
        float(row["selector_bce"])
        for row in rows
        if row.get("selector_bce") is not None
    ]

    # ---- threshold-conditioned safety decision (unchanged semantics) --------
    values = sorted(set(scores))
    candidates = [values[0] - 1e-6, *values, values[-1] + 1e-6]
    evaluated = [_metrics(rows, threshold) for threshold in candidates]
    safe = [
        item
        for item in evaluated
        if item["harmful_drop_precision"] >= args.min_harmful_precision
        and item["harmful_drop_recall"] >= args.min_harmful_recall
        and item["helpful_recall"] >= args.min_helpful_recall
        and item["tn_drop_harmful"] >= args.min_harmful_drops
    ]
    if safe:
        selected = max(
            safe,
            key=lambda item: (
                item["harmful_drop_recall"],
                item["balanced_accuracy"],
                -item["raw_logit_threshold"],
            ),
        )
        abstain_all = False
        no_prune_reason = None
    else:
        # No threshold demonstrates safe removal: explicitly recommend the
        # no-prune path rather than optimizing an unsafe aggregate accuracy.
        # NOTE: the fallback metrics are an identity, not evidence.
        selected = _metrics(rows, values[0] - 1e-6)
        abstain_all = True
        no_prune_reason = (
            "no candidate threshold satisfied "
            f"harmful_drop_precision>={args.min_harmful_precision}, "
            f"harmful_drop_recall>={args.min_harmful_recall}, "
            f"helpful_recall>={args.min_helpful_recall}, "
            f"tn_drop_harmful>={args.min_harmful_drops}"
        )

    report = {
        "metrics_path": str(args.metrics.resolve()),
        "probe_rows_total": len(probe_rows),
        "probe_rows_calibrated": len(rows),
        "min_confidence": args.min_confidence,
        "confidence_filter_effective": confidence_filter,
        "confidence_filter_relaxed": confidence_filter_relaxed,
        "label_regime": args.label_regime,
        "confidence_filter_caveat": (
            "confidence = |delta| / counterfactual_scale, so under "
            "label_regime='random_noise' this filter preferentially keeps probes "
            "whose |delta| was inflated by video-noise mismatch: it selects noise, "
            "not certainty.  Only label_regime='fixed_seed' makes it safe."
            if args.label_regime == "random_noise"
            else (
                "label regime not declared; treat the confidence filter as "
                "unverified"
                if args.label_regime == "unspecified"
                else "fixed-seed labels: the filter selects genuinely large effects"
            )
        ),
        "timesteps": sorted(selected_timesteps),
        "target_keep_ratio": _safe_div(
            sum(float(row["counterfactual_helpful_target"]) >= 0.5 for row in rows),
            len(rows),
        ),
        "constraints": {
            "min_harmful_drop_precision": args.min_harmful_precision,
            "min_harmful_drop_recall": args.min_harmful_recall,
            "min_harmful_drops": args.min_harmful_drops,
            "min_helpful_recall": args.min_helpful_recall,
        },
        "selected": selected,
        "recommended_abstain_margin": args.abstain_margin,
        "recommend_no_prune": abstain_all,
        "recommend_no_prune_reason": no_prune_reason,
        "discriminative_power": {
            "n": len(rows),
            "positive_rate": _safe_div(sum(labels), len(labels)),
            "auc": auc,
            "auc_ci95": [auc_low, auc_high],
            "auc_ci_excludes_chance": bool(ci_excludes_chance),
            "permutation_p_value": permutation_p,
            "bootstrap_resamples": args.bootstrap_resamples,
            "permutations": args.permutations,
            "significance_level": args.auc_significance,
            "verdict": verdict,
            "interpretation": interpretation,
            "note": (
                "recommend_no_prune is a SAFETY verdict about the absence of a "
                "threshold satisfying the constraints; it must never be quoted as "
                "evidence that the selector is at chance.  Read 'verdict' for that."
            ),
        },
        "unweighted_bce": (
            sum(unweighted_bce_values) / len(unweighted_bce_values)
            if unweighted_bce_values
            else None
        ),
        "confidence_weighted_bce": (
            sum(weighted_bce_values) / len(weighted_bce_values)
            if weighted_bce_values
            else None
        ),
        "report_threshold_operating_point": _metrics(rows, args.report_threshold),
        "oracle_best_accuracy": max(item["accuracy"] for item in evaluated),
        "oracle_best_balanced_accuracy": max(
            item["balanced_accuracy"] for item in evaluated
        ),
        "majority_class_accuracy": max(
            _safe_div(sum(labels), len(labels)),
            1.0 - _safe_div(sum(labels), len(labels)),
        ),
    }
    # Oracle accuracy is only meaningful relative to the trivial baseline; with
    # a 44/56 label split an oracle 0.574 is a 1.5 point lift, not 57% skill.
    report["oracle_accuracy_lift_over_majority"] = (
        report["oracle_best_accuracy"] - report["majority_class_accuracy"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

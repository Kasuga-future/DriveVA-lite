#!/usr/bin/env python3
"""Paired official-test comparison of token-selection arms.

Reuses the exact PDM CSVs produced by ``run_official_navsim_press.py`` so two
runs (or two methods inside one run) can be compared scene-by-scene.  Pairing is
by scene token, which is valid because every arm is evaluated on the identical
repaired 1,920-scene manifest with the identical metric cache.

Methodology uses paired bootstrap over scene-aligned PDM differences, 20,000
resamples.  Extreme compliance flips and exact-zero transitions are reported
separately so a small number of discontinuous scenes cannot silently dominate
the aggregate delta.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            # The official evaluator appends an aggregate row whose token is
            # literally "average".  Its `valid` is set to bool(all valid), so the
            # validity filter below does NOT exclude it, and it would enter every
            # paired statistic as one extra fake "scene" (audit 2026-09-12:
            # 7,877 keys instead of 7,876).
            if str(row.get("token", "")).strip() == "average":
                continue
            if str(row.get("valid", "True")).lower() not in {"true", "1"}:
                continue
            value = row.get("pdm_score")
            if value in (None, ""):
                continue
            out[row["token"]] = float(value)
    return out


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def find_method_csv(root: Path, method: str) -> Path | None:
    matches = sorted(glob.glob(str(root / "round*" / method / "pdm_score_*.csv")))
    return Path(matches[-1]) if matches else None


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    *,
    resamples: int,
    seed: int,
    chunk: int = 512,
) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}
    if resamples <= 0 or chunk <= 0:
        raise ValueError("resamples and chunk must be positive")
    diffs = np.asarray([a[token] - b[token] for token in shared], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = diffs.size
    means = np.empty(resamples, dtype=np.float64)
    done = 0
    while done < resamples:
        take = min(chunk, resamples - done)
        indices = rng.integers(0, n, size=(take, n))
        means[done : done + take] = diffs[indices].mean(axis=1)
        done += take
    means.sort()
    lo = float(means[int(0.025 * resamples)])
    hi = float(means[min(resamples - 1, int(0.975 * resamples))])
    return {
        "n": int(n),
        "delta": float(diffs.mean()),
        "ci95": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "wins": int((diffs > 0).sum()),
        "losses": int((diffs < 0).sum()),
    }


def paired_robustness(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    resamples: int,
    seed: int,
    extreme_threshold: float,
) -> dict:
    """Decompose paired quality into ordinary and discontinuous transitions."""

    shared = sorted(set(candidate) & set(reference))
    extreme = [
        token
        for token in shared
        if abs(candidate[token] - reference[token]) > extreme_threshold
    ]
    ordinary = set(shared) - set(extreme)
    candidate_zero = {token for token in shared if abs(candidate[token]) <= 1e-12}
    reference_zero = {token for token in shared if abs(reference[token]) <= 1e-12}
    return {
        "extreme_abs_delta_threshold": float(extreme_threshold),
        "extreme_count": len(extreme),
        "non_extreme": paired_bootstrap(
            {token: candidate[token] for token in ordinary},
            {token: reference[token] for token in ordinary},
            resamples=resamples,
            seed=seed,
        ),
        "reference_zero_count": len(reference_zero),
        "candidate_zero_count": len(candidate_zero),
        "rescued_zero_count": len(reference_zero - candidate_zero),
        "introduced_zero_count": len(candidate_zero - reference_zero),
        "extreme_transitions": [
            {
                "token": token,
                "reference": reference[token],
                "candidate": candidate[token],
                "delta": candidate[token] - reference[token],
            }
            for token in sorted(
                extreme,
                key=lambda token: abs(candidate[token] - reference[token]),
                reverse=True,
            )
        ],
    }


def summarise_run(root: Path, method: str, resamples: int, seed: int) -> dict:
    csv_path = find_method_csv(root, method)
    if csv_path is None:
        return {"method": method, "missing": True}
    pdm = load_csv(csv_path)
    records = load_records(root / "round01" / method / "records.jsonl")
    entry: dict = {
        "method": method,
        "csv": str(csv_path),
        "n_scenes": len(pdm),
        "pdm_mean": sum(pdm.values()) / len(pdm) if pdm else None,
    }
    ks = [float(r["K"]) for r in records if r.get("K") is not None]
    if ks:
        entry["mean_K"] = sum(ks) / len(ks)
        entry["K_min"] = min(ks)
        entry["K_max"] = max(ks)
    candidate_counts = [
        float(r["n_candidate"])
        for r in records
        if r.get("K") is not None and r.get("n_candidate") not in (None, 0)
    ]
    keep_ratios = [
        float(r["K"]) / float(r["n_candidate"])
        for r in records
        if r.get("K") is not None and r.get("n_candidate") not in (None, 0)
    ]
    if candidate_counts:
        entry["candidate_tokens_mean"] = sum(candidate_counts) / len(candidate_counts)
        entry["candidate_tokens_min"] = min(candidate_counts)
        entry["candidate_tokens_max"] = max(candidate_counts)
        if min(candidate_counts) == max(candidate_counts):
            # Preserve the legacy key only when it truthfully describes every
            # record; history-domain runs have 780 candidates, not 390.
            entry["candidate_tokens"] = candidate_counts[0]
    if keep_ratios:
        entry["candidate_kept_pct"] = 100.0 * sum(keep_ratios) / len(keep_ratios)
    ratios = [
        r["metadata"]["hidden_sequence_ratio_mean_across_steps"]
        for r in records
        if isinstance(r.get("metadata"), dict)
        and r["metadata"].get("hidden_sequence_ratio_mean_across_steps") is not None
    ]
    if ratios:
        entry["hidden_sequence_remaining_pct"] = 100.0 * sum(ratios) / len(ratios)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    if latencies:
        latencies.sort()
        entry["latency_median_ms"] = latencies[len(latencies) // 2]
        entry["latency_mean_ms"] = sum(latencies) / len(latencies)
    entry["_pdm"] = pdm
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-method", type=str, default="physical_no_press")
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--new-methods", type=str, required=True)
    parser.add_argument("--baseline-method", type=str, default=None,
                        help="method inside reference-run that the new arms must match")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--extreme-threshold", type=float, default=0.5)
    args = parser.parse_args()

    reference = summarise_run(
        args.reference_run, args.reference_method, args.resamples, args.seed
    )
    report: dict = {
        "reference_run": str(args.reference_run),
        "reference": {k: v for k, v in reference.items() if not k.startswith("_")},
        "new_run": str(args.new_run),
        "comparisons": {},
    }

    baseline_pdm = None
    if args.baseline_method:
        baseline = summarise_run(
            args.reference_run, args.baseline_method, args.resamples, args.seed
        )
        report["baseline_arm"] = {
            k: v for k, v in baseline.items() if not k.startswith("_")
        }
        baseline_pdm = baseline.get("_pdm")

    for method in [m.strip() for m in args.new_methods.split(",") if m.strip()]:
        entry = summarise_run(args.new_run, method, args.resamples, args.seed)
        if entry.get("missing"):
            report["comparisons"][method] = {"missing": True}
            continue
        pdm = entry.pop("_pdm")
        if reference.get("_pdm"):
            comparison = paired_bootstrap(
                pdm, reference["_pdm"], resamples=args.resamples, seed=args.seed
            )
            robustness = paired_robustness(
                pdm,
                reference["_pdm"],
                resamples=args.resamples,
                seed=args.seed + 2,
                extreme_threshold=args.extreme_threshold,
            )
            entry["vs_reference"] = comparison
            entry["vs_reference_robustness"] = robustness
            if args.reference_method == "physical_no_press":
                # Backward-compatible aliases for the overwhelmingly common
                # default.  Do not mislabel an arbitrary reference as no-press.
                entry["vs_no_press"] = comparison
                entry["vs_no_press_robustness"] = robustness
        if baseline_pdm:
            entry["vs_baseline_arm"] = paired_bootstrap(
                pdm, baseline_pdm, resamples=args.resamples, seed=args.seed + 1
            )
        report["comparisons"][method] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"reference {args.reference_method}: PDM={reference.get('pdm_mean')}")
    if report.get("baseline_arm"):
        print(f"baseline  {args.baseline_method}: PDM={report['baseline_arm'].get('pdm_mean')} "
              f"meanK={report['baseline_arm'].get('mean_K')}")
    for method, entry in report["comparisons"].items():
        if entry.get("missing"):
            print(f"{method:60s} MISSING")
            continue
        vs_reference = entry.get("vs_reference", {})
        vs_base = entry.get("vs_baseline_arm", {})
        line = (f"{method:60s} PDM={entry['pdm_mean']:.6f} n={entry['n_scenes']} "
                f"meanK={entry.get('mean_K')}")
        if vs_reference:
            line += (
                f" | d({args.reference_method})={vs_reference['delta']:+.6f} "
                f"CI[{vs_reference['ci95'][0]:+.6f},{vs_reference['ci95'][1]:+.6f}]"
            )
        if vs_base:
            line += f" | d(base)={vs_base['delta']:+.6f} CI[{vs_base['ci95'][0]:+.6f},{vs_base['ci95'][1]:+.6f}]"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

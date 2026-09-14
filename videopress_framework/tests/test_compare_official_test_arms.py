import csv
import json

import pytest

from videopress_framework.scripts.compare_official_test_arms import (
    paired_robustness,
    summarise_run,
)


def test_summarise_run_uses_record_candidate_count(tmp_path):
    method = "physical_dynamic"
    method_dir = tmp_path / "round01" / method
    method_dir.mkdir(parents=True)
    with (method_dir / "pdm_score_test.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "valid", "pdm_score"])
        writer.writeheader()
        writer.writerows(
            [
                {"token": "a", "valid": "True", "pdm_score": 0.8},
                {"token": "b", "valid": "True", "pdm_score": 1.0},
            ]
        )
    rows = [
        {"K": 390, "n_candidate": 780, "latency_ms": 10.0},
        {"K": 546, "n_candidate": 780, "latency_ms": 11.0},
    ]
    (method_dir / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarise_run(tmp_path, method, resamples=10, seed=1)

    assert summary["candidate_tokens"] == 780
    assert summary["candidate_tokens_mean"] == 780
    assert summary["candidate_kept_pct"] == pytest.approx(60.0)


def test_paired_robustness_separates_extremes_and_zero_transitions():
    reference = {"ordinary": 0.8, "rescue": 0.0, "introduced": 0.9}
    candidate = {"ordinary": 0.81, "rescue": 0.9, "introduced": 0.0}

    result = paired_robustness(
        candidate,
        reference,
        resamples=100,
        seed=1,
        extreme_threshold=0.5,
    )

    assert result["extreme_count"] == 2
    assert result["non_extreme"]["n"] == 1
    assert result["non_extreme"]["delta"] == pytest.approx(0.01)
    assert result["reference_zero_count"] == 1
    assert result["candidate_zero_count"] == 1
    assert result["rescued_zero_count"] == 1
    assert result["introduced_zero_count"] == 1

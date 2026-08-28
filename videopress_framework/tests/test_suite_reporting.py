import json

from evaluation.statistics import aggregate_suite, write_suite_tables
from evaluation.visualization import generate_suite_visualizations


def _make_run(root, round_index, method, protocol, pdm):
    run = root / f"round_{round_index:02d}" / method
    run.mkdir(parents=True)
    record = {
        "scene_token": f"scene-{round_index}",
        "log_id": "log-0",
        "press_name": "scorer_press",
        "scorer": "random",
        "selector": "topk",
        "operator": "zero" if protocol == "causal" else "kv_prune",
        "domain": "last_history",
        "K": 2,
        "n_candidate": 4,
        "n_history": 8,
        "eligible_keep_ratio": 0.5 if protocol == "causal" else None,
        "history_keep_ratio": 0.25,
        "pdm": pdm,
        "trajectory_l2": 0.5,
        "endpoint_l2": 0.6,
        "latency_ms": 3.0,
        "selector_latency_ms": 0.5,
        "model_latency_ms": 2.0,
        "e2e_latency_ms": 3.0,
        "peak_memory_mb": 12.0,
        "valid": True,
        "metadata": {
            "k_length_before": 8 if protocol == "physical" else None,
            "k_length_after": 6 if protocol == "physical" else None,
            "v_length_after": 6 if protocol == "physical" else None,
            "theoretical_attn_ratio": 0.75 if protocol == "physical" else None,
        },
    }
    (run / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "mode": protocol,
                "backend": "synthetic",
                "press": {
                    "name": "scorer_press",
                    "domain": "last_history",
                    "scorer": {"name": "random"},
                    "selector": {"name": "topk"},
                    "operator": {"name": record["operator"]},
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_multi_round_statistics_and_visualizations(tmp_path):
    causal0 = _make_run(tmp_path, 0, "causal_random_zero", "causal", 0.8)
    causal1 = _make_run(tmp_path, 1, "causal_random_zero", "causal", 0.6)
    manifest = {
        "suite_name": "test",
        "runs": [
            {"round": 0, "method": "causal_random_zero", "protocol": "causal", "output_dir": str(causal0.relative_to(tmp_path))},
            {"round": 1, "method": "causal_random_zero", "protocol": "causal", "output_dir": str(causal1.relative_to(tmp_path))},
        ],
    }
    (tmp_path / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = aggregate_suite(tmp_path)
    assert len(summary["run_rows"]) == 2
    assert summary["method_rows"][0]["rounds"] == 2
    assert summary["method_rows"][0]["n_scenes"] == 2
    assert summary["method_rows"][0]["pdm"] == 0.7

    tables = write_suite_tables(summary, tmp_path / "statistics")
    assert all(path.endswith((".csv", ".md")) for path in tables.values())
    assert (tmp_path / "statistics" / "method_statistics.csv").read_text(encoding="utf-8").count("causal_random_zero") == 1

    plots = generate_suite_visualizations(summary, tmp_path / "visualizations")
    assert {"pdm_by_method", "latency_by_method", "compression_ratios", "pdm_vs_latency", "round_stability", "summary_table"} <= set(plots)
    assert all((tmp_path / "visualizations" / f"{name}.png").is_file() for name in plots)


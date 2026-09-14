import json

import pytest

from evaluation.statistics import aggregate_records, aggregate_suite, write_suite_tables
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


def test_record_aggregation_excludes_invalid_placeholders_and_uses_true_k_extrema():
    rows = [
        {
            "valid": True,
            "pdm": 0.8,
            "K": 100,
            "metadata": {"dynamic_n_kept_min": 80, "dynamic_n_kept_max": 120},
        },
        {
            "valid": True,
            "pdm": 0.6,
            "K": 200,
            "metadata": {"dynamic_n_kept_min": 140, "dynamic_n_kept_max": 240},
        },
        {
            "valid": False,
            "pdm": 0.0,
            "K": 0,
            "metadata": {"dynamic_n_kept_min": 0, "dynamic_n_kept_max": 0},
        },
    ]
    summary = aggregate_records(rows)
    assert summary["n_scenes"] == 3
    assert summary["valid_scenes"] == 2
    assert summary["pdm"] == 0.7
    assert summary["K_mean"] == 150.0
    assert summary["dynamic_K_min"] == 80.0
    assert summary["dynamic_K_max"] == 240.0



def _make_run_with_persistence(root, name, end_layer, pdm, *, threshold=0.4):
    """One run whose press carries an explicit cross-layer persistence block."""

    run = root / name
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text(
        json.dumps(
            {
                "scene_token": "scene-a",
                "press_name": "scorer_press",
                "scorer": "learned_planning_selector",
                "selector": "threshold",
                "operator": "kv_prune",
                "domain": "last_history",
                "K": 2,
                "n_candidate": 4,
                "pdm": pdm,
                "valid": True,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "mode": "physical",
                "backend": "synthetic",
                "press": {
                    "name": "scorer_press",
                    "domain": "last_history",
                    "scorer": {"name": "learned_planning_selector", "layer": 15},
                    "selector": {"name": "threshold", "threshold": threshold},
                    "operator": {"name": "kv_prune"},
                    "cross_layer_persistence": {
                        "enabled": True,
                        "end_layer": end_layer,
                        "mode": "hidden_sequence",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_suite_refuses_to_pool_two_interventions_under_one_method_name(tmp_path):
    """A method NAME is not a unique key for the intervention.

    `physical_<scorer>_<mode>_persistent_layer_<NN>` is built from
    scorer/mode/layer only, so an arm with `end_layer == source_layer` (which
    silently disables cross-layer persistence) collides with the true-persistence
    arm of the same name.  Pooling them would merge two different experiments, so
    the aggregator must refuse instead.
    """

    plain = _make_run_with_persistence(tmp_path, "run_plain", None, 0.90)
    early = _make_run_with_persistence(tmp_path, "run_early_stop", 15, 0.50)
    shared_name = "physical_learned_planning_selector_hidden_persistent_layer_15"
    manifest = {
        "suite_name": "collision",
        "runs": [
            {"round": 0, "method": shared_name, "protocol": "physical", "output_dir": str(plain.relative_to(tmp_path))},
            {"round": 1, "method": shared_name, "protocol": "physical", "output_dir": str(early.relative_to(tmp_path))},
        ],
    }
    (tmp_path / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        aggregate_suite(tmp_path)
    except ValueError as exc:
        assert "DIFFERENT interventions" in str(exc)
    else:
        raise AssertionError("two different interventions must not be pooled by name")


def test_suite_pools_rounds_of_the_same_intervention(tmp_path):
    """The guard must not break the legitimate case: same intervention, 2 rounds."""

    first = _make_run_with_persistence(tmp_path, "run_r0", None, 0.90)
    second = _make_run_with_persistence(tmp_path, "run_r1", None, 0.70)
    name = "physical_learned_planning_selector_hidden_persistent_layer_15"
    manifest = {
        "suite_name": "legit",
        "runs": [
            {"round": 0, "method": name, "protocol": "physical", "output_dir": str(first.relative_to(tmp_path))},
            {"round": 1, "method": name, "protocol": "physical", "output_dir": str(second.relative_to(tmp_path))},
        ],
    }
    (tmp_path / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = aggregate_suite(tmp_path)
    assert len(summary["method_rows"]) == 1
    assert summary["method_rows"][0]["rounds"] == 2


def test_suite_identity_includes_selector_parameters(tmp_path):
    first = _make_run_with_persistence(tmp_path, "run_t04", None, 0.90, threshold=0.4)
    second = _make_run_with_persistence(tmp_path, "run_t05", None, 0.70, threshold=0.5)
    name = "physical_learned_planning_selector_hidden_persistent_layer_15"
    manifest = {
        "suite_name": "threshold-collision",
        "runs": [
            {"round": 0, "method": name, "protocol": "physical", "output_dir": str(first.relative_to(tmp_path))},
            {"round": 1, "method": name, "protocol": "physical", "output_dir": str(second.relative_to(tmp_path))},
        ],
    }
    (tmp_path / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="DIFFERENT interventions"):
        aggregate_suite(tmp_path)

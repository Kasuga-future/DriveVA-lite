"""Token-level, set-level future oracle: library, selector, runner wiring.

These tests cover the instrument the 2026-09-20 future-oracle conclusion asked
for (``outputs/future_oracle_multiseed4_conclusion_20260920.md``): token-level
atoms, set-level scoring, and the matching search / selector / runner plumbing.
Nothing here touches a GPU.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from videopress.core.context import TokenContext
from videopress.core.domain import build_domain
from videopress.core.layout import build_driveva_layout
from videopress.factory import build_press
from videopress.oracle.metrics import (
    OBJECTIVES,
    combined_harm,
    objective_higher_is_better,
    objective_value,
    planning_harm,
    trajectory_displacement,
    zscore,
)
from videopress.oracle.token_set import (
    SearchStep,
    SetScorer,
    SetSearchResult,
    beam_search,
    build_token_groups,
    entry_to_flat,
    flat_from_latent_local,
    greedy_backward_elimination,
    greedy_forward_selection,
    latent_local_mask,
    mean_over_scenes,
    per_scene_oracle,
    random_search,
    random_token_masks,
    read_token_mask_json,
    write_token_mask_json,
)
from videopress.selectors import OracleFutureTokenMaskSelector
from scripts.run_official_navsim_press import (
    _future_oracle_token_mask_paths,
    _mask_method_tag,
    _method_specs_for_run,
)
from scripts.search_future_token_set_oracle import (
    _run_runner,
    build_baseline_runner_command,
    build_evaluate_mask_candidates,
    build_mask_runner_command,
    compose_top_groups,
    decision_stats,
    mask_for_panel,
    resolve_actual_suite_root,
)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_token_groups_partition_every_token():
    groups = build_token_groups(4, 2, mode="token")
    assert len(groups) == 8
    assert all(len(group) == 1 for group in groups)
    assert sorted(token for group in groups for token in group) == list(range(8))
    with pytest.raises(ValueError, match="group_size"):
        build_token_groups(4, 2, mode="token", group_size=2)


def test_linear_groups_never_span_latents_and_tolerate_short_tail():
    groups = build_token_groups(5, 2, mode="linear", group_size=2)
    assert groups == ((0, 1), (2, 3), (4,), (5, 6), (7, 8), (9,))
    with pytest.raises(ValueError, match="positive group_size"):
        build_token_groups(5, 2, mode="linear")


def test_block_groups_follow_the_2d_grid():
    # 4x3 latent grid, 2x2 blocks: 2 columns x 2 rows with edge blocks clipped.
    groups = build_token_groups(12, 1, mode="block", block_h=2, block_w=2, width=4)
    assert groups == ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9), (10, 11))
    with pytest.raises(ValueError, match="width"):
        build_token_groups(12, 1, mode="block", block_h=2, block_w=2)
    with pytest.raises(ValueError, match="divisible"):
        build_token_groups(12, 1, mode="block", block_h=1, block_w=1, width=5)


def test_random_groups_are_seeded_and_disjoint():
    first = build_token_groups(12, 2, mode="random", group_size=4, seed=3)
    second = build_token_groups(12, 2, mode="random", group_size=4, seed=3)
    other = build_token_groups(12, 2, mode="random", group_size=4, seed=4)
    assert first == second
    assert first != other
    assert sorted(token for group in first for token in group) == list(range(24))
    seen: set[int] = set()
    for group in first:
        assert not (set(group) & seen)
        seen |= set(group)


def test_unknown_group_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown group mode"):
        build_token_groups(4, 1, mode="pixel")


# ---------------------------------------------------------------------------
# interchange format
# ---------------------------------------------------------------------------


def test_latent_local_mask_round_trip():
    flat = [0, 1, 389, 390, 391, 779]
    mask = latent_local_mask(flat, tokens_per_latent=390, num_latents=2)
    assert mask == {"future_latent_0": [0, 1, 389], "future_latent_1": [0, 1, 389]}
    assert flat_from_latent_local(mask, tokens_per_latent=390, num_latents=2) == flat


def test_entry_to_flat_accepts_mapping_and_flat_list():
    mapping = {"future_latent_0": [2, 0], "future_latent_1": [5]}
    assert entry_to_flat(mapping, tokens_per_latent=10, num_latents=2) == [0, 2, 15]
    assert entry_to_flat([3, 1, 3], tokens_per_latent=10, num_latents=2) == [1, 3]
    with pytest.raises(ValueError, match="outside"):
        entry_to_flat({"future_latent_0": [10]}, tokens_per_latent=10, num_latents=2)
    with pytest.raises(ValueError, match="unknown"):
        entry_to_flat({"latent_0": [1]}, tokens_per_latent=10, num_latents=2)


def test_token_mask_json_round_trip(tmp_path):
    path = write_token_mask_json({"scene": {"future_latent_0": [1]}}, tmp_path / "m.json")
    assert read_token_mask_json(path) == {"scene": {"future_latent_0": [1]}}


# ---------------------------------------------------------------------------
# set-level search
# ---------------------------------------------------------------------------


def test_greedy_forward_selection_finds_additive_optimum():
    weights = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    scorer = SetScorer(score_fn=lambda tokens: sum(weights[token] for token in tokens))
    groups = build_token_groups(6, 1, mode="token")
    result = greedy_forward_selection(groups, 2, scorer)
    assert result.selected == (4, 5)
    assert result.score == pytest.approx(14.0)
    assert result.metadata["budget_met"] is True
    assert result.steps[0].action == "init"
    assert [step.action for step in result.steps[1:]] == ["add", "add"]
    assert result.n_evaluations == 1 + 6 + 5


def test_greedy_forward_respects_min_gain_and_budget():
    scorer = SetScorer(score_fn=lambda tokens: float(len(tokens)))
    groups = build_token_groups(6, 1, mode="token")
    result = greedy_forward_selection(groups, 3, scorer, min_gain=0.5)
    # Every addition gains exactly 1.0, so all three rounds run.
    assert result.selected == (0, 1, 2)
    stopped = greedy_forward_selection(groups, 3, scorer, min_gain=2.0)
    assert stopped.selected == ()
    assert stopped.metadata["rounds"] == 0


def test_greedy_backward_elimination_keeps_the_heaviest_tokens():
    weights = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    scorer = SetScorer(score_fn=lambda tokens: sum(weights[token] for token in tokens))
    groups = build_token_groups(6, 1, mode="token")
    result = greedy_backward_elimination(groups, 2, scorer)
    assert result.selected == (4, 5)
    assert result.score == pytest.approx(14.0)
    assert [step.action for step in result.steps[1:]] == ["remove"] * 4


def test_greedy_backward_stops_before_a_harmful_removal():
    weights = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    scorer = SetScorer(score_fn=lambda tokens: sum(weights[token] for token in tokens))
    groups = build_token_groups(6, 1, mode="token")
    result = greedy_backward_elimination(groups, 0, scorer, max_harm=2.0)
    # Removing the weight-1 tokens costs 1.0 each and is allowed; the next
    # removal would cost 3.0 > max_harm, so the search stops above the budget.
    assert result.metadata["budget_met"] is False
    assert result.score == pytest.approx(21.0)
    assert result.selected == (0, 2, 4, 5)


def test_beam_search_beats_greedy_on_a_non_additive_objective():
    # The classic interaction case the tile oracle failed on: singleton scores
    # mislead greedy, and only the joint {2,3} pair is optimal.
    joint = {
        frozenset({0}): 0.30,
        frozenset({1}): 0.0,
        frozenset({2}): 0.25,
        frozenset({3}): 0.0,
        frozenset({0, 1}): 0.50,
        frozenset({2, 3}): 1.00,
    }
    scorer_fn = lambda tokens: joint.get(frozenset(tokens), 0.0)  # noqa: E731
    groups = build_token_groups(4, 1, mode="token")
    greedy = greedy_forward_selection(groups, 2, SetScorer(score_fn=scorer_fn))
    beam = beam_search(groups, 2, SetScorer(score_fn=scorer_fn), beam_width=2)
    assert greedy.selected == (0, 1)
    assert greedy.score == pytest.approx(0.50)
    assert beam.selected == (2, 3)
    assert beam.score == pytest.approx(1.00)


def test_beam_search_spends_the_budget_on_score_ties():
    # PDM is discrete: an all-tie round must not let the oracle "win" by
    # stopping at a smaller subset than the matched budget.
    scorer = SetScorer(score_fn=lambda tokens: 0.0)
    groups = build_token_groups(6, 1, mode="token")
    result = beam_search(groups, 3, scorer, beam_width=2)
    assert len(result.selected) == 3
    assert result.metadata["budget_met"] is True
    assert result.score == pytest.approx(0.0)


def test_random_search_is_seeded_and_budget_matched():
    calls: list[tuple[int, ...]] = []

    def score_many(token_sets):
        calls.extend(token_sets)
        return [float(len(tokens)) for tokens in token_sets]

    groups = build_token_groups(8, 1, mode="linear", group_size=2)
    first = random_search(groups, 4, SetScorer(score_many=score_many), n_samples=3, seed=1)
    second = random_search(groups, 4, SetScorer(score_many=score_many), n_samples=3, seed=1)
    assert first.selected == second.selected
    assert len(first.selected) <= 4
    assert len(calls) >= 6


def test_set_scorer_caches_and_batches():
    calls: list[tuple[int, ...]] = []

    def score_many(token_sets):
        calls.extend(token_sets)
        return [float(sum(tokens)) for tokens in token_sets]

    scorer = SetScorer(score_many=score_many)
    values = scorer.evaluate([(2, 1), (1, 2), (0,)])
    assert values == [3.0, 3.0, 0.0]
    assert calls == [(1, 2), (0,)]
    assert scorer.n_unique_evaluations == 2
    assert scorer.n_requests == 3
    with pytest.raises(ValueError, match="candidate sets"):
        SetScorer(score_many=lambda token_sets: []).evaluate([(1,)])


def test_nan_scores_are_always_worst():
    scorer = SetScorer(score_fn=lambda tokens: float("nan") if tokens == (0,) else 0.5)
    groups = build_token_groups(2, 1, mode="token")
    result = greedy_forward_selection(groups, 1, scorer)
    assert result.selected == (1,)
    assert result.score == pytest.approx(0.5)


def test_search_result_serialises():
    result = SetSearchResult(
        selected=(1, 2),
        score=0.5,
        steps=[SearchStep(0, "init", None, 0.0, 0)],
        n_evaluations=3,
        metadata={"search": "unit"},
    )
    payload = result.to_dict()
    assert payload["selected"] == [1, 2]
    assert payload["steps"][0]["action"] == "init"
    assert payload["metadata"]["search"] == "unit"


# ---------------------------------------------------------------------------
# per-scene best-of-N random masks
# ---------------------------------------------------------------------------


def test_random_token_masks_are_per_scene_and_budget_exact():
    samples = random_token_masks(
        ["scene-a", "scene-b"],
        tokens_per_latent=4,
        num_latents=2,
        budget=3,
        n_samples=2,
        seed=5,
    )
    assert set(samples) == {0, 1}
    for masks in samples.values():
        for scene in ("scene-a", "scene-b"):
            flat = flat_from_latent_local(
                masks[scene], tokens_per_latent=4, num_latents=2
            )
            assert len(flat) == 3
    assert samples[0]["scene-a"] != samples[1]["scene-a"]


def test_per_scene_oracle_takes_the_best_candidate_per_scene():
    scores = {
        "a": {"s1": 0.1, "s2": 0.9},
        "b": {"s1": 0.5, "s2": 0.2},
    }
    oracle = per_scene_oracle(scores, higher_is_better=True)
    assert oracle["n_scenes"] == 2
    assert oracle["oracle_mean"] == pytest.approx(0.7)
    assert oracle["candidate_win_counts"] == {"a": 1, "b": 1}
    assert oracle["scene_winners"] == {"s1": "b", "s2": "a"}
    assert mean_over_scenes(scores["a"]) == pytest.approx(0.5)


def test_per_scene_oracle_intersects_scene_coverage():
    scores = {"a": {"s1": 1.0}, "b": {"s1": 0.0, "s2": 1.0}}
    oracle = per_scene_oracle(scores, higher_is_better=True)
    assert oracle["n_scenes"] == 1
    assert set(oracle["scene_winners"]) == {"s1"}


# ---------------------------------------------------------------------------
# trajectory objectives
# ---------------------------------------------------------------------------


def test_trajectory_metrics_are_consistent():
    baseline = np.zeros((4, 3), dtype=np.float64)
    masked = np.ones((4, 3), dtype=np.float64)
    target = np.zeros((4, 3), dtype=np.float64)
    assert trajectory_displacement(baseline, masked) == pytest.approx(np.sqrt(3.0))
    assert planning_harm(baseline, masked, target) == pytest.approx(np.sqrt(3.0))
    assert zscore([0.0, 1.0, 2.0])[1] == pytest.approx(0.0)
    combined = combined_harm([0.0, 1.0], [0.0, 2.0], [0.0, 3.0])
    assert combined[0] < combined[1]


def test_objective_metadata_is_consistent():
    assert OBJECTIVES["pdm"]["higher_is_better"] is True
    assert OBJECTIVES["pdm_harm"]["higher_is_better"] is False
    assert objective_higher_is_better("pdm") is True
    assert objective_higher_is_better("combined") is False
    assert objective_value("planning_harm", {"planning_harm": 0.25}) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="unknown objective"):
        objective_value("nope", {})


# ---------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------


def _future_context(layout, scene_token="scene-A"):
    domain = build_domain("future_video", layout, "cpu")
    ctx = TokenContext(
        tokens=torch.zeros(1, layout.total_length, 1),
        layout=layout,
        domain=domain,
        scene_token=scene_token,
    )
    return domain, ctx


def test_oracle_future_token_mask_keeps_exactly_the_listed_tokens(tmp_path):
    layout = build_driveva_layout(4, 15, 26, 2, 2, 1)
    domain, ctx = _future_context(layout)
    mask_path = tmp_path / "tokens.json"
    mask_path.write_text(
        json.dumps(
            {
                "scene-A": {
                    "future_latent_0": [0, 5, 389],
                    "future_latent_1": [1],
                }
            }
        ),
        encoding="utf-8",
    )
    result = OracleFutureTokenMaskSelector(str(mask_path)).select(
        torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
    )
    assert result.K == 4
    assert result.metadata["per_scene_keep"] == [4]
    frame0 = layout.frame_range(2)
    frame1 = layout.frame_range(3)
    assert set(result.keep_global_indices[0].tolist()) == {
        frame0.start + 0,
        frame0.start + 5,
        frame0.start + 389,
        frame1.start + 1,
    }
    assert result.drop_candidate_indices.shape[1] == domain.n_candidate - 4


def test_oracle_future_token_mask_missing_latent_keeps_nothing_there(tmp_path):
    layout = build_driveva_layout(4, 15, 26, 2, 2, 1)
    domain, ctx = _future_context(layout)
    mask_path = tmp_path / "partial.json"
    mask_path.write_text(
        json.dumps({"scene-A": {"future_latent_1": [0, 1]}}), encoding="utf-8"
    )
    result = OracleFutureTokenMaskSelector(str(mask_path)).select(
        torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
    )
    frame1 = layout.frame_range(3)
    assert result.K == 2
    assert set(result.keep_global_indices[0].tolist()) == {frame1.start, frame1.start + 1}


def test_oracle_future_token_mask_accepts_flat_offsets(tmp_path):
    layout = build_driveva_layout(4, 15, 26, 2, 2, 1)
    domain, ctx = _future_context(layout)
    mask_path = tmp_path / "flat.json"
    mask_path.write_text(json.dumps({"scene-A": [0, 390]}), encoding="utf-8")
    result = OracleFutureTokenMaskSelector(str(mask_path)).select(
        torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
    )
    assert set(result.keep_global_indices[0].tolist()) == {
        layout.frame_range(2).start,
        layout.frame_range(3).start,
    }


def test_oracle_future_token_mask_rejects_bad_masks(tmp_path):
    layout = build_driveva_layout(4, 15, 26, 2, 2, 1)
    domain, ctx = _future_context(layout)
    out_of_range = tmp_path / "bad.json"
    out_of_range.write_text(
        json.dumps({"scene-A": {"future_latent_0": [390]}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="outside"):
        OracleFutureTokenMaskSelector(str(out_of_range)).select(
            torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
        )
    missing_scene = tmp_path / "missing.json"
    missing_scene.write_text(
        json.dumps({"other": {"future_latent_0": [0]}}), encoding="utf-8"
    )
    with pytest.raises(KeyError, match="scene-A"):
        OracleFutureTokenMaskSelector(str(missing_scene)).select(
            torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
        )
    wrong_domain = tmp_path / "wrong.json"
    wrong_domain.write_text(
        json.dumps({"scene-A": {"future_latent_0": [0]}}), encoding="utf-8"
    )
    history_domain = build_domain("history", layout, "cpu")
    with pytest.raises(ValueError, match="future_video"):
        OracleFutureTokenMaskSelector(str(wrong_domain)).select(
            torch.zeros(1, history_domain.n_candidate), history_domain, K=None, ctx=ctx
        )


def test_factory_builds_token_oracle_selector(tmp_path):
    mask_path = tmp_path / "tokens.json"
    mask_path.write_text(json.dumps({"scene-A": {"future_latent_0": [0]}}), encoding="utf-8")
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "future_video",
            "scorer": {"name": "random", "seed": 0},
            "selector": {"name": "oracle_future_token_mask", "path": str(mask_path)},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "ratio", "value": 1.0, "reference": "eligible"},
        }
    )
    selector = getattr(press, "selector", None)
    assert isinstance(selector, OracleFutureTokenMaskSelector)


# ---------------------------------------------------------------------------
# runner wiring
# ---------------------------------------------------------------------------


def _token_mask_args(paths, **overrides):
    base = {
        "future_oracle_token_mask_json": None,
        "future_oracle_token_mask_jsons": ",".join(str(path) for path in paths),
        "future_counterfactual_layer": 15,
        "persistent_skip_baseline": False,
        "methods": None,
        "retention_policy": None,
        "domain": "future_video",
        "seed_base": 7,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_future_token_mask_paths_split_dedupe_and_resolve(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        future_oracle_token_mask_json=first,
        future_oracle_token_mask_jsons=f"{first}, {second},",
    )
    paths = _future_oracle_token_mask_paths(args)
    assert [path.name for path in paths] == ["a.json", "b.json"]


def test_future_token_mask_specs_build_one_method_per_mask(tmp_path):
    first = tmp_path / "cand_000.json"
    second = tmp_path / "cand_001.json"
    for path in (first, second):
        path.write_text(json.dumps({"scene-A": {"future_latent_0": [0]}}), encoding="utf-8")
    specs = _method_specs_for_run(_token_mask_args([first, second]), round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_no_press",
        "physical_oracle_future_token_mask_cand_000_hidden_persistent_layer_15",
        "physical_oracle_future_token_mask_cand_001_hidden_persistent_layer_15",
    ]
    selector = specs[1]["press"]["selector"]
    assert selector["name"] == "oracle_future_token_mask"
    assert selector["path"] == str(first.resolve())
    assert specs[1]["press"]["cross_layer_persistence"]["mode"] == "hidden_sequence"
    assert specs[1]["press"]["operator"]["name"] == "kv_prune"


def test_future_token_mask_specs_skip_baseline_and_guard_methods(tmp_path):
    mask = tmp_path / "cand_000.json"
    mask.write_text("{}", encoding="utf-8")
    specs = _method_specs_for_run(
        _token_mask_args([mask], persistent_skip_baseline=True), round_seed=7
    )
    assert [spec["name"] for spec in specs] == [
        "physical_oracle_future_token_mask_cand_000_hidden_persistent_layer_15"
    ]
    with pytest.raises(ValueError, match="cannot be combined with --methods"):
        _method_specs_for_run(
            _token_mask_args([mask], methods="physical_no_press"), round_seed=7
        )
    missing = SimpleNamespace(
        future_oracle_token_mask_json=None,
        future_oracle_token_mask_jsons=str(tmp_path / "nope.json"),
        future_counterfactual_layer=15,
        persistent_skip_baseline=False,
        methods=None,
        retention_policy=None,
        domain="future_video",
        seed_base=7,
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        _method_specs_for_run(missing, round_seed=7)


def test_mask_method_tag_is_filesystem_safe():
    assert _mask_method_tag(__import__("pathlib").Path("/tmp/Cand 000-A.json")) == "cand_000_a"


def test_mask_for_panel_and_compose_top_groups():
    panel = ["s1", "s2"]
    mask = mask_for_panel([0, 390], panel, tokens_per_latent=390, num_latents=2)
    assert mask == {
        "s1": {"future_latent_0": [0], "future_latent_1": [0]},
        "s2": {"future_latent_0": [0], "future_latent_1": [0]},
    }
    groups = ((0, 1), (2, 3), (4, 5))
    selected = compose_top_groups(groups, [0.1, 0.9, 0.5], 4)
    assert selected == [2, 3, 4, 5]


def test_runner_commands_carry_token_masks_and_protocol_flags(tmp_path):
    args = SimpleNamespace(
        python="/usr/bin/python",
        runner=tmp_path / "runner.py",
        nproc_per_node=4,
        layer=15,
        dump_target=True,
        max_eval_tokens=64,
        poc_test_derived=True,
        force_full_scene_set=False,
        skip_plots=True,
        sample_seed_diffusion=None,
    )
    masks = [tmp_path / "a.json", tmp_path / "b.json"]
    command = build_mask_runner_command(args, masks, tmp_path / "out")
    assert "--future-oracle-token-mask-jsons" in command
    assert command[command.index("--future-oracle-token-mask-jsons") + 1] == (
        f"{masks[0]},{masks[1]}"
    )
    assert command[command.index("--max-eval-tokens") + 1] == "64"
    assert "--dump-target-trajectories" in command
    # The driver holds an external baseline, so candidate batches skip the
    # in-suite NoPress arm by default.
    assert "--persistent-skip-baseline" in command
    baseline = build_baseline_runner_command(args, tmp_path / "base")
    assert baseline[baseline.index("--domain") + 1] == "future_video"
    assert baseline[baseline.index("--methods") + 1] == "physical_no_press"
    assert "--future-oracle-token-mask-jsons" not in baseline
    assert "--persistent-skip-baseline" not in baseline
    assert "--persistent-skip-baseline" not in build_mask_runner_command(
        SimpleNamespace(**{**vars(args), "keep_suite_baseline": True}),
        masks,
        tmp_path / "out2",
    )


def test_resolve_actual_suite_root_follows_runner_reruns(tmp_path):
    requested = tmp_path / "suite"
    assert resolve_actual_suite_root(requested) == requested
    (requested / "round01").mkdir(parents=True)
    assert resolve_actual_suite_root(requested) == requested
    rerun = tmp_path / "suite_rerun01"
    (rerun / "round01").mkdir(parents=True)
    (rerun / "suite_summary.json").write_text("{}", encoding="utf-8")
    assert resolve_actual_suite_root(requested) == rerun


def test_run_runner_tolerates_only_the_truncated_poc_exit(tmp_path, monkeypatch):
    output_root = tmp_path / "suite"
    (output_root / "round01").mkdir(parents=True)
    (output_root / "suite_summary.json").write_text("{}", encoding="utf-8")

    def fake_run(cmd, cwd=None, env=None):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        "scripts.search_future_token_set_oracle.subprocess.run", fake_run
    )
    # Truncated POC suites exit 1 by design after writing every artifact.
    _run_runner(["runner"], output_root, {}, allow_truncated_poc=True)
    with pytest.raises(Exception):
        _run_runner(["runner"], output_root, {}, allow_truncated_poc=False)
    empty = tmp_path / "empty"
    with pytest.raises(Exception):
        _run_runner(["runner"], empty, {}, allow_truncated_poc=True)


def test_decision_stats_report_paired_ci_and_tails():
    per_scene = {
        "s1": {"pdm": 1.0},
        "s2": {"pdm": 0.5},
        "s3": {"pdm": 0.0},
        "s4": {"pdm": None},
    }
    baseline = {"s1": 1.0, "s2": 0.0, "s3": 0.25}
    stats = decision_stats(per_scene, baseline, resamples=2000, seed=1)
    assert stats["n"] == 3
    assert stats["delta"] == pytest.approx((0.0 + 0.5 - 0.25) / 3)
    assert stats["improved"] == 1
    assert stats["tied"] == 1
    assert stats["worse"] == 1
    assert stats["zero_candidate"] == 1
    assert stats["zero_baseline"] == 1
    assert stats["extreme_flips"] == 0
    assert len(stats["ci95"]) == 2
    assert stats["ci95"][0] <= stats["delta"] <= stats["ci95"][1]


def test_oracle_future_token_mask_accepts_a_broadcast_entry(tmp_path):
    layout = build_driveva_layout(4, 15, 26, 2, 2, 1)
    domain, ctx = _future_context(layout, scene_token="any-scene")
    mask_path = tmp_path / "broadcast.json"
    mask_path.write_text(
        json.dumps({"*": {"future_latent_0": [0, 1], "future_latent_1": []}}),
        encoding="utf-8",
    )
    result = OracleFutureTokenMaskSelector(str(mask_path)).select(
        torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
    )
    assert result.K == 2
    frame0 = layout.frame_range(2)
    assert set(result.keep_global_indices[0].tolist()) == {
        frame0.start,
        frame0.start + 1,
    }
    # an explicit scene entry still wins over the broadcast entry
    mask_path.write_text(
        json.dumps(
            {
                "*": {"future_latent_0": [0, 1]},
                "any-scene": {"future_latent_1": [0]},
            }
        ),
        encoding="utf-8",
    )
    result = OracleFutureTokenMaskSelector(str(mask_path)).select(
        torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
    )
    assert set(result.keep_global_indices[0].tolist()) == {layout.frame_range(3).start}
    # neither scene nor broadcast entry: loud failure
    mask_path.write_text(json.dumps({"other": {"future_latent_0": [0]}}), encoding="utf-8")
    with pytest.raises(KeyError, match="broadcast"):
        OracleFutureTokenMaskSelector(str(mask_path)).select(
            torch.zeros(1, domain.n_candidate), domain, K=None, ctx=ctx
        )


def test_build_evaluate_mask_candidates_adds_matched_random_control(tmp_path):
    named = tmp_path / "search_final.json"
    named.write_text(
        json.dumps({"*": {"future_latent_0": list(range(390)), "future_latent_1": []}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        include_random_masks=3,
        tokens_per_latent=390,
        num_future_latents=2,
        sample_seed=7,
    )
    candidates = build_evaluate_mask_candidates(
        args, ["s1", "s2"], 390, [named]
    )
    assert sorted(candidates) == ["mask000", "random000", "random001", "random002"]
    assert set(candidates["mask000"]) == {"*"}
    for name in ("random000", "random001", "random002"):
        for scene in ("s1", "s2"):
            flat = flat_from_latent_local(
                candidates[name][scene], tokens_per_latent=390, num_latents=2
            )
            assert len(flat) == 390
    # no random control when the flag is absent
    assert sorted(
        build_evaluate_mask_candidates(
            SimpleNamespace(
                include_random_masks=0,
                tokens_per_latent=390,
                num_future_latents=2,
                sample_seed=7,
            ),
            ["s1"],
            390,
            [named],
        )
    ) == ["mask000"]

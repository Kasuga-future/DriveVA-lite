from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.artifacts import ArtifactWriter
from evaluation.evaluator import Evaluator, SceneSample, build_matched_random_press
from evaluation.navsim_evaluator import DriveVANavsimBackend
from videopress.adapters import DriveVAAdapter
from videopress.adapters.scene_boundary import window_is_single_scene
from videopress.core.budget import TokenBudget
from videopress.core.context import TokenContext
from videopress.core.domain import LastHistoryDomain, build_domain
from videopress.core.layout import build_driveva_layout, decode_video_index_checked
from videopress.core.plan import validate_protocol
from videopress.core.plan import ProbeMode, build_execution_plan
from videopress.core.runtime import EvaluationMode, VideoPressRuntime
from videopress.factory import build_press
from videopress.objectives import EndpointObjective, TrajectoryObjective
from videopress.operators import (
    KVMergeOperator,
    KVPruneOperator,
    ShuffleDroppedOperator,
    ZeroMaskOperator,
)
from videopress.presses import NoPress, ScorerPress, SimilarityMergePress
from videopress.probes import ScoreCache, ScoreKey
from videopress.scorers import (
    ActionContributionStabilityScorer,
    ActionAttentionVNormScorer,
    ActionAttentionVNormTemporalScorer,
    ActionAttentionScorer,
    GradientNormScorer,
    PlanningGradientInputScorer,
    RandomScorer,
    TokenNormScorer,
    original_gradient_input_reduction,
    trajectory_projection_objective,
)
from videopress.selectors import (
    AdaptiveMassSelector,
    AdaptiveSpatialMassSelector,
    HistoryQuotaSelector,
    HistoryThresholdSelector,
    ProtectedTokenSelector,
    SignedRiskSelector,
    ThresholdSelector,
    TopKSelector,
)
from examples.wanvideo.driveva_train.navsim_dataset import _selector_command_3way
from scripts.pack_learned_condition_teacher import _selection_mask
from scripts.analyze_multiseed_panel import latency_summary, load_arm, robustness_summary
from scripts.cache_navsim_split import _atomic_save_buffer
from scripts.run_full_compression_suite import method_specs, persistent_attention_vnorm_spec
from scripts.run_official_navsim_press import (
    _method_specs_for_run,
    _runtime_event_summary,
    parse_layer_sweep,
)


def make_context(batch=1, hidden=8, *, with_qkv=False):
    layout = build_driveva_layout(3, 2, 2, 2, 3, 1)
    domain = LastHistoryDomain().build(layout, "cpu")
    tokens = torch.arange(batch * layout.total_length * hidden, dtype=torch.float32).reshape(
        batch, layout.total_length, hidden
    )
    ctx = TokenContext(
        tokens=tokens,
        layout=layout,
        domain=domain,
        scene_token="scene-A",
        log_id="log-A",
        diffusion_rank=0,
    )
    if with_qkv:
        generator = torch.Generator().manual_seed(123)
        ctx.q = torch.randn(1, 2, layout.total_length, hidden // 2, generator=generator)
        ctx.k = torch.randn(1, 2, layout.total_length, hidden // 2, generator=generator)
        ctx.v = torch.randn(1, 2, layout.total_length, hidden // 2, generator=generator)
    return ctx


def make_press(operator, *, scorer=None, domain="last_history", point="video_input", budget=2):
    return ScorerPress(
        scorer=scorer or RandomScorer(7),
        selector=TopKSelector(),
        operator=operator,
        budget=TokenBudget("absolute", budget),
        domain=domain,
        injection_point=point,
    )


def test_split_cache_atomic_writer_replaces_empty_partial(tmp_path):
    destination = tmp_path / "nested" / "metric_cache.pkl"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"")

    _atomic_save_buffer(destination, b"complete-cache")

    assert destination.read_bytes() == b"complete-cache"
    assert list(destination.parent.glob(".metric_cache.pkl.tmp-*")) == []


def test_multiseed_loader_accepts_suite_per_seed_layout(tmp_path):
    method = "physical_no_press"
    run_dir = tmp_path / "seed1001" / "round01" / method
    run_dir.mkdir(parents=True)
    (run_dir / "pdm_score_test.csv").write_text(
        "token,valid,pdm_score,infer_time_ms\nscene-a,True,0.75,12.5\n",
        encoding="utf-8",
    )

    loaded = load_arm(tmp_path, 1001, method)

    assert loaded == {"scene-a": {"pdm_score": 0.75, "infer_time_ms": 12.5}}


def test_multiseed_latency_summary_is_seed_aware():
    base = np.array([[100.0, 100.0], [200.0, 200.0]])
    method = np.array([[90.0, 90.0], [180.0, 180.0]])

    result = latency_summary(
        base,
        method,
        seeds=[0, 1],
        rng=np.random.default_rng(7),
        resamples=100,
    )

    assert result["post_warmup_count"] == 4
    assert result["speedup_percent"] == pytest.approx(10.0)
    assert result["speedup_random_seed_scene_ci95"] == pytest.approx([10.0, 10.0])


def test_selector_command_folds_navsim_default_into_straight():
    assert _selector_command_3way([0, 0, 0, 1], "sample-default").tolist() == [0.0, 1.0, 0.0]
    assert _selector_command_3way([0, 0, 1, 0], "sample-right").tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="Expected 3- or 4-way"):
        _selector_command_3way([1, 0], "sample-bad")


def test_multiseed_robustness_separates_extremes_and_zero_transitions():
    base = np.array([[0.0, 0.90], [0.80, 0.90]])
    method = np.array([[0.90, 0.89], [0.0, 0.91]])
    result = robustness_summary(
        base,
        method,
        seeds=[1, 2],
        rng=np.random.default_rng(7),
        resamples=200,
    )
    assert result["extreme_count"] == 2
    assert result["non_extreme_count"] == 2
    assert result["base_zero_count"] == 1
    assert result["method_zero_count"] == 1
    assert result["rescued_zero_count"] == 1
    assert result["introduced_zero_count"] == 1
    assert result["non_extreme_seed_averaged_delta"] == pytest.approx(0.0)


def test_threshold_selector_rejects_out_of_range_hard_cap():
    ctx = make_context()
    scores = torch.ones(1, ctx.domain.n_candidate)
    selector = ThresholdSelector(threshold=0.5)
    with pytest.raises(ValueError, match="outside"):
        selector.select(scores, ctx.domain, K=-1)
    with pytest.raises(ValueError, match="outside"):
        selector.select(scores, ctx.domain, K=ctx.domain.n_candidate + 1)


def test_threshold_selector_uses_conservative_rectangular_batch_k():
    ctx = make_context(batch=2)
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.0], [0.9, 0.2, 0.1, 0.0]])
    result = ThresholdSelector(threshold=0.5).select(scores, ctx.domain, K=4)
    assert result.K == 2
    assert result.metadata["proposed_K_per_batch"] == [2, 1]
    assert result.metadata["batch_conservative_fill"] is True
    assert result.keep_candidate_indices.tolist() == [[0, 1], [0, 1]]


def test_history_threshold_applies_independent_oldest_to_newest_thresholds():
    ctx = make_context()
    ctx = replace(ctx, domain=build_domain("history", ctx.layout, "cpu"))
    scores = torch.tensor([[0.6, 0.4, 0.9, 0.1, 0.7, 0.85, 0.2, 0.95]])
    result = HistoryThresholdSelector([0.5, 0.8]).select(
        scores, ctx.domain, K=8, ctx=ctx
    )
    assert result.K == 4
    assert result.keep_candidate_indices.tolist() == [[0, 2, 5, 7]]
    assert result.metadata["proposed_K_per_latent"] == [[2, 2]]
    assert result.metadata["thresholds_oldest_to_newest"] == [0.5, 0.8]


def test_history_quota_enforces_independent_per_latent_counts():
    ctx = make_context()
    ctx = replace(ctx, domain=build_domain("history", ctx.layout, "cpu"))
    scores = torch.arange(8, dtype=torch.float32).view(1, -1)
    result = HistoryQuotaSelector([0.5, 0.25]).select(
        scores, ctx.domain, K=3, ctx=ctx
    )
    assert result.K == 3
    assert result.keep_candidate_indices.tolist() == [[3, 2, 7]]
    assert result.metadata["quota_oldest_to_newest"] == [2, 1]
    with pytest.raises(ValueError, match="does not match"):
        HistoryQuotaSelector([0.5, 0.25]).select(
            scores, ctx.domain, K=4, ctx=ctx
        )


def test_protected_selector_forces_context_mask_without_changing_k():
    ctx = make_context()
    ctx.metadata["critical_token_mask"] = [False, False, False, True]
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    selector = ProtectedTokenSelector(TopKSelector())
    result = selector.select(scores, ctx.domain, K=2, ctx=ctx)
    assert result.K == 2
    assert result.keep_candidate_indices.tolist() == [[0, 3]]
    assert result.metadata["protected_replacements_per_batch"] == [1]
    assert result.metadata["protection_budget_preserved"] is True


def test_protected_selector_accepts_global_mask_and_factory_wrapper():
    ctx = make_context()
    global_mask = torch.zeros(ctx.layout.total_length, dtype=torch.bool)
    global_mask[ctx.domain.candidate_indices[-1]] = True
    ctx.metadata["object_mask"] = global_mask
    press = build_press(
        {
            "name": "scorer_press",
            "scorer": {"name": "token_norm"},
            "selector": {
                "name": "protected",
                "metadata_key": "object_mask",
                "base": {"name": "topk"},
            },
            "operator": {"name": "zero"},
            "domain": "last_history",
            "budget": {"type": "absolute", "value": 2},
        }
    )
    assert isinstance(press.selector, ProtectedTokenSelector)
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1]])
    result = press.selector.select(scores, ctx.domain, K=2, ctx=ctx)
    assert set(result.keep_candidate_indices[0].tolist()) == {0, 3}


def test_protected_selector_fails_closed_when_mask_missing_or_over_budget():
    ctx = make_context()
    selector = ProtectedTokenSelector(TopKSelector())
    scores = torch.ones(1, ctx.domain.n_candidate)
    with pytest.raises(KeyError, match="critical_token_mask"):
        selector.select(scores, ctx.domain, K=2, ctx=ctx)
    ctx.metadata["critical_token_mask"] = [True, True, True, False]
    with pytest.raises(ValueError, match="exceeds selected K"):
        selector.select(scores, ctx.domain, K=2, ctx=ctx)


def test_runtime_summary_averages_per_latent_counts_across_diffusion_steps():
    def event(rank, counts):
        return SimpleNamespace(
            key=SimpleNamespace(layer_idx=15, diffusion_rank=rank),
            result=SimpleNamespace(
                metadata={
                    "n_kept": sum(counts),
                    "persistent_selection_reused": False,
                    "selected_history_latent_counts": [counts],
                    "selected_history_latent_ratios": [
                        [value / 390.0 for value in counts]
                    ],
                    "effective_history_latent_kept_counts": [counts],
                    "effective_history_latent_keep_ratios": [
                        [value / 390.0 for value in counts]
                    ],
                }
            ),
        )

    runtime = SimpleNamespace(
        events=[event(0, [100, 200]), event(1, [140, 180])],
        selector_latency_ms=0.0,
    )
    summary = _runtime_event_summary(runtime)
    assert summary["selected_history_latent_counts_mean"] == [120.0, 190.0]
    assert summary["effective_history_latent_kept_counts_mean"] == [120.0, 190.0]


def test_noop_artifacts_are_independent_of_scores(tmp_path):
    ctx = make_context()
    result = NoPress().apply(ctx)
    writer = ArtifactWriter(tmp_path)
    writer.add_result(ctx, result)
    writer.write_tokens()
    assert list((tmp_path / "artifacts" / "masks").glob("*.pt"))
    assert list((tmp_path / "artifacts" / "mappings").glob("*.json"))
    assert not list((tmp_path / "artifacts" / "scores").glob("*.pt"))


def test_mapping_artifact_is_written_without_selection_and_is_not_overwritten(tmp_path):
    ctx = make_context(with_qkv=True)
    writer = ArtifactWriter(tmp_path)
    for layer in (0, 1):
        layer_ctx = replace(ctx, layer_idx=layer)
        result = SimilarityMergePress(TokenBudget("absolute", 2), domain="last_history").apply(layer_ctx)
        writer.add_result(layer_ctx, result)
    files = sorted((tmp_path / "artifacts" / "mappings").glob("*.json"))
    assert len(files) == 2
    assert all('"mapping"' in path.read_text() for path in files)


def test_video_index_decode_rejects_trajectory_token():
    layout = make_context().layout
    with pytest.raises(ValueError, match="not a video token"):
        decode_video_index_checked(layout.traj_all.start, layout)


def test_gradient_objective_requires_explicit_context_target():
    ctx = make_context()
    outputs = torch.ones(1, ctx.layout.future_action.length, 2)
    ctx.metadata["target_trajectory"] = torch.zeros_like(outputs)
    assert torch.equal(TrajectoryObjective().compute(outputs, ctx), torch.tensor(1.0))
    assert torch.equal(EndpointObjective().compute(outputs, ctx), torch.tensor(1.0))
    ctx.metadata.pop("target_trajectory")
    with pytest.raises(RuntimeError, match="target_trajectory"):
        TrajectoryObjective().compute(outputs, ctx)


def test_gradient_typeerror_from_callback_is_not_reinterpreted():
    ctx = make_context()

    def forward(_tokens, _ctx):
        raise TypeError("internal callback error")

    scorer = GradientNormScorer(forward_fn=forward, objective=lambda output, _ctx: output)
    with pytest.raises(TypeError, match="internal callback error"):
        scorer.score(ctx)


def test_original_trajectory_projection_objective_and_reduction():
    prediction = torch.tensor([[[9.0, 9.0, 9.0], [3.0, 4.0, 0.0]]], requires_grad=True)
    objective, points = trajectory_projection_objective(prediction, 1)
    assert points.shape == (1, 1, 3)
    assert torch.allclose(objective, torch.tensor(5.0))
    objective.backward()
    assert torch.allclose(prediction.grad[:, 0], torch.zeros(1, 3))
    gradient = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    inputs = torch.tensor([[[1.0, 1.0], [5.0, 3.0]]])
    scores = original_gradient_input_reduction(gradient, inputs)
    assert scores.shape == (1, 2)
    assert torch.allclose(scores, torch.tensor([[5.0, 6.0]]))
    assert PlanningGradientInputScorer().describe()["objective_type"] == "detached_unit_trajectory_projection_v1"


def test_random_mask_is_independent_of_batch_position():
    single = make_context()
    single_scores = RandomScorer(seed=9, scope="scene").score(single)
    batch = make_context(batch=2)
    batch.scene_token = "other"
    batch.metadata["scene_tokens"] = ["other", "scene-A"]
    batch_scores = RandomScorer(seed=9, scope="scene").score(batch)
    assert torch.equal(single_scores[0], batch_scores[1])
    assert RandomScorer(seed=9, scope="scene_layer_step").describe()["scope"] == "scene_layer_step"


def test_matched_random_copies_layer_routing():
    method = make_press(
        ZeroMaskOperator(),
        scorer=ActionAttentionScorer(layer=15),
    )
    method.random_scope = "scene_layer"
    control = build_matched_random_press(method, seed=4)
    assert control.scorer.layer == 15
    assert control.scorer.scope == "scene_layer"
    assert control.operator.describe() == method.operator.describe()
    assert control.budget == method.budget


def test_shuffle_drop_obeys_selection_and_preserves_protected_tokens():
    ctx = make_context()
    scores = torch.arange(ctx.domain.n_candidate, dtype=torch.float32).view(1, -1)
    selection = TopKSelector().select(scores, ctx.domain, 0, ctx)
    result = ShuffleDroppedOperator(seed=1).apply(ctx, selection)
    protected = torch.where(ctx.domain.protected_mask)[0]
    assert torch.equal(result.output[:, protected], ctx.tokens[:, protected])
    full_selection = TopKSelector().select(scores, ctx.domain, ctx.domain.n_candidate, ctx)
    full_result = ShuffleDroppedOperator(seed=1).apply(ctx, full_selection)
    assert torch.equal(full_result.output, ctx.tokens)
    assert result.metadata["selection_dependent"] is True


def test_zero_operator_zeros_unselected_domain_only():
    ctx = make_context()
    scores = torch.arange(ctx.domain.n_candidate, dtype=torch.float32).view(1, -1)
    selection = TopKSelector().select(scores, ctx.domain, 2, ctx)
    result = ZeroMaskOperator().apply(ctx, selection)
    keep = selection.keep_global_indices[0]
    dropped = ctx.domain.candidate_mask.clone()
    dropped[keep] = False
    protected = ctx.domain.protected_mask
    assert torch.equal(result.output[:, keep], ctx.tokens[:, keep])
    assert torch.count_nonzero(result.output[:, dropped]) == 0
    assert torch.equal(result.output[:, protected], ctx.tokens[:, protected])
    assert result.metadata["zero_scope"] == "unselected_within_domain"


def test_frozen_probe_cache_avoids_second_forward(tmp_path):
    ctx = make_context()
    calls = []

    def forward(tokens, probe_ctx):
        calls.append(1)
        return tokens[:, probe_ctx.domain.candidate_indices].pow(2).sum()

    press = make_press(
        ZeroMaskOperator(),
        scorer=GradientNormScorer(forward_fn=forward, objective=lambda output, _ctx: output),
    )
    runtime = VideoPressRuntime(press, score_cache=ScoreCache(tmp_path))
    sample = SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=0)
    runtime.begin_sample(sample, ctx.layout)
    runtime.set_context(ctx)
    first = runtime.execute_press(ctx)
    second = runtime.execute_press(ctx)
    assert first.selection.K == second.selection.K == 2
    assert len(calls) == 1
    assert runtime.score_cache.load_ranking(runtime.score_key(ctx)) is not None


def test_causal_attention_can_consume_a_precomputed_full_probe(tmp_path):
    ctx = make_context()
    scorer = ActionAttentionScorer(layer=15)
    press = make_press(ZeroMaskOperator(), scorer=scorer, budget=2)
    cache = ScoreCache(tmp_path)
    scores = torch.arange(ctx.domain.n_candidate, dtype=torch.float32).view(1, -1)
    runtime = VideoPressRuntime(press, score_cache=cache)
    runtime.begin_sample(SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=0), ctx.layout)
    key = runtime.score_key(ctx)
    cache.save(key, scores, ranking=scores.argsort(dim=-1, descending=True, stable=True))
    result = runtime.execute_press(ctx)
    assert result.selection.K == 2
    assert result.metadata["score_cache"]["ranking_frozen"] is True


def test_score_cache_key_separates_history_domains(tmp_path):
    ctx = make_context()
    press = make_press(ZeroMaskOperator(), scorer=ActionAttentionScorer(layer=15))
    runtime = VideoPressRuntime(press, score_cache=ScoreCache(tmp_path))
    last_key = runtime.score_key(ctx)
    history_ctx = TokenContext(
        tokens=ctx.tokens,
        layout=ctx.layout,
        domain=build_domain("history", ctx.layout, "cpu"),
        scene_token=ctx.scene_token,
        diffusion_rank=ctx.diffusion_rank,
    )
    history_key = runtime.score_key(history_ctx)
    assert last_key.scorer_signature != history_key.scorer_signature
    assert runtime.score_cache.path_for(last_key) != runtime.score_cache.path_for(history_key)
    assert "domain=last_history" in last_key.scorer_signature
    assert "domain=history" in history_key.scorer_signature


def test_policy_suite_excludes_only_similarity_merge():
    for policy in (
        "drop_previous_keep_last_100",
        "drop_previous_keep_last_50",
        "joint_keep_50",
        "joint_keep_25",
        "per_latent_keep_50",
        "per_latent_keep_25",
    ):
        specs = method_specs(7, retention_policy=policy)
        assert specs
        assert all(spec["press"]["name"] != "similarity_merge" for spec in specs)
        for spec in specs:
            if spec["press"]["name"] == "noop":
                continue
            assert spec["press"]["retention_policy"] == policy
            assert spec["press"]["domain"] == "history"


def test_teacher_artifact_mask_rebuilds_grouped_policies():
    scores = torch.arange(8, dtype=torch.float32)
    positions = torch.tensor(
        [(latent, row, col) for latent in range(2) for row in range(2) for col in range(2)]
    )
    assert torch.where(
        _selection_mask(scores, positions, 2, "drop_previous_keep_last_50")
    )[0].tolist() == [6, 7]
    assert torch.where(
        _selection_mask(scores, positions, 2, "per_latent_keep_25")
    )[0].tolist() == [3, 7]
    assert torch.where(
        _selection_mask(scores, positions, 2, "joint_keep_25")
    )[0].tolist() == [6, 7]


def test_attention_vnorm_single_head_uses_matched_value_head():
    ctx = make_context(with_qkv=True)
    scorer = ActionAttentionVNormScorer(head_mode="single", head_index=1, action_mode="mean")
    scores = scorer.score(ctx)
    assert scorer.describe()["value_norm_head_mode"] == "matched"
    q = ctx.q
    k = ctx.k
    v = ctx.v
    action = torch.arange(ctx.layout.future_action.start, ctx.layout.future_action.end)
    logits = torch.matmul(q[:, 1:2, action].float(), k[:, 1:2].float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    expected_attn = torch.softmax(logits, dim=-1).mean(dim=2).squeeze(1).index_select(1, ctx.domain.candidate_indices)
    expected = expected_attn * torch.linalg.vector_norm(
        v[:, 1].index_select(1, ctx.domain.candidate_indices).float(), dim=-1
    )
    assert torch.allclose(scores, expected)


def test_action_contribution_preserves_head_pairing():
    ctx = make_context(with_qkv=True)
    scorer = ActionContributionStabilityScorer(
        layer=3,
        observation_start_layer=3,
        redundancy_weight=0.0,
        stability_weight=0.0,
    )
    scores = scorer.score(ctx)
    action = torch.arange(ctx.layout.future_action.start, ctx.layout.future_action.end)
    logits = torch.matmul(
        ctx.q[:, :, action].float(), ctx.k.float().transpose(-1, -2)
    ) / (ctx.q.shape[-1] ** 0.5)
    attention = torch.softmax(logits, dim=-1).index_select(
        -1, ctx.domain.candidate_indices
    )
    value_norm = torch.linalg.vector_norm(
        ctx.v.index_select(2, ctx.domain.candidate_indices).float(), dim=-1
    )
    expected = torch.sqrt(
        (attention.square() * value_norm.unsqueeze(2).square()).sum(dim=1)
        + scorer.eps
    ).mean(dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(scores, expected)


def test_adaptive_mass_selector_uses_confidence_tiers_and_batch_fallback():
    ctx = make_context()
    selector = AdaptiveMassSelector(
        ratios=(0.25, 0.5, 1.0),
        mass_thresholds=(0.6, 0.7),
        gap_thresholds=(0.1, 0.1),
        gap_window=1,
    )
    concentrated = torch.tensor([[10.0, 5.0, 0.0, 0.0]])
    selected = selector.select(concentrated, ctx.domain, 4, ctx)
    assert selected.K == 1
    assert selected.metadata["proposed_K_per_batch"] == [1]

    mixed = torch.tensor(
        [[10.0, 5.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    )
    batch_ctx = make_context(batch=2)
    selected = selector.select(mixed, batch_ctx.domain, 4, batch_ctx)
    assert selected.metadata["proposed_K_per_batch"] == [1, 4]
    assert selected.K == 4
    empty = selector.select(concentrated, ctx.domain, 0, ctx)
    assert empty.K == 0
    assert empty.keep_global_indices.shape == (1, 0)


def test_protocol_validator_rejects_semantic_mixes():
    causal = make_press(ZeroMaskOperator(), point="video_input")
    physical = make_press(KVPruneOperator(), point="self_attn_kv")
    bad_causal = make_press(KVPruneOperator(), point="video_input")
    bad_physical = make_press(ZeroMaskOperator(), point="self_attn_kv")
    validate_protocol(causal, EvaluationMode.CAUSAL)
    validate_protocol(physical, EvaluationMode.PHYSICAL)
    with pytest.raises(ValueError):
        validate_protocol(bad_causal, EvaluationMode.CAUSAL)
    with pytest.raises(ValueError):
        validate_protocol(bad_physical, EvaluationMode.PHYSICAL)
    with pytest.raises(NotImplementedError):
        validate_protocol(make_press(ZeroMaskOperator(), point="block_input"), EvaluationMode.CAUSAL)


def test_attention_causal_is_declared_as_a_forward_probe():
    press = make_press(
        ZeroMaskOperator(),
        scorer=ActionAttentionScorer(layer=15),
        point="video_input",
    )
    plan = build_execution_plan(press, EvaluationMode.CAUSAL)
    assert plan.probe_required is True
    assert plan.probe_mode is ProbeMode.FORWARD_PROBE


def test_driveva_video_input_hook_uses_configured_press_domain():
    class PatchModel:
        def patchify(self, x):
            return x

    model = PatchModel()

    def model_fn(**kwargs):
        return model.patchify(kwargs["latents"])

    pipe = SimpleNamespace(dit=model, dit2=None, model_fn=model_fn)
    layout = build_driveva_layout(3, 2, 2, 2, 3, 1)
    press = make_press(ZeroMaskOperator(), domain="history", point="video_input")
    runtime = VideoPressRuntime(press, mode="causal", adapter=DriveVAAdapter())
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={"domain": "last_history"}, diffusion_rank=0),
        layout,
    )
    video = torch.ones(1, 1, 3, 2, 2)
    with runtime.activate(pipe):
        output = pipe.model_fn(
            latents=video,
            longcat_latents=torch.ones(1, 1, 2, 2, 2),
            traj_tokens=torch.ones(1, 3, 1),
            traj_prefix_len=1,
            timestep=torch.tensor(7),
        )
    assert output.shape == video.shape
    event = runtime.events[0]
    assert event.result.metadata["configured_domain"] == "history"
    assert event.result.metadata["resolved_domain"] == "history"
    assert event.result.metadata["candidate_start"] == 0
    assert event.result.metadata["candidate_end"] == event.result.metadata["n_candidate"]
    assert event.result.metadata["selection_candidate_valid"] is True
    assert event.result.metadata["selection_candidate_unique"] is True
    assert event.result.metadata["scene_token"] == "scene-A"
    assert event.context.domain.n_candidate == 8
    assert torch.equal(output[:, :, :2], torch.zeros_like(output[:, :, :2])) is False
    assert event.key.diffusion_rank == 7


def test_scene_boundary_guard_requires_one_token_and_one_name():
    frames = [
        {"scene_token": "scene-A", "scene_name": "segment-1", "frame_idx": 4},
        {"scene_token": "scene-A", "scene_name": "segment-1", "frame_idx": 5},
    ]
    assert window_is_single_scene(frames) is True
    assert window_is_single_scene(frames + [{"scene_token": "scene-B", "scene_name": "segment-2"}]) is False
    assert window_is_single_scene([{"scene_token": "scene-A", "scene_name": None}]) is False


def test_similarity_merge_reduces_real_kv_length_in_adapter():
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context(with_qkv=False)
    attention = SelfAttention(dim=8, num_heads=2)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)]), dit2=None)
    press = SimilarityMergePress(TokenBudget("absolute", 2), domain="last_history")
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.layout = ctx.layout
    runtime.current_scene = "scene-A"
    runtime.current_sample = SimpleNamespace(metadata={"domain": "history"}, diffusion_rank=0)
    runtime.install(pipe)
    x = torch.randn(1, ctx.layout.total_length, 8)
    q2, k2, v2 = attention.tokenpress_hook(x, x, x, layer_idx=0)
    runtime.remove(pipe)
    assert q2.shape[1] == ctx.layout.total_length
    assert k2.shape[1] == v2.shape[1] == ctx.domain.n_protected + 2
    assert runtime.events[0].result.metadata["mapping_complete"] is True


def test_attention_vnorm_selection_persists_across_deeper_layers():
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context(with_qkv=False)
    attentions = [SelfAttention(dim=8, num_heads=2) for _ in range(4)]
    pipe = SimpleNamespace(
        dit=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
        ),
        dit2=None,
    )
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "last_history",
            "scorer": {
                "name": "action_attention_vnorm",
                "layer": 1,
                "head_mode": "mean",
                "action_mode": "mean",
            },
            "selector": {"name": "topk"},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "absolute", "value": 2},
            "cross_layer_persistence": {"enabled": True},
        }
    )
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.install(pipe)
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=7),
        ctx.layout,
    )
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(1, ctx.layout.total_length, 8, generator=generator)
    outputs = [
        attention.tokenpress_hook(x, x, x, layer_idx=layer)
        for layer, attention in enumerate(attentions)
    ]
    runtime.remove(pipe)

    assert outputs[0][1].shape[1] == ctx.layout.total_length
    assert all(
        outputs[layer][1].shape[1] == ctx.domain.n_protected + 2
        for layer in (1, 2, 3)
    )
    assert [event.key.layer_idx for event in runtime.events] == [1, 2, 3]
    source, reused_2, reused_3 = [event.result for event in runtime.events]
    assert all(result.output is None and result.aux == {} for result in (source, reused_2, reused_3))
    assert all(event.context.q is None for event in runtime.events)
    assert all(event.context.tokens.shape[-1] == 0 for event in runtime.events)
    assert source.scores is not None
    assert reused_2.scores is None and reused_3.scores is None
    assert source.metadata["persistent_selection_reused"] is False
    assert reused_2.metadata["persistent_selection_reused"] is True
    assert reused_3.metadata["persistent_selection_reused"] is True
    assert all(
        result.metadata["selection_source_layer"] == 1
        for result in (source, reused_2, reused_3)
    )
    assert torch.equal(
        source.mapping.output_to_input,
        reused_3.mapping.output_to_input,
    )


def _run_persistence_probe(config, *, n_blocks=4):
    """Run one source layer + deeper layers and return (press, runtime, outputs)."""
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context(with_qkv=False)
    attentions = [SelfAttention(dim=8, num_heads=2) for _ in range(n_blocks)]
    pipe = SimpleNamespace(
        dit=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
        ),
        dit2=None,
    )
    press = build_press(config)
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.install(pipe)
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=7),
        ctx.layout,
    )
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(1, ctx.layout.total_length, 8, generator=generator)
    outputs = [
        attention.tokenpress_hook(x, x, x, layer_idx=layer)
        for layer, attention in enumerate(attentions)
    ]
    runtime.remove(pipe)
    return ctx, runtime, outputs


def test_cross_layer_persistence_rejects_end_layer_equal_to_source_layer() -> None:
    """BUG-13: `enabled=True, end_layer == source_layer` used to persist nothing
    while still being stamped `cross_layer_persistent: True`.  It must now be
    rejected, with `enabled=False` as the explicit one-shot alternative."""
    from videopress.core.persistence import CrossLayerPersistence

    with pytest.raises(ValueError, match="persists nothing"):
        CrossLayerPersistence(
            enabled=True, end_layer=15, mode="kv_only"
        ).validate_for(15)
    with pytest.raises(ValueError, match="cannot precede"):
        CrossLayerPersistence(
            enabled=True, end_layer=14, mode="kv_only"
        ).validate_for(15)
    # The legitimate settings are untouched.
    CrossLayerPersistence(enabled=True, end_layer=None).validate_for(15)
    CrossLayerPersistence(enabled=True, end_layer=20).validate_for(15)
    CrossLayerPersistence(enabled=False, end_layer=15).validate_for(15)

    config = {
        "name": "scorer_press",
        "injection_point": "self_attn_kv",
        "domain": "last_history",
        "scorer": {"name": "action_attention_vnorm", "layer": 1},
        "selector": {"name": "topk"},
        "operator": {"name": "kv_prune"},
        "budget": {"type": "absolute", "value": 2},
        "cross_layer_persistence": {"enabled": True, "end_layer": 1},
    }
    with pytest.raises(ValueError, match="persists nothing"):
        build_press(config)


def test_disabled_persistence_is_an_explicit_one_shot_prune() -> None:
    """`enabled=False` must be distinguishable from cross-layer persistence:
    the source layer prunes once, deeper layers are untouched, and the metadata
    says so instead of claiming `cross_layer_persistent: True` (BUG-13)."""
    from videopress.core.persistence import CrossLayerPersistence

    persistence = CrossLayerPersistence(enabled=False)
    assert persistence.persists_beyond(1) is False
    assert CrossLayerPersistence(enabled=True, end_layer=None).persists_beyond(1) is True
    assert CrossLayerPersistence(enabled=True, end_layer=20).persists_beyond(1) is True
    assert CrossLayerPersistence(enabled=True, end_layer=20).persists_beyond(20) is False

    config = {
        "name": "scorer_press",
        "injection_point": "self_attn_kv",
        "domain": "last_history",
        "scorer": {"name": "action_attention_vnorm", "layer": 1},
        "selector": {"name": "topk"},
        "operator": {"name": "kv_prune"},
        "budget": {"type": "absolute", "value": 2},
        "cross_layer_persistence": {"enabled": False},
    }
    ctx, runtime, outputs = _run_persistence_probe(config)
    pruned = ctx.domain.n_protected + 2

    # One-shot: only the source layer prunes; deeper layers keep full length and
    # produce no event at all.
    assert outputs[1][1].shape[1] == pruned
    assert outputs[2][1].shape[1] == ctx.layout.total_length
    assert outputs[3][1].shape[1] == ctx.layout.total_length
    assert [event.key.layer_idx for event in runtime.events] == [1]

    metadata = runtime.events[0].result.metadata
    assert metadata["cross_layer_persistent"] is False
    assert metadata["cross_layer_persistence_mode"] == "one_shot"
    assert metadata["cross_layer_persistence_configured"] is False
    assert metadata["persistent_selection_reused"] is False
    assert metadata["selection_source_layer"] == 1


def test_enabled_persistence_still_claims_and_reuses_across_layers() -> None:
    """The contrast case: with persistence genuinely enabled the source event is
    stamped persistent and the deeper layers reuse it."""
    config = {
        "name": "scorer_press",
        "injection_point": "self_attn_kv",
        "domain": "last_history",
        "scorer": {"name": "action_attention_vnorm", "layer": 1},
        "selector": {"name": "topk"},
        "operator": {"name": "kv_prune"},
        "budget": {"type": "absolute", "value": 2},
        "cross_layer_persistence": {"enabled": True},
    }
    ctx, runtime, outputs = _run_persistence_probe(config)
    pruned = ctx.domain.n_protected + 2
    assert all(outputs[layer][1].shape[1] == pruned for layer in (1, 2, 3))
    source = runtime.events[0].result.metadata
    assert source["cross_layer_persistent"] is True
    assert source["cross_layer_persistence_mode"] == "kv_only"
    assert source["persistent_selection_reused"] is False
    assert runtime.events[1].result.metadata["persistent_selection_reused"] is True
    assert runtime.events[1].result.metadata["cross_layer_persistent"] is True


def test_contribution_scorer_observes_earlier_layers_without_compressing():
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context(with_qkv=False)
    attentions = [SelfAttention(dim=8, num_heads=2) for _ in range(5)]
    pipe = SimpleNamespace(
        dit=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
        ),
        dit2=None,
    )
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "last_history",
            "scorer": {
                "name": "action_contribution_stability",
                "layer": 3,
                "observation_start_layer": 1,
                "redundancy_weight": 0.1,
                "stability_weight": 0.25,
            },
            "selector": {"name": "topk"},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "absolute", "value": 2},
            "cross_layer_persistence": {"enabled": True},
        }
    )
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.install(pipe)
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=7),
        ctx.layout,
    )
    generator = torch.Generator().manual_seed(2027)
    x = torch.randn(1, ctx.layout.total_length, 8, generator=generator)
    outputs = [
        attention.tokenpress_hook(x, x, x, layer_idx=layer)
        for layer, attention in enumerate(attentions)
    ]
    runtime.remove(pipe)
    assert outputs[1][1].shape[1] == ctx.layout.total_length
    assert outputs[2][1].shape[1] == ctx.layout.total_length
    assert outputs[3][1].shape[1] == ctx.domain.n_protected + 2
    assert [event.key.layer_idx for event in runtime.events] == [3, 4]
    diagnostics = runtime.events[0].result.metadata["score_diagnostics"]
    assert diagnostics["observation_layers_seen"] == [1, 2]
    assert diagnostics["score_layers_used"] == 3
    assert press.scorer._observations == {}


def test_learned_scorer_reads_earlier_features_and_starts_compression_later(tmp_path):
    from diffsynth.models.wan_video_dit import SelfAttention
    from safetensors.torch import save_file
    from videopress.scorers.learned_selector import DynamicTokenSelector

    ctx = make_context(with_qkv=False)
    network = DynamicTokenSelector(token_dim=8)
    checkpoint = tmp_path / "selector.safetensors"
    save_file(
        {f"selector.{key}": value for key, value in network.state_dict().items()},
        checkpoint,
    )
    attentions = [SelfAttention(dim=8, num_heads=2) for _ in range(4)]
    model = SimpleNamespace(
        blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
    )
    pipe = SimpleNamespace(dit=model, dit2=None)
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "last_history",
            "scorer": {
                "name": "learned_planning_selector",
                "checkpoint": str(checkpoint),
                "token_dim": 8,
                "feature_layer": 1,
                "layer": 2,
            },
            "selector": {"name": "topk"},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "absolute", "value": 2},
            "cross_layer_persistence": {"enabled": True},
        }
    )
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.install(pipe)
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=1000),
        ctx.layout,
    )
    x = torch.randn(1, ctx.layout.total_length, 8)
    outputs = []
    for layer, attention in enumerate(attentions):
        model._tokenpress_pre_block_hidden = x
        model._tokenpress_pre_block_layer = layer
        outputs.append(attention.tokenpress_hook(x, x, x, layer_idx=layer))
    runtime.remove(pipe)

    assert outputs[1][1].shape[1] == ctx.layout.total_length
    assert outputs[2][1].shape[1] == ctx.domain.n_protected + 2
    assert outputs[3][1].shape[1] == ctx.domain.n_protected + 2
    assert [event.key.layer_idx for event in runtime.events] == [2, 3]
    source = runtime.events[0].result.metadata
    assert source["selection_source_layer"] == 2
    assert source["score_diagnostics"]["feature_layer"] == 1
    assert source["score_diagnostics"]["feature_cache_reused"] is True


def test_hidden_sequence_persistence_gathers_and_restores_original_positions():
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context(with_qkv=False)
    attentions = [SelfAttention(dim=8, num_heads=2) for _ in range(4)]
    model = SimpleNamespace(
        blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
    )
    pipe = SimpleNamespace(dit=model, dit2=None)
    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "last_history",
            "scorer": {
                "name": "action_attention_vnorm",
                "layer": 1,
                "head_mode": "mean",
                "action_mode": "mean",
            },
            "selector": {"name": "topk"},
            "operator": {"name": "kv_prune"},
            "budget": {"type": "absolute", "value": 2},
            "cross_layer_persistence": {
                "enabled": True,
                "mode": "hidden_sequence",
            },
        }
    )
    runtime = VideoPressRuntime(press, mode="physical", adapter=DriveVAAdapter())
    runtime.install(pipe)
    runtime.begin_sample(
        SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=7),
        ctx.layout,
    )
    controller = model._tokenpress_hidden_sequence_controller
    length = ctx.layout.total_length
    x = torch.arange(length * 8, dtype=torch.float32).reshape(1, length, 8)
    freqs = torch.arange(length * 4, dtype=torch.float32).reshape(length, 1, 4)
    t_mod = torch.arange(length * 6 * 8, dtype=torch.float32).reshape(
        1, length, 6, 8
    )
    controller.begin_forward(x, freqs, t_mod, num_blocks=len(attentions))
    _, source_k, _ = attentions[1].tokenpress_hook(x, x, x, layer_idx=1)
    short_x, short_freqs, short_t_mod = controller.after_block(
        1, x, freqs, t_mod
    )
    keep = runtime.last_result.mapping.output_to_input

    assert source_k.shape[1] == ctx.domain.n_protected + 2
    assert short_x.shape[1] == source_k.shape[1]
    assert torch.equal(short_x, x.gather(1, keep.unsqueeze(-1).expand_as(short_x)))
    expanded_freqs = freqs.unsqueeze(0)
    assert torch.equal(
        short_freqs,
        expanded_freqs.gather(
            1,
            keep.view(1, -1, 1, 1).expand_as(short_freqs),
        ),
    )
    assert torch.equal(
        short_t_mod,
        t_mod.gather(1, keep.view(1, -1, 1, 1).expand_as(short_t_mod)),
    )
    restored = controller.finish_forward(short_x + 1)
    expected = torch.zeros_like(x)
    expected.scatter_(1, keep.unsqueeze(-1).expand_as(short_x), short_x + 1)
    assert torch.equal(restored, expected)
    assert runtime.events[0].result.metadata["hidden_sequence_length_after"] == int(
        short_x.shape[1]
    )
    assert runtime.events[0].result.metadata[
        "hidden_sequence_compressed_layer_count"
    ] == 2
    runtime.remove(pipe)
    assert not hasattr(model, "_tokenpress_hidden_sequence_controller")


def test_hidden_sequence_persistence_rejects_non_pruning_operator():
    config = {
        "name": "scorer_press",
        "injection_point": "self_attn_kv",
        "domain": "last_history",
        "scorer": {"name": "action_attention_vnorm", "layer": 16},
        "selector": {"name": "topk"},
        "operator": {"name": "kv_merge"},
        "budget": {"type": "ratio", "value": 0.5},
        "cross_layer_persistence": {
            "enabled": True,
            "mode": "hidden_sequence",
        },
    }
    with pytest.raises(ValueError, match="requires operator=kv_prune"):
        build_press(config)


def test_persistent_layer_sweep_parser_and_default_method():
    assert parse_layer_sweep("all") == list(range(30))
    assert parse_layer_sweep("0,8,12-14,29") == [0, 8, 12, 13, 14, 29]
    with pytest.raises(ValueError, match="outside"):
        parse_layer_sweep("30")
    specs = {spec["name"]: spec for spec in method_specs(7)}
    persistent = specs["physical_attention_vnorm_kv_prune_persistent"]
    assert persistent["press"]["scorer"]["layer"] == 15
    assert persistent["press"]["cross_layer_persistence"] == {
        "enabled": True,
        "end_layer": None,
        "mode": "kv_only",
    }


def test_joint_keep_50_combines_with_hidden_sequence_persistence():
    args = SimpleNamespace(
        persistent_layer_sweep="16",
        methods=None,
        persistent_keep_ratio=0.5,
        persistent_end_layer=None,
        persistent_mode="hidden_sequence",
        retention_policy="joint_keep_50",
        domain="last_history",
    )
    baseline, persistent = _method_specs_for_run(args, round_seed=7)
    assert baseline["name"] == "physical_no_press"
    assert persistent["name"] == "physical_attention_vnorm_hidden_persistent_layer_16"
    press = persistent["press"]
    assert press["domain"] == "history"
    assert press["retention_policy"] == "joint_keep_50"
    assert press["selector"] == {"name": "topk"}
    assert press["budget"] == {
        "type": "ratio",
        "value": 0.5,
        "reference": "eligible",
    }
    assert press["scorer"] == {
        "name": "action_attention_vnorm",
        "layer": 16,
        "head_mode": "mean",
        "action_mode": "mean",
    }
    assert press["cross_layer_persistence"] == {
        "enabled": True,
        "end_layer": None,
        "mode": "hidden_sequence",
    }


def test_persistent_layer_sweep_can_reuse_an_existing_baseline():
    args = SimpleNamespace(
        persistent_layer_sweep="16",
        methods=None,
        persistent_keep_ratio=0.5,
        persistent_end_layer=16,
        persistent_mode="kv_only",
        persistent_skip_baseline=True,
        retention_policy="joint_keep_50",
        domain="last_history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_attention_vnorm_kv_persistent_layer_16"
    ]
    assert specs[0]["press"]["cross_layer_persistence"] == {
        "enabled": True,
        "end_layer": 16,
        "mode": "kv_only",
    }


def test_persistent_spec_can_express_explicit_one_shot_control():
    spec = persistent_attention_vnorm_spec(15, persistence_enabled=False)
    assert spec["name"] == "physical_attention_vnorm_kv_one_shot_layer_15"
    assert spec["press"]["cross_layer_persistence"] == {
        "enabled": False,
        "end_layer": None,
        "mode": "kv_only",
    }
    with pytest.raises(ValueError, match="does not accept end_layer"):
        persistent_attention_vnorm_spec(
            15, persistence_enabled=False, end_layer=20
        )
    with pytest.raises(ValueError, match="only supports"):
        persistent_attention_vnorm_spec(
            15, persistence_enabled=False, persistence_mode="hidden_sequence"
        )


def test_persistent_layer_sweep_oneshot_flag_builds_one_shot_arm():
    args = SimpleNamespace(
        persistent_layer_sweep="15",
        methods=None,
        persistent_keep_ratio=0.5,
        persistent_end_layer=None,
        persistent_mode="kv_only",
        persistent_oneshot=True,
        persistent_skip_baseline=True,
        retention_policy=None,
        domain="last_history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_attention_vnorm_kv_one_shot_layer_15"
    ]
    assert specs[0]["press"]["cross_layer_persistence"]["enabled"] is False


def test_learned_sweep_decouples_feature_and_compression_layers(tmp_path):
    checkpoint = tmp_path / "selector.safetensors"
    checkpoint.touch()
    args = SimpleNamespace(
        persistent_layer_sweep="20",
        methods=None,
        persistent_keep_ratio=0.5,
        persistent_end_layer=None,
        persistent_mode="kv_only",
        persistent_oneshot=False,
        persistent_skip_baseline=True,
        persistent_scorer="learned_planning_selector",
        persistent_learned_checkpoint=checkpoint,
        persistent_feature_layer=15,
        persistent_selector="history_quota",
        per_latent_keep_ratios="1.0,0.415",
        retention_policy=None,
        domain="history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_learned_planning_selector_feature_15_history_quota_kv_persistent_layer_20"
    ]
    press = specs[0]["press"]
    assert press["scorer"]["layer"] == 20
    assert press["scorer"]["feature_layer"] == 15
    assert press["selector"] == {
        "name": "history_quota",
        "ratios": [1.0, 0.415],
    }
    assert press["budget"] == {
        "type": "absolute",
        "value": 552,
        "reference": "eligible",
    }


def test_promising_full_matrix_is_fixed_and_auditable():
    args = SimpleNamespace(
        promising_full_matrix=True,
        persistent_layer_sweep=None,
        methods=None,
        retention_policy=None,
        domain="last_history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_no_press",
        "physical_attention_vnorm_hidden_fixed375_layer_16",
        "physical_attention_vnorm_hidden_fixed50_layer_16",
        "physical_attention_vnorm_hidden_adaptive_balanced_layer_16",
        "physical_attention_vnorm_hidden_adaptive_cautious_layer_16",
    ]
    presses = [spec["press"] for spec in specs[1:]]
    assert all(press["scorer"]["layer"] == 16 for press in presses)
    assert all(
        press["cross_layer_persistence"]["mode"] == "hidden_sequence"
        for press in presses
    )
    assert presses[0]["budget"]["value"] == 0.375
    assert presses[1]["budget"]["value"] == 0.5
    assert presses[2]["selector"]["gap_thresholds"] == [0.04, 0.025]
    assert presses[3]["selector"]["gap_thresholds"] == [0.06, 0.04]
    assert presses[2]["budget"]["value"] == presses[3]["budget"]["value"] == 1.0


def test_adaptive_spatial_mass_preserves_all_coarse_tiles():
    layout = build_driveva_layout(3, 15, 26, 2, 3, 1)
    domain = LastHistoryDomain().build(layout, "cpu")
    ctx = TokenContext(
        tokens=torch.zeros(1, layout.total_length, 2),
        layout=layout,
        domain=domain,
        scene_token="scene-spatial",
    )
    scores = torch.arange(domain.n_candidate, 0, -1, dtype=torch.float32).unsqueeze(0)
    selector = AdaptiveSpatialMassSelector(
        ratios=(0.375, 1.0),
        mass_thresholds=(0.0,),
        gap_thresholds=(0.0,),
        tile_h=3,
        tile_w=4,
    )
    selected = selector.select(scores, domain, domain.n_candidate, ctx)
    assert selected.K == 146
    assert selected.metadata["spatial_tile_count"] == 35
    assert selected.metadata["covered_tiles_before"] == [14]
    assert selected.metadata["covered_tiles_after"] == [35]
    assert selected.keep_global_indices.unique().numel() == selected.K
    assert selected.drop_candidate_indices.shape == (1, 390 - selected.K)


def test_signed_risk_selector_drops_only_confident_harmful_tail():
    ctx = make_context()
    scores = torch.tensor([[0.9, 0.8, 0.45, 0.35]])
    selected = SignedRiskSelector(
        threshold=0.5,
        min_keep_ratio=0.25,
        min_drop_ratio=0.25,
        abstain_margin=0.05,
    ).select(scores, ctx.domain, ctx.domain.n_candidate, ctx)
    assert selected.K == 2
    assert selected.keep_candidate_indices.tolist() == [[0, 1]]
    assert selected.metadata["abstained"] == [False]


def test_signed_risk_selector_abstains_when_harm_is_uncertain():
    ctx = make_context()
    scores = torch.tensor([[0.9, 0.8, 0.49, 0.48]])
    selected = SignedRiskSelector(
        threshold=0.5,
        min_keep_ratio=0.25,
        min_drop_ratio=0.25,
        abstain_margin=0.05,
    ).select(scores, ctx.domain, ctx.domain.n_candidate, ctx)
    assert selected.K == ctx.domain.n_candidate
    assert selected.metadata["abstained"] == [True]


def test_breakthrough_full_matrix_is_fixed_and_auditable():
    args = SimpleNamespace(
        breakthrough_full_matrix=True,
        promising_full_matrix=False,
        persistent_layer_sweep=None,
        methods=None,
        retention_policy=None,
        domain="last_history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_no_press",
        "physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
        "physical_attention_vnorm_hidden_adaptive_spatial_layer_15",
        "physical_attention_vnorm_hidden_adaptive_spatial_layer_16",
    ]
    assert [spec["press"]["scorer"]["layer"] for spec in specs[1:]] == [15, 15, 16]
    assert specs[1]["press"]["selector"]["name"] == "adaptive_mass"
    assert all(
        spec["press"]["selector"]["name"] == "adaptive_spatial_mass"
        for spec in specs[2:]
    )


def test_dynamic_teacher_matrix_includes_attention_vnorm_baseline(tmp_path):
    args = SimpleNamespace(
        dynamic_selector_checkpoint=tmp_path / "selector.safetensors",
        breakthrough_full_matrix=False,
        promising_full_matrix=False,
        temporal_motion_matrix=False,
        persistent_layer_sweep=None,
    )
    specs = _method_specs_for_run(args, round_seed=7)
    names = [spec["name"] for spec in specs]
    assert names[:2] == [
        "physical_no_press",
        "physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
    ]
    assert specs[1]["press"]["scorer"]["name"] == "action_attention_vnorm"
    assert specs[1]["press"]["selector"]["name"] == "adaptive_mass"


def test_c4_replica_triad_is_exact_and_frozen(tmp_path):
    checkpoint = tmp_path / "selector.safetensors"
    checkpoint.touch()
    args = SimpleNamespace(
        c4_replica_triad_checkpoint=checkpoint,
        dynamic_selector_checkpoint=None,
        breakthrough_full_matrix=False,
        promising_full_matrix=False,
        temporal_motion_matrix=False,
        persistent_layer_sweep=None,
        methods=None,
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_no_press",
        "physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
        "physical_learned_planning_selector_history_threshold_hidden_persistent_layer_15",
    ]
    attention = specs[1]["press"]
    assert attention["domain"] == "last_history"
    assert attention["scorer"] == {
        "name": "action_attention_vnorm",
        "layer": 15,
        "action_mode": "mean",
        "head_mode": "mean",
    }
    assert attention["selector"] == {
        "name": "adaptive_mass",
        "ratios": [0.375, 0.5, 1.0],
        "mass_thresholds": [0.60, 0.68],
        "gap_thresholds": [0.04, 0.025],
        "gap_window": 8,
    }
    dynamic = specs[2]["press"]
    assert dynamic["domain"] == "history"
    assert dynamic["selector"] == {
        "name": "history_threshold",
        "thresholds": [0.05, 0.40],
    }
    assert dynamic["scorer"] == {
        "name": "learned_planning_selector",
        "layer": 15,
        "action_mode": "mean",
        "checkpoint": str(checkpoint.resolve()),
        "feature_layer": 15,
    }
    assert all(
        spec["press"].get("cross_layer_persistence", {}).get("mode") == "hidden_sequence"
        for spec in specs[1:]
    )


def test_signed_teacher_matrix_includes_risk_gated_abstention(tmp_path):
    args = SimpleNamespace(
        dynamic_selector_checkpoint=tmp_path / "selector.safetensors",
        dynamic_selector_suite="signed",
        signed_selector_threshold=0.47,
        signed_selector_abstain_margin=0.06,
        signed_selector_min_keep_ratio=0.375,
        signed_selector_min_drop_ratio=0.1,
        breakthrough_full_matrix=False,
        promising_full_matrix=False,
        temporal_motion_matrix=False,
        persistent_layer_sweep=None,
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs][-2:] == [
        "physical_signed_teacher_hidden_threshold_layer_15",
        "physical_signed_teacher_hidden_risk_gated_layer_15",
    ]
    risk = specs[-1]["press"]["selector"]
    assert risk == {
        "name": "signed_risk",
        "threshold": 0.47,
        "abstain_margin": 0.06,
        "min_keep_ratio": 0.375,
        "min_drop_ratio": 0.1,
    }


def test_temporal_attention_scorer_uses_aligned_history_positions():
    ctx = make_context(with_qkv=True)
    scorer = ActionAttentionVNormTemporalScorer(layer=1, temporal_weight=0.25)
    scores = scorer.score(ctx)
    assert scores.shape == (1, ctx.domain.n_candidate)
    assert torch.all(scores >= 0)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(1))
    assert ctx.metadata["score_diagnostics"]["temporal_weight"] == 0.25


def test_temporal_motion_matrix_has_two_registered_blends():
    args = SimpleNamespace(
        temporal_motion_matrix=True,
        breakthrough_full_matrix=False,
        promising_full_matrix=False,
        persistent_layer_sweep=None,
        methods=None,
        retention_policy=None,
        domain="last_history",
    )
    specs = _method_specs_for_run(args, round_seed=7)
    assert [spec["name"] for spec in specs] == [
        "physical_no_press",
        "physical_attention_vnorm_hidden_adaptive_spatial_layer_15",
        "physical_attention_vnorm_temporal25_hidden_adaptive_spatial_layer_15",
        "physical_attention_vnorm_temporal50_hidden_adaptive_spatial_layer_15",
    ]
    weights = [spec["press"]["scorer"].get("temporal_weight") for spec in specs[2:]]
    assert weights == [0.25, 0.5]


def test_evaluator_generates_matched_random_baseline(tmp_path):
    ctx = make_context()
    sample = SceneSample(
        scene_token="scene-A",
        log_id="log-A",
        timestamp=0,
        tokens=ctx.tokens,
        target_trajectory=torch.zeros(1, ctx.layout.future_action.length, 2),
    )
    press = make_press(ZeroMaskOperator(), scorer=TokenNormScorer(), budget=2)

    def predict(result, _sample, context):
        return result.output[:, context.layout.future_action.start : context.layout.future_action.end, :2]

    result = Evaluator().evaluate(
        [sample],
        press,
        ctx.layout,
        output_dir=tmp_path / "matched",
        config={"evaluation": {"random_baseline": True, "random_seeds": [3, 4]}},
        predict_fn=predict,
        metric_fn=lambda prediction, _sample, _ctx, _result: {"pdm": 1.0, "valid": True},
    )
    comparison = result["summary"]["random_baseline"]
    assert comparison["seeds"] == [3, 4]
    assert comparison["paired"]["n_scenes"] == 1
    for seed in comparison["seeds"]:
        child = tmp_path / "matched" / "random_baseline" / f"seed_{seed}" / "records.jsonl"
        assert child.exists()
        assert '"K": 2' in child.read_text()


def test_official_backend_requires_explicit_metric_path_and_uses_same_evaluator(tmp_path):
    ctx = make_context()
    sample = SceneSample(scene_token="scene-A", log_id="log-A", timestamp=0, tokens=ctx.tokens)
    with pytest.raises(ValueError, match="official NAVSIM PDM"):
        DriveVANavsimBackend(
            pipe=SimpleNamespace(),
            samples=[sample],
            layout=ctx.layout,
            feature_builder=lambda value: value,
            inference_fn=lambda pipe, features, sample, runtime: features,
        )
    backend = DriveVANavsimBackend(
        pipe=SimpleNamespace(),
        samples=[sample],
        layout=ctx.layout,
        feature_builder=lambda value: value,
        inference_fn=lambda pipe, features, sample, runtime: features,
        metric_fn=lambda prediction, sample: {"pdm": 1.0, "valid": True},
    )
    result = Evaluator().evaluate_backend(
        backend,
        NoPress(),
        mode="causal",
        output_dir=tmp_path / "official-smoke",
    )
    assert result["summary"]["backend"] == "official_navsim"
    assert result["records"][0].pdm == 1.0

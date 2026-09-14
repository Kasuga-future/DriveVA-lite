from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from evaluation.artifacts import ArtifactWriter
from evaluation.evaluator import Evaluator, SceneSample, build_matched_random_press
from evaluation.navsim_evaluator import DriveVANavsimBackend
from videopress.adapters import DriveVAAdapter
from videopress.adapters.scene_boundary import window_is_single_scene
from videopress.core.budget import TokenBudget
from videopress.core.context import TokenContext
from videopress.core.domain import LastHistoryDomain
from videopress.core.layout import build_driveva_layout, decode_video_index_checked
from videopress.core.plan import validate_protocol
from videopress.core.plan import ProbeMode, build_execution_plan
from videopress.core.runtime import EvaluationMode, VideoPressRuntime
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
    ActionAttentionVNormScorer,
    ActionAttentionScorer,
    GradientNormScorer,
    PlanningGradientInputScorer,
    RandomScorer,
    TokenNormScorer,
    original_gradient_input_reduction,
    trajectory_projection_objective,
)
from videopress.selectors import TopKSelector


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
    key = ScoreKey("scene-A", 0, 15, scorer.signature())
    cache.save(key, scores, ranking=scores.argsort(dim=-1, descending=True, stable=True))
    runtime = VideoPressRuntime(press, score_cache=cache)
    runtime.begin_sample(SimpleNamespace(scene_token="scene-A", metadata={}, diffusion_rank=0), ctx.layout)
    result = runtime.execute_press(ctx)
    assert result.selection.K == 2
    assert result.metadata["score_cache"]["ranking_frozen"] is True


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

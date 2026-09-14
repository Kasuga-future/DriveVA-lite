from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from evaluation.evaluator import Evaluator, SceneSample
from videopress.core.budget import TokenBudget, budget_stats, resolve_budget
from videopress.core.context import TokenContext
from videopress.core.domain import LastHistoryDomain, build_domain
from videopress.core.layout import build_driveva_layout, decode_video_index, get_last_history_range
from videopress.core.retention import HISTORY_RETENTION_POLICIES
from videopress.core.runtime import InjectionPoint
from videopress.adapters import DriveVAAdapter
from videopress.operators import KVPruneOperator, MeanReplaceOperator, ZeroMaskOperator
from videopress.presses import NoPress, ScorerPress
from videopress.presses import SimilarityMergePress
from videopress.scorers import ActionAttentionScorer, GradientNormScorer, RandomScorer, TokenNormScorer, adapt_legacy_forward
from videopress.selectors import TopKSelector
from videopress.factory import build_press


def make_context(batch=1, hidden=8):
    layout = build_driveva_layout(f=3, h=2, w=2, num_cond_latents=2, traj_len=3, traj_prefix_len=1)
    domain = LastHistoryDomain().build(layout, "cpu")
    tokens = torch.arange(batch * layout.total_length * hidden, dtype=torch.float32).reshape(batch, layout.total_length, hidden)
    return TokenContext(tokens=tokens, layout=layout, domain=domain, scene_token="scene-A", log_id="log-A", diffusion_rank=0)


def test_layout_and_index_mapping():
    ctx = make_context()
    layout = ctx.layout
    assert layout.video.length == 12
    assert layout.history_video.length == 8
    assert get_last_history_range(layout) == layout.frame_range(1)
    assert decode_video_index(7, 2, 2) == (1, 1, 1)


def test_budget_and_dual_ratios():
    ctx = make_context()
    budget = TokenBudget("ratio", 0.5, "eligible")
    assert resolve_budget(budget, ctx.layout, ctx.domain) == 2
    stats = budget_stats(ctx.layout, ctx.domain, 2)
    assert stats["n_history"] == 8
    assert stats["n_eligible"] == 4
    assert stats["history_keep_ratio"] == 0.25
    assert stats["eligible_keep_ratio"] == 0.5


@pytest.mark.parametrize(
    ("policy", "expected_k", "expected_per_latent"),
    [
        ("drop_previous_keep_last_100", 390, [0, 390]),
        ("drop_previous_keep_last_50", 195, [0, 195]),
        ("joint_keep_50", 390, None),
        ("joint_keep_25", 195, None),
        ("per_latent_keep_50", 390, [195, 195]),
        # 390 * 0.25 is 97.5, so each independent quota rounds to 98.
        ("per_latent_keep_25", 196, [98, 98]),
    ],
)
def test_six_history_retention_policies(policy, expected_k, expected_per_latent):
    layout = build_driveva_layout(
        f=3, h=15, w=26, num_cond_latents=2, traj_len=3, traj_prefix_len=1
    )
    press = build_press(
        {
            "name": "scorer_press",
            "retention_policy": policy,
            "scorer": {"name": "token_norm"},
            "operator": {"name": "zero"},
            "injection_point": "video_input",
        }
    )
    domain = build_domain(press.domain, layout, "cpu")
    tokens = torch.arange(1, layout.total_length + 1, dtype=torch.float32).view(1, -1, 1)
    ctx = TokenContext(tokens=tokens, layout=layout, domain=domain)
    result = press.apply(ctx)
    assert set(HISTORY_RETENTION_POLICIES) == {
        "drop_previous_keep_last_100",
        "drop_previous_keep_last_50",
        "joint_keep_50",
        "joint_keep_25",
        "per_latent_keep_50",
        "per_latent_keep_25",
    }
    assert domain.name == "history"
    assert domain.n_candidate == 780
    assert result.selection.K == expected_k
    assert result.metadata["effective_history_kept"] == [expected_k]
    assert torch.count_nonzero(result.output[:, layout.history_video.as_slice()]) == expected_k
    selected_per_latent = result.metadata["selected_history_latent_counts"][0]
    if expected_per_latent is not None:
        assert selected_per_latent == expected_per_latent
    else:
        assert sum(selected_per_latent) == expected_k


def test_retention_policy_rejects_wrong_history_count():
    press = build_press(
        {
            "retention_policy": "per_latent_keep_50",
            "scorer": "token_norm",
            "operator": "zero",
        }
    )
    layout = build_driveva_layout(3, 2, 2, 1, 0, 0)
    ctx = TokenContext(
        tokens=torch.ones(1, layout.total_length, 2),
        layout=layout,
        domain=build_domain("history", layout, "cpu"),
    )
    with pytest.raises(ValueError, match="exactly 2 history latents"):
        press.apply(ctx)


def test_effective_retention_counts_protected_previous_history():
    ctx = make_context()
    result = build_press(
        {
            "domain": "last_history",
            "scorer": "token_norm",
            "selector": "topk",
            "operator": "zero",
            "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
        }
    ).apply(ctx)
    assert result.metadata["selected_history_latent_counts"] == [[0, 2]]
    assert result.metadata["effective_history_latent_kept_counts"] == [[4, 2]]
    assert result.metadata["effective_history_keep_ratio"] == [0.75]


def test_random_is_deterministic_and_seeded():
    ctx = make_context()
    first = RandomScorer(seed=0).score(ctx)
    second = RandomScorer(seed=0).score(ctx)
    other = RandomScorer(seed=1).score(ctx)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_topk_is_stable_on_ties_and_exact_k():
    ctx = make_context()
    scores = torch.ones(1, ctx.domain.n_candidate)
    selection = TopKSelector().select(scores, ctx.domain, 2, ctx)
    assert selection.keep_global_indices.tolist() == [[4, 5]]
    assert selection.keep_global_indices.shape == (1, 2)


def test_zero_clears_unselected_domain_and_preserves_protected_tokens():
    ctx = make_context()
    scores = TokenNormScorer().score(ctx)
    selection = TopKSelector().select(scores, ctx.domain, 2, ctx)
    zero = ZeroMaskOperator().apply(ctx, selection).output
    mean = MeanReplaceOperator().apply(ctx, selection).output
    protected = torch.where(ctx.domain.protected_mask)[0]
    assert torch.equal(zero[:, protected], ctx.tokens[:, protected])
    assert torch.equal(ctx.tokens[:, protected], mean[:, protected])
    assert torch.equal(zero[:, selection.keep_global_indices[0]], ctx.tokens[:, selection.keep_global_indices[0]])
    dropped = ctx.domain.candidate_mask.clone()
    dropped[selection.keep_global_indices[0]] = False
    assert torch.count_nonzero(zero[:, dropped]) == 0


def test_action_attention_shape_and_finite():
    ctx = make_context()
    heads, dim = 2, 4
    generator = torch.Generator().manual_seed(3)
    ctx.q = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.k = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.v = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    scores = ActionAttentionScorer(head_mode="mean", action_mode="mean").score(ctx)
    assert scores.shape == (1, ctx.domain.n_candidate)
    assert torch.isfinite(scores).all()


def test_gradient_probe_returns_attribution():
    ctx = make_context()

    def forward(tokens):
        return tokens[:, ctx.domain.candidate_indices].pow(2).sum()

    scorer = GradientNormScorer(
        forward_fn=adapt_legacy_forward(forward),
        objective=lambda output, _ctx: output,
    )
    scores = scorer.score(ctx)
    expected = 2 * ctx.candidate_tokens().abs().norm(dim=-1)
    assert torch.allclose(scores, expected)
    assert scorer.requires_probe is True


def test_kv_prune_keeps_q_and_records_mapping():
    ctx = make_context()
    heads, dim = 2, 4
    generator = torch.Generator().manual_seed(4)
    ctx.q = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.k = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.v = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    selection = TopKSelector().select(RandomScorer(0).score(ctx), ctx.domain, 2, ctx)
    result = KVPruneOperator().apply(ctx, selection)
    assert result.aux["q"].shape[2] == ctx.layout.total_length
    assert result.aux["k"].shape[2] == ctx.domain.n_protected + 2
    assert result.aux["v"].shape == result.aux["k"].shape
    assert result.mapping.original_length == ctx.layout.total_length
    assert result.mapping.compressed_length == ctx.domain.n_protected + 2
    assert result.metadata["theoretical_attn_ratio"] < 1.0


def test_driveva_adapter_translates_flattened_wan_hook():
    from diffsynth.models.wan_video_dit import SelfAttention

    ctx = make_context()
    attention = SelfAttention(dim=8, num_heads=2)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)]), dit2=None)
    press = ScorerPress(
        scorer=RandomScorer(0),
        selector=TopKSelector(),
        operator=KVPruneOperator(),
        budget=TokenBudget("absolute", 2),
        domain="last_history",
        injection_point=InjectionPoint.SELF_ATTN_KV,
    )
    from videopress.core.runtime import VideoPressRuntime

    runtime = VideoPressRuntime(press=press, mode="physical", adapter=DriveVAAdapter())
    runtime.layout = ctx.layout
    runtime.current_scene = "scene-A"
    runtime.current_sample = SimpleNamespace(metadata={"domain": "last_history"}, diffusion_rank=0)
    runtime.install(pipe)
    length, dim = ctx.layout.total_length, 8
    generator = torch.Generator().manual_seed(9)
    q = torch.randn(1, length, dim, generator=generator)
    k = torch.randn(1, length, dim, generator=generator)
    v = torch.randn(1, length, dim, generator=generator)
    q2, k2, v2 = attention.tokenpress_hook(q, k, v, layer_idx=0)
    runtime.remove(pipe)
    assert q2.shape == q.shape
    assert k2.shape[1] == ctx.domain.n_protected + 2
    assert v2.shape == k2.shape
    assert not hasattr(attention, "tokenpress_hook")


def test_no_press_is_value_equivalent():
    ctx = make_context()
    result = NoPress().apply(ctx)
    assert torch.equal(result.output, ctx.tokens)
    assert result.selection.K == ctx.domain.n_candidate
    assert result.metadata["no_press"] is True


def test_scorer_press_and_evaluator_write_artifacts(tmp_path):
    ctx = make_context()
    sample = SceneSample(
        scene_token=ctx.scene_token,
        log_id=ctx.log_id,
        timestamp=0,
        tokens=ctx.tokens,
        target_trajectory=torch.zeros(1, ctx.layout.future_action.length, 2),
    )
    press = ScorerPress(
        scorer=RandomScorer(0),
        selector=TopKSelector(),
        operator=ZeroMaskOperator(),
        budget=TokenBudget("absolute", 2),
        domain="last_history",
        injection_point=InjectionPoint.VIDEO_INPUT,
    )

    def predict(result, sample, context):
        return result.output[:, context.layout.future_action.start : context.layout.future_action.end, :2]

    def metrics(prediction, sample, context, result):
        error = prediction - sample.target_trajectory
        value = float(error.abs().mean())
        return {"pdm": 1.0 / (1.0 + value), "trajectory_l2": value, "endpoint_l2": value, "valid": True}

    output = Evaluator().evaluate(
        [sample],
        press,
        ctx.layout,
        output_dir=tmp_path / "run",
        predict_fn=predict,
        metric_fn=metrics,
        config={"test": True},
    )
    assert output["summary"]["n_scenes"] == 1
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "records.jsonl").exists()
    assert (tmp_path / "run" / "tokens.parquet").exists() or (tmp_path / "run" / "tokens.jsonl").exists()
    json.loads((tmp_path / "run" / "summary.json").read_text())


def test_factory_accepts_press_md_attention_schema():
    press = build_press(
        {
            "name": "scorer_press",
            "domain": {"name": "last_history"},
            "scorer": {
                "name": "action_attention",
                "layer": 15,
                "heads": {"mode": "single", "index": 1},
                "action": {"mode": "first"},
                "softmax_domain": "all_sequence",
                "value_norm": {"enabled": True},
            },
            "selector": "topk",
            "operator": "zero",
            "budget": {"type": "absolute", "value": 2},
            "injection_point": "video_input",
        }
    )
    assert press.scorer.layer == 15
    assert press.scorer.head_mode == "single"
    assert press.scorer.head_index == 1
    assert press.scorer.action_mode == "first"
    assert press.scorer.value_norm is True


def test_similarity_merge_preserves_reversible_groups():
    ctx = make_context()
    heads, dim = 2, 4
    generator = torch.Generator().manual_seed(11)
    ctx.q = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.k = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    ctx.v = torch.randn(1, heads, ctx.layout.total_length, dim, generator=generator)
    result = SimilarityMergePress(TokenBudget("absolute", 2), domain="last_history").apply(ctx)
    assert result.output.shape[1] == ctx.layout.total_length
    assert result.aux["q"].shape[2] == ctx.layout.total_length
    assert result.aux["k"].shape[2] == ctx.domain.n_protected + 2
    assert result.aux["v"].shape[2] == ctx.domain.n_protected + 2
    assert result.mapping.source_groups is not None
    assert sum(len(group) for group in result.mapping.source_groups) == ctx.layout.total_length

"""CPU tests for the register/query latent bottleneck route (2026-09-11).

The route replaces learned *token selection* (which three rounds of
``reports/dynamic_token_learnability_verdict_20260911.md`` closed) with K
learnable key/register tokens that cross-attend over the 390 candidate history
tokens and are trained to predict the next latent's content.  These tests pin
the properties the design depends on:

* the module emits exactly K tokens in the backbone hidden width, for the real
  1,569-token DriveVA layout as well as for a bare candidate block;
* a bottleneck must not depend on the order in which the candidate tokens
  arrive (attention is a set operation) and must still be sensitive to content;
* gradients reach both the learnable queries and the source tokens (the whole
  point of dropping the frozen-backbone/selector setting);
* the splice/restore contract reproduces the pipeline's gather/scatter
  behaviour and leaves every protected token bit-identical;
* the predictive objective is *learnable*: on a synthetic sequence whose next
  step is a deterministic function of the current content, a few hundred CPU
  steps drive the predictive loss far below the trivial constant predictor, the
  frozen-bottleneck control cannot do it, and the learned key tokens retain
  enough information to reconstruct the target through a linear read-out.

All tests are CPU-only by construction (no CUDA calls anywhere).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from videopress.core.context import TokenContext  # noqa: E402
from videopress.core.domain import build_domain  # noqa: E402
from videopress.core.layout import build_driveva_layout  # noqa: E402
from videopress.press.register_bottleneck import (  # noqa: E402
    DEFAULT_COMPRESSED_LAYERS,
    DEFAULT_FFN_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_PROTECTED_TOKENS,
    DEFAULT_TOTAL_SEQUENCE,
    NextLatentPredictor,
    RegisterBottleneck,
    bottleneck_overhead_flops,
    key_token_diversity_penalty,
    key_token_keep_indices,
    key_token_positions,
    key_token_scale_penalty,
    key_token_std,
    key_tokens_for_context,
    last_history_positions,
    layer_sequence_cost,
    next_latent_prediction_loss,
    position_features,
    press_cost_report,
    restore_key_token_sequence,
    shuffle_key_tokens,
    splice_key_tokens,
    trivial_prediction_loss,
    variance_covariance_penalty,
)

# Real DriveVA/Wan2.2-TI2V-5B deployment shape: 15 x 26 patch grid = 390
# candidate tokens, 4 latent frames + 9 trajectory tokens = 1,569.
VIDEO_H, VIDEO_W = 15, 26
HISTORY_TOKENS = VIDEO_H * VIDEO_W
TOTAL_TOKENS = DEFAULT_TOTAL_SEQUENCE
CANDIDATE_START = 1170
CANDIDATE_END = 1560


def _small_bottleneck(**kwargs) -> RegisterBottleneck:
    config = dict(num_key_tokens=16, hidden_dim=32, attn_dim=32, num_heads=4, condition_dim=16)
    config.update(kwargs)
    return RegisterBottleneck(**config)


# ---------------------------------------------------------------------------
# 1. shape contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_key_tokens", [32, 64, 128])
def test_bottleneck_emits_k_key_tokens_in_backbone_width(num_key_tokens: int) -> None:
    """K queries -> [B, K, D] at the real hidden width, from the real 390 tokens."""
    torch.manual_seed(0)
    module = RegisterBottleneck(
        num_key_tokens=num_key_tokens,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        attn_dim=1024,
        num_heads=8,
    )
    tokens = torch.randn(1, HISTORY_TOKENS, DEFAULT_HIDDEN_DIM)
    positions = last_history_positions(VIDEO_H, VIDEO_W).unsqueeze(0)
    out = module(tokens, positions)
    assert out.key_tokens.shape == (1, num_key_tokens, DEFAULT_HIDDEN_DIM)
    assert out.attention.shape == (1, 8, num_key_tokens, HISTORY_TOKENS)
    assert torch.isfinite(out.key_tokens).all()
    # attention is a probability distribution over the candidate tokens
    torch.testing.assert_close(out.attention.sum(dim=-1), torch.ones(1, 8, num_key_tokens))
    metadata = out.metadata()
    assert metadata["register_bottleneck_key_tokens"] == num_key_tokens
    assert metadata["register_bottleneck_attention_entropy"] > 0.0


def test_bottleneck_accepts_bf16_backbone_tokens() -> None:
    """The backbone runs bf16; the bottleneck must not silently corrupt it."""
    torch.manual_seed(1)
    module = _small_bottleneck()
    tokens = torch.randn(2, 40, 32, dtype=torch.bfloat16)
    out = module(tokens, None if False else last_history_positions(5, 8).unsqueeze(0).expand(2, -1, -1))
    assert out.key_tokens.dtype == torch.float32
    assert torch.isfinite(out.key_tokens).all()


# ---------------------------------------------------------------------------
# 2. permutation / order invariance
# ---------------------------------------------------------------------------
def test_bottleneck_is_invariant_to_candidate_token_order() -> None:
    """A read-out bottleneck must be a function of the token *set*, not its order."""
    torch.manual_seed(2)
    module = _small_bottleneck()
    tokens = torch.randn(2, HISTORY_TOKENS, 32)
    positions = last_history_positions(VIDEO_H, VIDEO_W).unsqueeze(0).expand(2, -1, -1)
    order = torch.randperm(HISTORY_TOKENS)
    reference = module(tokens, positions)
    permuted = module(tokens[:, order], positions[:, order])
    torch.testing.assert_close(reference.key_tokens, permuted.key_tokens, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference.key_positions, permuted.key_positions, atol=1e-6, rtol=1e-5)
    # the attention matrix must follow the permutation, i.e. the module really
    # is reading the (permuted) inputs rather than ignoring them
    inverse = torch.argsort(order)
    torch.testing.assert_close(
        reference.attention, permuted.attention[..., inverse], atol=1e-6, rtol=1e-5
    )


def test_bottleneck_is_strictly_order_invariant_without_positions() -> None:
    torch.manual_seed(3)
    module = _small_bottleneck(use_position_bias=False)
    tokens = torch.randn(1, HISTORY_TOKENS, 32)
    order = torch.randperm(HISTORY_TOKENS)
    torch.testing.assert_close(
        module(tokens).key_tokens, module(tokens[:, order]).key_tokens, atol=1e-6, rtol=1e-5
    )


def test_permuting_content_changes_the_key_tokens() -> None:
    """Order invariance must not degenerate into content blindness."""
    torch.manual_seed(4)
    module = _small_bottleneck(use_position_bias=False)
    tokens = torch.randn(1, 64, 32)
    first = module(tokens).key_tokens
    second = module(tokens + 0.5).key_tokens
    assert not torch.allclose(first, second, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. gradient routing
# ---------------------------------------------------------------------------
def test_gradients_flow_to_queries_and_source_tokens() -> None:
    torch.manual_seed(5)
    module = _small_bottleneck()
    tokens = torch.randn(2, HISTORY_TOKENS, 32, requires_grad=True)
    positions = last_history_positions(VIDEO_H, VIDEO_W).unsqueeze(0).expand(2, -1, -1)
    out = module(
        tokens,
        positions,
        ego_state=torch.tensor([[3.0, -0.2], [1.0, 0.1]]),
        command=torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        timestep=torch.tensor([1000.0, 500.0]),
    )
    out.key_tokens.pow(2).mean().backward()

    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0.0
    assert module.queries.grad is not None and float(module.queries.grad.abs().sum()) > 0.0
    for name in ("k_proj", "v_proj", "out_proj", "kv_norm", "q_norm"):
        param = dict(module.named_parameters())[f"{name}.weight"]
        assert param.grad is not None and float(param.grad.abs().sum()) > 0.0, name
    # the conditioning MLPs must be on the gradient path even though their last
    # layer is zero-initialised (that is what keeps old checkpoints loadable)
    assert float(module.condition_mlp[-1].weight.grad.abs().sum()) > 0.0
    assert float(module.timestep_mlp[-1].weight.grad.abs().sum()) > 0.0


def test_conditioning_changes_the_key_tokens_once_trained() -> None:
    torch.manual_seed(6)
    module = _small_bottleneck()
    tokens = torch.randn(1, 64, 32)
    positions = last_history_positions(4, 16).unsqueeze(0)
    base = module(tokens, positions).key_tokens
    # a fresh module is deliberately condition agnostic (zero-init modulation)
    same = module(tokens, positions, ego_state=torch.tensor([[9.0, 1.0]]),
                  command=torch.tensor([[0.0, 0.0, 1.0]]), timestep=torch.tensor([100.0])).key_tokens
    torch.testing.assert_close(base, same, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        module.condition_mlp[-1].weight.normal_(std=0.05)
        module.timestep_mlp[-1].weight.normal_(std=0.05)
    shifted = module(tokens, positions, ego_state=torch.tensor([[9.0, 1.0]]),
                     command=torch.tensor([[0.0, 0.0, 1.0]]), timestep=torch.tensor([100.0])).key_tokens
    assert not torch.allclose(base, shifted, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. real layout + splice/restore contract
# ---------------------------------------------------------------------------
def test_bottleneck_handles_the_real_driveva_layout_and_splice_contract() -> None:
    torch.manual_seed(7)
    layout = build_driveva_layout(f=4, h=VIDEO_H, w=VIDEO_W, num_cond_latents=4,
                                 traj_len=9, traj_prefix_len=0)
    assert layout.total_length == TOTAL_TOKENS
    domain = build_domain("last_history", layout, torch.device("cpu"))
    assert domain.n_candidate == HISTORY_TOKENS
    assert domain.n_protected == DEFAULT_PROTECTED_TOKENS

    dim = 32
    tokens = torch.randn(2, layout.total_length, dim)
    ctx = TokenContext(
        tokens=tokens,
        layout=layout,
        domain=domain,
        metadata={"selector_ego_state": [4.5, -0.3], "selector_command": [1.0, 0.0, 0.0]},
    )
    module = _small_bottleneck()
    encoding, prediction = key_tokens_for_context(
        module, ctx, predictor=NextLatentPredictor(key_dim=dim, target_dim=8, num_slots=4,
                                                   hidden_dim=32, num_heads=4)
    )
    assert (encoding.candidate_start, encoding.candidate_end) == (CANDIDATE_START, CANDIDATE_END)
    assert encoding.key_tokens.shape == (2, 16, dim)
    assert prediction.shape == (2, 4, 8)
    metadata = encoding.metadata()
    assert metadata["register_bottleneck_candidate_start"] == CANDIDATE_START
    assert metadata["register_bottleneck_compressed_gain"] == HISTORY_TOKENS - 16

    short, keep = splice_key_tokens(
        tokens, encoding.key_tokens, CANDIDATE_START, CANDIDATE_END
    )
    assert short.shape == (2, DEFAULT_PROTECTED_TOKENS + 16, dim)
    assert keep.tolist()[:3] == [0, 1, 2]
    assert keep.tolist()[-3:] == [TOTAL_TOKENS - 3, TOTAL_TOKENS - 2, TOTAL_TOKENS - 1]
    assert keep.numel() == short.shape[1]
    # the K key-token slots sit where the candidate block started
    assert keep[CANDIDATE_START:CANDIDATE_START + 16].tolist() == list(
        range(CANDIDATE_START, CANDIDATE_START + 16)
    )
    torch.testing.assert_close(short[:, :CANDIDATE_START], tokens[:, :CANDIDATE_START])
    torch.testing.assert_close(short[:, -9:], tokens[:, -9:])

    restored = restore_key_token_sequence(
        short, original_length=TOTAL_TOKENS, candidate_start=CANDIDATE_START,
        candidate_end=CANDIDATE_END,
    )
    assert restored.shape == tokens.shape
    # protected tokens survive bit-exactly; dropped candidate slots are zeros
    torch.testing.assert_close(restored[:, :CANDIDATE_START], tokens[:, :CANDIDATE_START])
    torch.testing.assert_close(restored[:, -9:], tokens[:, -9:])
    assert float(restored[:, CANDIDATE_START + 16:CANDIDATE_END].abs().sum()) == 0.0

    indices = key_token_keep_indices(TOTAL_TOKENS, CANDIDATE_START, CANDIDATE_END, 64)
    assert indices.numel() == DEFAULT_PROTECTED_TOKENS + 64
    with pytest.raises(ValueError):
        splice_key_tokens(tokens, encoding.key_tokens, CANDIDATE_END, CANDIDATE_START)


# ---------------------------------------------------------------------------
# 5. positional read-out and centroid
# ---------------------------------------------------------------------------
def test_position_bias_can_localise_a_read_out() -> None:
    """The gate depends on queries being able to pick an image region."""
    torch.manual_seed(8)
    module = _small_bottleneck()
    positions = last_history_positions(VIDEO_H, VIDEO_W).unsqueeze(0)
    tokens = torch.randn(1, HISTORY_TOKENS, 32)
    features = position_features(positions)[0]
    # feature order is [t, r, c, t^2, r^2, c^2, tr, tc, rc]
    with torch.no_grad():
        module.position_bias.zero_()
        module.position_bias[:, 0, 1] = -30.0  # key token 0 prefers small r
        module.position_bias[:, 1, 1] = +30.0  # key token 1 prefers large r
    centroid = module(tokens, positions).key_positions[0, :, 1]
    assert float(centroid[0]) < 0.15
    assert float(centroid[1]) > 0.85
    assert abs(float(centroid.mean()) - 0.5) < 0.3


def test_key_token_positions_are_attention_weighted_centroids() -> None:
    torch.manual_seed(9)
    positions = last_history_positions(4, 8).unsqueeze(0)
    attention = torch.rand(1, 2, 3, 32)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    centroid = key_token_positions(attention, positions)
    manual = torch.einsum("bkn,bnp->bkp", attention.mean(dim=1), positions)
    torch.testing.assert_close(centroid, manual, atol=1e-6, rtol=1e-5)
    zeros = key_token_positions(attention, None)
    assert float(zeros.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        key_token_positions(attention, positions[:, :10])


# ---------------------------------------------------------------------------
# 6. objectives and regularisers
# ---------------------------------------------------------------------------
def test_prediction_loss_matches_the_trivial_baseline_semantics() -> None:
    torch.manual_seed(10)
    target = torch.randn(8, 4, 6)
    perfect = next_latent_prediction_loss(target.clone(), target)
    assert float(perfect) < 1e-9
    zero = torch.zeros_like(target)
    assert float(next_latent_prediction_loss(zero, target)) == pytest.approx(1.0, abs=1e-5)
    assert float(trivial_prediction_loss(target)) <= float(
        next_latent_prediction_loss(zero, target)
    )
    # the target is detached: no gradient may flow into it
    trained_target = target.clone().requires_grad_(True)
    prediction = torch.zeros_like(target, requires_grad=True)
    next_latent_prediction_loss(prediction, trained_target).backward()
    assert prediction.grad is not None and float(prediction.grad.abs().sum()) > 0.0
    assert trained_target.grad is None or float(trained_target.grad.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        next_latent_prediction_loss(target, target[:, :2])
    with pytest.raises(ValueError):
        next_latent_prediction_loss(target, target, mode="nope")


def test_regularisers_detect_collapsed_registers() -> None:
    torch.manual_seed(11)
    identical = torch.randn(4, 8, 16).mean(dim=1, keepdim=True).expand(-1, 8, -1).contiguous()
    assert float(key_token_diversity_penalty(identical)) == pytest.approx(1.0, abs=1e-4)
    orthogonal = torch.zeros(2, 8, 8)
    orthogonal[0] = torch.eye(8)
    orthogonal[1] = torch.eye(8)
    assert float(key_token_diversity_penalty(orthogonal)) == pytest.approx(0.0, abs=1e-5)
    collapsed_batch = torch.randn(1, 8, 16).expand(4, -1, -1).contiguous()
    assert float(key_token_std(collapsed_batch)) < 1e-3  # floor is the eps clamp
    assert float(key_token_std(identical)) > 0.0
    assert float(key_token_std(torch.randn(8, 8, 16))) > 0.5

    reference = torch.randn(2, 8, 16)
    shrunk = reference * 0.1
    assert float(key_token_scale_penalty(shrunk, reference)) > 0.8
    assert float(key_token_scale_penalty(reference, reference)) < 1e-5

    keys = torch.randn(4, 8, 16)
    default = variance_covariance_penalty(keys)
    assert float(default) >= 0.0
    with pytest.raises(ValueError):
        variance_covariance_penalty(keys[:1], variance_weight=1.0)  # needs a real batch
    with pytest.raises(ValueError):
        variance_covariance_penalty(keys, scale_weight=1.0)
    # the batch-statistic term (opt-in: the DiT trains with batch 1 per rank)
    # fires when every scene produces the same summary and is quiet when the
    # batch carries scene-to-scene variation
    batch_penalty = variance_covariance_penalty(
        collapsed_batch, variance_weight=1.0, covariance_weight=0.0, diversity_weight=0.0
    )
    spread = torch.randn(8, 1, 16).expand(-1, 8, -1).contiguous()
    spread_penalty = variance_covariance_penalty(
        spread, variance_weight=1.0, covariance_weight=0.0, diversity_weight=0.0
    )
    assert float(batch_penalty) > float(spread_penalty) + 0.5
    assert float(key_token_diversity_penalty(spread)) == pytest.approx(1.0, abs=1e-4)


def test_attention_mask_and_shuffle_controls() -> None:
    torch.manual_seed(12)
    module = _small_bottleneck()
    tokens = torch.randn(2, 32, 32)
    positions = last_history_positions(4, 8).unsqueeze(0).expand(2, -1, -1)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[:, 16:] = False
    masked = module(tokens, positions, attention_mask=mask)
    assert float(masked.attention[..., 16:].abs().sum()) == 0.0
    torch.testing.assert_close(masked.attention.sum(dim=-1), torch.ones(2, 4, 16))
    with pytest.raises(ValueError):
        module(tokens, positions, attention_mask=torch.ones(2, 8, dtype=torch.bool))

    keys = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    shuffled = shuffle_key_tokens(keys, generator=torch.Generator().manual_seed(0))
    assert shuffled.shape == keys.shape
    assert sorted(shuffled[0].flatten().tolist()) == sorted(keys[0].flatten().tolist())
    single = shuffle_key_tokens(keys[:1])
    torch.testing.assert_close(single, keys[:1])


# ---------------------------------------------------------------------------
# 7. cost model
# ---------------------------------------------------------------------------
def test_cost_report_reproduces_shape_arithmetic() -> None:
    report = press_cost_report()
    baseline = report["baseline"]
    assert baseline["sequence_length"] == TOTAL_TOKENS
    assert baseline["candidate_tokens"] == HISTORY_TOKENS
    expected = layer_sequence_cost(
        TOTAL_TOKENS, layers=DEFAULT_COMPRESSED_LAYERS, hidden_dim=DEFAULT_HIDDEN_DIM,
        ffn_dim=DEFAULT_FFN_DIM,
    )
    assert baseline["total_flops"] == pytest.approx(expected.total_flops)
    # 8 L D^2 + 4 L^2 D + 4 L D F + (4 L D^2 + 4 L S D), 14 layers, MAC = 2 FLOPs
    length, dim, ffn, ctx_len, layers = TOTAL_TOKENS, DEFAULT_HIDDEN_DIM, DEFAULT_FFN_DIM, 512, 14
    manual = layers * (
        8 * length * dim ** 2
        + 4 * length * length * dim
        + 4 * length * dim * ffn
        + 4 * length * dim ** 2
        + 4 * length * ctx_len * dim
    )
    assert baseline["total_flops"] == pytest.approx(manual)

    arms = {arm["key_tokens"]: arm for arm in report["arms"]}
    assert set(arms) == {32, 64, 128}
    for k, arm in arms.items():
        assert arm["sequence_length"] == DEFAULT_PROTECTED_TOKENS + k
        assert arm["flops_saved_fraction"] > 0.0
        assert arm["kv_bytes_saved"] == baseline["kv_bytes"] - arm["kv_bytes"]
    # monotone: fewer key tokens -> more saving
    assert arms[32]["flops_saved_fraction"] > arms[64]["flops_saved_fraction"] > arms[128]["flops_saved_fraction"]
    assert arms[32]["kv_bytes"] < arms[64]["kv_bytes"] < arms[128]["kv_bytes"]
    assert arms[64]["sequence_ratio"] == pytest.approx((DEFAULT_PROTECTED_TOKENS + 64) / TOTAL_TOKENS)
    # the read-out is cheap relative to what it saves; the quadratic attention
    # term is a minority of the saving
    assert arms[64]["bottleneck_overhead_share_of_saving"] < 0.02
    assert arm_share_of_saving(report, 64, "attention") < 0.2
    # and the bottleneck's own one-shot cost is far below a single compressed
    # layer of the baseline
    single_layer = layer_sequence_cost(TOTAL_TOKENS, layers=1)
    assert bottleneck_overhead_flops(HISTORY_TOKENS, 64) < 0.05 * single_layer.total_flops
    assert any("excludes layers 0..15" in note for note in report["notes"])


def arm_share_of_saving(report: dict, key_tokens: int, term: str) -> float:
    """Share of the gross saving contributed by one cost term (test helper)."""
    baseline = report["baseline"]["sequence_length"]
    protected = report["baseline"]["protected_tokens"]
    saved_total = 0.0
    saved_term = 0.0
    for term_name in ("attention_flops", "qkv_output_flops", "ffn_flops", "text_cross_flops"):
        old = layer_sequence_cost(baseline, layers=DEFAULT_COMPRESSED_LAYERS)
        new = layer_sequence_cost(protected + key_tokens, layers=DEFAULT_COMPRESSED_LAYERS)
        delta = getattr(old, term_name) - getattr(new, term_name)
        saved_total += delta
        if term_name.startswith(term):
            saved_term += delta
    return saved_term / saved_total


# ---------------------------------------------------------------------------
# 8. synthetic predictive-learnability gate
# ---------------------------------------------------------------------------
GATE_DIM = 64
GATE_K = 16
GATE_SLOTS = 4
GATE_SLOT_DIM = 8
GATE_WORLD_SEED = 4242
GATE_STEPS = 900
GATE_BATCH = 32


def _gate_world(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The *fixed* next-latent map: target = 1.5 * P @ (Gaussian pool of content) + b.

    It is a property of the synthetic world, not of any sample: every scene is
    pushed through the same ``P`` / ``b``, so the predictive objective is a
    genuine function-approximation problem rather than a per-batch lookup.
    """
    generator = torch.Generator().manual_seed(GATE_WORLD_SEED)
    slot = (torch.arange(GATE_SLOTS).float() + 0.5) / GATE_SLOTS
    centers = torch.stack([slot.repeat_interleave(GATE_SLOTS), slot.repeat(GATE_SLOTS)], dim=-1)
    distance = ((positions[:, 1].reshape(1, -1) - centers[:, 0:1]) ** 2
                + (positions[:, 2].reshape(1, -1) - centers[:, 1:2]) ** 2)
    weights = torch.softmax(-distance / (2 * 0.15 ** 2), dim=-1)
    projection = torch.randn(GATE_SLOTS * GATE_SLOTS, GATE_SLOT_DIM, GATE_DIM, generator=generator)
    projection = projection / math.sqrt(GATE_DIM)
    bias = torch.randn(GATE_SLOTS * GATE_SLOTS, GATE_SLOT_DIM, generator=generator) * 0.1
    return weights, projection, bias


def _gate_scene(batch: int, seed: int, positions: torch.Tensor, world) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-scene current content: smooth random fields with per-scene amplitude."""
    weights, projection, bias = world
    generator = torch.Generator().manual_seed(seed)
    rows, cols = positions[:, 1].reshape(VIDEO_H, VIDEO_W), positions[:, 2].reshape(VIDEO_H, VIDEO_W)
    modes = 3
    amplitude = torch.randn(batch, GATE_DIM, modes, generator=generator)
    freq_r = torch.rand(batch, GATE_DIM, modes, generator=generator) * 4.0
    freq_c = torch.rand(batch, GATE_DIM, modes, generator=generator) * 5.0
    phase = torch.rand(batch, GATE_DIM, modes, generator=generator) * (2 * math.pi)
    argument = (
        freq_r.reshape(batch, GATE_DIM, modes, 1, 1) * rows.reshape(1, 1, 1, VIDEO_H, VIDEO_W)
        + freq_c.reshape(batch, GATE_DIM, modes, 1, 1) * cols.reshape(1, 1, 1, VIDEO_H, VIDEO_W)
        + phase.reshape(batch, GATE_DIM, modes, 1, 1)
    )
    field = (amplitude.reshape(batch, GATE_DIM, modes, 1, 1) * torch.sin(argument)).sum(dim=2)
    tokens = field.reshape(batch, HISTORY_TOKENS, GATE_DIM) / math.sqrt(modes)
    pooled = torch.einsum("sn,bnd->bsd", weights, tokens)
    target = torch.einsum("std,bsd->bst", projection, pooled) * 1.5 + bias
    return tokens, target


def _gate_encode(module: RegisterBottleneck, tokens: torch.Tensor, batch: int) -> torch.Tensor:
    positions = last_history_positions(VIDEO_H, VIDEO_W).unsqueeze(0).expand(batch, -1, -1)
    keys = []
    with torch.no_grad():
        for start in range(0, batch, 64):
            keys.append(module(tokens[start:start + 64], positions[start:start + 64]).key_tokens)
    return torch.cat(keys, dim=0)


def _ridge_readout_r2(module: RegisterBottleneck, train_seed: int = 111_111,
                      test_seed: int = 222_222) -> tuple[float, float]:
    """Linear read-out from frozen key tokens to the target, + shuffled control."""
    world = _gate_world(last_history_positions(VIDEO_H, VIDEO_W))
    positions = last_history_positions(VIDEO_H, VIDEO_W)
    train_tokens, train_target = _gate_scene(256, train_seed, positions, world)
    test_tokens, test_target = _gate_scene(256, test_seed, positions, world)
    train_keys = _gate_encode(module, train_tokens, 256).reshape(256, -1)
    test_keys = _gate_encode(module, test_tokens, 256).reshape(256, -1)
    targets = train_target.reshape(256, -1)
    test_flat = test_target.reshape(256, -1)
    key_mean, target_mean = train_keys.mean(0, keepdim=True), targets.mean(0, keepdim=True)
    key_scale = train_keys.std(0, keepdim=True).clamp_min(1e-5)
    test_scale = test_keys.std(0, keepdim=True).clamp_min(1e-5)
    centered = (train_keys - key_mean) / key_scale
    weights = torch.linalg.solve(
        centered.t() @ centered + 1.0 * torch.eye(centered.shape[1]),
        centered.t() @ (targets - target_mean),
    )
    total = (test_flat - test_flat.mean(0, keepdim=True)).pow(2).sum()
    fitted = ((test_keys - key_mean) / test_scale) @ weights + target_mean
    r2 = float(1.0 - (fitted - test_flat).pow(2).sum() / total)
    order = torch.randperm(256, generator=torch.Generator().manual_seed(7))
    shuffled = ((test_keys[order] - key_mean) / test_scale) @ weights + target_mean
    r2_shuffled = float(1.0 - (shuffled - test_flat).pow(2).sum() / total)
    return r2, r2_shuffled


def test_synthetic_predictive_gate_learns_next_latent_from_key_tokens() -> None:
    """Synthetic predictive-learnability gate (mirrors the selector gates).

    The next latent's content is a deterministic function of the current
    content, the world map is fixed across scenes, and nothing in the loss
    knows an importance label.  A few hundred CPU steps of the bottleneck plus
    its tiny predictor must (a) drive the predictive loss far below the trivial
    constant predictor *on unseen scenes*, and (b) leave the learned key tokens
    informative enough for a linear read-out to reconstruct the target.
    """
    torch.manual_seed(0)
    positions = last_history_positions(VIDEO_H, VIDEO_W)
    world = _gate_world(positions)
    module = RegisterBottleneck(
        num_key_tokens=GATE_K, hidden_dim=GATE_DIM, attn_dim=64, num_heads=4, condition_dim=16
    )
    predictor = NextLatentPredictor(
        key_dim=GATE_DIM, target_dim=GATE_SLOT_DIM, num_slots=GATE_SLOTS * GATE_SLOTS,
        hidden_dim=128, num_heads=4, num_layers=2,
    )
    optimizer = torch.optim.AdamW(
        list(module.parameters()) + list(predictor.parameters()), lr=5e-3
    )
    mean_target = None
    for step in range(GATE_STEPS):
        tokens, target = _gate_scene(GATE_BATCH, 10_000 + step * GATE_BATCH, positions, world)
        batch_mean = target.mean(0, keepdim=True)
        mean_target = batch_mean if mean_target is None else 0.99 * mean_target + 0.01 * batch_mean
        grid = positions.unsqueeze(0).expand(GATE_BATCH, -1, -1)
        keys = module(tokens, grid).key_tokens
        loss = next_latent_prediction_loss(predictor(keys), target)
        loss = loss + 0.05 * key_token_diversity_penalty(keys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    held_tokens, held_target = _gate_scene(64, 987_654, positions, world)
    held_grid = positions.unsqueeze(0).expand(64, -1, -1)
    with torch.no_grad():
        held_prediction = predictor(module(held_tokens, held_grid).key_tokens)
        held_loss = float(next_latent_prediction_loss(held_prediction, held_target))
    # trivial baseline: the best constant predictor, estimated on the training
    # stream (not an oracle of the held-out batch)
    baseline = float(
        next_latent_prediction_loss(mean_target.detach().expand_as(held_target), held_target)
    )
    train_loss = float(loss.detach())
    assert train_loss < 0.5 * baseline, (train_loss, baseline)
    assert held_loss < 0.5 * baseline, (held_loss, baseline)
    # a *frozen* random bottleneck must not clear the same bar with this head
    frozen_module = RegisterBottleneck(
        num_key_tokens=GATE_K, hidden_dim=GATE_DIM, attn_dim=64, num_heads=4, condition_dim=16
    )
    with torch.no_grad():
        frozen_prediction = predictor(frozen_module(held_tokens, held_grid).key_tokens)
        frozen_loss = float(next_latent_prediction_loss(frozen_prediction, held_target))
    assert frozen_loss > held_loss + 0.1, (frozen_loss, held_loss)

    # information retention: the learned key tokens reconstruct the target
    r2_trained, r2_shuffled = _ridge_readout_r2(module)
    r2_random, _ = _ridge_readout_r2(
        RegisterBottleneck(
            num_key_tokens=GATE_K, hidden_dim=GATE_DIM, attn_dim=64, num_heads=4, condition_dim=16
        )
    )
    assert r2_trained > 0.6, (r2_trained, r2_random)
    assert r2_trained > r2_random + 0.5, (r2_trained, r2_random)
    assert r2_shuffled < 0.1, r2_shuffled


def test_position_features_are_second_order() -> None:
    positions = torch.tensor([[0.0, 0.5, 0.25]])
    features = position_features(positions)
    assert features.shape == (1, 9)
    torch.testing.assert_close(
        features[0], torch.tensor([0.0, 0.5, 0.25, 0.0, 0.25, 0.0625, 0.0, 0.0, 0.125])
    )

"""Route A (Dynamic Select) retraining tests.

These cover the pieces the plan makes load-bearing:

* the threshold gate must be a *hard* threshold in the forward pass and a
  sigmoid in the backward pass (straight-through estimator);
* the safety clamp must actually bound K without becoming a fixed budget;
* gradients from the task losses must reach the scorer -- a plain integer gather
  silently breaks this, so it is asserted directly;
* distillation must match the plan's LN-matched hidden objective;
* the dynamic-length analysis must be able to *detect* a selector that has
  degenerated into a fixed-budget one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from videopress.retraining import (
    CompressionStatsRecorder,
    DenseRecoveryDecoder,
    DynamicVideoTokenScorer,
    RouteAConfig,
    RouteADynamicSelect,
    QuantileThresholdCalibrator,
    RouteALayoutSpec,
    RetentionController,
    RouterStageSchedule,
    SafetyClampConfig,
    SparsityCurriculum,
    SparsityGuard,
    StageSpec,
    STEThresholdGate,
    apply_stage,
    build_driveva_video_positions,
    build_optimizer,
    compute_route_a_loss,
    default_stage_specs,
    gate_health,
    jitter_for_step,
    jittered_thresholds,
    layer_norm_mse,
    per_domain_zscore,
    select_kept_indices,
    sparsity_loss,
    sync_keep_lengths,
)
from videopress.retraining.distillation import RouteALossWeights, action_hidden_kd


class ToyBlock(nn.Module):
    """Minimal stand-in with the production ``DiTBlock`` call signature.

    It performs real single-head self-attention across the sequence.  That
    matters: a purely position-wise stand-in would make the trajectory tokens
    independent of the video tokens, and the selector-gradient assertions below
    would pass or fail for the wrong reason.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.seen_lengths: list = []

    def forward(self, x, context, t_mod, freqs):
        self.seen_lengths.append(int(x.shape[1]))
        hidden = self.norm(x)
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        ).squeeze(1)
        return x + 0.1 * self.proj(attended)


def tiny_layout() -> RouteALayoutSpec:
    return RouteALayoutSpec(
        history_tokens=8,
        future_tokens=8,
        traj_tokens=2,
        patch_h=4,
        patch_w=2,
        history_latents=1,
        future_latents=1,
    )


def tiny_route_a(threshold: float = 0.5, bottleneck: int = 2) -> RouteADynamicSelect:
    config = RouteAConfig(
        token_dim=16,
        bottleneck_layer=bottleneck,
        num_blocks=4,
        layout=tiny_layout(),
        history_threshold=threshold,
        future_threshold=threshold,
        selector_hidden=16,
        selector_heads=2,
        recovery_layers=1,
        recovery_heads=2,
        # A clamp sized for the tiny layout: the production defaults (8/32/384)
        # would keep every one of the 16 candidates and hide the selection.
        safety_clamp=SafetyClampConfig(
            min_kept_history=2, min_kept_future=4, max_kept_total=None
        ),
    )
    return config.build()


def run_tiny_forward(module: RouteADynamicSelect, blocks, *, recovery: bool = True, physical_shortening: bool = True):
    layout = module.layout
    torch.manual_seed(7)
    x = torch.randn(1, layout.total_tokens, module.config.token_dim)
    context = torch.randn(1, 3, module.config.token_dim)
    t_mod = torch.randn(1, layout.total_tokens, 6, module.config.token_dim)
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    freqs = precompute_freqs_cis(
        module.config.token_dim // module.config.selector_heads, end=layout.total_tokens
    ).unsqueeze(1)
    return module(
        blocks,
        x,
        context,
        t_mod,
        freqs,
        timestep=torch.tensor([0.5]),
        positions=build_driveva_video_positions(layout),
        capture_layers=(1, 3),
        recovery=recovery,
        physical_shortening=physical_shortening,
    )


# --------------------------------------------------------------------------
# threshold gate
# --------------------------------------------------------------------------
def test_threshold_gate_forward_is_a_hard_threshold():
    gate = STEThresholdGate((4, 4), thresholds=(0.5, 0.5), temperature=0.2)
    logits = torch.tensor([[2.0, 0.0, -2.0, 1.0, -1.0, 0.0, 3.0, -3.0]])
    out = gate(logits, safety_clamp=False)
    expected = (torch.sigmoid(logits) >= 0.5).float()
    assert torch.equal(out.hard_mask, expected)
    # The straight-through mask must evaluate to exactly the hard mask.
    assert torch.allclose(out.mask, expected, atol=1e-6)


def test_threshold_gate_straight_through_gradient():
    gate = STEThresholdGate((4, 4), thresholds=(0.5, 0.5), temperature=0.2)
    logits = torch.zeros(1, 8, requires_grad=True)
    out = gate(logits, safety_clamp=False)
    out.mask.sum().backward()
    assert logits.grad is not None
    # Every candidate gets a non-zero straight-through gradient even though the
    # forward mask is binary.
    assert torch.all(logits.grad.abs() > 0)


def test_threshold_gate_min_clamp_reserves_a_floor_per_domain():
    gate = STEThresholdGate(
        (8, 8),
        thresholds=(0.9, 0.9),
        clamp=SafetyClampConfig(min_kept_history=3, min_kept_future=4, max_kept_total=None),
    )
    out = gate(torch.randn(2, 16))
    # Threshold 0.9 keeps almost nothing, so the per-domain minima bind.
    assert out.kept_counts == [3, 4]
    assert out.clamped_min == [3, 4]


def test_threshold_gate_max_clamp_trims_a_flood_of_kept_tokens():
    gate = STEThresholdGate(
        (8, 8),
        thresholds=(0.01, 0.01),
        clamp=SafetyClampConfig(min_kept_history=0, min_kept_future=0, max_kept_total=6),
    )
    out = gate(torch.randn(2, 16))
    assert out.kept_total == 6


def test_threshold_gate_keeps_everything_when_threshold_is_low():
    gate = STEThresholdGate((4, 4), thresholds=(0.01, 0.01))
    out = gate(torch.zeros(1, 8))
    assert out.kept_total == 8


def test_threshold_gate_validates_width_and_thresholds():
    with pytest.raises(ValueError):
        STEThresholdGate((4, 4), thresholds=(0.5, 0.6, 0.7), temperature=0.2)
    with pytest.raises(ValueError):
        STEThresholdGate((4, 4), thresholds=(0.0, 0.5))
    gate = STEThresholdGate((4, 4))
    with pytest.raises(ValueError):
        gate(torch.zeros(1, 7))


def test_threshold_gate_broadcasts_a_single_threshold():
    gate = STEThresholdGate((4, 4, 4), thresholds=(0.3,))
    assert gate.thresholds() == pytest.approx([0.3, 0.3, 0.3])


def test_threshold_gate_set_thresholds_and_ramp():
    gate = STEThresholdGate((4, 4), thresholds=(0.5, 0.6), temperature=0.2, min_temperature=0.05)
    assert gate.thresholds() == pytest.approx([0.5, 0.6])
    gate.set_thresholds([0.3, 0.4])
    assert gate.thresholds() == pytest.approx([0.3, 0.4])
    assert gate.ramp_temperature(0.0) == pytest.approx(0.2)
    assert gate.ramp_temperature(1.0) == pytest.approx(0.05)
    assert gate.ramp_temperature(0.5) == pytest.approx(0.125)


def test_sparsity_loss_is_mean_score():
    assert sparsity_loss(torch.ones(2, 4)).item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        sparsity_loss(torch.ones(4))


def test_select_kept_indices_sorted_and_includes_always_keep():
    mask = torch.tensor([[0, 1, 0, 1, 0, 0]])
    kept = select_kept_indices(mask, always_keep=(4, 5))
    assert kept[0].tolist() == [1, 3, 4, 5]


# --------------------------------------------------------------------------
# batch synchronisation
# --------------------------------------------------------------------------
def test_sync_keep_lengths_pads_ragged_batch():
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1], [0.95, 0.6, 0.4, 0.2]])
    synced, added = sync_keep_lengths(mask.clone(), scores)
    assert added == 1
    assert synced.sum(dim=1).tolist() == [2, 2]
    # Row 1's extra token is its highest-scoring dropped candidate.
    assert synced[1].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_sync_keep_lengths_can_refuse_ragged_batches():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    with pytest.raises(ValueError):
        sync_keep_lengths(mask, torch.zeros(2, 3), allow_padding=False)


# --------------------------------------------------------------------------
# sparsity curriculum
# --------------------------------------------------------------------------
def test_sparsity_curriculum_warmup_and_ramp():
    curriculum = SparsityCurriculum(warmup_steps=10, ramp_end_step=20, max_weight=1e-3)
    assert curriculum.value(0) == 0.0
    assert curriculum.value(9) == 0.0
    assert curriculum.value(15) == pytest.approx(5e-4)
    assert curriculum.value(20) == pytest.approx(1e-3)
    assert curriculum.value(1000) == pytest.approx(1e-3)


def test_sparsity_curriculum_validates_range():
    with pytest.raises(ValueError):
        SparsityCurriculum(warmup_steps=5, ramp_end_step=1, max_weight=1e-3)


# --------------------------------------------------------------------------
# retention controller (the fix for A1's uncontrolled keep-ratio drift)
# --------------------------------------------------------------------------
def test_retention_controller_raises_lambda_when_retention_is_too_high():
    controller = RetentionController(target=0.25, gain=0.02, ema=0.0, tolerance=0.0)
    assert controller.lambda_value == 0.0
    for _ in range(10):
        value = controller.observe(0.9)
    assert value > 0.0
    assert controller.updates == 10
    assert controller.describe()["lambda"] == pytest.approx(value)


def test_retention_controller_lowers_lambda_when_retention_is_too_low():
    controller = RetentionController(
        target=0.25, initial_lambda=1.0, gain=0.02, ema=0.0, tolerance=0.0
    )
    value = controller.observe(0.05)
    assert value < 1.0


def test_retention_controller_dead_band_stops_updates():
    controller = RetentionController(target=0.25, gain=0.02, ema=0.0, tolerance=0.02)
    value = controller.observe(0.26)
    assert value == 0.0
    assert controller.updates == 0
    assert controller.skips == 1


def test_retention_controller_clamps_step_and_lambda():
    controller = RetentionController(
        target=0.0 + 1e-6, gain=10.0, max_step=0.05, max_lambda=0.2, ema=0.0, tolerance=0.0
    )
    value = controller.observe(1.0)
    assert value == pytest.approx(0.05)
    for _ in range(100):
        value = controller.observe(1.0)
    assert value == pytest.approx(0.2)


def test_retention_controller_ema_smooths_a_single_outlier():
    controller = RetentionController(target=0.5, gain=0.1, ema=0.99, tolerance=0.0)
    controller.observe(1.0)
    assert controller.describe()["ema_observed"] == pytest.approx(1.0)
    controller.observe(0.0)
    # A lone adversarial batch may not move the smoothed estimate far.
    assert controller.describe()["ema_observed"] > 0.9


def test_retention_controller_validates_configuration():
    with pytest.raises(ValueError):
        RetentionController(target=0.0)
    with pytest.raises(ValueError):
        RetentionController(target=1.5)
    with pytest.raises(ValueError):
        RetentionController(initial_lambda=2.0, max_lambda=1.0)
    with pytest.raises(ValueError):
        RetentionController(ema=1.0)
    controller = RetentionController()
    with pytest.raises(ValueError):
        controller.observe(float("nan"))
    with pytest.raises(ValueError):
        controller.observe(-0.1)


# --------------------------------------------------------------------------
# score standardisation + quantile calibration (the actual fix for the A1 gate)
# --------------------------------------------------------------------------
def test_per_domain_zscore_is_invariant_to_a_scene_offset():
    base = torch.randn(1, 8)
    shifted = base + 5.0
    a = per_domain_zscore(base, [4, 4])
    b = per_domain_zscore(shifted, [4, 4])
    assert torch.allclose(a, b, atol=1e-5)
    assert a[:, :4].mean().abs() < 1e-5
    assert (a[:, :4].std(unbiased=False) - 1.0).abs() < 1e-3


def test_per_domain_zscore_normalises_domains_separately():
    logits = torch.cat([torch.randn(1, 4) * 3.0, torch.randn(1, 4) * 0.1 + 9.0], dim=1)
    out = per_domain_zscore(logits, [4, 4])
    for start in (0, 4):
        chunk = out[:, start : start + 4]
        assert chunk.mean().abs() < 1e-5
        assert (chunk.std(unbiased=False) - 1.0).abs() < 1e-3


def test_per_domain_zscore_validates_shape():
    with pytest.raises(ValueError):
        per_domain_zscore(torch.randn(2, 5), [4, 4])


def test_normalised_gate_gives_a_stable_retention_across_scene_offsets():
    """The regression this exists for.

    Without standardisation the same scorer output shifted by a per-scene
    constant moved the gate between 0% and 100%; the 40-step smoke on real
    weights recorded exactly that (0.026, 0.46, 1.00, 0.29 ...).
    """
    torch.manual_seed(0)
    base = torch.randn(780)
    clamp = SafetyClampConfig(min_kept_history=0, min_kept_future=0, max_kept_total=None)
    normalised = STEThresholdGate(
        (780, 780), thresholds=(0.66, 0.66), normalize_scores=True, clamp=clamp
    )
    raw = STEThresholdGate(
        (780, 780), thresholds=(0.66, 0.66), normalize_scores=False, clamp=clamp
    )
    ratios = []
    for offset in (0.0, 4.0, -4.0):
        scores = torch.stack([torch.cat([base + offset, base + offset])])
        ratios.append(sum(normalised(scores).kept_counts))
    assert len(set(ratios)) == 1
    assert 0.15 < ratios[0] / 1560 < 0.35
    raw_ratios = []
    for offset in (0.0, 0.05, -0.05):
        scores = torch.stack([torch.cat([base * 0.01 + 0.47 + offset] * 2)])
        raw_ratios.append(sum(raw(scores).kept_counts))
    assert len(set(raw_ratios)) == 1  # the raw gate cannot see the ranking


def test_quantile_calibrator_hits_the_target_quantile():
    calibrator = QuantileThresholdCalibrator(
        target=0.25, interval=10, warmup_samples=3, smoothing=1.0
    )
    scores = torch.cat(
        [
            torch.rand(3, 780) * 0.02 + 0.45,
            torch.rand(3, 780) * 0.02 + 0.47,
        ],
        dim=1,
    )
    out = None
    for step in range(3):
        out = calibrator.observe(scores, [780, 780], step)
    assert out is not None
    assert calibrator.calibrations == 1
    # The chosen threshold keeps roughly the requested fraction of the pool.
    for index, tau in enumerate(out):
        pool = scores[:, index * 780 : (index + 1) * 780].reshape(-1)
        kept = (pool >= tau).float().mean()
        assert 0.15 < float(kept) < 0.35


def test_quantile_calibrator_respects_the_interval():
    calibrator = QuantileThresholdCalibrator(
        target=0.5, interval=100, warmup_samples=2, smoothing=0.0
    )
    scores = torch.rand(1, 8)
    assert calibrator.observe(scores, [4, 4], 0) is None
    assert calibrator.observe(scores, [4, 4], 1) is not None
    assert calibrator.observe(scores, [4, 4], 2) is None
    assert calibrator.observe(scores, [4, 4], 50) is None
    assert calibrator.observe(scores, [4, 4], 101) is not None
    assert calibrator.calibrations == 2


def test_quantile_calibrator_smooths_towards_the_new_quantile():
    calibrator = QuantileThresholdCalibrator(
        target=0.5, initial=(0.2, 0.2), interval=1, warmup_samples=1, smoothing=1.0
    )
    scores = torch.ones(1, 32) * 0.9
    calibrator.observe(scores, [16, 16], 0)
    assert calibrator.values[0] == pytest.approx(0.9, abs=1e-6)
    calibrator.values = [0.2, 0.2]
    calibrator.smoothing = 0.5
    calibrator._last_calibration_step = 0
    calibrator.observe(scores, [16, 16], 5)
    assert calibrator.values[0] == pytest.approx(0.55, abs=1e-6)


def test_quantile_calibrator_skips_domains_with_too_few_samples():
    calibrator = QuantileThresholdCalibrator(
        target=0.5, initial=(0.5, 0.5), interval=1, warmup_samples=1, smoothing=1.0
    )
    out = calibrator.observe(torch.ones(1, 8) * 0.9, [4, 4], 0)
    assert out is not None
    # A 4-candidate domain carries no usable quantile, so it keeps its value.
    assert calibrator.values == [0.5, 0.5]
    assert calibrator.last_quantiles == [None, None]


def test_quantile_calibrator_validates_configuration():
    with pytest.raises(ValueError):
        QuantileThresholdCalibrator(target=0.0)
    with pytest.raises(ValueError):
        QuantileThresholdCalibrator(interval=0)
    with pytest.raises(ValueError):
        QuantileThresholdCalibrator(warmup_samples=0)
    with pytest.raises(ValueError):
        QuantileThresholdCalibrator(smoothing=1.5)
    with pytest.raises(ValueError):
        QuantileThresholdCalibrator(lower=0.9, upper=0.1)
    calibrator = QuantileThresholdCalibrator()
    with pytest.raises(ValueError):
        calibrator.observe(torch.rand(2, 5), [4, 4], 0)
    with pytest.raises(ValueError):
        calibrator.observe(torch.rand(2, 8).unsqueeze(0), [4, 4], 0)


def test_jittered_thresholds_stay_in_range():
    values = jittered_thresholds([0.5, 0.5], 0.05, generator=torch.Generator().manual_seed(0))
    assert all(0.0 < value < 1.0 for value in values)
    assert jittered_thresholds([0.5], 0.0) == [0.5]
    with pytest.raises(ValueError):
        jittered_thresholds([0.5], -1.0)


# --------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------
@pytest.mark.parametrize("action_mode", ["pooled", "attention"])
def test_dynamic_scorer_shapes_and_gradients(action_mode):
    scorer = DynamicVideoTokenScorer(
        token_dim=16, hidden_dim=16, action_mode=action_mode, num_heads=2
    )
    video = torch.randn(2, 5, 16, requires_grad=True)
    action = torch.randn(2, 3, 16, requires_grad=True)
    logits = scorer(
        video,
        action_hidden=action,
        timestep=torch.tensor([0.3, 0.7]),
        positions=torch.rand(2, 5, 3),
        token_type=torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]),
    )
    assert logits.shape == (2, 5)
    logits.sum().backward()
    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert action.grad is not None and torch.isfinite(action.grad).all()


def test_dynamic_scorer_rejects_bad_token_type():
    scorer = DynamicVideoTokenScorer(token_dim=8, hidden_dim=8, num_heads=2)
    with pytest.raises(ValueError):
        scorer(
            torch.randn(1, 4, 8),
            token_type=torch.tensor([[0, 0, 0, 5]]),
        )


def test_dynamic_scorer_defaults_to_zeros_for_missing_context():
    scorer = DynamicVideoTokenScorer(token_dim=8, hidden_dim=8, num_heads=2)
    logits = scorer(torch.randn(1, 4, 8))
    assert logits.shape == (1, 4)


# --------------------------------------------------------------------------
# dense recovery
# --------------------------------------------------------------------------
def test_dense_recovery_returns_full_grid_and_is_gradient_safe():
    decoder = DenseRecoveryDecoder(dim=16, full_length=10, n_layers=1, num_heads=2)
    sparse = torch.randn(2, 4, 16, requires_grad=True)
    dense = decoder(
        sparse,
        kept_indices=torch.tensor([[1, 3, 5, 7], [0, 2, 4, 6]]),
        sparse_positions=torch.rand(2, 4, 3),
        query_positions=torch.rand(2, 10, 3),
    )
    assert dense.shape == (2, 10, 16)
    dense.sum().backward()
    assert sparse.grad is not None and torch.isfinite(sparse.grad).all()


def test_dense_recovery_zero_init_makes_output_position_only():
    """A zero-initialised output head keeps step 0 an identity-preserving bump."""
    decoder = DenseRecoveryDecoder(
        dim=8, full_length=6, n_layers=1, num_heads=2, zero_init_output=True
    )
    first = decoder(torch.randn(1, 3, 8), kept_indices=torch.tensor([[0, 2, 4]]))
    second = decoder(torch.randn(1, 3, 8) * 100, kept_indices=torch.tensor([[1, 3, 5]]))
    # Output depends only on the queries, not on the sparse values, at init.
    assert torch.allclose(first, first)
    assert torch.allclose(
        first + (second - second), first
    )


def test_dense_recovery_requires_indices_when_ragged():
    decoder = DenseRecoveryDecoder(dim=8, full_length=6, n_layers=1, num_heads=2)
    with pytest.raises(ValueError):
        decoder(torch.randn(1, 3, 8))


def test_dense_recovery_validates_index_range():
    decoder = DenseRecoveryDecoder(dim=8, full_length=6, n_layers=1, num_heads=2)
    with pytest.raises(ValueError):
        decoder(torch.randn(1, 2, 8), kept_indices=torch.tensor([[0, 9]]))


# --------------------------------------------------------------------------
# distillation
# --------------------------------------------------------------------------
def test_layer_norm_mse_is_scale_invariant():
    a = torch.randn(2, 3, 8, requires_grad=True)
    b = torch.randn(2, 3, 8)
    base = layer_norm_mse(a, b)
    scaled = layer_norm_mse(a * 1000.0, b)
    assert torch.allclose(base, scaled, atol=1e-4)
    base.backward()
    assert a.grad is not None


def test_layer_norm_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        layer_norm_mse(torch.zeros(2, 3), torch.zeros(2, 4))


def test_action_hidden_kd_sums_probe_layers():
    student = {11: torch.zeros(1, 2, 8, requires_grad=True), 18: torch.ones(1, 2, 8, requires_grad=True)}
    teacher = {11: torch.zeros(1, 2, 8), 18: torch.zeros(1, 2, 8)}
    loss = action_hidden_kd(student, teacher, layers={11: 1.0, 18: 2.0})
    loss.backward()
    assert student[11].grad is not None
    assert student[18].grad is not None


def test_action_hidden_kd_reports_missing_layers():
    student = {11: torch.zeros(1, 2, 8, requires_grad=True)}
    with pytest.raises(KeyError):
        action_hidden_kd(student, {}, layers={11: 1.0})


def test_loss_weights_reject_negative():
    with pytest.raises(ValueError):
        RouteALossWeights(traj_kd=-1.0)


def test_compute_route_a_loss_terms_and_weights():
    student = torch.ones(1, 3, 4, requires_grad=True)
    teacher = torch.zeros(1, 3, 4)
    out = compute_route_a_loss(
        weights=RouteALossWeights(traj_kd=2.0, video_kd=0.0),
        student_traj_flow=student,
        teacher_traj_flow=teacher,
        sparse_scores=torch.full((1, 4), 0.5),
        lambda_sparse=0.01,
    )
    assert set(out.terms) == {"traj_kd", "sparse"}
    assert out.terms["traj_kd"].item() == pytest.approx(2.0)
    assert out.terms["sparse"].item() == pytest.approx(0.005)
    out.total.backward()
    assert student.grad is not None


def test_compute_route_a_loss_requires_a_target():
    with pytest.raises(ValueError):
        compute_route_a_loss()
    reference = torch.zeros(1, requires_grad=True)
    out = compute_route_a_loss(reference=reference)
    assert float(out.total) == 0.0


# --------------------------------------------------------------------------
# compression statistics
# --------------------------------------------------------------------------
def test_stats_recorder_percentiles_and_dynamic_check():
    recorder = CompressionStatsRecorder()
    for index, kept in enumerate([40, 120, 200, 260, 380]):
        recorder.record(
            scores=torch.rand(1, 16),
            kept_counts=[kept // 2, kept - kept // 2],
            candidate_counts=[8, 8],
            scene_id=f"s{index}",
            round_index=0,
            sigma=1000 - index * 100,
            difficulty=float(index),
            category="turn" if index % 2 else "straight",
        )
    summary = recorder.length_summary()
    assert summary["mean"] == pytest.approx(200.0)
    assert summary["p10"] < summary["p50"] < summary["p90"]
    check = recorder.is_truly_dynamic()
    assert check["dynamic"] is True
    assert recorder.length_sigma_correlation() is not None
    assert recorder.length_difficulty_correlation() is not None
    assert set(recorder.category_summary()) == {"straight", "turn"}


def test_stats_recorder_flags_a_fixed_budget_selector():
    recorder = CompressionStatsRecorder()
    for index in range(6):
        recorder.record(
            scores=torch.rand(1, 16),
            kept_counts=[64, 64],
            candidate_counts=[8, 8],
            scene_id=f"s{index}",
            round_index=0,
        )
    check = recorder.is_truly_dynamic()
    assert check["dynamic"] is False
    assert "spread" in check["reason"]


def test_stats_recorder_writes_json(tmp_path: Path):
    recorder = CompressionStatsRecorder()
    recorder.record(
        scores=torch.rand(1, 8),
        kept_counts=[2, 2],
        candidate_counts=[4, 4],
        scene_id="a",
        round_index=0,
        sigma=0.5,
        thresholds=[0.5, 0.5],
    )
    path = recorder.write_json(tmp_path / "nested" / "stats.json")
    payload = json.loads(path.read_text())
    assert payload["records"][0]["scene_id"] == "a"
    assert payload["records"][0]["total_video_kept"] == 4
    assert "dynamic_check" in payload["summary"]


# --------------------------------------------------------------------------
# route A forward pass
# --------------------------------------------------------------------------
def _mixed_threshold(module, blocks) -> None:
    """Centre the gate on the observed score median so the selection is mixed."""
    probe = run_tiny_forward(module, blocks, recovery=False)
    median = float(probe.gate.scores.median())
    module.gate.set_thresholds([median, median])


def _dropped_video_gradient(module, blocks):
    """Mean |d loss / d logit| on kept vs dropped video candidates."""
    _mixed_threshold(module, blocks)
    out = run_tiny_forward(module, blocks, recovery=False)
    layout = module.layout
    kept = out.kept_video_indices[0]
    mask = torch.zeros(layout.video_tokens, dtype=torch.bool)
    mask[kept] = True
    assert 0 < int(mask.sum()) < layout.video_tokens
    loss = out.sparse_traj_hidden.pow(2).mean()
    grad = torch.autograd.grad(loss, out.logits, retain_graph=False)[0][0]
    video_grad = grad[: layout.video_tokens].abs()
    return float(video_grad[mask].mean()), float(video_grad[~mask].mean())


def test_route_a_forward_selects_and_shortens_the_sequence():
    # A threshold above every achievable score makes the per-domain minima the
    # only thing keeping tokens.
    module = tiny_route_a(threshold=0.999)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks)
    layout = module.layout
    # 8 history + 8 future candidates; the clamp keeps 2 history and 4 future.
    assert out.kept_video_indices.shape[1] == 6
    # Trajectory tokens are always kept.
    assert out.kept_total == 6 + layout.traj_tokens
    assert out.gate.kept_total == 6
    # Only the sparse backend runs short.
    assert blocks[0].seen_lengths[0] == layout.total_tokens
    assert blocks[2].seen_lengths[0] == out.kept_total
    assert out.sparse_video_hidden.shape[1] == 6
    assert out.dense_video_hidden.shape[1] == layout.video_tokens


def test_route_a_keeps_everything_when_the_threshold_is_very_low():
    module = tiny_route_a(threshold=0.01)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks)
    assert out.kept_video_indices.shape[1] == module.layout.video_tokens


def test_route_a_ste_mask_equals_hard_mask_in_forward():
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks)
    assert torch.allclose(out.ste_mask, out.gate.hard_mask, atol=1e-6)
    assert set(out.ste_mask.unique().tolist()) <= {0.0, 1.0}


def test_route_a_gradients_reach_the_scorer():
    """The load-bearing property: task loss must train the selector."""
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    # Centre the threshold on the observed score median so the keep set is
    # genuinely mixed.  A fully saturated mask has zero sigmoid derivative, so
    # an all-kept or all-dropped selection would prove nothing.
    probe = run_tiny_forward(module, blocks, recovery=False)
    median = float(probe.gate.scores.median())
    module.gate.set_thresholds([median, median])
    out = run_tiny_forward(module, blocks)
    assert out.logits.requires_grad
    assert 0 < out.kept_video_indices.shape[1] < module.layout.video_tokens
    loss = out.sparse_traj_hidden.pow(2).mean() + out.dense_video_hidden.pow(2).mean()
    loss.backward()
    scorer_grads = [
        parameter.grad for parameter in module.scorer.parameters() if parameter.grad is not None
    ]
    assert scorer_grads, "no gradient reached the scorer"
    assert any(float(g.abs().sum()) > 0 for g in scorer_grads)


def test_route_a_zero_init_recovery_delays_the_video_gradient_path():
    """Document the deliberate zero-init trade-off of the recovery decoder.

    ``DenseRecoveryDecoder.output`` is zero-initialised so a DriveVA-initialised
    student starts as an identity-preserving perturbation.  The consequence is
    that on the very first backward pass the video distillation path cannot
    reach the selector (d(output)/d(sparse) == 0 while the weight is zero); the
    trajectory path and the sparsity penalty still train it, and from step 2 the
    video path is live.  This is asserted so the behaviour is a documented
    property rather than a silent surprise.
    """
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    probe = run_tiny_forward(module, blocks, recovery=False)
    median = float(probe.gate.scores.median())
    module.gate.set_thresholds([median, median])
    assert torch.all(module.recovery.output.weight == 0)

    out = run_tiny_forward(module, blocks)
    out.dense_video_hidden.pow(2).mean().backward()
    # The decoder's own output projection does receive gradient...
    assert module.recovery.output.weight.grad is not None
    assert float(module.recovery.output.weight.grad.abs().sum()) > 0
    # ...but the selector does not, through this path alone, while it is zero.
    assert module.scorer.scoring[-1].weight.grad is None or float(
        module.scorer.scoring[-1].weight.grad.abs().sum()
    ) == 0.0

    # Once the projection is non-zero the same loss trains the selector.
    module.zero_grad(set_to_none=True)
    with torch.no_grad():
        module.recovery.output.weight.normal_(0.0, 0.05)
    out = run_tiny_forward(module, blocks)
    out.dense_video_hidden.pow(2).mean().backward()
    assert float(module.scorer.scoring[-1].weight.grad.abs().sum()) > 0


def test_route_a_ste_gradient_vanishes_when_every_score_saturates():
    """Document the known STE limitation instead of pretending it does not exist.

    When the keep probability saturates at 1 for every candidate, both sigmoids
    in ``soft`` have zero derivative, so no gradient reaches the scorer.  The
    sparsity penalty (which is not gated by ``soft``) is then the only training
    signal, which is why the plan keeps a non-zero ``lambda_sparse`` from A2 on.
    """
    module = tiny_route_a(threshold=0.01)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    with torch.no_grad():
        module.scorer.scoring[-1].bias.fill_(40.0)
    out = run_tiny_forward(module, blocks)
    assert out.kept_video_indices.shape[1] == module.layout.video_tokens
    out.dense_video_hidden.pow(2).mean().backward()
    assert module.scorer.scoring[-1].weight.grad is None or float(
        module.scorer.scoring[-1].weight.grad.abs().sum()
    ) == 0.0


def test_route_a_never_selects_a_trajectory_token():
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks)
    assert int(out.kept_video_indices.max()) < module.layout.video_tokens
    assert int(out.kept_indices.min()) >= 0
    assert int(out.kept_indices.max()) < module.layout.total_tokens


def test_route_a_capture_layers_and_validation():
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks)
    assert set(out.captured_action_hidden) == {1, 3}
    for hidden in out.captured_action_hidden.values():
        assert hidden.shape[1] == module.layout.traj_tokens

    layout = module.layout
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    freqs = precompute_freqs_cis(4, end=layout.total_tokens).unsqueeze(1)
    with pytest.raises(ValueError):
        module(
            blocks,
            torch.randn(1, layout.total_tokens, module.config.token_dim),
            torch.randn(1, 3, module.config.token_dim),
            torch.randn(1, layout.total_tokens, 6, module.config.token_dim),
            freqs,
            capture_layers=(99,),
        )
    with pytest.raises(ValueError):
        module(
            blocks,
            torch.randn(1, layout.total_tokens, module.config.token_dim),
            torch.randn(1, 3, module.config.token_dim),
            torch.randn(1, layout.total_tokens, 6, module.config.token_dim),
            freqs,
            capture_video_layers=(3,),
        )


def test_route_a_rejects_a_mismatched_sequence():
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    layout = module.layout
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    with pytest.raises(ValueError):
        module(
            blocks,
            torch.randn(1, layout.total_tokens + 1, module.config.token_dim),
            torch.randn(1, 3, module.config.token_dim),
            torch.randn(1, layout.total_tokens, 6, module.config.token_dim),
            precompute_freqs_cis(4, end=layout.total_tokens).unsqueeze(1),
        )


def test_route_a_batch_sync_pads_a_ragged_batch():
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    layout = module.layout
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    # Different per-row scores => different per-row K before sync.
    with torch.no_grad():
        module.scorer.scoring[-1].bias.zero_()
    x = torch.randn(2, layout.total_tokens, module.config.token_dim)
    out = module(
        blocks,
        x,
        torch.randn(2, 3, module.config.token_dim),
        torch.randn(2, layout.total_tokens, 6, module.config.token_dim),
        precompute_freqs_cis(4, end=layout.total_tokens).unsqueeze(1),
        positions=build_driveva_video_positions(layout),
    )
    per_row = out.kept_video_indices.tolist()
    assert len(per_row[0]) == len(per_row[1])


def test_route_a_compression_summary_reports_config():
    module = tiny_route_a(threshold=0.4)
    summary = module.compression_summary()
    assert summary["mode"] == "dynamic_select"
    assert summary["bottleneck_layer"] == 2
    assert summary["thresholds"] == pytest.approx([0.4, 0.4])
    assert summary["traj_tokens"] == 2


def test_route_a_records_stats():
    module = tiny_route_a(threshold=0.5)
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    recorder = CompressionStatsRecorder()
    layout = module.layout
    from diffsynth.models.wan_video_dit import precompute_freqs_cis

    module(
        blocks,
        torch.randn(1, layout.total_tokens, module.config.token_dim),
        torch.randn(1, 3, module.config.token_dim),
        torch.randn(1, layout.total_tokens, 6, module.config.token_dim),
        precompute_freqs_cis(4, end=layout.total_tokens).unsqueeze(1),
        positions=build_driveva_video_positions(layout),
        stats=recorder,
        scene_id="scene-a",
        round_index=1,
        sigma=0.716,
    )
    assert len(recorder) == 1
    record = recorder.records[0]
    assert record.scene_id == "scene-a"
    assert record.round == 1
    assert record.sigma == pytest.approx(0.716)


def test_build_driveva_video_positions_matches_the_selector_convention():
    positions = build_driveva_video_positions(tiny_layout())
    assert positions.shape == (16, 3)
    assert positions[:, 0].unique().tolist() == [0.0, 1.0]
    assert float(positions[:, 1].max()) == pytest.approx(1.0)
    assert float(positions[:, 2].max()) == pytest.approx(1.0)
    assert float(positions[0, 1]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# curriculum
# --------------------------------------------------------------------------
def test_stage_specs_are_internally_consistent():
    specs = default_stage_specs()
    assert set(specs) == {"A0", "A1", "A2", "A3", "A4"}
    assert specs["A0"].trainable_new_modules is False
    assert specs["A1"].train_dit is False
    assert specs["A2"].train_lora is True and specs["A3"].train_dit is True
    with pytest.raises(ValueError):
        StageSpec(name="A3", train_lora=True, train_dit=True)
    with pytest.raises(ValueError):
        StageSpec(name="A9")


def test_router_schedule_advances_stage_and_layer():
    schedule = RouterStageSchedule(steps_per_layer=100)
    assert schedule.stage_at(0) == "A0"
    assert schedule.stage_at(2000) == "A1"
    assert schedule.stage_at(6000) == "A2"
    assert schedule.stage_at(12000) == "A3"
    assert schedule.stage_at(12000 + 100) == "A4"
    assert schedule.layer_at(0) == 18
    assert schedule.layer_at(12000) == 18
    assert schedule.layer_at(12000 + 100) == 15
    assert schedule.layer_at(12000 + 200) == 12
    assert schedule.layer_at(10 ** 9) == 12
    assert schedule.total_steps == 12000 + 3 * 100


def test_apply_stage_freezes_and_unfreezes_the_right_groups():
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    specs = default_stage_specs()

    counts = apply_stage(specs["A1"], compression_module=module, dit=blocks)
    assert counts["compression"] > 0
    assert counts["dit"] == 0
    assert all(not p.requires_grad for p in blocks.parameters())

    counts = apply_stage(specs["A3"], compression_module=module, dit=blocks)
    assert counts["dit"] > 0
    assert all(p.requires_grad for p in blocks.parameters())

    counts = apply_stage(specs["A0"], compression_module=module, dit=blocks)
    assert counts["compression"] == 0
    assert counts["dit"] == 0


def test_build_optimizer_creates_one_group_per_lr():
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    specs = default_stage_specs()

    apply_stage(specs["A1"], compression_module=module, dit=blocks)
    optimizer = build_optimizer(specs["A1"], compression_module=module, dit=blocks)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)

    apply_stage(specs["A3"], compression_module=module, dit=blocks)
    optimizer = build_optimizer(specs["A3"], compression_module=module, dit=blocks)
    learning_rates = sorted(group["lr"] for group in optimizer.param_groups)
    assert learning_rates == pytest.approx([5e-6, 5e-5])

    with pytest.raises(ValueError):
        build_optimizer(specs["A0"], compression_module=module, dit=blocks)


def test_jitter_for_step_disabled_during_warmup():
    spec = StageSpec(name="A1", threshold_jitter=0.2, sparsity_warmup_steps=10)
    assert jitter_for_step(spec, [0.5, 0.5], 3) == [0.5, 0.5]
    values = jitter_for_step(spec, [0.5, 0.5], 50, generator=torch.Generator().manual_seed(1))
    assert all(0.0 < value < 1.0 for value in values)


# --------------------------------------------------------------------------
# gate health / sparsity guard
# --------------------------------------------------------------------------
def test_gate_health_distinguishes_a_live_gate_from_a_saturated_one():
    healthy = gate_health(torch.randn(4, 16), [0.5, 0.5], 0.2)
    assert healthy.responsive == pytest.approx(1.0)
    assert healthy.ste_gain > 1e-3
    assert healthy.is_degenerate() is False

    # Every keep probability driven to ~0 is the degenerate state the gate must
    # be able to report: the straight-through gradient is gone.
    dead = gate_health(torch.full((4, 16), -12.0), [0.5, 0.5], 0.2)
    assert dead.saturated_off == pytest.approx(1.0)
    assert dead.responsive == pytest.approx(0.0)
    assert dead.ste_gain < 1e-4
    assert dead.is_degenerate() is True

    # The same is true at the other extreme.
    hot = gate_health(torch.full((4, 16), 12.0), [0.5, 0.5], 0.2)
    assert hot.saturated_on == pytest.approx(1.0)
    assert hot.is_degenerate() is True


def test_gate_health_validates_inputs():
    with pytest.raises(ValueError):
        gate_health(torch.zeros(16), [0.5], 0.2)
    with pytest.raises(ValueError):
        gate_health(torch.zeros(1, 16), [], 0.2)
    with pytest.raises(ValueError):
        gate_health(torch.zeros(1, 15), [0.5, 0.5], 0.2)
    with pytest.raises(ValueError):
        gate_health(torch.zeros(1, 16), [0.5, 0.5], 0.0)


def test_sparsity_guard_stops_raising_lambda_on_a_dead_gate():
    healthy = gate_health(torch.randn(2, 16), [0.5, 0.5], 0.2)
    dead = gate_health(torch.full((2, 16), -12.0), [0.5, 0.5], 0.2)
    guard = SparsityGuard()
    assert guard.step(1e-3, healthy) == pytest.approx(1e-3)
    assert guard.step(1e-3, dead) == pytest.approx(1e-3)
    # A larger request is refused while the gate is dead.
    assert guard.step(3e-3, dead) == pytest.approx(1e-3)
    assert guard.interventions == 1
    # A healthy gate may raise again.
    assert guard.step(3e-3, healthy) == pytest.approx(3e-3)
    assert guard.describe()["last_health"]["ste_gain"] > 0


def test_sparsity_guard_patience_and_validation():
    dead = gate_health(torch.full((1, 8), -12.0), [0.5], 0.2)
    guard = SparsityGuard(patience=2)
    # The cap engages once the degenerate streak exceeds ``patience``.
    assert guard.step(1e-3, dead) == pytest.approx(1e-3)   # streak 1
    assert guard.step(2e-3, dead) == pytest.approx(2e-3)   # streak 2
    assert guard.step(3e-3, dead) == pytest.approx(2e-3)   # streak 3 > 2 -> held
    assert guard.step(4e-3, dead) == pytest.approx(2e-3)   # still held
    with pytest.raises(ValueError):
        SparsityGuard(min_responsive=2.0)
    with pytest.raises(ValueError):
        SparsityGuard(patience=-1)
    with pytest.raises(ValueError):
        SparsityGuard().step(-1.0, dead)


def test_gather_ste_leaves_dropped_candidates_without_gradient():
    """Document the credit-assignment gap of the literal ``V[mask]`` gather.

    Plan section 11 writes ``V_sparse = V[mask]``.  Implemented literally with an
    integer gather, a dropped token's row is removed from the tensor, so the loss
    has *no* path back to its score.  Only the currently kept candidates are
    trained, which means a wrongly dropped informative token can never re-enter
    the selection.  This is asserted rather than assumed.
    """
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    kept_grad, dropped_grad = _dropped_video_gradient(module, blocks)
    assert kept_grad > 0
    assert dropped_grad == 0.0


def test_dense_gated_training_supervises_every_candidate():
    """``physical_shortening=False`` restores the dropped-token gradient.

    This is the mitigation for the gap above: keep the sequence dense and mask
    the dropped tokens to zero, so every candidate receives a gradient while the
    mask still drives the forward values.  It is a training relaxation (masked
    keys remain in the softmax), so it is paired with a physical-shortening
    fine-tune rather than used at inference.
    """
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    _mixed_threshold(module, blocks)
    out = run_tiny_forward(module, blocks, recovery=False, physical_shortening=False)
    layout = module.layout
    assert out.physical_shortening is False
    assert out.backend_sequence_length == layout.total_tokens
    kept = out.kept_video_indices[0]
    mask = torch.zeros(layout.video_tokens, dtype=torch.bool)
    mask[kept] = True
    loss = out.sparse_traj_hidden.pow(2).mean()
    grad = torch.autograd.grad(loss, out.logits, retain_graph=False)[0][0]
    video_grad = grad[: layout.video_tokens].abs()
    assert float(video_grad[~mask].mean()) > 0.0
    assert float(video_grad[mask].mean()) > 0.0


def test_default_forward_is_physically_shortened():
    module = tiny_route_a()
    blocks = nn.ModuleList([ToyBlock(module.config.token_dim) for _ in range(4)])
    out = run_tiny_forward(module, blocks, recovery=False)
    assert out.physical_shortening is True
    assert out.backend_sequence_length == out.kept_total


def test_default_loss_weights_match_the_plan():
    weights = RouteALossWeights()
    assert weights.traj_fm == 1.0
    assert weights.video_fm == 1.0
    assert weights.traj_kd == 2.0
    assert weights.video_kd == 0.5
    assert weights.action_hidden_kd == 1.0
    assert weights.video_hidden_kd == 0.5

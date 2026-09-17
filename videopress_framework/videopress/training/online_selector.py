"""Lightweight online teacher selector used by DriveVA training.

This implementation lives in the standalone VideoPress package so training,
offline analysis, and inference use one canonical selector implementation.  It
deliberately contains no checkpoint or offline-teacher logic.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


class DynamicTokenSelector(nn.Module):
    """Token/position/condition MLP selector returning ``[B, N]`` logits."""

    def __init__(self, token_dim: int = 3072, ego_dim: int = 2, command_dim: int = 3,
                 hidden_dim: int = 256, position_dim: int = 64,
                 feature_mode: str = "all"):
        super().__init__()
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in {"all", "condition_position_time"}:
            raise ValueError(f"unsupported selector feature mode: {self.feature_mode}")
        self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU())
        self.position_mlp = nn.Sequential(nn.Linear(3, position_dim), nn.GELU(), nn.Linear(position_dim, hidden_dim))
        self.condition_mlp = nn.Sequential(nn.Linear(ego_dim + command_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.scoring = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        # A signed contribution is scene-relative: the same local feature can
        # help in one scene and distract in another.  This residual head gives
        # every token access to a pooled scene representation while preserving
        # exact compatibility with older checkpoints through zero init.
        self.context_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.context_scoring = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.context_scoring[-1].weight)
        nn.init.zeros_(self.context_scoring[-1].bias)
        # Official DriveVA inference uses only three diffusion stages.  Expose
        # the stage explicitly instead of asking the selector to infer it from
        # noisy hidden states.  A zero-initialised residual keeps older
        # checkpoints bit-compatible when no timestep weights are present.
        self.timestep_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.timestep_scoring = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.timestep_scoring[-1].weight)
        nn.init.zeros_(self.timestep_scoring[-1].bias)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor | None = None,
                ego_state: torch.Tensor | None = None, command: torch.Tensor | None = None,
                timestep: torch.Tensor | float | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
        b, n, _ = tokens.shape
        device = tokens.device
        param_dtype = next(self.parameters()).dtype
        model_tokens = tokens.to(dtype=param_dtype)
        positions = torch.zeros(b, n, 3, device=device, dtype=param_dtype) if positions is None else positions
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(b, -1, -1)
        ego_state = torch.zeros(b, 2, device=device, dtype=param_dtype) if ego_state is None else ego_state
        command = torch.zeros(b, 3, device=device, dtype=param_dtype) if command is None else command
        condition = self.condition_mlp(torch.cat([ego_state.to(param_dtype), command.to(param_dtype)], dim=-1)).unsqueeze(1).expand(-1, n, -1)
        local = self.token_proj(model_tokens)
        if self.feature_mode == "condition_position_time":
            local = torch.zeros_like(local)
        pos = self.position_mlp(positions.to(dtype=param_dtype))
        interaction = local * condition
        # Keep logits in the selector parameter dtype (fp32 by default).  Casting
        # them back to the bf16 backbone dtype makes BCE unnecessarily noisy.
        base_logits = self.scoring(
            torch.cat([local, pos, condition, interaction], dim=-1)
        )
        scene = self.context_mlp(local.mean(dim=1)).unsqueeze(1).expand(-1, n, -1)
        context_logits = self.context_scoring(
            torch.cat([local, pos, condition, scene, local * scene], dim=-1)
        )
        timestep = torch.zeros(b, device=device, dtype=param_dtype) if timestep is None else torch.as_tensor(
            timestep, device=device, dtype=param_dtype
        ).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(b)
        if timestep.numel() != b:
            raise ValueError(f"timestep must be scalar or length {b}, got {timestep.numel()}")
        phase = timestep / 1000.0
        time_features = torch.stack(
            [phase, torch.sin(math.pi * phase), torch.cos(math.pi * phase),
             torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)],
            dim=-1,
        )
        time = self.timestep_mlp(time_features).unsqueeze(1).expand(-1, n, -1)
        timestep_logits = self.timestep_scoring(
            torch.cat([local, pos, condition, scene, time, local * time], dim=-1)
        )
        return (base_logits + context_logits + timestep_logits).squeeze(-1)


def gradient_input_scores(
    planning_loss: torch.Tensor,
    history_tokens: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute detached Gradient x Input importance without creating a graph."""
    if not torch.is_tensor(planning_loss) or not planning_loss.requires_grad:
        raise ValueError("planning_loss must be a differentiable scalar")
    if not history_tokens.requires_grad:
        raise ValueError("history_tokens must require gradients")
    grad = torch.autograd.grad(
        planning_loss,
        history_tokens,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    return (history_tokens.detach() * grad.detach()).abs().sum(dim=-1)


def signed_removal_scores(
    planning_loss: torch.Tensor,
    history_tokens: torch.Tensor,
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    """First-order loss change caused by removing each token.

    For a removal perturbation ``dx=-x``, ``dL ~= -<grad, x>``.  Positive
    values therefore mean that removing the token is predicted to hurt the
    planning objective (keep it), while negative values identify candidates
    whose removal may improve the objective.
    """
    if not torch.is_tensor(planning_loss) or not planning_loss.requires_grad:
        raise ValueError("planning_loss must be a differentiable scalar")
    if not history_tokens.requires_grad:
        raise ValueError("history_tokens must require gradients")
    grad = torch.autograd.grad(
        planning_loss,
        history_tokens,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )[0]
    return -(history_tokens.detach() * grad.detach()).sum(dim=-1)


def signed_soft_keep_labels(
    signed_scores: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map signed removal deltas to robust, per-sample soft keep labels."""
    if signed_scores.ndim != 2:
        raise ValueError("signed_scores must have shape [B,N]")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    centered = signed_scores.detach()
    scale = centered.abs().mean(dim=1, keepdim=True).clamp_min(float(eps))
    return torch.sigmoid(centered / (scale * float(temperature))).float()


def spatial_counterfactual_probe(
    positions: torch.Tensor,
    group_index: int,
    *,
    tile_h: int = 3,
    tile_w: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a keep mask and membership mask for one deterministic tile."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,N,3]")
    if int(tile_h) <= 0 or int(tile_w) <= 0:
        raise ValueError("tile_h and tile_w must be positive")
    groups = int(tile_h) * int(tile_w)
    index = int(group_index)
    if not 0 <= index < groups:
        raise ValueError(f"group_index must be in [0, {groups - 1}]")
    y = torch.floor(positions[..., 1].float() * int(tile_h)).long().clamp(0, int(tile_h) - 1)
    x = torch.floor(positions[..., 2].float() * int(tile_w)).long().clamp(0, int(tile_w) - 1)
    membership = y * int(tile_w) + x == index
    if not membership.any(dim=1).all():
        raise ValueError(f"counterfactual tile {index} contains no tokens")
    return (~membership).to(dtype=positions.dtype), membership


def counterfactual_group_bce(
    logits: torch.Tensor,
    membership: torch.Tensor,
    baseline_loss: torch.Tensor,
    masked_loss: torch.Tensor,
    *,
    relative_scale: float = 0.05,
    abstain_eps: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Supervise one tile using the measured signed removal loss change.

    ``abstain_eps`` implements the review's dead-zone proposal (P1).  When it is
    positive, probes whose relative effect is inside ``[-abstain_eps,
    +abstain_eps]`` contribute zero weight instead of becoming a hard 0/1
    label, because at that magnitude the sign is measurement noise.  The default
    ``0.0`` reproduces the previous weighted-mean behaviour exactly.
    """
    if logits.shape != membership.shape:
        raise ValueError("logits and membership must have identical [B,N] shape")
    if not math.isfinite(float(relative_scale)) or float(relative_scale) <= 0:
        raise ValueError("relative_scale must be finite and positive")
    if not math.isfinite(float(abstain_eps)) or float(abstain_eps) < 0:
        raise ValueError("abstain_eps must be finite and non-negative")
    group_logits = torch.stack(
        [logits[row][membership[row]].mean() for row in range(logits.shape[0])]
    )
    baseline = baseline_loss.detach().float().reshape(-1)
    masked = masked_loss.detach().float().reshape(-1)
    if baseline.numel() == 1 and group_logits.numel() > 1:
        baseline = baseline.expand_as(group_logits)
        masked = masked.expand_as(group_logits)
    relative_delta = (masked - baseline) / baseline.abs().clamp_min(1e-6)
    target = (relative_delta >= 0).to(dtype=group_logits.dtype)
    confidence = (relative_delta.abs() / float(relative_scale)).clamp(max=1.0)
    # Tiny measured changes are uncertain.  Keep a small floor so neutral
    # probes still teach the scorer to abstain around a 0.5 probability.
    weight = confidence.clamp_min(0.05)
    if float(abstain_eps) > 0.0:
        abstained = relative_delta.abs() <= float(abstain_eps)
        weight = torch.where(abstained, torch.zeros_like(weight), weight)
        per_row = torch.nn.functional.binary_cross_entropy_with_logits(
            group_logits, target, reduction="none"
        )
        loss = (per_row * weight).sum() / weight.sum().clamp_min(1e-6)
        supervised = ~abstained
        unweighted_bce = (
            per_row[supervised].mean()
            if bool(supervised.any())
            else per_row.new_zeros(())
        )
    else:
        # Kept verbatim so abstain_eps=0 stays bit-compatible with the runs that
        # produced the 2026-09-11 pilot numbers.  Note that PyTorch divides a
        # weighted BCE by the number of elements, not by sum(weight), so this
        # quantity is bounded by the mean confidence and must not be read as a
        # classification metric (review defect D).
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            group_logits, target, weight=weight, reduction="mean"
        )
        unweighted_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            group_logits, target, reduction="mean"
        )
        abstained = torch.zeros_like(weight, dtype=torch.bool)
    return loss, {
        "counterfactual_relative_delta": float(relative_delta.mean()),
        "counterfactual_helpful_target": float(target.mean()),
        "counterfactual_confidence": float(confidence.mean()),
        "counterfactual_group_logit": float(group_logits.detach().mean()),
        "counterfactual_unweighted_bce": float(unweighted_bce.detach()),
        "counterfactual_abstain_ratio": float(abstained.float().mean()),
        "counterfactual_group_logit_std": float(
            group_logits.detach().std(unbiased=False)
        )
        if group_logits.numel() > 1
        else 0.0,
    }


def displacement_token_bce(
    logits: torch.Tensor,
    membership: torch.Tensor,
    relative_displacements: torch.Tensor,
    *,
    disp_scale: float = 0.01,
    normalize: str = "scene",
    sample_weight: Optional[torch.Tensor] = None,
    min_spread: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Per-token BCE on the measured plan displacement caused by removal.

    The signed loss-delta teacher asks "does removing this help or hurt", and the
    2026-09-11 verdict run showed that sign is not learnable from the frozen
    hidden states at any granularity or noise-averaging depth. Earlier apparent
    displacement learnability was later traced to perturbation dose/removal-count
    confounding; this remains an experimental legacy objective, not evidence that
    same-scale token importance is learnable.

    This teacher therefore supervises every token in a removed tile with the
    displacement that removing that tile actually caused, scaled into (0, 1) by
    ``disp_scale`` so the selector logit keeps the same "probability of being
    worth keeping" semantics as the existing threshold rule.

    ``min_spread`` guards the scene-relative normalisation against amplifying
    numerical noise.  Min-max maps the smallest measured harm to 0 and the
    largest to 1 no matter how small that range is, so a step whose whole tile
    spread is 1e-5 still produces full-confidence 0/1 labels.  When the observed
    spread is below ``min_spread`` the step carries no usable ordering
    information and is abstained from instead of being stretched to [0, 1].

    Supervision is per token, not per tile mean: pooling to the tile mean is
    review defect 3.4 and destroys exactly the within-tile discrimination that
    token selection needs.
    """
    if logits.shape != membership.shape:
        raise ValueError("logits and membership must have identical [B,N] shape")
    if not math.isfinite(float(disp_scale)) or float(disp_scale) <= 0:
        raise ValueError("disp_scale must be finite and positive")
    if not math.isfinite(float(min_spread)) or float(min_spread) < 0:
        raise ValueError("min_spread must be finite and non-negative")
    displacement = relative_displacements.detach().float().reshape(-1)
    if displacement.numel() == 1 and logits.shape[0] > 1:
        displacement = displacement.expand(logits.shape[0])
    if displacement.numel() != logits.shape[0]:
        raise ValueError(
            f"expected one displacement per row, got {displacement.numel()} for {logits.shape[0]} rows"
        )
    if normalize not in {"scene", "absolute"}:
        raise ValueError(f"unsupported displacement normalisation: {normalize}")
    spread = 0.0
    abstained = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    if normalize == "scene" and displacement.numel() > 1:
        # Selection only ever compares tiles *inside one scene*, so the target is
        # the tile's relative displacement rank, not its absolute magnitude.
        # The absolute scale is dominated by scene/timestep and is both hard to
        # predict and irrelevant to which tiles to keep; a measured run that
        # regressed the absolute value sat 0.039 ABOVE the constant-predictor
        # floor (0.684 vs 0.644), i.e. it had learned nothing about ordering.
        low = displacement.min()
        high = displacement.max()
        spread = float((high - low).clamp_min(0.0).item())
        if spread < float(min_spread):
            # No measurable ordering signal: supervise nothing this step rather
            # than turning floating-point dust into a confident 0/1 label.
            target = torch.zeros_like(displacement)
            abstained = torch.ones_like(abstained)
        else:
            target = ((displacement - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    else:
        if normalize == "scene":
            spread = float(displacement.max().item()) if displacement.numel() else 0.0
        target = (displacement / float(disp_scale)).clamp(0.0, 1.0)
    target = target.to(dtype=logits.dtype)
    selected_logits = logits[membership]
    selected_target = torch.repeat_interleave(
        target, membership.sum(dim=1), dim=0
    ).to(dtype=logits.dtype)
    if selected_logits.numel() == 0:
        raise ValueError("membership selected no tokens")
    per_token = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits, selected_target, reduction="none"
    )
    unweighted_bce = per_token.mean()
    token_weight = torch.ones_like(per_token)
    if sample_weight is not None:
        weight = sample_weight.detach().float().reshape(-1)
        if weight.numel() == 1 and logits.shape[0] > 1:
            weight = weight.expand(logits.shape[0])
        token_weight = token_weight * torch.repeat_interleave(
            weight.to(dtype=per_token.dtype), membership.sum(dim=1), dim=0
        )
    if bool(abstained.any()):
        row_weight = (~abstained).to(dtype=per_token.dtype)
        token_weight = token_weight * torch.repeat_interleave(
            row_weight, membership.sum(dim=1), dim=0
        )
    if sample_weight is not None or bool(abstained.any()):
        loss = (per_token * token_weight).sum() / token_weight.sum().clamp_min(1e-6)
        # Keep the term connected to the graph even when every row abstained so
        # DDP never sees an unused-parameter mismatch.
        loss = loss + selected_logits.sum() * 0.0
    else:
        loss = unweighted_bce
    return loss, {
        "counterfactual_displacement_target_mean": float(target.mean()),
        "counterfactual_displacement_target_max": float(target.max()),
        "counterfactual_displacement_target_std": float(
            target.std(unbiased=False)
        ),
        "counterfactual_displacement_scale": float(disp_scale),
        "counterfactual_displacement_normalize": normalize,
        "counterfactual_displacement_spread": float(spread),
        "counterfactual_displacement_abstain_ratio": float(abstained.float().mean()),
        "counterfactual_unweighted_bce": float(unweighted_bce.detach()),
        "counterfactual_supervised_tokens": float(selected_logits.numel()),
    }



def parse_keep_schedule(spec: str | None) -> List[Tuple[int, float]]:
    """Parse ``ratio@step,ratio@step`` (milestones sorted by step)."""
    if not spec:
        return []
    out = []
    for item in str(spec).split(","):
        ratio, step = item.strip().split("@", 1)
        ratio, step = float(ratio), int(step)
        if not 0.0 < ratio <= 1.0 or step < 0:
            raise ValueError(f"invalid keep schedule item: {item}")
        out.append((step, ratio))
    return sorted(out)


def parse_horizon_weights(spec: str | None) -> List[Tuple[float, float]]:
    """Parse ``seconds:weight`` pairs and normalize positive weights."""
    if not spec:
        return []
    out = []
    for item in str(spec).split(","):
        seconds, weight = item.strip().split(":", 1)
        seconds, weight = float(seconds), float(weight)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"invalid horizon seconds: {item}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"invalid horizon weight: {item}")
        out.append((seconds, weight))
    if not out or sum(weight for _, weight in out) <= 0:
        raise ValueError("horizon weights must have positive total weight")
    if len({seconds for seconds, _ in out}) != len(out):
        raise ValueError("horizon seconds must be unique")
    total = sum(weight for _, weight in out)
    return [(seconds, weight / total) for seconds, weight in sorted(out)]


def horizon_weighted_trajectory_displacement(
    baseline: torch.Tensor,
    altered: torch.Tensor,
    horizons: Sequence[Tuple[float, float]],
    *,
    target_fps: float,
) -> tuple[torch.Tensor, List[int]]:
    """Per-row weighted plan displacement at pre-registered time horizons."""
    if baseline.shape != altered.shape or baseline.ndim != 3:
        raise ValueError("trajectory tensors must have identical [B,T,D] shape")
    if not math.isfinite(float(target_fps)) or float(target_fps) <= 0:
        raise ValueError("target_fps must be finite and positive")
    if not horizons:
        raise ValueError("at least one horizon is required")
    indices = [int(round(seconds * float(target_fps))) - 1 for seconds, _ in horizons]
    if min(indices) < 0 or max(indices) >= baseline.shape[1]:
        raise ValueError(
            f"horizons map to indices {indices}, outside trajectory length {baseline.shape[1]}"
        )
    weights = baseline.new_tensor([weight for _, weight in horizons])
    weights = weights / weights.sum().clamp_min(1e-12)
    displacement = (baseline[:, indices] - altered[:, indices]).norm(dim=-1)
    return (displacement * weights.unsqueeze(0)).sum(dim=1), indices


def keep_ratio_at_step(step: int, warmup_steps: int, schedule: Sequence[Tuple[int, float]] = ()) -> float:
    if int(step) < int(warmup_steps):
        return 1.0
    ratio = 1.0
    for milestone, value in schedule:
        if int(step) >= int(milestone):
            ratio = float(value)
    return ratio


def attention_to_scores(attention: torch.Tensor, history_tokens: int | None = None) -> torch.Tensor:
    """Aggregate action-to-history attention to one score per history token."""
    if attention.ndim == 4:  # [B, heads, action_queries, history]
        scores = attention.mean(dim=(1, 2))
    elif attention.ndim == 3:
        scores = attention.mean(dim=1)
    elif attention.ndim == 2:
        scores = attention
    else:
        raise ValueError(f"unsupported attention shape: {tuple(attention.shape)}")
    if history_tokens is not None and scores.shape[-1] != int(history_tokens):
        raise ValueError(f"attention history dimension {scores.shape[-1]} != {history_tokens}")
    return scores


def online_topk_labels(teacher_scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if teacher_scores.ndim != 2:
        raise ValueError("teacher_scores must have shape [B,N]")
    k = max(1, min(teacher_scores.shape[1], int(round(teacher_scores.shape[1] * float(keep_ratio)))))
    labels = torch.zeros_like(teacher_scores, dtype=torch.float32)
    labels.scatter_(1, teacher_scores.detach().topk(k, dim=1).indices, 1.0)
    return labels


def critical_box_token_mask(
    positions: torch.Tensor,
    boxes_yxyx: torch.Tensor,
    *,
    box_valid: Optional[torch.Tensor] = None,
    dilation: float = 0.0,
) -> torch.Tensor:
    """Project normalized image-space boxes to a candidate-token mask.

    ``positions`` uses the selector's ``(t, y, x)`` convention and boxes use
    ``(y_min, x_min, y_max, x_max)`` in ``[0, 1]``.  Projection of tracked
    objects is deliberately kept outside the selector: NAVSIM, Bench2Drive and
    other datasets can supply their own calibrated camera projection while the
    safety invariant below remains dataset-independent.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,N,3]")
    if boxes_yxyx.ndim == 2:
        boxes_yxyx = boxes_yxyx.unsqueeze(0)
    if boxes_yxyx.ndim != 3 or boxes_yxyx.shape[-1] != 4:
        raise ValueError("boxes_yxyx must have shape [B,M,4] or [M,4]")
    if boxes_yxyx.shape[0] == 1 and positions.shape[0] > 1:
        boxes_yxyx = boxes_yxyx.expand(positions.shape[0], -1, -1)
    if boxes_yxyx.shape[0] != positions.shape[0]:
        raise ValueError("boxes and positions must have the same batch size")
    if not math.isfinite(float(dilation)) or float(dilation) < 0:
        raise ValueError("dilation must be finite and non-negative")
    boxes = boxes_yxyx.to(device=positions.device, dtype=positions.dtype)
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes_yxyx contains non-finite values")
    ymin, xmin, ymax, xmax = boxes.unbind(dim=-1)
    if bool(((ymin > ymax) | (xmin > xmax)).any()):
        raise ValueError("box minima must not exceed maxima")
    if box_valid is None:
        valid = torch.ones_like(ymin, dtype=torch.bool)
    else:
        valid = torch.as_tensor(box_valid, device=positions.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape[0] == 1 and positions.shape[0] > 1:
            valid = valid.expand(positions.shape[0], -1)
        if valid.shape != ymin.shape:
            raise ValueError("box_valid must have shape [B,M] matching boxes")
    pad = float(dilation)
    y = positions[..., 1].unsqueeze(-1)
    x = positions[..., 2].unsqueeze(-1)
    covered = (
        (y >= (ymin - pad).unsqueeze(1))
        & (y <= (ymax + pad).unsqueeze(1))
        & (x >= (xmin - pad).unsqueeze(1))
        & (x <= (xmax + pad).unsqueeze(1))
        & valid.unsqueeze(1)
    )
    return covered.any(dim=-1)


def protected_topk_labels(
    scores: torch.Tensor,
    keep_ratio: float,
    protected_mask: torch.Tensor,
) -> torch.Tensor:
    """Top-k mask that reserves budget for every protected candidate.

    The total K is unchanged: protected tokens replace the lowest-ranked
    unprotected selections.  Failing when protection exceeds K is intentional;
    silently dropping a protected object would violate the safety contract.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [B,N]")
    protected = torch.as_tensor(protected_mask, device=scores.device, dtype=torch.bool)
    if protected.ndim == 1:
        protected = protected.unsqueeze(0).expand(scores.shape[0], -1)
    if protected.shape != scores.shape:
        raise ValueError("protected_mask must have shape [B,N] or [N]")
    k = max(1, min(scores.shape[1], int(round(scores.shape[1] * float(keep_ratio)))))
    counts = protected.sum(dim=1)
    if bool((counts > k).any()):
        raise ValueError(
            f"protected token count exceeds fixed budget K={k}: {counts.tolist()}"
        )
    ranked = scores.detach().masked_fill(protected, float("-inf")).argsort(
        dim=1, descending=True, stable=True
    )
    labels = protected.clone()
    for row in range(scores.shape[0]):
        needed = k - int(counts[row])
        if needed:
            labels[row, ranked[row, :needed]] = True
    return labels.to(dtype=torch.float32)


def selector_pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.0,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Ranking-aligned loss over teacher-positive/negative token pairs.

    BCE calibrates individual probabilities but the deployed selector consumes
    their ordering.  This auxiliary loss directly penalizes a positive token
    scoring below a negative one.  The hardest deterministic pairs are used to
    bound memory; a zero caller weight preserves the historical objective.
    """
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must have identical [B,N] shape")
    if not math.isfinite(float(margin)) or float(margin) < 0:
        raise ValueError("margin must be finite and non-negative")
    if int(max_pairs) <= 0:
        raise ValueError("max_pairs must be positive")
    losses = []
    binary = targets.detach() >= 0.5
    for row in range(logits.shape[0]):
        positive = logits[row][binary[row]]
        negative = logits[row][~binary[row]]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        # Low positives and high negatives are the ranking boundary that can
        # change top-k membership.  Use a square block within max_pairs.
        side = max(1, int(math.sqrt(int(max_pairs))))
        positive = positive.topk(min(side, positive.numel()), largest=False).values
        negative = negative.topk(min(side, negative.numel()), largest=True).values
        pair_delta = positive.unsqueeze(1) - negative.unsqueeze(0)
        losses.append(torch.nn.functional.softplus(float(margin) - pair_delta).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def hard_topk_mask(
    logits: torch.Tensor,
    keep_ratio: float,
    protected_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    labels = (
        online_topk_labels(logits.detach(), keep_ratio)
        if protected_mask is None
        else protected_topk_labels(logits.detach(), keep_ratio, protected_mask)
    )
    return labels.to(dtype=logits.dtype)


def selector_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    pred = logits.detach().topk(max(1, int(labels.sum(dim=1).round().mean().item())), dim=1).indices
    teacher = labels.detach().bool()
    overlap = torch.stack([(teacher[i, pred[i]].float().mean() if pred.shape[1] else teacher.new_tensor(0.0)) for i in range(labels.shape[0])]).mean()
    pairwise = []
    ndcg = []
    discounts = 1.0 / torch.log2(
        torch.arange(pred.shape[1], device=logits.device, dtype=torch.float32) + 2.0
    )
    for row in range(labels.shape[0]):
        positive = logits.detach()[row][teacher[row]]
        negative = logits.detach()[row][~teacher[row]]
        if positive.numel() and negative.numel():
            comparisons = positive.unsqueeze(1) - negative.unsqueeze(0)
            pairwise.append(
                ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean()
            )
        gains = teacher[row, pred[row]].float()
        ideal_count = min(int(teacher[row].sum()), pred.shape[1])
        ideal = discounts[:ideal_count].sum().clamp_min(1e-12)
        ndcg.append((gains * discounts).sum() / ideal)
    return {"selector_topk_overlap": float(overlap),
            "selector_pairwise_accuracy": float(torch.stack(pairwise).mean()) if pairwise else 0.0,
            "selector_ndcg_at_k": float(torch.stack(ndcg).mean()) if ndcg else 0.0,
            "selector_mean_positive_logit": float(logits.detach()[teacher].mean()) if teacher.any() else 0.0,
            "selector_mean_negative_logit": float(logits.detach()[~teacher].mean()) if (~teacher).any() else 0.0}

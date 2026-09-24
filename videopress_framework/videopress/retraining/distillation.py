"""Route A distillation and flow-matching losses (plan sections 13, 31, 32).

The teacher is always the *frozen original DriveVA / NoPress* model, never the
deployed Press checkpoint: the goal is to preserve the original planning
function, not to imitate a compressed model.

Distilled quantities
--------------------

``traj_flow`` / ``video_flow``
    The final flow-matching predictions of the teacher for the same
    ``(z_sigma, a_sigma, sigma)``, which carry both the world-model and the
    trajectory distribution.
``action_hidden @ L11 / L18 / L29``
    The three probe layers chosen from the 2026-09-22 DiT probes: L11 is where
    the trajectory planning semantics form, L18 is where the future video latent
    becomes linearly decodable, L29 is the last block.  Matching all three
    anchors the whole depth rather than only the output.  Following the plan,
    both sides are LayerNorm-ed before the L2 comparison so the loss measures
    direction, not per-token magnitude drift.

Every term is optional; the builder fails loudly when nothing was supplied so a
misconfigured A0/A1 stage cannot silently train on zero loss.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F

from .threshold_gate import sparsity_loss


@dataclass
class RouteALossWeights:
    """Default weights from plan section 32."""

    traj_fm: float = 1.0
    video_fm: float = 1.0
    traj_kd: float = 2.0
    video_kd: float = 0.5
    action_hidden_kd: float = 1.0
    video_hidden_kd: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} weight must be finite and non-negative")

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass
class RouteALossOutput:
    total: torch.Tensor
    terms: Dict[str, torch.Tensor] = field(default_factory=dict)

    def detached(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.terms.items()}


def layer_norm_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """LayerNorm-matched L2 used for hidden-state distillation."""
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    left = student.float()
    right = teacher.detach().float()
    if normalize:
        left = F.layer_norm(left, (left.shape[-1],), eps=float(eps))
        right = F.layer_norm(right, (right.shape[-1],), eps=float(eps))
    return F.mse_loss(left, right)


def weighted_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE with an optional per-element or per-sample weight."""
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    target = teacher.detach().float()
    diff = (student.float() - target) ** 2
    if weight is None:
        return diff.mean()
    weight = weight.to(device=diff.device, dtype=diff.dtype)
    if weight.ndim == 1 and diff.ndim > 1:
        weight = weight.reshape(-1, *([1] * (diff.ndim - 1)))
    if weight.shape[-1] == 1 and diff.shape[-1] != 1:
        weight = weight.expand_as(diff)
    return (diff * weight).sum() / weight.sum().clamp_min(1e-6)


def action_hidden_kd(
    student_hidden: Mapping[int, torch.Tensor],
    teacher_hidden: Mapping[int, torch.Tensor],
    *,
    layers: Optional[Mapping[int, float]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sum over the probe layers of ``LN(H_s) - LN(H_t)`` L2 (plan section 31)."""
    if not student_hidden:
        raise ValueError("student_hidden is empty")
    weights = dict(layers) if layers else {int(k): 1.0 for k in student_hidden}
    total = None
    for layer, weight in sorted(weights.items()):
        if layer not in student_hidden:
            raise KeyError(f"student hidden for layer {layer} is missing")
        if layer not in teacher_hidden:
            raise KeyError(f"teacher hidden for layer {layer} is missing")
        term = layer_norm_mse(student_hidden[layer], teacher_hidden[layer], eps=eps)
        total = term * float(weight) if total is None else total + term * float(weight)
    assert total is not None
    return total


def compute_route_a_loss(
    *,
    weights: Optional[RouteALossWeights] = None,
    student_traj_flow: Optional[torch.Tensor] = None,
    teacher_traj_flow: Optional[torch.Tensor] = None,
    traj_fm_target: Optional[torch.Tensor] = None,
    student_video_flow: Optional[torch.Tensor] = None,
    teacher_video_flow: Optional[torch.Tensor] = None,
    video_fm_target: Optional[torch.Tensor] = None,
    video_fm_weight: Optional[torch.Tensor] = None,
    student_action_hidden: Optional[Mapping[int, torch.Tensor]] = None,
    teacher_action_hidden: Optional[Mapping[int, torch.Tensor]] = None,
    action_hidden_layers: Optional[Mapping[int, float]] = None,
    student_video_hidden: Optional[torch.Tensor] = None,
    teacher_video_hidden: Optional[torch.Tensor] = None,
    sparse_scores: Optional[torch.Tensor] = None,
    lambda_sparse: float = 0.0,
    reference: Optional[torch.Tensor] = None,
) -> RouteALossOutput:
    """Assemble ``L_A`` from plan section 32.

    ``sparse_scores`` must be the per-candidate keep *probabilities*
    ``sigma(logit)``, because plan section 5 defines ``L_sparse = mean(s_i)``.

    ``reference`` is any tensor that participates in the graph; it is used to
    keep ``total`` connected when every configured term happens to be disabled,
    which prevents DDP from seeing a parameter that never received a gradient.
    """
    w = weights or RouteALossWeights()
    terms: Dict[str, torch.Tensor] = {}

    def add(name: str, value: Optional[torch.Tensor], weight: float):
        if value is None:
            return
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        terms[name] = value * float(weight)

    # --- ground-truth flow matching -------------------------------------
    if traj_fm_target is not None:
        if student_traj_flow is None:
            raise ValueError("traj_fm_target given but student_traj_flow is missing")
        add("traj_fm", F.mse_loss(student_traj_flow.float(), traj_fm_target.detach().float()), w.traj_fm)
    if video_fm_target is not None:
        if student_video_flow is None:
            raise ValueError("video_fm_target given but student_video_flow is missing")
        add(
            "video_fm",
            weighted_mse(student_video_flow, video_fm_target, video_fm_weight),
            w.video_fm,
        )

    # --- teacher distillation -------------------------------------------
    if teacher_traj_flow is not None:
        if student_traj_flow is None:
            raise ValueError("teacher_traj_flow given but student_traj_flow is missing")
        add("traj_kd", F.mse_loss(student_traj_flow.float(), teacher_traj_flow.detach().float()), w.traj_kd)
    if teacher_video_flow is not None:
        if student_video_flow is None:
            raise ValueError("teacher_video_flow given but student_video_flow is missing")
        add(
            "video_kd",
            weighted_mse(student_video_flow, teacher_video_flow, video_fm_weight),
            w.video_kd,
        )
    if student_action_hidden is not None and teacher_action_hidden is not None:
        add(
            "action_hidden_kd",
            action_hidden_kd(
                student_action_hidden, teacher_action_hidden, layers=action_hidden_layers
            ),
            w.action_hidden_kd,
        )
    elif student_action_hidden is not None or teacher_action_hidden is not None:
        raise ValueError(
            "action hidden distillation needs both student_action_hidden and teacher_action_hidden"
        )
    if student_video_hidden is not None and teacher_video_hidden is not None:
        add("video_hidden_kd", layer_norm_mse(student_video_hidden, teacher_video_hidden), w.video_hidden_kd)

    # --- sparsity ---------------------------------------------------------
    if sparse_scores is not None and float(lambda_sparse) != 0.0:
        terms["sparse"] = sparsity_loss(sparse_scores.float()) * float(lambda_sparse)

    if not terms:
        if reference is None:
            raise ValueError(
                "compute_route_a_loss received no active term; pass at least one target "
                "or a reference tensor"
            )
        zero = reference.sum() * 0.0
        return RouteALossOutput(total=zero, terms={"inactive": zero})

    total = None
    for name in sorted(terms):
        total = terms[name] if total is None else total + terms[name]
    assert total is not None
    if reference is not None:
        total = total + reference.sum() * 0.0
    return RouteALossOutput(total=total, terms=terms)

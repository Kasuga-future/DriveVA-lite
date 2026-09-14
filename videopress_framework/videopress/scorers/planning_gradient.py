"""Original DriveVA trajectory-projection Gradient x Input semantics."""

from __future__ import annotations

import torch

from ..core.registry import register_scorer
from .base import TokenScorer


OBJECTIVE_TYPE = "detached_unit_trajectory_projection_v1"
SCORE_REDUCTION = "l2_embedding_then_batch_mean"


def trajectory_projection_objective(
    trajectory_prediction: torch.Tensor,
    trajectory_prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce DriveVATrajectoryGradientProbe's default scalar objective."""

    if not torch.is_tensor(trajectory_prediction):
        raise TypeError("trajectory_prediction must be a tensor")
    prefix_len = int(trajectory_prefix_len)
    if prefix_len < 0 or prefix_len >= trajectory_prediction.shape[1]:
        raise ValueError("no future trajectory points remain after prefix removal")
    trajectory_points = trajectory_prediction[:, prefix_len:]
    direction = trajectory_points.detach().float()
    direction_norm = torch.linalg.vector_norm(direction).clamp_min(1e-12)
    objective = (trajectory_points.float() * direction).sum() / direction_norm
    if objective.ndim != 0:
        raise RuntimeError("trajectory projection objective must be scalar")
    return objective, trajectory_points


def original_gradient_input_reduction(
    gradient: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply the original embedding-L2 then batch-mean reduction exactly.

    The original teacher emits one shared ranking after reducing the batch.
    Official NAVSIM evaluation has batch size one; retain a leading singleton
    dimension so VideoTokenPress receives its required ``[B,N]`` shape.
    """

    if gradient.shape != inputs.shape or gradient.ndim != 3:
        raise ValueError("gradient and inputs must share shape [B,N,C]")
    reduced = torch.linalg.vector_norm(
        gradient.detach().float() * inputs.detach().float(), dim=-1
    ).mean(dim=0)
    return reduced.unsqueeze(0)


@register_scorer("planning_gradient_input")
class PlanningGradientInputScorer(TokenScorer):
    """Protocol marker for the original DriveVA trajectory-gradient teacher."""

    name = "planning_gradient_input"
    requires_probe = True
    requires_grad = True
    probe_mode = "backward_probe"

    def score(self, ctx) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError("planning_gradient_input requires the official DriveVA gradient probe")

    def describe(self) -> dict:
        return {
            **super().describe(),
            "objective_type": OBJECTIVE_TYPE,
            "score_reduction": SCORE_REDUCTION,
            "teacher_source": "diffsynth/analysis/driveva_trajectory_gradient.py",
        }

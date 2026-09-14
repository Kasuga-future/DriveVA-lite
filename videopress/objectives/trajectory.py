from __future__ import annotations

import torch

from .base import PlanningObjective


def _trajectory(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        return outputs["trajectory"]
    return outputs.trajectory


class TrajectoryObjective(PlanningObjective):
    """Simple differentiable trajectory objective for gradient probes."""

    def __init__(self, target_key: str = "target_trajectory"):
        self.target_key = target_key

    def _target(self, ctx, trajectory):
        target = ctx.metadata.get(self.target_key)
        if target is None:
            raise RuntimeError(
                f"{type(self).__name__} requires ctx.metadata[{self.target_key!r}]"
            )
        target = target.to(device=trajectory.device, dtype=trajectory.dtype)
        if target.shape != trajectory.shape:
            raise RuntimeError(
                f"trajectory shape mismatch: pred={tuple(trajectory.shape)}, "
                f"target={tuple(target.shape)}"
            )
        return target

    def compute(self, outputs, ctx):
        trajectory = _trajectory(outputs)
        target = self._target(ctx, trajectory)
        return (trajectory - target).float().pow(2).mean()


class EndpointObjective(TrajectoryObjective):
    def compute(self, outputs, ctx):
        trajectory = _trajectory(outputs)
        target = self._target(ctx, trajectory)
        return (trajectory[:, -1] - target[:, -1]).float().pow(2).mean()

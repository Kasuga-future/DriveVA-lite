"""Offline gradient attribution scorers.

The scorer only produces a ranking.  Evaluator code can run it on a probe
context and pass the frozen scores to ``ScorerPress.apply_with_scores`` for the
intervention pass.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ..core.registry import register_scorer
from ..utils.validation import ensure_finite
from .base import TokenScorer


def adapt_legacy_forward(fn: Callable) -> Callable:
    """Explicitly adapt an old one-argument callback to the V1 API."""

    def wrapped(value, ctx):
        return fn(value)

    return wrapped


class _GradientScorer(TokenScorer):
    requires_probe = True
    requires_grad = True
    probe_mode = "backward_probe"

    def __init__(self, forward_fn: Optional[Callable] = None, objective: Any = None):
        self.forward_fn = forward_fn
        self.objective = objective

    def _loss(self, outputs: Any, ctx) -> torch.Tensor:
        objective = self.objective if self.objective is not None else ctx.metadata.get("gradient_objective")
        if objective is None:
            raise ValueError("GradientScorer requires an objective callable/object")
        if hasattr(objective, "compute"):
            loss = objective.compute(outputs, ctx)
        else:
            loss = objective(outputs, ctx)
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError("gradient objective must return a scalar tensor")
        return loss

    def score(self, ctx) -> torch.Tensor:
        forward_fn = self.forward_fn or ctx.metadata.get("gradient_forward")
        if forward_fn is None:
            raise ValueError("GradientScorer requires forward_fn or ctx.metadata['gradient_forward']")
        x = ctx.tokens.detach().clone().requires_grad_(True)
        outputs = forward_fn(x, ctx)
        loss = self._loss(outputs, ctx)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False, allow_unused=False)[0]
        candidate = ctx.domain.candidate_indices
        grad_candidate = grad.index_select(1, candidate)
        x_candidate = x.index_select(1, candidate)
        scores = self._reduce(grad_candidate, x_candidate)
        ensure_finite(scores, "gradient scores")
        return scores.float()

    def _reduce(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> dict:
        return {**super().describe(), "objective": type(self.objective).__name__ if self.objective is not None else "context"}


@register_scorer("gradient_norm")
class GradientNormScorer(_GradientScorer):
    name = "gradient_norm"

    def _reduce(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(grad.float(), dim=-1)


@register_scorer("gradient_input")
class GradientInputScorer(_GradientScorer):
    name = "gradient_input"

    def __init__(self, reduction: str = "l2", **kwargs):
        super().__init__(**kwargs)
        self.reduction = str(reduction).lower()
        if self.reduction not in {"l2", "abs_sum"}:
            raise ValueError("gradient_input reduction must be l2 or abs_sum")

    def _reduce(self, grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attribution = grad.float() * x.float()
        if self.reduction == "l2":
            return torch.linalg.vector_norm(attribution, dim=-1)
        return attribution.abs().sum(dim=-1)

    def describe(self) -> dict:
        return {**super().describe(), "reduction": self.reduction}

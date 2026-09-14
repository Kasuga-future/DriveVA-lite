from __future__ import annotations

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("topk")
class TopKSelector(TokenSelector):
    name = "topk"

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        if scores.ndim != 2:
            raise ValueError(f"scores must have shape [B,N_candidate], got {tuple(scores.shape)}")
        if scores.shape[1] != domain.n_candidate:
            raise ValueError("score candidate dimension differs from domain")
        if not 0 <= int(K) <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")
        ensure_finite(scores, "scores")
        # stable=True gives deterministic original-candidate-index tie breaking.
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order: torch.Tensor, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        if not 0 <= int(K) <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")
        keep_local = order[:, :K]
        drop_local = order[:, K:]
        candidate = domain.candidate_indices
        keep_global = candidate[keep_local]
        drop_global = candidate[drop_local]
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=keep_global,
            K=int(K),
            metadata={"selector": self.name, "stable": True},
        )

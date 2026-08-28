from __future__ import annotations

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("threshold")
class ThresholdSelector(TokenSelector):
    name = "threshold"

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def select(self, scores: torch.Tensor, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if scores.ndim != 2 or scores.shape[1] != domain.n_candidate:
            raise ValueError("scores must have shape [B,N_candidate]")
        ensure_finite(scores, "scores")
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order: torch.Tensor, scores: torch.Tensor, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        keep_mask = scores >= self.threshold
        if K is not None:
            # Threshold selection is allowed to return fewer tokens, but an explicit
            # K is treated as a hard cap so the result remains budget-bounded.
            rank_mask = torch.zeros_like(keep_mask)
            rank_mask.scatter_(1, order[:, :K], True)
            keep_mask &= rank_mask
        keep_local_rows, drop_local_rows = [], []
        for row in keep_mask:
            keep_local_rows.append(torch.where(row)[0])
            drop_local_rows.append(torch.where(~row)[0])
        lengths = {int(row.numel()) for row in keep_local_rows}
        if len(lengths) > 1:
            raise ValueError("ThresholdSelector needs equal keep counts across a batch")
        keep_local = torch.stack(keep_local_rows) if keep_local_rows else scores.new_empty((0, 0), dtype=torch.long)
        drop_local = torch.stack(drop_local_rows) if drop_local_rows else scores.new_empty((0, 0), dtype=torch.long)
        candidate = domain.candidate_indices
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=int(keep_local.shape[1]),
            metadata={"selector": self.name, "threshold": self.threshold},
        )

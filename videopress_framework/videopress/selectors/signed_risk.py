"""Risk-gated selection for signed keep-utility predictions."""

from __future__ import annotations

import math

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("signed_risk")
class SignedRiskSelector(TokenSelector):
    """Drop only confidently harmful tokens, otherwise retain the full input."""

    name = "signed_risk"

    def __init__(
        self,
        threshold: float = 0.5,
        min_keep_ratio: float = 0.375,
        min_drop_ratio: float = 0.05,
        abstain_margin: float = 0.03,
    ):
        self.threshold = float(threshold)
        self.min_keep_ratio = float(min_keep_ratio)
        self.min_drop_ratio = float(min_drop_ratio)
        self.abstain_margin = float(abstain_margin)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if not 0.0 < self.min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be within (0, 1]")
        if not 0.0 <= self.min_drop_ratio <= 1.0:
            raise ValueError("min_drop_ratio must be within [0, 1]")
        if not 0.0 <= self.abstain_margin <= 1.0:
            raise ValueError("abstain_margin must be within [0, 1]")

    def select(self, scores: torch.Tensor, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if scores.ndim != 2 or scores.shape[1] != domain.n_candidate:
            raise ValueError("scores must have shape [B,N_candidate]")
        ensure_finite(scores, "scores")
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order, scores, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        ensure_finite(scores, "scores")
        n = int(domain.n_candidate)
        budget_cap = n if K is None else int(K)
        if not 0 <= budget_cap <= n:
            raise ValueError(f"K={budget_cap} outside [0, {n}]")
        min_keep = min(budget_cap, max(1, int(math.ceil(n * self.min_keep_ratio))))
        harmful = scores < self.threshold
        harmful_count = harmful.sum(dim=1)
        confidence_sum = ((self.threshold - scores).clamp_min(0.0) * harmful).sum(dim=1)
        mean_confidence = confidence_sum / harmful_count.clamp_min(1)
        min_drop = int(math.ceil(n * self.min_drop_ratio))
        compress = (harmful_count >= min_drop) & (mean_confidence >= self.abstain_margin)
        proposed = (n - harmful_count).clamp(min=min_keep, max=budget_cap)
        proposed = torch.where(compress, proposed, proposed.new_full((), budget_cap))

        # A batch shares the least aggressive decision so tensors remain
        # rectangular. Official evaluation uses batch size one.
        actual_k = int(proposed.max().item()) if proposed.numel() else budget_cap
        keep_local = order[:, :actual_k]
        drop_local = order[:, actual_k:]
        candidate = domain.candidate_indices.to(order.device)
        diagnostics = {}
        if ctx is not None and isinstance(getattr(ctx, "metadata", None), dict):
            diagnostics = dict(ctx.metadata.get("score_diagnostics", {}))
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata={
                "selector": self.name,
                "dynamic": True,
                "threshold": self.threshold,
                "min_keep_ratio": self.min_keep_ratio,
                "min_drop_ratio": self.min_drop_ratio,
                "abstain_margin": self.abstain_margin,
                "harmful_count": harmful_count.detach().cpu().tolist(),
                "mean_harmful_confidence": mean_confidence.detach().cpu().tolist(),
                "abstained": (~compress).detach().cpu().tolist(),
                "proposed_K_per_batch": proposed.detach().cpu().tolist(),
                "actual_K": actual_k,
                "score_diagnostics": diagnostics,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "min_keep_ratio": self.min_keep_ratio,
            "min_drop_ratio": self.min_drop_ratio,
            "abstain_margin": self.abstain_margin,
        }

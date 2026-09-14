"""Risk-gated dynamic token budget selection from score concentration."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("adaptive_mass")
class AdaptiveMassSelector(TokenSelector):
    """Choose the smallest confident K, otherwise retain the full fallback.

    Confidence requires both sufficient cumulative score mass and a local
    separation at the tier boundary.  ``K`` supplied by the press is a hard
    maximum; configurations intended to permit the full fallback should use a
    1.0 budget.  A batch shares the most conservative proposed K so tensors
    remain rectangular.
    """

    name = "adaptive_mass"

    def __init__(
        self,
        ratios: Sequence[float] = (0.375, 0.5, 1.0),
        mass_thresholds: Sequence[float] = (0.60, 0.68),
        gap_thresholds: Sequence[float] = (0.04, 0.025),
        gap_window: int = 8,
        eps: float = 1e-8,
    ):
        self.ratios = tuple(float(value) for value in ratios)
        self.mass_thresholds = tuple(float(value) for value in mass_thresholds)
        self.gap_thresholds = tuple(float(value) for value in gap_thresholds)
        self.gap_window = int(gap_window)
        self.eps = float(eps)
        if len(self.ratios) < 2 or self.ratios[-1] != 1.0:
            raise ValueError("ratios must contain at least one compressed tier and end in 1.0")
        if any(not 0.0 < value <= 1.0 for value in self.ratios):
            raise ValueError("ratios must be within (0, 1]")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("ratios must be strictly increasing")
        if len(self.mass_thresholds) != len(self.ratios) - 1:
            raise ValueError("mass_thresholds must match compressed tiers")
        if len(self.gap_thresholds) != len(self.ratios) - 1:
            raise ValueError("gap_thresholds must match compressed tiers")
        if any(not 0.0 <= value <= 1.0 for value in self.mass_thresholds):
            raise ValueError("mass_thresholds must be within [0, 1]")
        if any(not 0.0 <= value <= 1.0 for value in self.gap_thresholds):
            raise ValueError("gap_thresholds must be within [0, 1]")
        if self.gap_window < 1:
            raise ValueError("gap_window must be positive")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def _tier_statistics(self, sorted_scores: torch.Tensor, tier_k: int):
        n = sorted_scores.shape[1]
        mass = sorted_scores[:, :tier_k].sum(dim=1) / sorted_scores.sum(dim=1).clamp_min(self.eps)
        if tier_k >= n:
            gap = torch.ones_like(mass)
        else:
            window = min(self.gap_window, tier_k, n - tier_k)
            above = sorted_scores[:, tier_k - window : tier_k].mean(dim=1)
            below = sorted_scores[:, tier_k : tier_k + window].mean(dim=1)
            gap = ((above - below) / above.clamp_min(self.eps)).clamp_min(0.0)
        return mass, gap

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        if scores.ndim != 2 or scores.shape[1] != domain.n_candidate:
            raise ValueError("scores must have shape [B, N_candidate]")
        ensure_finite(scores, "scores")
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order, scores, domain, K: int, ctx=None) -> SelectionResult:
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        n = int(domain.n_candidate)
        K = int(K)
        if not 0 <= K <= n:
            raise ValueError(f"K={K} outside [0, {n}]")
        ensure_finite(scores, "scores")
        if K == 0:
            empty = order[:, :0]
            return SelectionResult(
                keep_candidate_indices=empty,
                drop_candidate_indices=order,
                keep_global_indices=empty,
                K=0,
                metadata={
                    "selector": self.name,
                    "stable": True,
                    "dynamic": True,
                    "proposed_K_per_batch": [0] * scores.shape[0],
                    "actual_K": 0,
                    "fallback_K": 0,
                    "tiers": [],
                },
            )
        sorted_scores = torch.gather(scores.float(), 1, order.long())
        # Score concentration is meaningful for non-negative utility. Shifting
        # preserves ranking while making the selector safe for future signed
        # predictors; the fallback handles an all-zero distribution.
        sorted_scores = (sorted_scores - sorted_scores[:, -1:]).clamp_min(0.0)
        fallback_k = K
        tier_ks = [min(K, max(1, int(round(n * ratio)))) for ratio in self.ratios[:-1]]
        proposed = torch.full(
            (scores.shape[0],), fallback_k, device=scores.device, dtype=torch.long
        )
        unresolved = torch.ones_like(proposed, dtype=torch.bool)
        tier_metadata = []
        for tier_k, mass_threshold, gap_threshold in zip(
            tier_ks, self.mass_thresholds, self.gap_thresholds
        ):
            mass, gap = self._tier_statistics(sorted_scores, tier_k)
            confident = unresolved & (mass >= mass_threshold) & (gap >= gap_threshold)
            proposed = torch.where(confident, proposed.new_full((), tier_k), proposed)
            unresolved &= ~confident
            tier_metadata.append(
                {
                    "K": int(tier_k),
                    "ratio": float(tier_k / max(1, n)),
                    "mass": mass.detach().cpu().tolist(),
                    "gap": gap.detach().cpu().tolist(),
                    "mass_threshold": float(mass_threshold),
                    "gap_threshold": float(gap_threshold),
                }
            )
        actual_k = int(proposed.max().item()) if proposed.numel() else fallback_k
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
                "stable": True,
                "dynamic": True,
                "proposed_K_per_batch": proposed.detach().cpu().tolist(),
                "actual_K": actual_k,
                "fallback_K": fallback_k,
                "tiers": tier_metadata,
                "score_diagnostics": diagnostics,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "ratios": list(self.ratios),
            "mass_thresholds": list(self.mass_thresholds),
            "gap_thresholds": list(self.gap_thresholds),
            "gap_window": self.gap_window,
        }

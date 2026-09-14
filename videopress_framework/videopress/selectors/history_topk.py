"""Top-K selectors with explicit two-history-latent constraints."""

from __future__ import annotations

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("history_topk")
class HistoryTopKSelector(TokenSelector):
    """Select only the last latent or enforce an equal quota per latent."""

    name = "history_topk"

    def __init__(self, mode: str, require_num_history_latents: int = 2):
        mode = str(mode).strip().lower()
        if mode not in {"last_only", "per_latent"}:
            raise ValueError("HistoryTopKSelector mode must be last_only or per_latent")
        self.mode = mode
        self.require_num_history_latents = int(require_num_history_latents)
        if self.require_num_history_latents < 1:
            raise ValueError("require_num_history_latents must be positive")

    def _validate(self, scores, domain, K: int, ctx) -> None:
        if ctx is None:
            raise ValueError("HistoryTopKSelector requires a TokenContext")
        if scores.ndim != 2 or scores.shape != (ctx.batch_size, domain.n_candidate):
            raise ValueError("scores must have shape [B,N_candidate]")
        if ctx.layout.num_cond_latents != self.require_num_history_latents:
            raise ValueError(
                f"history retention policies require exactly {self.require_num_history_latents} history latents, "
                f"got {ctx.layout.num_cond_latents}"
            )
        if domain.name not in {"history", "all_history"}:
            raise ValueError("HistoryTopKSelector requires domain=history/all_history")
        if domain.n_candidate != ctx.layout.history_video.length:
            raise ValueError("history selector domain must contain every history token and no other tokens")
        if not 0 <= int(K) <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")
        ensure_finite(scores, "scores")

    def _quota(self, K: int, ctx) -> list[int]:
        tokens_per_latent = ctx.layout.tokens_per_latent
        if self.mode == "last_only":
            if K > tokens_per_latent:
                raise ValueError(f"last_only K={K} exceeds last-history size {tokens_per_latent}")
            return [0] * (ctx.layout.num_cond_latents - 1) + [int(K)]
        if K % ctx.layout.num_cond_latents:
            raise ValueError(
                f"per_latent K={K} is not divisible by {ctx.layout.num_cond_latents}; "
                "use budget reference=each_history"
            )
        per_latent = K // ctx.layout.num_cond_latents
        if per_latent > tokens_per_latent:
            raise ValueError("per-latent quota exceeds tokens_per_latent")
        return [per_latent] * ctx.layout.num_cond_latents

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        self._validate(scores, domain, K, ctx)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(
        self, order: torch.Tensor, scores: torch.Tensor, domain, K: int, ctx=None
    ) -> SelectionResult:
        self._validate(scores, domain, K, ctx)
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        candidate = domain.candidate_indices.to(order.device)
        quotas = self._quota(int(K), ctx)
        keep_rows = []
        for batch_order in order.long():
            ranked_global = candidate.index_select(0, batch_order)
            pieces = []
            for latent_index, quota in enumerate(quotas):
                frame = ctx.layout.frame_range(latent_index)
                within = (ranked_global >= frame.start) & (ranked_global < frame.end)
                ranked_local = batch_order[within]
                if ranked_local.numel() != ctx.layout.tokens_per_latent:
                    raise RuntimeError("history candidate/ranking does not cover one complete latent")
                pieces.append(ranked_local[:quota])
            keep_rows.append(torch.cat(pieces))
        keep_local = torch.stack(keep_rows)
        if keep_local.shape[1] != int(K):
            raise RuntimeError("history selector produced a K inconsistent with the budget")
        keep_candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
        if keep_local.numel():
            keep_candidate_mask.scatter_(1, keep_local, True)
        all_local = torch.arange(domain.n_candidate, device=order.device).expand(scores.shape[0], -1)
        drop_local = all_local[~keep_candidate_mask].reshape(scores.shape[0], domain.n_candidate - int(K))
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=int(K),
            metadata={
                "selector": self.name,
                "mode": self.mode,
                "stable": True,
                "per_history_latent_quota": quotas,
            },
        )

    def describe(self) -> dict:
        return {
            **super().describe(),
            "mode": self.mode,
            "require_num_history_latents": self.require_num_history_latents,
        }

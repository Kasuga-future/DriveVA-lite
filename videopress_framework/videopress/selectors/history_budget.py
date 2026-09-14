"""Independent threshold and quota selectors for complete history latents."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


def _validate_complete_history(scores, domain, ctx, require_num_history_latents: int) -> None:
    if ctx is None:
        raise ValueError("per-latent history selectors require a TokenContext")
    if scores.ndim != 2 or scores.shape != (ctx.batch_size, domain.n_candidate):
        raise ValueError("scores must have shape [B,N_candidate]")
    if ctx.layout.num_cond_latents != int(require_num_history_latents):
        raise ValueError(
            f"expected {require_num_history_latents} history latents, "
            f"got {ctx.layout.num_cond_latents}"
        )
    if domain.name not in {"history", "all_history"}:
        raise ValueError("per-latent history selectors require domain=history/all_history")
    if domain.n_candidate != ctx.layout.history_video.length:
        raise ValueError("history domain must contain every history token and no other tokens")
    ensure_finite(scores, "scores")


def _latent_local_masks(domain, ctx, device) -> list[torch.Tensor]:
    candidate = domain.candidate_indices.to(device)
    masks = []
    for latent_index in range(ctx.layout.num_cond_latents):
        frame = ctx.layout.frame_range(latent_index)
        mask = (candidate >= frame.start) & (candidate < frame.end)
        if int(mask.sum()) != ctx.layout.tokens_per_latent:
            raise RuntimeError("history candidates do not cover one complete latent")
        masks.append(mask)
    return masks


@register_selector("history_threshold")
class HistoryThresholdSelector(TokenSelector):
    """Apply one absolute score threshold per history latent.

    Threshold order follows storage order: oldest to newest. A multi-sample
    batch uses the largest proposed total K and conservatively fills other rows
    by global score so physical tensors remain rectangular. Official DriveVA
    evaluation uses batch size one, where no fill occurs.
    """

    name = "history_threshold"

    def __init__(self, thresholds: Sequence[float], require_num_history_latents: int = 2):
        self.thresholds = tuple(float(value) for value in thresholds)
        self.require_num_history_latents = int(require_num_history_latents)
        if len(self.thresholds) != self.require_num_history_latents:
            raise ValueError("thresholds must contain one value per history latent")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.thresholds):
            raise ValueError("history thresholds must be finite and within [0, 1]")

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        _validate_complete_history(
            scores, domain, ctx, self.require_num_history_latents
        )
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order, scores, domain, K: int, ctx=None) -> SelectionResult:
        _validate_complete_history(
            scores, domain, ctx, self.require_num_history_latents
        )
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        K = int(K)
        if not 0 <= K <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")
        latent_masks = _latent_local_masks(domain, ctx, scores.device)
        local_thresholds = scores.new_empty((domain.n_candidate,))
        for threshold, mask in zip(self.thresholds, latent_masks):
            local_thresholds[mask] = threshold
        keep_mask = scores >= local_thresholds.unsqueeze(0)
        cap_mask = torch.zeros_like(keep_mask)
        if K:
            cap_mask.scatter_(1, order[:, :K], True)
        keep_mask &= cap_mask
        proposed = keep_mask.sum(dim=1)
        proposed_per_latent = torch.stack(
            [keep_mask[:, mask].sum(dim=1) for mask in latent_masks], dim=1
        )
        actual_k = int(proposed.max().item()) if proposed.numel() else 0
        rectangular = keep_mask.clone()
        for batch_index in range(scores.shape[0]):
            needed = actual_k - int(proposed[batch_index])
            if needed <= 0:
                continue
            ranked = order[batch_index]
            fill = ranked[~rectangular[batch_index, ranked]][:needed]
            rectangular[batch_index, fill] = True
        all_local = torch.arange(domain.n_candidate, device=scores.device).expand(
            scores.shape[0], -1
        )
        keep_local = all_local[rectangular].reshape(scores.shape[0], actual_k)
        drop_local = all_local[~rectangular].reshape(
            scores.shape[0], domain.n_candidate - actual_k
        )
        candidate = domain.candidate_indices.to(scores.device)
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata={
                "selector": self.name,
                "dynamic": True,
                "thresholds_oldest_to_newest": list(self.thresholds),
                "proposed_K_per_batch": proposed.detach().cpu().tolist(),
                "proposed_K_per_latent": proposed_per_latent.detach().cpu().tolist(),
                "actual_K": actual_k,
                "batch_conservative_fill": bool(
                    proposed.numel() and not torch.all(proposed == actual_k)
                ),
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "thresholds": list(self.thresholds),
            "latent_order": "oldest_to_newest",
            "require_num_history_latents": self.require_num_history_latents,
        }


@register_selector("history_quota")
class HistoryQuotaSelector(TokenSelector):
    """Keep an exact, independently ranked quota from each history latent."""

    name = "history_quota"

    def __init__(self, ratios: Sequence[float], require_num_history_latents: int = 2):
        self.ratios = tuple(float(value) for value in ratios)
        self.require_num_history_latents = int(require_num_history_latents)
        if len(self.ratios) != self.require_num_history_latents:
            raise ValueError("ratios must contain one value per history latent")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.ratios):
            raise ValueError("history quota ratios must be finite and within [0, 1]")

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        _validate_complete_history(
            scores, domain, ctx, self.require_num_history_latents
        )
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order, scores, domain, K: int, ctx=None) -> SelectionResult:
        _validate_complete_history(
            scores, domain, ctx, self.require_num_history_latents
        )
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        quotas = [
            int(round(ctx.layout.tokens_per_latent * ratio)) for ratio in self.ratios
        ]
        if int(K) != sum(quotas):
            raise ValueError(
                f"resolved K={K} does not match per-latent quota total {sum(quotas)}"
            )
        latent_masks = _latent_local_masks(domain, ctx, scores.device)
        candidate = domain.candidate_indices.to(scores.device)
        keep_rows = []
        for batch_order in order.long():
            pieces = []
            for quota, latent_mask in zip(quotas, latent_masks):
                ranked_local = batch_order[latent_mask[batch_order]]
                pieces.append(ranked_local[:quota])
            keep_rows.append(torch.cat(pieces))
        keep_local = torch.stack(keep_rows)
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        if keep_local.numel():
            keep_mask.scatter_(1, keep_local, True)
        all_local = torch.arange(domain.n_candidate, device=scores.device).expand(
            scores.shape[0], -1
        )
        drop_local = all_local[~keep_mask].reshape(
            scores.shape[0], domain.n_candidate - int(K)
        )
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=int(K),
            metadata={
                "selector": self.name,
                "ratios_oldest_to_newest": list(self.ratios),
                "quota_oldest_to_newest": quotas,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "ratios": list(self.ratios),
            "latent_order": "oldest_to_newest",
            "require_num_history_latents": self.require_num_history_latents,
        }

"""Independent threshold and quota selectors for complete video latent groups.

Although the original module name and the registered ``history_*`` selectors
are history-specific, the same audited per-latent mechanics are used for
future video latents.  Each latent keeps its own ranking and quota/threshold;
the mapping between latent slots and global frame indices is supplied by the
concrete selector subclass.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


def _validate_latent_context(
    scores: torch.Tensor,
    domain,
    ctx,
    *,
    require_num_latents: int,
    accepted_domain_names: set[str],
    latent_indices: Sequence[int],
    label: str,
) -> None:
    """Validate that ``domain`` covers exactly the requested complete latents."""

    if ctx is None:
        raise ValueError(f"per-latent {label} selectors require a TokenContext")
    if scores.ndim != 2 or scores.shape != (ctx.batch_size, domain.n_candidate):
        raise ValueError("scores must have shape [B,N_candidate]")
    if len(latent_indices) != int(require_num_latents):
        raise ValueError(
            f"expected {require_num_latents} {label} latents, "
            f"got {len(latent_indices)}"
        )
    if domain.name not in accepted_domain_names:
        choices = "/".join(sorted(accepted_domain_names))
        raise ValueError(f"per-latent {label} selectors require domain={choices}")
    if not latent_indices:
        raise ValueError(f"{label} domain must contain at least one complete latent")
    first_start = ctx.layout.frame_range(int(latent_indices[0])).start
    last_end = ctx.layout.frame_range(int(latent_indices[-1])).end
    expected_length = int(last_end) - int(first_start)
    if domain.n_candidate != expected_length:
        raise ValueError(
            f"{label} domain must contain every {label} token and no other tokens"
        )
    expected = torch.arange(
        first_start, last_end, device=domain.candidate_indices.device
    )
    if not torch.equal(domain.candidate_indices, expected):
        raise ValueError(
            f"{label} domain must be the contiguous storage-order range "
            f"[{first_start}, {last_end})"
        )
    ensure_finite(scores, "scores")


def _latent_local_masks(
    domain, ctx, latent_indices: Sequence[int], device
) -> list[torch.Tensor]:
    candidate = domain.candidate_indices.to(device)
    masks = []
    for latent_index in latent_indices:
        frame = ctx.layout.frame_range(int(latent_index))
        mask = (candidate >= frame.start) & (candidate < frame.end)
        if int(mask.sum()) != ctx.layout.tokens_per_latent:
            raise RuntimeError("candidates do not cover one complete latent")
        masks.append(mask)
    return masks


class _LatentThresholdSelector(TokenSelector):
    """Common implementation for independent per-latent score thresholds."""

    name = "latent_threshold"
    accepted_domain_names: tuple[str, ...] = ()
    label = "latent"
    threshold_metadata_key = "thresholds"
    latent_order_label = "storage_order"

    def __init__(self, thresholds: Sequence[float], require_num_latents: int = 2):
        self.thresholds = tuple(float(value) for value in thresholds)
        self.require_num_latents = int(require_num_latents)
        if len(self.thresholds) != self.require_num_latents:
            raise ValueError("thresholds must contain one value per latent")
        if self.require_num_latents < 1:
            raise ValueError("require_num_latents must be positive")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.thresholds
        ):
            raise ValueError("thresholds must be finite and within [0, 1]")

    def _latent_indices_for_layout(self, layout) -> list[int]:
        raise NotImplementedError

    def _validate(self, scores, domain, ctx, K: int) -> None:
        latent_indices = (
            self._latent_indices_for_layout(ctx.layout) if ctx is not None else []
        )
        _validate_latent_context(
            scores,
            domain,
            ctx,
            require_num_latents=self.require_num_latents,
            accepted_domain_names=set(self.accepted_domain_names),
            latent_indices=latent_indices,
            label=self.label,
        )
        if not 0 <= int(K) <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        self._validate(scores, domain, ctx, K)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(
        self, order, scores, domain, K: int, ctx=None
    ) -> SelectionResult:
        self._validate(scores, domain, ctx, K)
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        K = int(K)
        latent_indices = self._latent_indices_for_layout(ctx.layout)
        latent_masks = _latent_local_masks(domain, ctx, latent_indices, scores.device)
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
                self.threshold_metadata_key: list(self.thresholds),
                "latent_order": self.latent_order_label,
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
            "latent_order": self.latent_order_label,
            "require_num_latents": self.require_num_latents,
        }


class _LatentQuotaSelector(TokenSelector):
    """Common implementation for exact independent per-latent Top-K quotas."""

    name = "latent_quota"
    accepted_domain_names: tuple[str, ...] = ()
    label = "latent"
    quota_metadata_key = "quota"
    latent_order_label = "storage_order"

    def __init__(self, ratios: Sequence[float], require_num_latents: int = 2):
        self.ratios = tuple(float(value) for value in ratios)
        self.require_num_latents = int(require_num_latents)
        if len(self.ratios) != self.require_num_latents:
            raise ValueError("ratios must contain one value per latent")
        if self.require_num_latents < 1:
            raise ValueError("require_num_latents must be positive")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.ratios
        ):
            raise ValueError("quota ratios must be finite and within [0, 1]")

    def _latent_indices_for_layout(self, layout) -> list[int]:
        raise NotImplementedError

    def _validate(self, scores, domain, ctx, K: int) -> None:
        latent_indices = (
            self._latent_indices_for_layout(ctx.layout) if ctx is not None else []
        )
        _validate_latent_context(
            scores,
            domain,
            ctx,
            require_num_latents=self.require_num_latents,
            accepted_domain_names=set(self.accepted_domain_names),
            latent_indices=latent_indices,
            label=self.label,
        )

    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        self._validate(scores, domain, ctx, K)
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(
        self, order, scores, domain, K: int, ctx=None
    ) -> SelectionResult:
        self._validate(scores, domain, ctx, K)
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        quotas = [
            int(round(ctx.layout.tokens_per_latent * ratio))
            for ratio in self.ratios
        ]
        if int(K) != sum(quotas):
            raise ValueError(
                f"resolved K={K} does not match per-latent quota total {sum(quotas)}"
            )
        latent_indices = self._latent_indices_for_layout(ctx.layout)
        latent_masks = _latent_local_masks(domain, ctx, latent_indices, scores.device)
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
                self.quota_metadata_key: quotas,
                "latent_order": self.latent_order_label,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "ratios": list(self.ratios),
            "latent_order": self.latent_order_label,
            "require_num_latents": self.require_num_latents,
        }


@register_selector("history_threshold")
class HistoryThresholdSelector(_LatentThresholdSelector):
    """Apply one absolute score threshold per history latent (oldest,newest)."""

    name = "history_threshold"
    accepted_domain_names = ("history", "all_history")
    label = "history"
    threshold_metadata_key = "thresholds_oldest_to_newest"
    latent_order_label = "oldest_to_newest"

    def __init__(self, thresholds: Sequence[float], require_num_history_latents: int = 2):
        super().__init__(thresholds, require_num_latents=require_num_history_latents)

    def _latent_indices_for_layout(self, layout) -> list[int]:
        return list(range(int(layout.num_cond_latents)))


@register_selector("future_threshold")
class FutureThresholdSelector(_LatentThresholdSelector):
    """Apply one absolute score threshold per future latent (near,far)."""

    name = "future_threshold"
    accepted_domain_names = ("future_video",)
    label = "future"
    threshold_metadata_key = "thresholds_near_to_far"
    latent_order_label = "near_to_far"

    def __init__(self, thresholds: Sequence[float], require_num_future_latents: int = 2):
        super().__init__(thresholds, require_num_latents=require_num_future_latents)

    def _latent_indices_for_layout(self, layout) -> list[int]:
        return list(range(int(layout.num_cond_latents), int(layout.video_f)))


@register_selector("history_quota")
class HistoryQuotaSelector(_LatentQuotaSelector):
    """Keep an exact, independently ranked quota from each history latent."""

    name = "history_quota"
    accepted_domain_names = ("history", "all_history")
    label = "history"
    quota_metadata_key = "quota_oldest_to_newest"
    latent_order_label = "oldest_to_newest"

    def __init__(self, ratios: Sequence[float], require_num_history_latents: int = 2):
        super().__init__(ratios, require_num_latents=require_num_history_latents)

    def _latent_indices_for_layout(self, layout) -> list[int]:
        return list(range(int(layout.num_cond_latents)))


@register_selector("future_quota")
class FutureQuotaSelector(_LatentQuotaSelector):
    """Keep an exact, independently ranked quota from each future latent."""

    name = "future_quota"
    accepted_domain_names = ("future_video",)
    label = "future"
    quota_metadata_key = "quota_near_to_far"
    latent_order_label = "near_to_far"

    def __init__(self, ratios: Sequence[float], require_num_future_latents: int = 2):
        super().__init__(ratios, require_num_latents=require_num_future_latents)

    def _latent_indices_for_layout(self, layout) -> list[int]:
        return list(range(int(layout.num_cond_latents), int(layout.video_f)))

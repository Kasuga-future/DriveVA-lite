"""History-guided future token selection.

The learned planning selector is trained on clean history tokens.  Early future
latents are still dominated by diffusion noise, so scoring them directly is
close to a random selection.  This selector uses the trustworthy history scores
to decide which spatial positions to keep in the future:

1. apply the configured per-history-latent threshold to the history scores;
2. map the selected history local positions to future latent positions;
3. return a single rectangular selection over the complete ``all_video``
   candidate set.

The future mask is copied from history decisions, not recomputed from future
features.  With ``future_mapping="same_latent"`` and equal history/future
latent counts, the number of physically removed tokens is exactly doubled
relative to history-only compression.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


_FUTURE_MAPPINGS = {
    "same_latent",
    "reverse_latent",
    "nearest_history",
    "oldest_history",
    "union_history",
    "intersection_history",
    "majority_history",
}


@register_selector("history_guided_future")
class HistoryGuidedFutureSelector(TokenSelector):
    """Copy a threshold-selected history spatial mask to future latents."""

    name = "history_guided_future"

    def __init__(
        self,
        thresholds: Sequence[float],
        future_mapping: str = "same_latent",
        require_num_history_latents: int = 2,
        require_num_future_latents: int = 2,
    ):
        self.thresholds = tuple(float(value) for value in thresholds)
        self.require_num_history_latents = int(require_num_history_latents)
        self.require_num_future_latents = int(require_num_future_latents)
        if len(self.thresholds) != self.require_num_history_latents:
            raise ValueError("thresholds must contain one value per history latent")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.thresholds
        ):
            raise ValueError("history thresholds must be finite and within [0, 1]")
        mapping = str(future_mapping).strip().lower()
        if mapping not in _FUTURE_MAPPINGS:
            raise ValueError(
                f"future_mapping must be one of {sorted(_FUTURE_MAPPINGS)}, got {future_mapping!r}"
            )
        self.future_mapping = mapping

    def _latent_masks(self, domain, ctx, latent_indices, device):
        candidate = domain.candidate_indices.to(device)
        masks = []
        for latent_index in latent_indices:
            frame = ctx.layout.frame_range(int(latent_index))
            mask = (candidate >= frame.start) & (candidate < frame.end)
            if int(mask.sum()) != int(ctx.layout.tokens_per_latent):
                raise RuntimeError(
                    "history-guided selection requires every latent to be complete; "
                    f"latent {latent_index} has {int(mask.sum())} candidates"
                )
            masks.append(mask)
        return masks

    def _future_masks(self, history_masks, device):
        stacked = torch.stack(history_masks, dim=0)
        num_history = int(stacked.shape[0])
        num_future = int(self.require_num_future_latents)
        future = []
        for future_index in range(num_future):
            if self.future_mapping == "same_latent":
                source = future_index % num_history
                future.append(stacked[source])
            elif self.future_mapping == "reverse_latent":
                source = num_history - 1 - (future_index % num_history)
                future.append(stacked[source])
            elif self.future_mapping == "nearest_history":
                future.append(stacked[-1])
            elif self.future_mapping == "oldest_history":
                future.append(stacked[0])
            elif self.future_mapping == "union_history":
                future.append(stacked.any(dim=0))
            elif self.future_mapping == "intersection_history":
                future.append(stacked.all(dim=0))
            elif self.future_mapping == "majority_history":
                majority = (num_history + 1) // 2
                future.append(stacked.sum(dim=0) >= majority)
            else:  # pragma: no cover - constructor validates.
                raise RuntimeError(f"unsupported future_mapping {self.future_mapping!r}")
        return future

    def select(self, scores: torch.Tensor, domain, K: int | None = None, ctx=None) -> SelectionResult:
        if ctx is None:
            raise ValueError("history_guided_future requires a TokenContext")
        if scores.ndim != 2 or scores.shape != (ctx.batch_size, domain.n_candidate):
            raise ValueError("scores must have shape [B,N_candidate]")
        if domain.name not in {"all_video", "video"}:
            raise ValueError("history_guided_future requires domain=all_video")
        if domain.n_candidate != int(ctx.layout.video.length):
            raise ValueError("history_guided_future requires the complete video candidate set")
        if int(ctx.layout.num_cond_latents) != self.require_num_history_latents:
            raise ValueError(
                f"expected {self.require_num_history_latents} history latents, "
                f"got {ctx.layout.num_cond_latents}"
            )
        num_future = int(ctx.layout.video_f) - int(ctx.layout.num_cond_latents)
        if num_future != self.require_num_future_latents:
            raise ValueError(
                f"expected {self.require_num_future_latents} future latents, got {num_future}"
            )
        if K is not None and int(K) < 0:
            raise ValueError("K must be non-negative when provided")
        ensure_finite(scores, "scores")

        device = scores.device
        history_indices = list(range(self.require_num_history_latents))
        future_indices = list(
            range(
                int(ctx.layout.num_cond_latents),
                int(ctx.layout.num_cond_latents) + num_future,
            )
        )
        history_candidate_masks = self._latent_masks(
            domain, ctx, history_indices, device
        )
        future_candidate_masks = self._latent_masks(
            domain, ctx, future_indices, device
        )

        history_latent_keep = []
        for threshold, mask in zip(self.thresholds, history_candidate_masks):
            history_latent_keep.append(scores[:, mask] >= float(threshold))
        future_latent_keep = self._future_masks(history_latent_keep, device)

        keep_candidate = torch.zeros(
            (ctx.batch_size, domain.n_candidate), dtype=torch.bool, device=device
        )
        for keep, mask in zip(history_latent_keep, history_candidate_masks):
            keep_candidate[:, mask] = keep
        for keep, mask in zip(future_latent_keep, future_candidate_masks):
            keep_candidate[:, mask] = keep

        proposed = keep_candidate.sum(dim=1)
        actual_k = int(proposed.max().item()) if proposed.numel() else 0
        # The official evaluator uses batch size one.  For a larger batch we
        # conservatively pad to the largest proposed K using the same global
        # score ordering, mirroring the other dynamic selectors.
        if actual_k:
            for batch_index in range(ctx.batch_size):
                needed = actual_k - int(proposed[batch_index])
                if needed <= 0:
                    continue
                unselected = ~keep_candidate[batch_index]
                order = torch.argsort(
                    scores[batch_index], descending=True, stable=True
                )
                fill = order[unselected[order]][:needed]
                keep_candidate[batch_index, fill] = True

        all_local = torch.arange(domain.n_candidate, device=device).expand(
            ctx.batch_size, -1
        )
        keep_local = all_local[keep_candidate].reshape(ctx.batch_size, actual_k)
        drop_local = all_local[~keep_candidate].reshape(
            ctx.batch_size, domain.n_candidate - actual_k
        )
        candidate = domain.candidate_indices.to(device)
        history_counts = torch.stack(
            [keep.sum(dim=1) for keep in history_latent_keep],
            dim=1,
        )
        future_counts = torch.stack(
            [keep.sum(dim=1) for keep in future_latent_keep],
            dim=1,
        )
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata={
                "selector": self.name,
                "dynamic": True,
                "history_thresholds_oldest_to_newest": list(self.thresholds),
                "future_mapping": self.future_mapping,
                "proposed_K_per_batch": proposed.detach().cpu().tolist(),
                "proposed_K_per_history_latent": history_counts.detach().cpu().tolist(),
                "proposed_K_per_future_latent": future_counts.detach().cpu().tolist(),
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
            "future_mapping": self.future_mapping,
            "latent_order": "history oldest_to_newest; future near_to_far",
            "require_num_history_latents": self.require_num_history_latents,
            "require_num_future_latents": self.require_num_future_latents,
        }

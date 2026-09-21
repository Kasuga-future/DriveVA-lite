"""Per-block dynamic token retention.

Motivation (2026-09-21).  A single global ``topk`` over the ``all_video``
candidate pool lets one block absorb the entire cut.  Measured at K=1149/1560,
the compositional selector kept **780/780 future tokens** and pushed every one
of the 411 dropped tokens onto history, so the "joint" arm never compressed the
future block at all.  The same is true in reverse for the ``only_future`` arms,
which leave history untouched.

This selector forces the cut to be shared between the video blocks, and can do
so with a **dynamic** (per-scene varying) count inside each block:

* ``mode="quota"``   -- fixed top-k inside each block.  The block split is
  controlled, but the count is still fixed per scene.
* ``mode="dynamic"`` -- keep every token whose score reaches
  ``score_threshold``, clamped to ``[ceil(floor_ratio * quota), quota]``.  The
  count therefore varies per scene *and* per block, while the quota stays a hard
  upper bound so the sequence-length budget is still respected.

Rectangularity: ``SelectionResult`` must be rectangular, so a batch takes the
largest proposed count and rows that proposed fewer are conservatively filled
with their next-best dropped tokens (the same rule ``threshold`` uses).  For the
official evaluator the batch is one scene, so the per-scene dynamic count is
realised exactly.
"""

from __future__ import annotations

import math

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from ..utils.validation import ensure_finite
from .base import TokenSelector


@register_selector("block_quota")
class BlockQuotaSelector(TokenSelector):
    """Split the candidate pool into blocks and give each block its own budget."""

    name = "block_quota"

    def __init__(
        self,
        block_weights: dict[str, float] | None = None,
        mode: str = "dynamic",
        score_threshold: float = 0.0,
        floor_ratio: float = 0.0,
        eps: float = 1e-8,
    ):
        weights = {"history": 0.5, "future": 0.5} if block_weights is None else dict(block_weights)
        if not weights:
            raise ValueError("block_weights must not be empty")
        self.block_weights = {str(k): float(v) for k, v in weights.items()}
        if any(
            not math.isfinite(v) or v < 0.0 for v in self.block_weights.values()
        ):
            raise ValueError("block_weights must be finite and non-negative")
        if sum(self.block_weights.values()) <= 0.0:
            raise ValueError("block_weights must not sum to zero")
        unknown = set(self.block_weights) - {"history", "future"}
        if unknown:
            raise ValueError(
                "block_weights keys must be 'history' and/or 'future', got "
                f"{sorted(unknown)}"
            )
        self.mode = str(mode).strip().lower()
        if self.mode not in {"quota", "dynamic"}:
            raise ValueError("block_quota mode must be 'quota' or 'dynamic'")
        self.score_threshold = float(score_threshold)
        if not math.isfinite(self.score_threshold):
            raise ValueError("score_threshold must be finite")
        self.floor_ratio = float(floor_ratio)
        if not 0.0 <= self.floor_ratio <= 1.0:
            raise ValueError("floor_ratio must be within [0, 1]")
        self.eps = float(eps)

    # -- block geometry -----------------------------------------------------
    def _block_of_candidate(self, domain, ctx) -> dict[str, torch.Tensor]:
        if ctx is None or getattr(ctx, "layout", None) is None:
            raise ValueError(
                "block_quota requires ctx.layout to locate the history and future blocks"
            )
        layout = ctx.layout
        candidate = domain.candidate_indices
        labels: dict[str, torch.Tensor] = {}
        spans = {
            "history": layout.history_video,
            "future": layout.future_video,
        }
        for name in self.block_weights:
            span = spans[name]
            labels[name] = (candidate >= int(span.start)) & (candidate < int(span.end))
        covered = torch.zeros_like(candidate, dtype=torch.bool)
        for name, mask in labels.items():
            covered |= mask
            if not bool(mask.any()):
                # An empty block would still be handed its weight share, so the
                # budget for the blocks that ARE present would be silently cut
                # in half.  Fail loudly instead.
                raise ValueError(
                    f"block_quota named the '{name}' block but the candidate "
                    f"domain {domain.name!r} contains no {name} tokens; use a "
                    "plain topk/threshold selector for a single-block domain"
                )
        if not bool(covered.all()):
            raise ValueError(
                "block_quota requires the candidate domain to cover exactly the "
                "history and future video blocks"
            )
        return labels

    def _block_quota(self, K: int, n_block: int, weight: float) -> int:
        total = sum(self.block_weights.values())
        share = int(round(int(K) * (weight / total)))
        return max(0, min(int(n_block), share))

    # -- selection ----------------------------------------------------------
    def select(self, scores: torch.Tensor, domain, K: int, ctx=None) -> SelectionResult:
        if scores.ndim != 2 or scores.shape[1] != domain.n_candidate:
            raise ValueError("scores must have shape [B, N_candidate]")
        ensure_finite(scores, "scores")
        K = int(K)
        if not 0 <= K <= domain.n_candidate:
            raise ValueError(f"K={K} outside [0, {domain.n_candidate}]")
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return self.select_from_order(order, scores, domain, K, ctx)

    def select_from_order(self, order, scores, domain, K: int, ctx=None) -> SelectionResult:
        if order.ndim != 2 or order.shape != scores.shape:
            raise ValueError("frozen ranking must have the same shape as scores")
        ensure_finite(scores, "scores")
        K = int(K)
        labels = self._block_of_candidate(domain, ctx)
        n = int(domain.n_candidate)

        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        quotas: dict[str, int] = {}
        kept_counts: dict[str, torch.Tensor] = {}
        for name, mask in labels.items():
            n_block = int(mask.sum().item())
            quota = self._block_quota(K, n_block, self.block_weights[name])
            quotas[name] = quota
            block_scores = scores[:, mask]                      # [B, n_block]
            block_rank = torch.argsort(
                block_scores, dim=-1, descending=True, stable=True
            )                                                    # local ranks
            if self.mode == "quota" or quota == 0:
                block_keep = torch.zeros_like(block_scores, dtype=torch.bool)
                if quota:
                    block_keep.scatter_(1, block_rank[:, :quota], True)
            else:
                # Dynamic: threshold-qualified, clamped into [floor, quota].
                qualified = block_scores >= self.score_threshold
                floor = int(math.ceil(self.floor_ratio * quota))
                proposed = qualified.sum(dim=1).clamp(min=floor, max=quota)
                block_keep = torch.zeros_like(block_scores, dtype=torch.bool)
                # Take the best `proposed` tokens of this block, so the dynamic
                # count can never exceed the quota nor fall under the floor.
                for row in range(block_scores.shape[0]):
                    take = int(proposed[row].item())
                    if take:
                        block_keep[row, block_rank[row, :take]] = True
            keep_mask[:, mask] = block_keep
            kept_counts[name] = block_keep.sum(dim=1)

        proposed_per_row = keep_mask.sum(dim=1)
        actual_k = int(proposed_per_row.max().item()) if proposed_per_row.numel() else 0
        # Conservative rectangular fill: every row keeps its own proposals, and
        # short rows are topped up from their best remaining tokens.  Batch size
        # one is therefore exact.
        priority = torch.empty_like(order)
        for row in range(scores.shape[0]):
            row_order = order[row]
            row_kept = keep_mask[row][row_order]
            chosen = row_order[row_kept]      # kept, in descending score order
            rest = row_order[~row_kept]       # dropped, in descending score order
            priority[row] = torch.cat([chosen, rest])
        keep_local = priority[:, :actual_k]
        drop_local = priority[:, actual_k:]
        candidate = domain.candidate_indices
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata={
                "selector": self.name,
                "mode": self.mode,
                "score_threshold": self.score_threshold,
                "floor_ratio": self.floor_ratio,
                "block_weights": dict(self.block_weights),
                "per_block_quota": quotas,
                "per_block_kept_mean": {
                    k: float(v.float().mean().item()) for k, v in kept_counts.items()
                },
                "dynamic": self.mode == "dynamic",
                "proposed_K_per_batch": proposed_per_row.detach().cpu().tolist(),
                "actual_K": actual_k,
            },
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "block_weights": dict(self.block_weights),
            "score_threshold": self.score_threshold,
            "floor_ratio": self.floor_ratio,
            "dynamic": self.mode == "dynamic",
        }

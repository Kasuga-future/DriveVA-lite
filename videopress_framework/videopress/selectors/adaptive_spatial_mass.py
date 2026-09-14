"""Dynamic score-mass budgeting with a minimum spatial coverage constraint."""

from __future__ import annotations

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from .adaptive_mass import AdaptiveMassSelector


@register_selector("adaptive_spatial_mass")
class AdaptiveSpatialMassSelector(AdaptiveMassSelector):
    """Use adaptive K while retaining the strongest token in every spatial tile."""

    name = "adaptive_spatial_mass"

    def __init__(self, *args, tile_h: int = 3, tile_w: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_h = int(tile_h)
        self.tile_w = int(tile_w)
        if self.tile_h < 1 or self.tile_w < 1:
            raise ValueError("tile_h and tile_w must be positive")

    def _block_ids(self, ctx, domain, device) -> tuple[torch.Tensor, int]:
        if ctx is None:
            raise ValueError("adaptive_spatial_mass requires a TokenContext")
        height = int(ctx.layout.video_h)
        width = int(ctx.layout.video_w)
        if domain.n_candidate != height * width:
            raise ValueError(
                "adaptive_spatial_mass requires a domain containing exactly one video latent"
            )
        candidate = domain.candidate_indices.to(device)
        if candidate.numel() and not torch.equal(
            candidate,
            torch.arange(
                int(candidate[0]), int(candidate[0]) + candidate.numel(), device=device
            ),
        ):
            raise ValueError("adaptive_spatial_mass requires a contiguous video latent")
        local = torch.arange(domain.n_candidate, device=device)
        y = torch.div(local, width, rounding_mode="floor")
        x = local.remainder(width)
        blocks_h = (height + self.tile_h - 1) // self.tile_h
        blocks_w = (width + self.tile_w - 1) // self.tile_w
        block_ids = torch.div(y, self.tile_h, rounding_mode="floor") * blocks_w + torch.div(
            x, self.tile_w, rounding_mode="floor"
        )
        return block_ids, blocks_h * blocks_w

    def select_from_order(self, order, scores, domain, K: int, ctx=None) -> SelectionResult:
        gated = AdaptiveMassSelector.select_from_order(self, order, scores, domain, K, ctx)
        actual_k = int(gated.K)
        if actual_k == 0:
            gated.metadata.update(
                {
                    "selector": self.name,
                    "spatial_coverage": True,
                    "tile_h": self.tile_h,
                    "tile_w": self.tile_w,
                    "covered_tiles_before": [0] * scores.shape[0],
                    "covered_tiles_after": [0] * scores.shape[0],
                }
            )
            return gated

        block_ids, num_blocks = self._block_ids(ctx, domain, order.device)
        keep_rows = []
        drop_rows = []
        before_coverage = []
        after_coverage = []
        for batch_order in order.long():
            before_coverage.append(int(torch.unique(block_ids[batch_order[:actual_k]]).numel()))
            # Find the earliest global-ranking position for every tile in one
            # scatter reduction. This is selection-equivalent to a per-tile
            # Python loop but avoids dozens of small GPU kernel launches.
            rank_by_candidate = torch.empty_like(batch_order)
            rank_by_candidate.scatter_(
                0,
                batch_order,
                torch.arange(domain.n_candidate, device=order.device),
            )
            anchor_ranks = torch.full(
                (num_blocks,),
                domain.n_candidate,
                dtype=torch.long,
                device=order.device,
            )
            anchor_ranks.scatter_reduce_(
                0, block_ids, rank_by_candidate, reduce="amin", include_self=True
            )
            anchors = batch_order[torch.sort(anchor_ranks).values]
            anchor_mask = torch.zeros(
                domain.n_candidate, dtype=torch.bool, device=order.device
            )
            anchor_mask[anchors] = True
            if anchors.numel() >= actual_k:
                keep = anchors[:actual_k]
            else:
                remainder = batch_order[~anchor_mask[batch_order]]
                keep = torch.cat([anchors, remainder[: actual_k - anchors.numel()]])
            selected_mask = torch.zeros_like(anchor_mask)
            selected_mask[keep] = True
            drop = batch_order[~selected_mask[batch_order]]
            keep_rows.append(keep)
            drop_rows.append(drop)
            after_coverage.append(int(torch.unique(block_ids[keep]).numel()))

        keep_local = torch.stack(keep_rows)
        drop_local = torch.stack(drop_rows)
        candidate = domain.candidate_indices.to(order.device)
        metadata = dict(gated.metadata)
        metadata.update(
            {
                "selector": self.name,
                "spatial_coverage": True,
                "tile_h": self.tile_h,
                "tile_w": self.tile_w,
                "spatial_tile_count": num_blocks,
                "covered_tiles_before": before_coverage,
                "covered_tiles_after": after_coverage,
            }
        )
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=actual_k,
            metadata=metadata,
        )

    def describe(self) -> dict:
        return {
            **super().describe(),
            "name": self.name,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
        }

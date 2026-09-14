from __future__ import annotations

import torch

from ..core.registry import register_operator
from ..core.result import OperatorResult
from .base import TokenOperator


@register_operator("zero")
class ZeroMaskOperator(TokenOperator):
    name = "zero"
    driveva_compatible = True

    def apply(self, ctx, selection) -> OperatorResult:
        x = ctx.tokens.clone()
        keep = selection.keep_global_indices
        keep_mask = torch.zeros(
            (ctx.batch_size, ctx.layout.total_length), dtype=torch.bool, device=x.device
        )
        if keep.shape[1]:
            keep_mask.scatter_(1, keep.to(x.device), True)
        candidate_mask = ctx.domain.candidate_mask.unsqueeze(0).expand(ctx.batch_size, -1)
        zero_mask = candidate_mask & ~keep_mask
        x = x.masked_fill(zero_mask.unsqueeze(-1), 0)
        return OperatorResult(
            output=x,
            metadata={
                "operator": self.name,
                "drop_count": int(zero_mask.sum().item()),
                "zero_scope": "unselected_within_domain",
            },
        )

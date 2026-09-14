from __future__ import annotations

import torch

from ..core.registry import register_operator
from ..core.result import OperatorResult
from ..utils.validation import assert_protected_unchanged
from .base import TokenOperator


@register_operator("zero")
class ZeroMaskOperator(TokenOperator):
    name = "zero"
    driveva_compatible = True

    def apply(self, ctx, selection) -> OperatorResult:
        x = ctx.tokens.clone()
        candidate = ctx.domain.candidate_indices
        keep = selection.keep_global_indices
        keep_mask = torch.zeros(
            (ctx.batch_size, ctx.layout.total_length), dtype=torch.bool, device=x.device
        )
        if keep.shape[1]:
            keep_mask.scatter_(1, keep.to(x.device), True)
        candidate_mask = ctx.domain.candidate_mask.to(x.device).expand(ctx.batch_size, -1)
        drop_mask = candidate_mask & ~keep_mask
        x = x.masked_fill(drop_mask.unsqueeze(-1), 0)
        assert_protected_unchanged(ctx.tokens, x, ctx.domain.protected_mask)
        return OperatorResult(
            output=x,
            metadata={"operator": self.name, "drop_count": int(drop_mask.sum().item())},
        )

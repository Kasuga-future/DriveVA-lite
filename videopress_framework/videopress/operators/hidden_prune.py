"""Physical residual-token pruning at the DiT block-input boundary."""

from __future__ import annotations

import torch

from ..core.registry import register_operator
from ..core.result import OperatorResult, TokenMapping
from .base import TokenOperator


@register_operator("hidden_prune")
class HiddenPruneOperator(TokenOperator):
    """Gather selected hidden tokens while preserving every protected token.

    Unlike :class:`KVPruneOperator`, this operator runs before the first DiT
    block and therefore shortens Q, K, V and the residual/MLP path together.
    The DriveVA adapter retains the returned mapping and restores the original
    layout immediately before the video and trajectory heads.
    """

    name = "hidden_prune"
    preserves_sequence_length = False
    physical_compression = True
    driveva_compatible = True
    block_input_compatible = True

    def apply(self, ctx, selection) -> OperatorResult:
        protected = torch.where(ctx.domain.protected_mask.to(ctx.tokens.device))[0]
        keep_rows = [
            torch.sort(torch.cat([protected, row.to(protected.device).long()]))[0]
            for row in selection.keep_global_indices
        ]
        keep = torch.stack(keep_rows)
        output = torch.gather(
            ctx.tokens,
            1,
            keep.unsqueeze(-1).expand(keep.shape[0], keep.shape[1], ctx.hidden_dim),
        )

        input_to_output = torch.full(
            (ctx.batch_size, ctx.layout.total_length),
            -1,
            dtype=torch.long,
            device=ctx.tokens.device,
        )
        if keep.shape[1]:
            output_positions = torch.arange(
                keep.shape[1], device=ctx.tokens.device
            ).expand(ctx.batch_size, -1)
            input_to_output.scatter_(1, keep, output_positions)
        mapping = TokenMapping(
            output_to_input=keep,
            input_to_output=input_to_output,
            original_length=ctx.layout.total_length,
            compressed_length=int(keep.shape[1]),
        )
        return OperatorResult(
            output=output,
            mapping=mapping,
            metadata={
                "operator": self.name,
                "protected_count": int(protected.numel()),
                "selected_count": int(selection.K),
                "hidden_length_before": int(ctx.layout.total_length),
                "hidden_length_after": int(keep.shape[1]),
                "hidden_length_ratio": float(keep.shape[1] / ctx.layout.total_length),
                "pre_dit": True,
            },
            aux={"keep_indices": keep},
        )

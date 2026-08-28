"""Post-RoPE physical K/V pruning."""

from __future__ import annotations

import torch

from ..core.registry import register_operator
from ..core.result import OperatorResult, TokenMapping
from ..utils.tensor import batch_gather_seq
from .base import TokenOperator


@register_operator("kv_prune")
class KVPruneOperator(TokenOperator):
    name = "kv_prune"
    preserves_sequence_length = False
    physical_compression = True
    driveva_compatible = True

    def apply(self, ctx, selection) -> OperatorResult:
        if ctx.k is None or ctx.v is None:
            raise ValueError("KVPruneOperator requires ctx.k and ctx.v")
        if ctx.k.ndim != 4 or ctx.v.ndim != 4:
            raise ValueError("KVPruneOperator expects canonical [B,H,L,D] k/v")
        if ctx.k.shape != ctx.v.shape:
            raise ValueError("k and v shapes differ")
        if ctx.k.shape[0] != ctx.batch_size or ctx.k.shape[2] != ctx.layout.total_length:
            raise ValueError("k sequence does not match TokenContext layout")
        protected = torch.where(ctx.domain.protected_mask.to(ctx.k.device))[0]
        keep_rows = []
        for row in selection.keep_global_indices.to(ctx.k.device):
            # Sorted original order is important for stable attention positions.
            keep_rows.append(torch.sort(torch.cat([protected, row.long()]))[0])
        keep = torch.stack(keep_rows) if keep_rows else torch.empty((0, protected.numel()), dtype=torch.long, device=ctx.k.device)
        k_new = batch_gather_seq(ctx.k, keep)
        v_new = batch_gather_seq(ctx.v, keep)
        b = keep.shape[0]
        input_to_output = torch.full(
            (b, ctx.layout.total_length), -1, dtype=torch.long, device=keep.device
        )
        if keep.shape[1]:
            output_positions = torch.arange(keep.shape[1], device=keep.device).expand(b, -1)
            input_to_output.scatter_(1, keep, output_positions)
        mapping = TokenMapping(
            output_to_input=keep,
            input_to_output=input_to_output,
            original_length=ctx.layout.total_length,
            compressed_length=keep.shape[1],
        )
        q_length = ctx.q.shape[2] if ctx.q is not None and ctx.q.ndim == 4 else ctx.layout.total_length
        metadata = {
            "operator": self.name,
            "protected_count": int(protected.numel()),
            "selected_count": int(selection.K),
            "q_length": int(q_length),
            "k_length_before": int(ctx.k.shape[2]),
            "k_length_after": int(k_new.shape[2]),
            "v_length_after": int(v_new.shape[2]),
            "theoretical_attn_ratio": float(k_new.shape[2] / ctx.k.shape[2]),
            "post_rope": True,
        }
        return OperatorResult(
            output=ctx.tokens,
            metadata=metadata,
            mapping=mapping,
            aux={"q": ctx.q, "k": k_new, "v": v_new, "keep_indices": keep},
        )

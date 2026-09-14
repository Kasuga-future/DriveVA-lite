from __future__ import annotations

import torch

from ..core.registry import register_operator
from ..core.result import OperatorResult
from ..utils.seed import stable_seed
from ..utils.validation import assert_protected_unchanged
from .base import TokenOperator


@register_operator("mean")
class MeanReplaceOperator(TokenOperator):
    name = "mean"
    driveva_compatible = True

    def apply(self, ctx, selection) -> OperatorResult:
        if ctx.domain.n_candidate == 0:
            raise ValueError("Mean replacement needs at least one candidate token")
        x = ctx.tokens.clone()
        candidate_values = ctx.candidate_tokens()
        replacement = candidate_values.mean(dim=1, keepdim=True)
        keep_mask = torch.zeros(
            (ctx.batch_size, ctx.layout.total_length), dtype=torch.bool, device=x.device
        )
        if selection.keep_global_indices.shape[1]:
            keep_mask.scatter_(1, selection.keep_global_indices.to(x.device), True)
        drop_mask = ctx.domain.candidate_mask.to(x.device).expand(ctx.batch_size, -1) & ~keep_mask
        x = torch.where(drop_mask.unsqueeze(-1), replacement.expand(-1, x.shape[1], -1), x)
        assert_protected_unchanged(ctx.tokens, x, ctx.domain.protected_mask)
        return OperatorResult(
            output=x,
            metadata={"operator": self.name, "drop_count": int(drop_mask.sum().item())},
        )


class _ShuffleBase(TokenOperator):

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def _permutation(self, count: int, ctx, batch_index: int, label: str, device):
        if count <= 1:
            return torch.arange(count, device=device, dtype=torch.long)
        scene_tokens = ctx.metadata.get("scene_tokens")
        if scene_tokens is not None and (
            not isinstance(scene_tokens, (list, tuple)) or len(scene_tokens) != ctx.batch_size
        ):
            raise ValueError("ctx.metadata['scene_tokens'] must contain one value per batch item")
        scene = scene_tokens[batch_index] if scene_tokens is not None else ctx.scene_token
        seed = stable_seed(self.seed, scene, ctx.log_id, ctx.diffusion_rank, ctx.layer_idx, label)
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randperm(count, generator=generator, device=device)


@register_operator("shuffle_all")
class ShuffleAllOperator(_ShuffleBase):
    name = "shuffle_all"

    def apply(self, ctx, selection) -> OperatorResult:
        x = ctx.tokens.clone()
        candidate = ctx.domain.candidate_indices
        for batch_index in range(ctx.batch_size):
            permutation = self._permutation(candidate.numel(), ctx, batch_index, self.name, x.device)
            if candidate.numel() > 1:
                x[batch_index, candidate] = x[batch_index, candidate[permutation]]
        assert_protected_unchanged(ctx.tokens, x, ctx.domain.protected_mask)
        return OperatorResult(
            output=x,
            metadata={"operator": self.name, "seed": self.seed, "selection_dependent": False},
        )


@register_operator("shuffle_drop")
class ShuffleDroppedOperator(_ShuffleBase):
    name = "shuffle_drop"

    def apply(self, ctx, selection) -> OperatorResult:
        x = ctx.tokens.clone()
        candidate = ctx.domain.candidate_indices.to(x.device)
        drop_global = candidate[selection.drop_candidate_indices.to(x.device)]
        for batch_index, row in enumerate(drop_global):
            permutation = self._permutation(row.numel(), ctx, batch_index, self.name, x.device)
            if row.numel() > 1:
                x[batch_index, row] = x[batch_index, row[permutation]]
        assert_protected_unchanged(ctx.tokens, x, ctx.domain.protected_mask)
        return OperatorResult(
            output=x,
            metadata={"operator": self.name, "seed": self.seed, "selection_dependent": True},
        )


@register_operator("shuffle_keep")
class ShuffleKeptOperator(_ShuffleBase):
    name = "shuffle_keep"

    def apply(self, ctx, selection) -> OperatorResult:
        x = ctx.tokens.clone()
        for batch_index, row in enumerate(selection.keep_global_indices.to(x.device)):
            permutation = self._permutation(row.numel(), ctx, batch_index, self.name, x.device)
            if row.numel() > 1:
                x[batch_index, row] = x[batch_index, row[permutation]]
        assert_protected_unchanged(ctx.tokens, x, ctx.domain.protected_mask)
        return OperatorResult(
            output=x,
            metadata={"operator": self.name, "seed": self.seed, "selection_dependent": True},
        )


@register_operator("shuffle")
class ShuffleOperator(ShuffleDroppedOperator):
    """Compatibility alias for the K-controlled drop shuffle."""

    name = "shuffle_drop"

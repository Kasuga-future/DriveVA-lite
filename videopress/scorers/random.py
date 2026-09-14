from __future__ import annotations

import torch

from ..core.registry import register_scorer
from ..utils.seed import stable_seed
from .base import TokenScorer


@register_scorer("random")
class RandomScorer(TokenScorer):
    name = "random"

    def __init__(self, seed: int = 0, scope: str = "scene", layer: int | None = None):
        self.seed = int(seed)
        self.scope = str(scope).strip().lower()
        self.layer = None if layer is None else int(layer)
        if self.scope not in {"scene", "scene_step", "scene_layer", "scene_layer_step"}:
            raise ValueError(
                "random scope must be scene, scene_step, scene_layer or scene_layer_step"
            )

    @staticmethod
    def _batch_metadata(ctx, name: str, batch_index: int):
        values = ctx.metadata.get(name)
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or len(values) != ctx.batch_size:
            raise ValueError(f"ctx.metadata[{name!r}] must contain one value per batch item")
        return values[batch_index]

    def _seed_for(self, ctx, batch_index: int) -> int:
        scene = self._batch_metadata(ctx, "scene_tokens", batch_index) or ctx.scene_token
        parts = [self.seed, scene, ctx.log_id]
        if self.scope in {"scene_step", "scene_layer_step"}:
            parts.append(ctx.diffusion_rank)
        if self.scope in {"scene_layer", "scene_layer_step"}:
            parts.append(ctx.layer_idx if ctx.layer_idx is not None else self.layer)
        # batch_index is intentionally not part of the seed.  Batch packing or
        # distributed ordering must not alter a scene's random control mask.
        return stable_seed(*parts)

    def score(self, ctx) -> torch.Tensor:
        values = []
        for batch_index in range(ctx.batch_size):
            seed = self._seed_for(ctx, batch_index)
            generator = torch.Generator(device=ctx.tokens.device)
            generator.manual_seed(seed)
            values.append(
                torch.rand(
                    ctx.domain.n_candidate,
                    generator=generator,
                    device=ctx.tokens.device,
                    dtype=torch.float32,
                )
            )
        if not values:
            return ctx.tokens.new_empty((0, ctx.domain.n_candidate), dtype=torch.float32)
        return torch.stack(values)

    def signature(self) -> str:
        return f"{self.name}:seed={self.seed}:scope={self.scope}:layer={self.layer}"

    def describe(self) -> dict:
        return {**super().describe(), "seed": self.seed, "scope": self.scope, "layer": self.layer}

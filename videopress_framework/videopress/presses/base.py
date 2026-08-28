from __future__ import annotations

from typing import Any

import torch

from ..core.budget import budget_stats
from ..core.registry import register_press
from ..core.result import CompressionResult, SelectionResult, TokenMapping
from ..core.runtime import InjectionPoint


class BaseVideoPress:
    name = "press"
    injection_point = InjectionPoint.VIDEO_INPUT

    def prepare(self, runtime) -> None:
        return None

    def apply(self, ctx) -> CompressionResult:
        raise NotImplementedError

    def finalize(self, runtime) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "injection_point": self.injection_point.value
            if isinstance(self.injection_point, InjectionPoint)
            else str(self.injection_point),
        }


@register_press("noop")
class NoPress(BaseVideoPress):
    name = "noop"

    def apply(self, ctx) -> CompressionResult:
        candidate = ctx.domain.candidate_indices
        keep_local = torch.arange(candidate.numel(), device=ctx.tokens.device).expand(ctx.batch_size, -1).clone()
        empty = torch.empty((ctx.batch_size, 0), dtype=torch.long, device=ctx.tokens.device)
        keep_global = candidate[keep_local]
        selection = SelectionResult(keep_local, empty, keep_global, candidate.numel(), {"selector": "all"})
        return CompressionResult(
            output=ctx.tokens,
            scores=None,
            selection=selection,
            metadata={
                "press": self.name,
                "operator": "identity",
                "no_press": True,
                **budget_stats(ctx.layout, ctx.domain, candidate.numel()),
            },
            mapping=TokenMapping.identity(ctx.layout.total_length, ctx.batch_size, ctx.tokens.device),
        )

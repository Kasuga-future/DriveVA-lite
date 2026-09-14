from __future__ import annotations

import torch

from ..core.registry import register_scorer
from .base import TokenScorer


@register_scorer("token_norm")
class TokenNormScorer(TokenScorer):
    name = "token_norm"

    def __init__(self, ord: float = 2.0):
        self.ord = ord

    def score(self, ctx) -> torch.Tensor:
        values = ctx.candidate_tokens().float()
        return torch.linalg.vector_norm(values, ord=self.ord, dim=-1)

    def describe(self) -> dict:
        return {**super().describe(), "ord": self.ord}

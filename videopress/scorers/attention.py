"""Action-query to video-key attention scorers."""

from __future__ import annotations

import math

import torch

from ..core.registry import register_scorer
from ..utils.tensor import canonicalize_qkv
from .base import TokenScorer


class _ActionAttentionBase(TokenScorer):
    probe_mode = "online"

    def __init__(
        self,
        num_heads: int | None = None,
        layer: int | None = None,
        head_mode: str = "mean",
        head_index: int | None = None,
        action_mode: str = "mean",
        softmax_domain: str = "all_sequence",
        value_norm: bool = False,
        value_norm_head_mode: str | None = None,
    ):
        self.num_heads = num_heads
        self.layer = layer
        self.head_mode = str(head_mode).lower()
        self.head_index = head_index
        self.action_mode = str(action_mode).lower()
        self.softmax_domain = str(softmax_domain).lower()
        self.value_norm = bool(value_norm)
        self.value_norm_head_mode = (
            "matched" if self.head_mode == "single" and value_norm_head_mode is None
            else str(value_norm_head_mode or "mean").lower()
        )
        if self.head_mode not in {"mean", "single"}:
            raise ValueError("head_mode must be 'mean' or 'single'")
        if self.action_mode not in {"mean", "max", "first"}:
            raise ValueError("action_mode must be mean, max or first")
        if self.softmax_domain not in {"all_sequence", "all"}:
            raise ValueError("only all_sequence softmax is supported in V1")
        if self.head_mode == "single" and self.head_index is None:
            raise ValueError("head_index is required for single-head attention")
        if self.value_norm_head_mode not in {"mean", "matched"}:
            raise ValueError("value_norm_head_mode must be mean or matched")
        if self.value_norm_head_mode == "matched" and self.head_mode != "single":
            raise ValueError("matched value norm requires head_mode=single")

    def _aggregate_actions(self, values: torch.Tensor) -> torch.Tensor:
        # values: [B,H,A,N]
        if self.action_mode == "mean":
            return values.mean(dim=2)
        if self.action_mode == "max":
            return values.max(dim=2).values
        return values[:, :, :1].squeeze(2)

    def score(self, ctx) -> torch.Tensor:
        if ctx.q is None or ctx.k is None:
            raise ValueError("ActionAttentionScorer requires ctx.q and ctx.k")
        q, _ = canonicalize_qkv(ctx.q, self.num_heads)
        k, _ = canonicalize_qkv(ctx.k, q.shape[1])
        if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k head dimensions differ")
        action = torch.arange(
            ctx.layout.future_action.start,
            ctx.layout.future_action.end,
            device=q.device,
            dtype=torch.long,
        )
        if action.numel() == 0:
            raise ValueError("future_action range is empty; attention scorer needs action queries")
        if action.max() >= q.shape[2] or ctx.domain.candidate_indices.numel() and ctx.domain.candidate_indices.max() >= k.shape[2]:
            raise ValueError("layout indices exceed q/k sequence length")
        q_action = q.index_select(2, action)
        logits = torch.matmul(q_action.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attn = torch.softmax(logits, dim=-1)
        candidate_attn = attn.index_select(-1, ctx.domain.candidate_indices)
        if self.head_mode == "single":
            if self.head_index is None or not 0 <= self.head_index < candidate_attn.shape[1]:
                raise ValueError(f"head_index {self.head_index} outside head range")
            scores = self._aggregate_actions(candidate_attn[:, self.head_index : self.head_index + 1]).squeeze(1)
        else:
            scores = self._aggregate_actions(candidate_attn).mean(dim=1)
        if self.value_norm:
            if ctx.v is None:
                raise ValueError("value_norm attention scorer requires ctx.v")
            v, _ = canonicalize_qkv(ctx.v, q.shape[1])
            v_candidate = v.index_select(2, ctx.domain.candidate_indices).float()
            if self.head_mode == "single" and self.value_norm_head_mode == "matched":
                v_norm = torch.linalg.vector_norm(v_candidate[:, self.head_index], dim=-1)
            else:
                v_norm = torch.linalg.vector_norm(v_candidate, dim=-1).mean(dim=1)
            scores = scores * v_norm
        ctx.metadata["last_attention_metadata"] = self.describe()
        return scores

    def describe(self) -> dict:
        return {
            **super().describe(),
            "num_heads": self.num_heads,
            "layer": self.layer,
            "head_mode": self.head_mode,
            "head_index": self.head_index,
            "action_mode": self.action_mode,
            "softmax_domain": self.softmax_domain,
            "value_norm": self.value_norm,
            "value_norm_head_mode": self.value_norm_head_mode,
        }

    def signature(self) -> str:
        return (
            f"{self.name}:layer={self.layer}:heads={self.head_mode}:{self.head_index}:action={self.action_mode}:"
            f"vnorm={int(self.value_norm)}:{self.value_norm_head_mode}"
        )


@register_scorer("action_attention")
class ActionAttentionScorer(_ActionAttentionBase):
    name = "action_attention"


@register_scorer("action_attention_vnorm")
class ActionAttentionVNormScorer(_ActionAttentionBase):
    name = "action_attention_vnorm"

    def __init__(self, **kwargs):
        kwargs["value_norm"] = True
        super().__init__(**kwargs)

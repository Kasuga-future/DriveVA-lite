"""Action-query to video-key attention scorers."""

from __future__ import annotations

import math
from collections import defaultdict

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


@register_scorer("action_attention_vnorm_temporal")
class ActionAttentionVNormTemporalScorer(ActionAttentionVNormScorer):
    """Blend action salience with same-position change across history latents."""

    name = "action_attention_vnorm_temporal"

    def __init__(self, *, temporal_weight: float = 0.25, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.temporal_weight = float(temporal_weight)
        self.eps = float(eps)
        if not 0.0 <= self.temporal_weight <= 1.0:
            raise ValueError("temporal_weight must be within [0, 1]")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def score(self, ctx) -> torch.Tensor:
        attention_score = super().score(ctx).float()
        candidate = ctx.domain.candidate_indices.to(ctx.tokens.device)
        tokens_per_latent = int(ctx.layout.tokens_per_latent)
        expected_start = int(ctx.layout.history_video.end - tokens_per_latent)
        expected = torch.arange(
            expected_start,
            expected_start + tokens_per_latent,
            device=candidate.device,
        )
        if candidate.numel() != tokens_per_latent or not torch.equal(candidate, expected):
            raise ValueError(
                "action_attention_vnorm_temporal requires domain=last_history"
            )
        previous = candidate - tokens_per_latent
        current_tokens = ctx.tokens.index_select(1, candidate).float()
        previous_tokens = ctx.tokens.index_select(1, previous).float()
        temporal_change = torch.linalg.vector_norm(
            current_tokens - previous_tokens, dim=-1
        )
        attention_score = attention_score / attention_score.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        temporal_score = temporal_change / temporal_change.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        score = (
            (1.0 - self.temporal_weight) * attention_score
            + self.temporal_weight * temporal_score
        )
        ctx.metadata["score_diagnostics"] = {
            "temporal_weight": self.temporal_weight,
            "temporal_change_mean": float(temporal_change.mean().item()),
            "temporal_change_max": float(temporal_change.max().item()),
        }
        return score

    def describe(self) -> dict:
        return {
            **super().describe(),
            "name": self.name,
            "temporal_weight": self.temporal_weight,
        }

    def signature(self) -> str:
        return f"{super().signature()}:temporal={self.temporal_weight}"


@register_scorer("action_contribution_stability")
class ActionContributionStabilityScorer(_ActionAttentionBase):
    """Rank tokens by action-output contribution and cross-layer consistency.

    Unlike ``attention * mean(||V_h||)``, the contribution term preserves the
    per-head pairing before reducing heads::

        ||concat_h(A[h, q, i] * V[h, i])||_2

    The score is then discounted when a token's mean-head value vector is very
    similar to another candidate and when its normalized importance is
    unstable across the observation window.  Earlier layers are observation
    only; compression still begins at ``layer``.
    """

    name = "action_contribution_stability"

    def __init__(
        self,
        *,
        layer: int,
        observation_start_layer: int | None = None,
        num_heads: int | None = None,
        action_mode: str = "mean",
        redundancy_weight: float = 0.15,
        stability_weight: float = 0.25,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            layer=layer,
            head_mode="mean",
            action_mode=action_mode,
            value_norm=False,
            **kwargs,
        )
        self.observation_start_layer = (
            max(0, int(layer) - 3)
            if observation_start_layer is None
            else int(observation_start_layer)
        )
        self.redundancy_weight = float(redundancy_weight)
        self.stability_weight = float(stability_weight)
        self.eps = float(eps)
        if self.observation_start_layer < 0 or self.observation_start_layer > int(layer):
            raise ValueError("observation_start_layer must be within [0, layer]")
        if not 0.0 <= self.redundancy_weight <= 1.0:
            raise ValueError("redundancy_weight must be within [0, 1]")
        if not 0.0 <= self.stability_weight <= 1.0:
            raise ValueError("stability_weight must be within [0, 1]")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        self._observations = defaultdict(dict)

    def observation_layers(self) -> tuple[int, ...]:
        return tuple(range(self.observation_start_layer, int(self.layer)))

    def reset_observations(self) -> None:
        self._observations.clear()

    @staticmethod
    def _context_key(ctx) -> tuple[str, str, int | None]:
        metadata = ctx.metadata if isinstance(ctx.metadata, dict) else {}
        rank = None if ctx.diffusion_rank is None else int(ctx.diffusion_rank)
        return str(ctx.scene_token), str(metadata.get("model_name", "")), rank

    def _contribution_and_redundancy(self, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.q is None or ctx.k is None or ctx.v is None:
            raise ValueError("contribution scorer requires ctx.q, ctx.k and ctx.v")
        q, _ = canonicalize_qkv(ctx.q, self.num_heads)
        k, _ = canonicalize_qkv(ctx.k, q.shape[1])
        v, _ = canonicalize_qkv(ctx.v, q.shape[1])
        if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k head dimensions differ")
        if v.shape[:3] != k.shape[:3]:
            raise ValueError("v batch/head/sequence dimensions differ from k")
        action = torch.arange(
            ctx.layout.future_action.start,
            ctx.layout.future_action.end,
            device=q.device,
            dtype=torch.long,
        )
        if action.numel() == 0:
            raise ValueError("future_action range is empty")
        candidate = ctx.domain.candidate_indices.to(q.device)
        logits = torch.matmul(
            q.index_select(2, action).float(), k.float().transpose(-1, -2)
        ) / math.sqrt(q.shape[-1])
        attention = torch.softmax(logits, dim=-1).index_select(-1, candidate)
        candidate_v = v.index_select(2, candidate).float()
        value_norm = torch.linalg.vector_norm(candidate_v, dim=-1)
        # [B,H,A,N] paired with [B,H,N], then reduce heads as one concatenated
        # contribution vector without materializing [B,A,N,H*D].
        contribution = torch.sqrt(
            torch.sum(attention.square() * value_norm.unsqueeze(2).square(), dim=1)
            + self.eps
        )
        if self.action_mode == "mean":
            contribution = contribution.mean(dim=1)
        elif self.action_mode == "max":
            contribution = contribution.max(dim=1).values
        else:
            contribution = contribution[:, 0]

        if candidate_v.shape[2] <= 1 or self.redundancy_weight == 0:
            redundancy = contribution.new_zeros(contribution.shape)
        else:
            # Mean-head values keep this O(B*N^2*head_dim), avoiding an
            # expensive full hidden-dimension candidate similarity matrix.
            features = torch.nn.functional.normalize(
                candidate_v.mean(dim=1), dim=-1, eps=self.eps
            )
            similarity = torch.matmul(features, features.transpose(-1, -2))
            diagonal = torch.eye(
                similarity.shape[-1], device=similarity.device, dtype=torch.bool
            ).unsqueeze(0)
            similarity = similarity.masked_fill(diagonal, -1.0)
            redundancy = similarity.max(dim=-1).values.clamp_(0.0, 1.0)
        return contribution, redundancy

    def _base_score(self, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        contribution, redundancy = self._contribution_and_redundancy(ctx)
        score = contribution * (1.0 - self.redundancy_weight * redundancy)
        normalized = score / score.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return normalized, redundancy

    def observe(self, ctx) -> None:
        layer = int(ctx.layer_idx)
        if layer not in self.observation_layers():
            return
        score, _ = self._base_score(ctx)
        self._observations[self._context_key(ctx)][layer] = score.detach()

    def score(self, ctx) -> torch.Tensor:
        current, redundancy = self._base_score(ctx)
        key = self._context_key(ctx)
        history = self._observations.pop(key, {})
        expected = self.observation_layers()
        observed = [history[layer].to(current.device) for layer in expected if layer in history]
        stack = torch.stack([*observed, current], dim=0)
        mean = stack.mean(dim=0)
        if stack.shape[0] > 1 and self.stability_weight > 0:
            variation = stack.std(dim=0, unbiased=False) / mean.clamp_min(self.eps)
            consistency = 1.0 / (1.0 + variation)
            scores = mean * (
                (1.0 - self.stability_weight)
                + self.stability_weight * consistency
            )
        else:
            variation = mean.new_zeros(mean.shape)
            scores = mean
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        ctx.metadata["score_diagnostics"] = {
            "observation_layers_expected": list(expected),
            "observation_layers_seen": sorted(int(layer) for layer in history),
            "score_layers_used": int(stack.shape[0]),
            "redundancy_mean": float(redundancy.mean().item()),
            "redundancy_max": float(redundancy.max().item()),
            "cross_layer_cv_mean": float(variation.mean().item()),
        }
        return scores

    def describe(self) -> dict:
        return {
            **super().describe(),
            "observation_start_layer": self.observation_start_layer,
            "redundancy_weight": self.redundancy_weight,
            "stability_weight": self.stability_weight,
            "contribution_reduction": "paired_head_l2",
        }

    def signature(self) -> str:
        return (
            f"{self.name}:layer={self.layer}:observe={self.observation_start_layer}:"
            f"action={self.action_mode}:redundancy={self.redundancy_weight}:"
            f"stability={self.stability_weight}"
        )

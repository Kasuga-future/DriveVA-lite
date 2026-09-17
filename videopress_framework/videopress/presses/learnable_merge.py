"""Learnable pre-DiT merge: ``N`` history candidates -> ``K`` register tokens.

This is the training-oriented counterpart of :class:`SimilarityMergePress`.  A
hand-designed grouping has to trade quality against grouping cost (greedy is
accurate but ~16 ms, k-means is fast but loses ~0.05 PDM).  Here the grouping is
a :class:`RegisterBottleneck` cross-attention read-out, so the whole operation
is one attention over the candidate set, costs < 1 ms, and -- crucially -- is
differentiable, which lets the DiT and the bottleneck be trained jointly under
the trajectory loss instead of being stitched on at inference time.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..core.budget import budget_stats, resolve_budget
from ..core.registry import register_press
from ..core.result import CompressionResult, OperatorResult, TokenMapping
from ..core.runtime import InjectionPoint
from ..operators.base import TokenOperator
from ..press.register_bottleneck import (
    RegisterBottleneck,
    key_tokens_for_context,
    splice_key_tokens,
)
from .base import BaseVideoPress


def _normalise_state_dict(state):
    """Accept both a bare module state dict and the trainer's export.

    ``export_trainable_state_dict`` keeps the submodule prefix, so a checkpoint
    trained by the framework-local wrapper has keys like
    ``learnable_merge.queries``; the press itself expects ``queries``.
    """

    if isinstance(state, dict) and "module" in state:
        state = state["module"]
    prefix = "learnable_merge."
    if state and any(str(key).startswith(prefix) for key in state):
        return {
            str(key)[len(prefix):]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
    return state


def _load_state(path):
    """Load a press checkpoint from either ``.pt`` or ``.safetensors``."""

    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


class RegisterMergeOperator(TokenOperator):
    """Finalise key tokens into a short residual sequence with a token mapping."""

    name = "register_merge"
    preserves_sequence_length = False
    physical_compression = True
    driveva_compatible = True
    block_input_compatible = True

    def apply(self, ctx, selection):  # pragma: no cover - press drives the operator
        raise NotImplementedError(
            "register_merge builds key tokens in the press; use apply_key_tokens"
        )

    def apply_key_tokens(
        self,
        ctx,
        key_tokens: torch.Tensor,
        candidate_start: int,
        candidate_end: int,
        *,
        attention: Optional[torch.Tensor] = None,
    ) -> OperatorResult:
        if key_tokens.ndim != 3:
            raise ValueError("key_tokens must be [B,K,D]")
        key_tokens = key_tokens.to(dtype=ctx.tokens.dtype)
        short, keep = splice_key_tokens(
            ctx.tokens, key_tokens, int(candidate_start), int(candidate_end)
        )
        batch = int(ctx.tokens.shape[0])
        original_length = int(ctx.tokens.shape[1])
        keep = keep.reshape(1, -1).expand(batch, -1).contiguous()
        input_to_output = self._input_to_output(
            ctx, key_tokens.shape[1], int(candidate_start), int(candidate_end), attention
        )
        mapping = TokenMapping(
            output_to_input=keep,
            input_to_output=input_to_output,
            original_length=original_length,
            compressed_length=int(short.shape[1]),
        )
        return OperatorResult(
            output=short,
            mapping=mapping,
            metadata={
                "operator": self.name,
                "physical_compression": True,
                "register_merge_k": int(key_tokens.shape[1]),
                "register_merge_candidate_start": int(candidate_start),
                "register_merge_candidate_end": int(candidate_end),
                "register_merge_original_length": original_length,
                "register_merge_compressed_length": int(short.shape[1]),
            },
        )

    @staticmethod
    def _input_to_output(
        ctx,
        num_key_tokens: int,
        candidate_start: int,
        candidate_end: int,
        attention: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-input short-sequence position, for auditing and drain-out checks."""

        device = ctx.tokens.device
        total = int(ctx.tokens.shape[1])
        batch = int(ctx.tokens.shape[0])
        prefix = torch.arange(0, candidate_start, device=device)
        suffix = torch.arange(candidate_end, total, device=device) + (
            num_key_tokens - (candidate_end - candidate_start)
        )
        candidate_len = candidate_end - candidate_start
        if attention is None:
            key_of_candidate = (
                torch.arange(candidate_len, device=device)
                .mul(num_key_tokens)
                .div(candidate_len, rounding_mode="floor")
            )
            key_of_candidate = key_of_candidate.reshape(1, -1).expand(batch, -1)
        else:
            # Attention is [B,H,K,N]; each candidate is assigned to the key that
            # reads it most strongly.
            key_of_candidate = attention.detach().mean(dim=1).argmax(dim=1)
        candidate = candidate_start + key_of_candidate
        return torch.cat([prefix.expand(batch, -1), candidate, suffix.expand(batch, -1)], dim=1)


@register_press("register_merge")
class RegisterBottleneckPress(BaseVideoPress):
    """Cross-attention merge of the ``last_history`` latent at the block input."""

    name = "register_merge"

    def __init__(
        self,
        budget,
        domain=None,
        num_key_tokens: int = 64,
        hidden_dim: int = 3072,
        attn_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_position_bias: bool = True,
        value_norm: bool = False,
        checkpoint: Optional[str] = None,
        trainable: bool = True,
        injection_point=InjectionPoint.BLOCK_INPUT,
    ):
        self.budget = budget
        self.domain = domain
        self.num_key_tokens = int(num_key_tokens)
        self.injection_point = InjectionPoint.parse(injection_point)
        if self.injection_point is not InjectionPoint.BLOCK_INPUT:
            raise ValueError("register_merge currently supports injection_point=block_input only")
        self.operator = RegisterMergeOperator()
        self.module = RegisterBottleneck(
            num_key_tokens=self.num_key_tokens,
            hidden_dim=int(hidden_dim),
            attn_dim=int(attn_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            use_position_bias=bool(use_position_bias),
            value_norm=bool(value_norm),
        )
        if checkpoint:
            state = _load_state(checkpoint)
            self.module.load_state_dict(_normalise_state_dict(state), strict=True)
        self.module.train(bool(trainable))
        self._device: Optional[torch.device] = None

    def apply(self, ctx) -> CompressionResult:
        if ctx.batch_size != 1:
            raise NotImplementedError("register_merge v1 supports batch_size=1 only")
        if ctx.domain.name != "last_history":
            raise NotImplementedError(
                "register_merge v1 supports domain=last_history only "
                "(the key-token layout assumes one contiguous final latent)"
            )
        k = resolve_budget(self.budget, ctx.layout, ctx.domain)
        if k != self.num_key_tokens:
            raise ValueError(
                f"register_merge was built with K={self.num_key_tokens} but the "
                f"budget resolves to K={k}"
            )
        if self._device != ctx.tokens.device:
            self.module.to(ctx.tokens.device)
            self._device = ctx.tokens.device

        encoding, _ = key_tokens_for_context(self.module, ctx)
        result = self.operator.apply_key_tokens(
            ctx,
            encoding.key_tokens,
            encoding.candidate_start,
            encoding.candidate_end,
            attention=encoding.attention,
        )
        result.metadata.update(
            {
                "press": self.name,
                "domain": ctx.domain.name,
                "injection_point": self.injection_point.value,
                "register_merge_trainable": bool(
                    any(p.requires_grad for p in self.module.parameters())
                ),
                "register_merge_num_parameters": int(
                    sum(p.numel() for p in self.module.parameters())
                ),
                **budget_stats(ctx.layout, ctx.domain, k),
                **encoding.metadata(),
            }
        )
        return CompressionResult(
            output=result.output,
            scores=None,
            selection=None,
            metadata=result.metadata,
            mapping=result.mapping,
            aux={
                "key_tokens": encoding.key_tokens,
                "attention": encoding.attention,
            },
        )

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def save_checkpoint(self, path) -> None:
        """Persist exactly the bottleneck weights plus shape metadata."""

        torch.save(
            {
                "module": self.module.state_dict(),
                "describe": self.module.describe(),
                "num_key_tokens": self.num_key_tokens,
            },
            str(path),
        )

    def load_checkpoint(self, path, *, strict: bool = True) -> None:
        state = _load_state(path)
        self.module.load_state_dict(_normalise_state_dict(state), strict=strict)
        self._device = None

    def trainable_parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "domain": getattr(self.domain, "name", self.domain),
            "num_key_tokens": self.num_key_tokens,
            "operator": self.operator.name,
            **self.module.describe(),
        }

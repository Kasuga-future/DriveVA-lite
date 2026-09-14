"""Shared model/sample state passed to every scorer and operator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch

from .domain import TokenDomain
from .layout import TokenLayout


@dataclass
class TokenContext:
    tokens: torch.Tensor
    layout: TokenLayout
    domain: TokenDomain
    scene_token: str = ""
    frame_token: Optional[str] = None
    log_id: str = ""
    timestamp: Optional[int] = None
    timestep: Optional[int] = None
    diffusion_rank: Optional[int] = None
    layer_idx: Optional[int] = None
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    trajectory_pred: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.tokens) or self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, N, D]")
        if self.tokens.shape[1] != self.layout.total_length:
            raise ValueError(
                f"tokens length {self.tokens.shape[1]} does not match layout.total_length "
                f"{self.layout.total_length}"
            )
        if self.domain.total_length != self.layout.total_length:
            raise ValueError("domain and layout lengths differ")
        for name in ("q", "k", "v"):
            value = getattr(self, name)
            if value is not None and value.ndim not in {3, 4}:
                raise ValueError(f"{name} must be [B,L,D] or [B,H,L,D]")
            if value is not None and value.shape[0] != self.tokens.shape[0]:
                raise ValueError(f"{name} batch dimension differs from tokens")

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.tokens.shape[-1])

    def candidate_tokens(self) -> torch.Tensor:
        return self.tokens.index_select(1, self.domain.candidate_indices)

    def protected_tokens(self) -> torch.Tensor:
        indices = torch.where(self.domain.protected_mask)[0]
        return self.tokens.index_select(1, indices)

    def clone_for_probe(self) -> "TokenContext":
        """Make a detached context for a backward/forward probe pass."""

        return replace(
            self,
            tokens=self.tokens.detach().clone(),
            q=None if self.q is None else self.q.detach().clone(),
            k=None if self.k is None else self.k.detach().clone(),
            v=None if self.v is None else self.v.detach().clone(),
            metadata=dict(self.metadata),
        )

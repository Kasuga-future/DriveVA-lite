"""Route A scorer: planning-conditioned token importance (plan section 10).

The plan's central claim is that a token's importance cannot be judged from the
video token alone.  The scorer therefore takes four inputs and lets them
interact:

* ``video_hidden``  - ``[B, N, D]`` candidate video tokens after the dense
  front-end blocks;
* ``action_hidden`` - ``[B, T, D]`` trajectory/action tokens from the same
  layer, used as the planning query;
* ``timestep``      - the flow-matching sigma for the current round;
* ``positions``     - normalized ``(t, y, x)`` per token;
* ``token_type``    - 0 for history, 1 for future (plan section 6 wants
  separate thresholds, which requires the scorer to know the identity).

Output is one logit per candidate; :class:`~videopress.retraining.threshold_gate.STEThresholdGate`
turns it into a mask.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, *, final_gelu: bool = True) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)]
    if final_gelu:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class DynamicVideoTokenScorer(nn.Module):
    """Token + position + action + timestep interaction scorer.

    ``action_mode`` controls how the action tokens become context:

    ``pooled``
        mean-pool the action tokens.  Cheapest, and the default from the plan
        pseudocode.
    ``attention``
        one learned query cross-attends over the action tokens, which lets the
        scorer read a specific planning sub-state (useful because the
        trajectory semantic forms around L10-12 while future video semantics
        only settle at L16-18).
    """

    def __init__(
        self,
        token_dim: int = 3072,
        hidden_dim: int = 256,
        position_dim: int = 64,
        time_dim: int = 64,
        action_mode: str = "pooled",
        num_heads: int = 4,
        n_token_types: int = 2,
    ):
        super().__init__()
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if action_mode not in {"pooled", "attention"}:
            raise ValueError(f"unsupported action_mode: {action_mode}")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_mode = str(action_mode)

        self.token_proj = _mlp(self.token_dim, self.hidden_dim, self.hidden_dim)
        self.action_proj = _mlp(self.token_dim, self.hidden_dim, self.hidden_dim)
        self.time_proj = _mlp(5, self.hidden_dim, self.hidden_dim)
        self.pos_proj = _mlp(3, int(position_dim), self.hidden_dim)
        self.type_embedding = nn.Embedding(int(n_token_types), self.hidden_dim)

        self.attention_action = None
        if self.action_mode == "attention":
            if self.hidden_dim % int(num_heads):
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.action_query = nn.Parameter(
                torch.randn(1, 1, self.hidden_dim) / math.sqrt(self.hidden_dim)
            )
            self.attention_action = nn.MultiheadAttention(
                self.hidden_dim, int(num_heads), batch_first=True
            )

        # Interaction term: the product of the token feature and the planning
        # context is what makes this a *planning* scorer rather than a magnitude
        # scorer (plan section 22 warns against raw ||register|| style scores).
        self.interaction = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.scoring = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _time_features(timestep, batch: int, n: int, dtype: torch.dtype) -> torch.Tensor:
        """Time features as ``[B, 5]`` (per sample) or ``[B, N, 5]`` (per token).

        DriveVA passes a *per-token* flow timestep to ``model_fn``: conditioned
        history latents carry 0 while the future latents carry the current
        sigma, so the tensor has length ``N`` (or ``B*N``) rather than ``B``.
        That distinction is informative rather than a nuisance -- a token's own
        noise level is exactly the kind of context the plan asks the scorer to
        condition on -- so per-token timesteps are supported natively instead of
        being collapsed to a scalar.
        """
        if timestep is None:
            value = torch.zeros(batch, dtype=torch.float32)
        else:
            value = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1)
            if value.numel() == 1:
                value = value.expand(batch)
            elif value.numel() == batch * n:
                value = value.reshape(batch, n)
            elif value.numel() != batch:
                raise ValueError(
                    f"timestep must be scalar, length {batch} (per sample) or "
                    f"length {batch * n} (per token), got {value.numel()}"
                )
        # DriveVA flow-matching timesteps are in [0, 1000]; Phase keeps the
        # frozen sinusoidal features from the existing selector so the two are
        # numerically comparable.
        phase = value / 1000.0
        return torch.stack(
            [
                phase,
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.sin(2.0 * math.pi * phase),
                torch.cos(2.0 * math.pi * phase),
            ],
            dim=-1,
        ).to(dtype=dtype)

    def _action_context(self, action_hidden: torch.Tensor, batch: int, n: int, dtype: torch.dtype) -> torch.Tensor:
        if action_hidden is None:
            return torch.zeros(batch, n, self.hidden_dim, device=self.type_embedding.weight.device, dtype=dtype)
        if action_hidden.ndim == 2:
            action_hidden = action_hidden.unsqueeze(0)
        if action_hidden.ndim != 3:
            raise ValueError("action_hidden must be [B,T,D] or [T,D]")
        action = action_hidden.to(dtype=dtype)
        projected = self.action_proj(action)
        if self.action_mode == "pooled":
            pooled = projected.mean(dim=1)
        else:
            query = self.action_query.to(dtype=dtype).expand(batch, -1, -1)
            pooled, _ = self.attention_action(query, projected, projected, need_weights=False)
            pooled = pooled.squeeze(1)
        return pooled.unsqueeze(1).expand(-1, n, -1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        video_hidden: torch.Tensor,
        *,
        action_hidden: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        token_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_hidden.ndim != 3:
            raise ValueError(
                f"video_hidden must be [B,N,D], got {tuple(video_hidden.shape)}"
            )
        batch, n, _ = video_hidden.shape
        param_dtype = self.type_embedding.weight.dtype

        tokens = video_hidden.to(dtype=param_dtype)
        token_feature = self.token_proj(tokens)

        if positions is None:
            positions = torch.zeros(batch, n, 3, device=video_hidden.device, dtype=param_dtype)
        else:
            positions = positions.to(device=video_hidden.device, dtype=param_dtype)
            if positions.ndim == 2:
                positions = positions.unsqueeze(0).expand(batch, -1, -1)
            if positions.shape[:2] != (batch, n):
                raise ValueError(
                    f"positions must be [B,N,3] matching video_hidden, got {tuple(positions.shape)}"
                )
        position_feature = self.pos_proj(positions)

        if token_type is None:
            token_type = torch.zeros(batch, n, device=video_hidden.device, dtype=torch.long)
        else:
            token_type = token_type.to(device=video_hidden.device)
            if token_type.ndim == 1:
                token_type = token_type.unsqueeze(0).expand(batch, -1)
            if token_type.shape != (batch, n):
                raise ValueError("token_type must be [B,N] or [N]")
            token_type = token_type.long()
            if int(token_type.min().item()) < 0 or int(token_type.max().item()) >= self.type_embedding.num_embeddings:
                raise ValueError("token_type contains an out-of-range id")
        type_feature = self.type_embedding(token_type)

        time_feature = self.time_proj(
            self._time_features(timestep, batch, n, param_dtype)
        )
        if time_feature.ndim == 2:
            time_feature = time_feature.unsqueeze(1).expand(-1, n, -1)

        context = self._action_context(action_hidden, batch, n, param_dtype) + time_feature
        local = token_feature + position_feature + type_feature
        interaction = self.interaction(local * context)
        return self.scoring(
            torch.cat([local, context, interaction, local * context], dim=-1)
        ).squeeze(-1)


def build_token_type_vector(
    domain_sizes,
    *,
    history_type: int = 0,
    future_type: int = 1,
) -> torch.Tensor:
    """Flat token-type vector matching a ``domain_sizes`` layout."""
    parts = []
    for index, size in enumerate(domain_sizes):
        parts.append(
            torch.full((int(size),), int(history_type if index == 0 else future_type), dtype=torch.long)
        )
    return torch.cat(parts, dim=0)

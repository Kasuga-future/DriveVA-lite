"""Checkpoint-backed planning-aware token selector scorer."""

from __future__ import annotations

from pathlib import Path
import math

import torch
from torch import nn

from ..core.registry import register_scorer
from .base import TokenScorer


class DynamicTokenSelector(nn.Module):
    """Token/position/condition MLP used by online teacher distillation."""

    def __init__(
        self,
        token_dim: int = 3072,
        ego_dim: int = 2,
        command_dim: int = 3,
        hidden_dim: int = 256,
        position_dim: int = 64,
        feature_mode: str = "all",
    ):
        super().__init__()
        self.feature_mode = str(feature_mode)
        if self.feature_mode not in {"all", "condition_position_time"}:
            raise ValueError(f"unsupported selector feature mode: {self.feature_mode}")
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(3, position_dim),
            nn.GELU(),
            nn.Linear(position_dim, hidden_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(ego_dim + command_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scoring = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.context_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.context_scoring = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.context_scoring[-1].weight)
        nn.init.zeros_(self.context_scoring[-1].bias)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.timestep_scoring = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.timestep_scoring[-1].weight)
        nn.init.zeros_(self.timestep_scoring[-1].bias)

    def forward(self, tokens, positions=None, ego_state=None, command=None, timestep=None):
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
        b, n, _ = tokens.shape
        device = tokens.device
        dtype = next(self.parameters()).dtype
        positions = (
            torch.zeros(b, n, 3, device=device, dtype=dtype)
            if positions is None
            else positions.to(device=device, dtype=dtype)
        )
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(b, -1, -1)
        ego_state = (
            torch.zeros(b, 2, device=device, dtype=dtype)
            if ego_state is None
            else ego_state.to(device=device, dtype=dtype)
        )
        command = (
            torch.zeros(b, 3, device=device, dtype=dtype)
            if command is None
            else command.to(device=device, dtype=dtype)
        )
        condition = self.condition_mlp(torch.cat([ego_state, command], dim=-1))
        condition = condition.unsqueeze(1).expand(-1, n, -1)
        local = self.token_proj(tokens.to(dtype=dtype))
        if self.feature_mode == "condition_position_time":
            local = torch.zeros_like(local)
        position = self.position_mlp(positions)
        base_logits = self.scoring(
            torch.cat([local, position, condition, local * condition], dim=-1)
        )
        scene = self.context_mlp(local.mean(dim=1)).unsqueeze(1).expand(-1, n, -1)
        context_logits = self.context_scoring(
            torch.cat([local, position, condition, scene, local * scene], dim=-1)
        )
        timestep = torch.zeros(b, device=device, dtype=dtype) if timestep is None else torch.as_tensor(
            timestep, device=device, dtype=dtype
        ).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(b)
        if timestep.numel() != b:
            raise ValueError(f"timestep must be scalar or length {b}, got {timestep.numel()}")
        phase = timestep / 1000.0
        time_features = torch.stack(
            [phase, torch.sin(math.pi * phase), torch.cos(math.pi * phase),
             torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)],
            dim=-1,
        )
        time = self.timestep_mlp(time_features).unsqueeze(1).expand(-1, n, -1)
        timestep_logits = self.timestep_scoring(
            torch.cat([local, position, condition, scene, time, local * time], dim=-1)
        )
        return (base_logits + context_logits + timestep_logits).squeeze(-1)


def _load_selector_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"selector checkpoint must contain a state dict: {path}")
    normalized = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "selector."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


@register_scorer("learned_planning_selector")
class LearnedPlanningSelectorScorer(TokenScorer):
    """Predict Gradient x Input planning utility without an online probe."""

    name = "learned_planning_selector"
    uses_pre_block_hidden = True

    def __init__(
        self,
        checkpoint: str,
        layer: int = 15,
        feature_layer: int | None = None,
        token_dim: int = 3072,
        action_mode: str | None = None,
        feature_mode: str = "all",
        future_position_mode: str = "storage",
    ):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.layer = int(layer)
        self.feature_layer = self.layer if feature_layer is None else int(feature_layer)
        if self.layer < 0 or self.feature_layer < 0:
            raise ValueError("learned selector layers must be non-negative")
        if self.feature_layer > self.layer:
            raise ValueError(
                "learned selector feature_layer cannot follow its compression layer"
            )
        # Accepted for compatibility with the common persistent scorer spec;
        # the learned network already pools planning supervision in its teacher.
        self.action_mode = action_mode
        self.feature_mode = str(feature_mode)
        self.future_position_mode = str(future_position_mode).strip().lower()
        if self.future_position_mode not in {"storage", "history_compatible"}:
            raise ValueError(
                "future_position_mode must be 'storage' or 'history_compatible'"
            )
        path = Path(self.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"learned selector checkpoint not found: {path}")
        self.network = DynamicTokenSelector(
            token_dim=int(token_dim), feature_mode=self.feature_mode
        )
        missing, unexpected = self.network.load_state_dict(
            _load_selector_state(path), strict=False
        )
        incompatible_missing = [
            key for key in missing if not key.startswith(
                ("context_mlp.", "context_scoring.", "timestep_mlp.", "timestep_scoring.")
            )
        ]
        if incompatible_missing or unexpected:
            raise ValueError(
                f"incompatible selector checkpoint {path}: "
                f"missing={incompatible_missing}, unexpected={unexpected}"
            )
        self.network.eval().requires_grad_(False)
        self._feature_scores: dict[tuple, tuple[torch.Tensor, dict]] = {}

    def reset_observations(self) -> None:
        self._feature_scores.clear()

    def observation_layers(self) -> tuple[int, ...]:
        return (self.feature_layer,) if self.feature_layer < self.layer else ()

    @staticmethod
    def _feature_key(ctx) -> tuple:
        metadata = ctx.metadata if isinstance(ctx.metadata, dict) else {}
        candidate = ctx.domain.candidate_indices
        return (
            str(ctx.scene_token),
            str(metadata.get("model_name", "")),
            None if ctx.diffusion_rank is None else int(ctx.diffusion_rank),
            str(ctx.domain.name),
            int(ctx.batch_size),
            int(ctx.layout.total_length),
            int(candidate.min().item()) if candidate.numel() else -1,
            int(candidate.max().item()) + 1 if candidate.numel() else -1,
        )

    @staticmethod
    def _conditions(ctx, dtype: torch.dtype):
        metadata = ctx.metadata if isinstance(ctx.metadata, dict) else {}

        def condition(key: str, default, width: int) -> torch.Tensor:
            value = torch.as_tensor(
                metadata.get(key, default),
                device=ctx.tokens.device,
                dtype=dtype,
            )
            if value.ndim == 1 and value.numel() == width:
                return value.unsqueeze(0).expand(ctx.batch_size, -1)
            if value.ndim == 2 and tuple(value.shape) == (ctx.batch_size, width):
                return value
            raise ValueError(
                f"learned selector {key} must have shape [{width}] or "
                f"[{ctx.batch_size},{width}], got {tuple(value.shape)}"
            )

        return condition("selector_ego_state", [0.0, 0.0], 2), condition(
            "selector_command", [0.0, 1.0, 0.0], 3
        )

    @staticmethod
    def _positions(ctx, dtype: torch.dtype, future_position_mode: str = "storage"):
        """Position features ``(t, y, x)`` for every candidate token.

        History domains retain the training-time convention exactly: the
        temporal coordinate is the latent's storage index (oldest = 0,
        newest = 1 for the default two-history-latent layout).

        Future domains support two documented modes:

        * ``"storage"`` -- use the real storage index ``t = 2, 3`` for the two
          future latents.  This is the honest coordinate, but it is out of the
          training distribution because history checkpoints only saw ``t`` in
          ``{0, 1}``.
        * ``"history_compatible"`` -- subtract ``num_cond_latents`` so the
          future latents reuse the history temporal range ``t = 0, 1``.  The
          relative order (near future = 0, far future = 1) is preserved while
          the selector sees coordinates closer to its history training data.

        Future latent order is always storage/temporal order: ``future_latent_0``
        is nearest to the history block, ``future_latent_1`` is farther ahead.
        """

        mode = str(future_position_mode).strip().lower()
        if mode not in {"storage", "history_compatible"}:
            raise ValueError(
                "future_position_mode must be 'storage' or 'history_compatible'"
            )
        indices = ctx.domain.candidate_indices.to(ctx.tokens.device)
        if indices.numel() == 0:
            raise ValueError("learned planning selector requires a non-empty video domain")
        layout = ctx.layout
        per_latent = int(layout.tokens_per_latent)
        video_start = int(layout.video.start)
        video_end = int(layout.video.end)
        history_start = int(layout.history_video.start)
        history_end = int(layout.history_video.end)
        future_start = int(layout.future_video.start)
        future_end = int(layout.future_video.end)
        num_cond = int(layout.num_cond_latents)

        in_history = bool(
            int(indices.min()) >= history_start and int(indices.max()) < history_end
        )
        in_future = bool(
            int(indices.min()) >= future_start and int(indices.max()) < future_end
        )
        in_video = bool(
            int(indices.min()) >= video_start and int(indices.max()) < video_end
        )
        if not in_video:
            raise ValueError(
                "learned planning selector requires a video domain "
                "(history, future_video, future_latent_i or all_video)"
            )

        offset_in_video = indices - video_start
        t_storage = torch.div(offset_in_video, per_latent, rounding_mode="floor")
        local = offset_in_video.remainder(per_latent)
        if in_history or (not in_future and mode == "storage"):
            t = t_storage
        elif in_future and mode == "storage":
            t = t_storage
        elif in_future and mode == "history_compatible":
            # Near future -> 0, next future -> 1, matching the history range.
            t = t_storage - num_cond
        else:
            # all_video with history_compatible: preserve history storage
            # coordinates and remap future latents into the same 0.. series.
            t = torch.where(
                t_storage < num_cond, t_storage, t_storage - num_cond
            )

        y = torch.div(local, int(layout.video_w), rounding_mode="floor")
        x = local.remainder(int(layout.video_w))
        positions = torch.stack(
            [
                t.to(dtype=dtype),
                (y / max(int(layout.video_h) - 1, 1)).to(dtype=dtype),
                (x / max(int(layout.video_w) - 1, 1)).to(dtype=dtype),
            ],
            dim=-1,
        )
        return positions.unsqueeze(0).expand(ctx.batch_size, -1, -1)

    @torch.no_grad()
    def _score_network(self, ctx):
        candidate = ctx.candidate_tokens()
        if candidate.shape[-1] != self.network.token_proj[1].in_features:
            raise ValueError(
                f"selector expected hidden dim {self.network.token_proj[1].in_features}, "
                f"got {candidate.shape[-1]}"
            )
        self.network.to(candidate.device)
        dtype = next(self.network.parameters()).dtype
        positions = self._positions(ctx, dtype, self.future_position_mode)
        ego, command = self._conditions(ctx, dtype)
        timestep = 0.0 if ctx.diffusion_rank is None else float(ctx.diffusion_rank)
        scores = torch.sigmoid(
            self.network(candidate, positions, ego, command, timestep=timestep)
        )
        ctx.metadata["score_diagnostics"] = {
            "checkpoint": self.checkpoint,
            "score_mean": float(scores.mean().item()),
            "score_std": float(scores.std().item()),
            "condition_source": "ego_velocity_and_prompt_command",
            "diffusion_timestep": timestep,
            "feature_layer": self.feature_layer,
            "compression_source_layer": self.layer,
        }
        return scores

    @torch.no_grad()
    def observe(self, ctx) -> None:
        if int(ctx.layer_idx) != self.feature_layer:
            raise ValueError(
                f"learned selector expected feature layer {self.feature_layer}, "
                f"got {ctx.layer_idx}"
            )
        scores = self._score_network(ctx)
        self._feature_scores[self._feature_key(ctx)] = (
            scores.detach(),
            dict(ctx.metadata.get("score_diagnostics", {})),
        )

    @torch.no_grad()
    def score(self, ctx):
        if self.feature_layer == self.layer:
            return self._score_network(ctx)
        key = self._feature_key(ctx)
        cached = self._feature_scores.get(key)
        if cached is None:
            raise RuntimeError(
                "learned selector has no cached feature-layer scores for "
                f"scene={ctx.scene_token}, diffusion={ctx.diffusion_rank}, "
                f"feature_layer={self.feature_layer}, source_layer={self.layer}"
            )
        scores, diagnostics = cached
        ctx.metadata["score_diagnostics"] = {
            **diagnostics,
            "feature_cache_reused": True,
        }
        return scores.to(ctx.tokens.device)

    def describe(self) -> dict:
        return {
            **super().describe(),
            "layer": self.layer,
            "feature_layer": self.feature_layer,
            "checkpoint": self.checkpoint,
            "uses_pre_block_hidden": True,
            "action_mode": self.action_mode,
            "feature_mode": self.feature_mode,
            "future_position_mode": self.future_position_mode,
        }

    def signature(self) -> str:
        signature = (
            f"{self.name}:layer={self.layer}:checkpoint={Path(self.checkpoint).name}:"
            f"features={self.feature_mode}"
        )
        if self.feature_layer != self.layer:
            signature += (
                f":feature_layer={self.feature_layer}:source_layer={self.layer}"
            )
        if self.future_position_mode != "storage":
            signature += f":future_position_mode={self.future_position_mode}"
        return signature

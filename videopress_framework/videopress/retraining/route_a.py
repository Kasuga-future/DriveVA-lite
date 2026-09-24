"""Route A - Dynamic Select (plan sections 7-15, 39-42).

Route A keeps the model's own spatial patch tokens and selects a dynamic subset
of them::

    dense noisy video latent
        -> patch embedding
        -> dense DiT blocks 0 .. Lb-1
        -> DynamicVideoTokenScorer
        -> STEThresholdGate          (score >= tau, dynamic K)
        -> sparse DiT blocks Lb .. 29 (short residual sequence)
        -> trajectory head            (planning output)
        -> DenseRecoveryDecoder       (video flow target, training only)
        -> Wan head

This module owns the whole forward pass so the compression is a first-class
``nn.Module`` that can be checkpointed, rather than an inference-time hook
(plan section 36).  It is deliberately model-agnostic: it takes a block list and
calls ``block(x, context, t_mod, freqs)``, which is the Wan ``DiTBlock``
signature.  That makes the exact same code path testable with a tiny stand-in
DiT and runnable against the production blocks.

Why a middle bottleneck layer (plan section 9)
---------------------------------------------

The 2026-09-22 DiT probes showed early hidden states are still close to VAE
patch texture, history semantics settle around L8-15, and future video latent
semantics only become linearly decodable around L16-18.  Selecting at L0 would
ask the scorer to judge a patch's planning value twenty layers before that value
exists.  ``bottleneck_layer`` is therefore a curriculum parameter, moving
18 -> 15 -> 12 only after the previous setting is stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .compression_stats import CompressionStatsRecorder, RoundRecord
from .dense_recovery import DenseRecoveryDecoder
from .dynamic_selector import DynamicVideoTokenScorer, build_token_type_vector
from .threshold_gate import (
    SafetyClampConfig,
    STEThresholdGate,
    ThresholdGateOutput,
    binding_row_domain_counts,
    select_kept_indices,
)


DRIVEVA_HISTORY_LATENTS = 2
DRIVEVA_FUTURE_LATENTS = 2
DRIVEVA_PATCH_H = 13
DRIVEVA_PATCH_W = 30


@dataclass
class RouteALayoutSpec:
    """Token layout of one DriveVA DiT forward."""

    history_tokens: int = DRIVEVA_HISTORY_LATENTS * DRIVEVA_PATCH_H * DRIVEVA_PATCH_W
    future_tokens: int = DRIVEVA_FUTURE_LATENTS * DRIVEVA_PATCH_H * DRIVEVA_PATCH_W
    traj_tokens: int = 9
    patch_h: int = DRIVEVA_PATCH_H
    patch_w: int = DRIVEVA_PATCH_W
    history_latents: int = DRIVEVA_HISTORY_LATENTS
    future_latents: int = DRIVEVA_FUTURE_LATENTS

    def __post_init__(self) -> None:
        for name in ("history_tokens", "future_tokens", "traj_tokens", "patch_h", "patch_w"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def domain_sizes(self) -> Tuple[int, int]:
        return (int(self.history_tokens), int(self.future_tokens))

    @property
    def video_tokens(self) -> int:
        return int(self.history_tokens) + int(self.future_tokens)

    @property
    def total_tokens(self) -> int:
        return self.video_tokens + int(self.traj_tokens)

    @property
    def video_slice(self) -> slice:
        return slice(0, self.video_tokens)

    @property
    def traj_slice(self) -> slice:
        return slice(self.video_tokens, self.total_tokens)


@dataclass
class RouteAConfig:
    """Everything that defines one Route A compression setting."""

    token_dim: int = 3072
    bottleneck_layer: int = 18
    num_blocks: int = 30
    layout: RouteALayoutSpec = field(default_factory=RouteALayoutSpec)
    history_threshold: float = 0.5
    future_threshold: float = 0.5
    temperature: float = 0.2
    min_temperature: float = 0.05
    selector_hidden: int = 256
    action_mode: str = "pooled"
    selector_heads: int = 4
    recovery_layers: int = 2
    recovery_heads: int = 16
    recovery_mlp_ratio: float = 2.0
    safety_clamp: SafetyClampConfig = field(default_factory=SafetyClampConfig)

    def __post_init__(self) -> None:
        if not 0 <= int(self.bottleneck_layer) < int(self.num_blocks):
            raise ValueError(
                f"bottleneck_layer must be in [0, {int(self.num_blocks) - 1}], "
                f"got {self.bottleneck_layer}"
            )
        for name in ("history_threshold", "future_threshold"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")

    def build(self) -> "RouteADynamicSelect":
        return RouteADynamicSelect(self)

    def with_bottleneck(self, layer: int) -> "RouteAConfig":
        """Compression-layer curriculum step (plan sections 9, 14-A4)."""
        return replace(self, bottleneck_layer=int(layer))


@dataclass
class RouteAForwardOutput:
    """Everything one Route A forward produces."""

    video_flow: Optional[torch.Tensor]
    traj_pred: Optional[torch.Tensor]
    dense_video_hidden: Optional[torch.Tensor]
    sparse_video_hidden: torch.Tensor
    sparse_traj_hidden: torch.Tensor
    logits: torch.Tensor
    gate: ThresholdGateOutput
    kept_indices: torch.Tensor
    kept_video_indices: torch.Tensor
    captured_action_hidden: Dict[int, torch.Tensor] = field(default_factory=dict)
    captured_video_hidden: Dict[int, torch.Tensor] = field(default_factory=dict)
    round_record: Optional[RoundRecord] = None
    batch_sync_added: int = 0
    ste_mask: Optional[torch.Tensor] = None
    #: ``True`` when the backend ran on the physically gathered short sequence
    #: (inference-equivalent); ``False`` for the dense-gated training relaxation.
    physical_shortening: bool = True
    backend_sequence_length: int = 0

    @property
    def kept_total(self) -> int:
        return int(self.kept_indices.shape[1])


def build_driveva_video_positions(
    layout: Optional[RouteALayoutSpec] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized ``(t, y, x)`` positions matching the deployed selector.

    Replicates the coordinate convention used by the training capture code and
    ``LearnedPlanningSelectorScorer._positions``: ``t`` is the storage latent
    index, ``y``/``x`` are divided by ``h-1`` / ``w-1``.
    """
    spec = layout or RouteALayoutSpec()
    h, w = int(spec.patch_h), int(spec.patch_w)
    per_latent = h * w
    t = torch.arange(spec.history_latents + spec.future_latents).repeat_interleave(per_latent)
    y = torch.arange(h).repeat_interleave(w).repeat(spec.history_latents + spec.future_latents)
    x = torch.arange(w).repeat(h).repeat(spec.history_latents + spec.future_latents)
    denom = torch.tensor([1.0, max(h - 1, 1), max(w - 1, 1)], dtype=torch.float32)
    positions = torch.stack([t.float(), y.float(), x.float()], dim=-1) / denom
    return positions.to(device=device, dtype=dtype)


def _domain_offsets(sizes: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    offset = 0
    for size in sizes:
        out.append((offset, int(size)))
        offset += int(size)
    return out


def _gather_rows(tensor: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """``torch.gather`` along the sequence dimension for any trailing rank."""
    if tensor.shape[0] != keep.shape[0]:
        raise ValueError("batch dimension differs from the keep index")
    view = (keep.shape[0], keep.shape[1]) + (1,) * (tensor.ndim - 2)
    expand = (keep.shape[0], keep.shape[1]) + tuple(tensor.shape[2:])
    return torch.gather(tensor, dim=1, index=keep.view(view).expand(expand))


def sync_keep_lengths(
    hard_mask: torch.Tensor,
    scores: torch.Tensor,
    *,
    allow_padding: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Make a per-row dynamic keep set rectangular for the batched forward.

    A threshold selector produces a *different* K for every scene, so a batch of
    scenes cannot be gathered into one rectangular tensor without a decision.
    The plan's training recipe uses ``micro_batch_per_gpu = 1`` (section 28), in
    which case this is a no-op.  For larger batches we pad shorter rows with
    their highest-scoring dropped tokens up to the batch maximum.  That only
    *adds* tokens, so it can never make a row less faithful; it does mean the
    effective compression of a batch is set by its least-compressible scene.

    Set ``allow_padding=False`` to make a ragged batch a hard error instead.
    """
    if hard_mask.ndim != 2 or scores.shape != hard_mask.shape:
        raise ValueError("hard_mask and scores must be matching [B,N] tensors")
    counts = (hard_mask > 0).sum(dim=1)
    target = int(counts.max().item())
    added = 0
    if bool((counts != target).any()):
        if not allow_padding:
            raise ValueError(
                "dynamic per-scene K produced a ragged batch "
                f"(counts={counts.tolist()}); use batch_size=1 or allow padding"
            )
        padding = target - counts
        for row in range(hard_mask.shape[0]):
            need = int(padding[row].item())
            if need <= 0:
                continue
            masked = scores[row].detach().clone()
            masked[hard_mask[row] > 0] = float("-inf")
            pick = masked.topk(need, dim=-1).indices
            hard_mask[row, pick] = True
            added += need
    return hard_mask, added


class RouteADynamicSelect(nn.Module):
    """Trained compression module implementing Route A.

    The wrapper owns only the *new* parameters (scorer, gate, recovery decoder).
    The DiT blocks are passed in at call time so the same instance can drive the
    student backbone, be checkpointed next to it, and be unit tested with a
    small stand-in block list.
    """

    def __init__(self, config: Optional[RouteAConfig] = None):
        super().__init__()
        self.config = config or RouteAConfig()
        spec = self.config.layout
        self.scorer = DynamicVideoTokenScorer(
            token_dim=self.config.token_dim,
            hidden_dim=self.config.selector_hidden,
            action_mode=self.config.action_mode,
            num_heads=self.config.selector_heads,
        )
        self.gate = STEThresholdGate(
            domain_sizes=spec.domain_sizes,
            thresholds=(self.config.history_threshold, self.config.future_threshold),
            temperature=self.config.temperature,
            min_temperature=self.config.min_temperature,
            clamp=self.config.safety_clamp,
        )
        self.recovery = DenseRecoveryDecoder(
            dim=self.config.token_dim,
            full_length=spec.video_tokens,
            n_layers=self.config.recovery_layers,
            num_heads=self.config.recovery_heads,
            mlp_ratio=self.config.recovery_mlp_ratio,
        )
        self.register_buffer(
            "token_type",
            build_token_type_vector(spec.domain_sizes),
            persistent=True,
        )

    # ------------------------------------------------------------------ props
    @property
    def bottleneck_layer(self) -> int:
        return int(self.config.bottleneck_layer)

    @property
    def layout(self) -> RouteALayoutSpec:
        return self.config.layout

    def compression_summary(self) -> Dict[str, object]:
        return {
            "mode": "dynamic_select",
            "bottleneck_layer": self.bottleneck_layer,
            "domain_sizes": list(self.layout.domain_sizes),
            "thresholds": self.gate.thresholds(),
            "temperature": float(self.config.temperature),
            "safety_clamp": {
                "min_kept_history": self.config.safety_clamp.min_kept_history,
                "min_kept_future": self.config.safety_clamp.min_kept_future,
                "max_kept_total": self.config.safety_clamp.max_kept_total,
            },
            "traj_tokens": int(self.layout.traj_tokens),
        }

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        blocks: Sequence[nn.Module],
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        timestep: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        trajectory_head: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        head_t_mod: Optional[torch.Tensor] = None,
        capture_layers: Sequence[int] = (),
        capture_video_layers: Sequence[int] = (),
        use_checkpoint: bool = False,
        temperature: Optional[float] = None,
        safety_clamp: bool = True,
        allow_batch_padding: bool = True,
        physical_shortening: bool = True,
        stats: Optional[CompressionStatsRecorder] = None,
        scene_id: str = "",
        round_index: int = 0,
        sigma: Optional[float] = None,
        difficulty: Optional[float] = None,
        category: Optional[str] = None,
        recovery: bool = True,
    ) -> RouteAForwardOutput:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,S,D], got {tuple(x.shape)}")
        spec = self.layout
        if int(x.shape[1]) != spec.total_tokens:
            raise ValueError(
                f"x sequence length {int(x.shape[1])} != layout total {spec.total_tokens}"
            )
        n_blocks = len(blocks)
        if self.bottleneck_layer >= n_blocks:
            raise ValueError(
                f"bottleneck_layer {self.bottleneck_layer} >= available blocks {n_blocks}"
            )

        capture = {int(v) for v in capture_layers}
        capture_video = {int(v) for v in capture_video_layers}
        invalid = sorted(v for v in capture if not 0 <= v < n_blocks)
        if invalid:
            raise ValueError(f"capture_layers outside [0, {n_blocks - 1}]: {invalid}")
        # Video hidden is only dense before the bottleneck; afterwards it is the
        # sparse set and its length is dynamic, so it cannot be distilled
        # position-wise against a dense teacher.
        invalid = sorted(v for v in capture_video if not 0 <= v < self.bottleneck_layer)
        if invalid:
            raise ValueError(
                "capture_video_layers must be dense layers in "
                f"[0, {self.bottleneck_layer - 1}], got {invalid}"
            )
        captured_action: Dict[int, torch.Tensor] = {}
        captured_video: Dict[int, torch.Tensor] = {}

        def note(block_index: int, hidden: torch.Tensor, *, dense: bool) -> None:
            if int(block_index) in capture:
                captured_action[int(block_index)] = hidden[:, -spec.traj_tokens :]
            if dense and int(block_index) in capture_video:
                captured_video[int(block_index)] = hidden[:, : spec.video_tokens]

        # ---------------------------------------------------- dense front-end
        front = list(blocks[: self.bottleneck_layer])
        for index, block in enumerate(front):
            if use_checkpoint and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs, use_reentrant=False
                )
            else:
                x = block(x, context, t_mod, freqs)
            # Probe layer indices count DiT blocks 0..29, so the hidden *after*
            # block ``i`` is layer ``i``.  Truncation below would silently
            # renumber layers, so capture is keyed on the true block index.
            note(index, x, dense=True)

        video_hidden = x[:, spec.video_slice]
        traj_hidden = x[:, spec.traj_slice]

        if positions is None:
            positions = build_driveva_video_positions(
                spec, device=x.device, dtype=video_hidden.dtype
            )
        else:
            positions = positions.to(device=x.device, dtype=video_hidden.dtype)
            if positions.ndim == 2:
                positions = positions.unsqueeze(0).expand(x.shape[0], -1, -1)

        # -------------------------------------------------------- thresholding
        logits = self.scorer(
            video_hidden,
            action_hidden=traj_hidden,
            timestep=timestep,
            positions=positions,
            token_type=self.token_type.to(x.device),
        )
        gate_out = self.gate(
            logits, temperature=temperature, safety_clamp=safety_clamp
        )
        hard_mask, batch_sync_added = sync_keep_lengths(
            gate_out.hard_mask.clone(), logits, allow_padding=allow_batch_padding
        )
        # Straight-through mask for the *synced* selection.  In the forward pass
        # this evaluates to exactly 0/1 (``hard - soft.detach() + soft`` is
        # ``hard`` at every element), so the deployed hard threshold is what
        # runs.  In the backward pass ``d(mask)/d(logits)`` is the sigmoid
        # derivative, which is the only path by which the flow-matching and
        # distillation losses can reach the scorer -- a plain integer gather is
        # not differentiable and would leave the scorer trained by the sparsity
        # term alone.
        ste_mask = (
            hard_mask.to(gate_out.soft_mask.dtype)
            - gate_out.soft_mask.detach()
            + gate_out.soft_mask
        )
        kept_counts = binding_row_domain_counts(hard_mask, spec.domain_sizes)
        index_matrix = torch.stack(
            [
                hard_mask[row, : spec.video_tokens].nonzero(as_tuple=False).flatten()
                for row in range(x.shape[0])
            ]
        ).to(x.device)
        n_kept_video = int(index_matrix.shape[1])

        kept_rows = select_kept_indices(
            hard_mask,
            video_start=0,
            always_keep=range(spec.video_tokens, spec.total_tokens),
        )
        kept = torch.stack(kept_rows).to(x.device)

        # ----------------------------------------------------- sparse backend
        # Scale the video tokens by the straight-through mask.
        gated_video = video_hidden * ste_mask.unsqueeze(-1)
        x_gated = torch.cat([gated_video, traj_hidden], dim=1)

        if freqs.ndim == 3:
            freqs_full = freqs.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        else:
            freqs_full = freqs
        if t_mod.ndim not in (3, 4):
            raise ValueError(f"unsupported t_mod rank: {t_mod.ndim}")

        if physical_shortening:
            # Inference-equivalent path: the selected tokens are physically
            # gathered, so the backend really runs a short sequence.
            x_run = _gather_rows(x_gated, kept)
            freqs_run = _gather_rows(freqs_full, kept)
            t_mod_run = _gather_rows(t_mod, kept)
        else:
            # Training relaxation (``physical_shortening=False``): the whole
            # sequence runs, with dropped video tokens masked to zero.
            #
            # This exists because of a hard credit-assignment fact about the
            # gather formulation: once a token is dropped, its row is removed
            # from the gathered tensor, so ``d loss / d mask_i == 0`` for every
            # dropped candidate and *only the currently kept tokens are ever
            # trained*.  A token that is wrongly dropped can then never recover,
            # and the selected set can only erode.  Keeping the sequence dense
            # gives every candidate a gradient while the mask still controls the
            # forward values.
            #
            # This is a relaxation, not an inference-equivalent forward: masking
            # a token's residual to zero keeps its (bias-driven) key in the
            # softmax, whereas physical removal deletes it.  Use it for the A1
            # warm-up (and optionally to re-rank candidates after a K collapse),
            # then fine-tune with ``physical_shortening=True`` to close the gap.
            x_run = x_gated
            freqs_run = freqs_full
            t_mod_run = t_mod

        back = list(blocks[self.bottleneck_layer :])
        for index, block in enumerate(back):
            true_index = self.bottleneck_layer + index
            if use_checkpoint and self.training and x_run.requires_grad:
                x_run = torch.utils.checkpoint.checkpoint(
                    block, x_run, context, t_mod_run, freqs_run, use_reentrant=False
                )
            else:
                x_run = block(x_run, context, t_mod_run, freqs_run)
            note(true_index, x_run, dense=False)

        if physical_shortening:
            # Kept rows are sorted, so every kept video row precedes the
            # always-kept trajectory block.
            sparse_video = x_run[:, :n_kept_video]
            sparse_traj = x_run[:, n_kept_video:]
            recovery_index = index_matrix
        else:
            sparse_video = x_run[:, : spec.video_tokens]
            sparse_traj = x_run[:, spec.traj_slice]
            recovery_index = (
                torch.arange(spec.video_tokens, device=x.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )

        traj_pred = None
        if trajectory_head is not None:
            traj_pred = trajectory_head(sparse_traj)

        dense_video_hidden = None
        video_flow = None
        if recovery:
            sparse_positions = positions.gather(
                1, recovery_index.unsqueeze(-1).expand(-1, -1, positions.shape[-1])
            )
            dense_video_hidden = self.recovery(
                sparse_video,
                kept_indices=recovery_index,
                sparse_positions=sparse_positions,
                query_positions=positions,
                batch_size=x.shape[0],
            )
            if head is not None:
                # The Wan head is position-wise over the *video* grid only: the
                # production pipeline strips the trajectory tokens before
                # calling it, so its time embedding is video-length.
                head_time = head_t_mod if head_t_mod is not None else t_mod
                if head_time.shape[1] == spec.total_tokens:
                    head_time = head_time[:, : spec.video_tokens]
                elif head_time.shape[1] != spec.video_tokens:
                    raise ValueError(
                        "head time embedding must be video-length "
                        f"({spec.video_tokens}) or full-length ({spec.total_tokens}), "
                        f"got {int(head_time.shape[1])}"
                    )
                video_flow = head(dense_video_hidden, head_time)

        record = None
        if stats is not None:
            record = stats.record(
                scores=gate_out.scores,
                kept_counts=kept_counts,
                candidate_counts=gate_out.candidate_counts,
                scene_id=scene_id,
                round_index=round_index,
                sigma=sigma,
                thresholds=gate_out.thresholds,
                difficulty=difficulty,
                category=category,
            )

        return RouteAForwardOutput(
            video_flow=video_flow,
            traj_pred=traj_pred,
            dense_video_hidden=dense_video_hidden,
            sparse_video_hidden=sparse_video,
            sparse_traj_hidden=sparse_traj,
            logits=logits,
            gate=gate_out,
            kept_indices=kept,
            kept_video_indices=index_matrix,
            captured_action_hidden=captured_action,
            captured_video_hidden=captured_video,
            round_record=record,
            batch_sync_added=batch_sync_added,
            ste_mask=ste_mask,
            physical_shortening=bool(physical_shortening),
            backend_sequence_length=int(x_run.shape[1]),
        )

    # ------------------------------------------------------- loss integration
    def sparsity_term(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable sparsity pressure on the current batch."""
        return torch.sigmoid(logits.float()).mean()


def parameter_groups(
    module: RouteADynamicSelect,
    *,
    compression_lr: float = 1.0e-4,
    dit_lr: Optional[float] = None,
    weight_decay: float = 0.01,
) -> List[Dict[str, object]]:
    """Optimizer groups with the plan's separate LR for new modules (section 28)."""
    new_params = [p for p in module.parameters() if p.requires_grad]
    groups: List[Dict[str, object]] = [
        {"params": new_params, "lr": float(compression_lr), "weight_decay": float(weight_decay)}
    ]
    return groups

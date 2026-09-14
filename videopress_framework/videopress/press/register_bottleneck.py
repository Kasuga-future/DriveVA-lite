"""Learnable register/query bottleneck for the DriveVA history latent.

Motivation
----------
Every compression route evaluated in this project up to 2026-09-11 was a
*selector*: score the 390 candidate history tokens with a learned or heuristic
importance scalar, keep the top-K, physically prune the rest at
``selector_layer`` and restore the layout before the heads.  Three rounds of
experiments (``reports/dynamic_token_learnability_verdict_20260911.md``) closed
that family:

* the signed causal label "does removing this token help or hurt" is not a
  function of the frozen hidden state that any learner can recover
  (within-scene pairwise 0.38-0.52 over 5 granularities x 4 label
  constructions);
* the only learnable target (geometric plan displacement) still lost to the
  existing gradient teacher on the official 1,920-scene test at every
  compression level;
* a hard 0/1 deletion on a frozen backbone destroys information and asks the
  frozen model to be invariant to a loss it was never trained on.

This module implements the alternative the project owner proposed: stop
*selecting* tokens with a learned importance target and instead learn a small
set of **key tokens** as a latent bottleneck that is trained to predict its own
future.  Concretely, ``K`` learnable queries cross-attend over the 390 candidate
history tokens and emit ``K`` key tokens; the remaining layers consume only
those ``K`` tokens; the backbone is LoRA-finetuned jointly; and the objective is
the existing trajectory loss + the existing video loss + a self-supervised
predictive loss on the key tokens (predict the next latent's content from the
current one).  No importance label appears anywhere.

This mirrors the *register token* mechanism of DrivoR ("Driving on Registers",
CVPR 2026, valeo.ai, arXiv 2601.05083: per-camera learnable registers + a
LoRA-finetuned ViT + separate trajectory/scoring decoders) and the memory
extension MemoryDrivoR (Bosch), transplanted into this repo's DriveVA/Wan layout
instead of a DINOv2 multi-camera ViT.

Design choices
--------------
K
    ``num_key_tokens`` is the compression level.  The 2026-09-11 official test
    measured ``K = 158.30`` for the best selector at PDM 0.898970; the
    pre-registered ablation grid here is ``K in {32, 64, 128}`` with 64 as the
    centre arm, i.e. 3.4x-12x fewer candidate tokens than the selector needed.

Where gradients flow
    The candidate tokens are **not** detached: this bottleneck is meant to be
    trained jointly with LoRA adapters on the frozen 5B backbone, so the
    trajectory/video losses must reach the backbone through the key tokens
    (gradient path: head -> layers 16..29 -> key tokens -> cross-attention ->
    layer-15 candidate hidden states -> LoRA/backbone).  The public
    ``DynamicTokenSelector`` detaches because it is trained against a frozen
    teacher; that is exactly the setting this module leaves behind.

What is detached
    Only the *predictive target* is detached, and it should be **data**, not
    model activations: the clean next-latent VAE latent of the same training
    window.  A detached-but-learned target (hidden states of the same backbone)
    would let the encoder and the target collapse onto each other, which is the
    strongest counter-argument to this design (see the report) and the reason
    ``next_latent_prediction_loss`` calls ``.detach()`` on the target and the
    plan mandates a frozen/offline target plus a next-frame input-ablation
    control.

Deployment at inference
    At ``selector_layer`` (default 15, the same capture point the counterfactual
    mask uses) the pipeline already materialises the 390 candidate hidden states
    ``x[:, 1170:1560]`` for the 1,569-token trajectory-only sequence
    (4 latents x 390 video tokens + 9 trajectory tokens; the candidate range is
    ``last_history`` and the other 1,179 tokens are protected).  The bottleneck
    runs once there and emits ``K`` key tokens.  ``key_token_keep_indices`` /
    ``splice_key_tokens`` then build the short sequence
    ``[0,1170) + K key tokens + [1560,1569)`` and ``restore_key_token_sequence``
    scatters the result back to the original 1,569 positions with zeros in the
    dropped candidate slots - the same gather/scatter contract the existing
    ``HiddenSequencePersistenceController`` uses, so the eventual pipeline hook
    is a drop-in of the key tokens for the selected tokens.  The K key tokens
    need RoPE frequencies; ``key_token_positions`` returns the
    attention-weighted mean source position per key token so the hook can build
    them from the existing ``rope_params`` grid.  ``key_tokens_for_context``
    wires this to the repo's ``TokenContext``/``TokenDomain`` conventions.

How the compute saving is realised
    Sequence length drops from 1,569 to ``1179 + K`` for the 14 compressed
    layers (16..29).  Every sequence-dependent term shrinks: the quadratic
    attention term, but much more importantly the Q/K/V/O projections and the
    FFN, which dominate.  ``press_cost_report`` computes this from shapes; it is
    shape arithmetic, not a kernel measurement.  The bottleneck itself costs one
    extra KV projection of the 390 source tokens plus a K x 390 attention at a
    single layer - about 0.4% of the gross FLOP saving (see ``__main__``).

What this is *not*
    The saving is not a 5B-model saving, a VAE saving, a text-encoder saving or
    a feature-prep saving.  Layers 0..15, the VAE encode/decode, the text
    encoder, the trajectory head, the video decode and the surrounding data
    pipeline are untouched.  Measured end to end, the repo's own
    hidden-sequence compression of -15.55% bought -2.91% wall clock, so a
    -21.8% hidden sequence should be expected to buy roughly -3% to -4%, not
    -22%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DEFAULT_HISTORY_TOKENS",
    "DEFAULT_PROTECTED_TOKENS",
    "DEFAULT_TOTAL_SEQUENCE",
    "DEFAULT_COMPRESSED_LAYERS",
    "DEFAULT_SELECTOR_LAYER",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_FFN_DIM",
    "DEFAULT_TEXT_CONTEXT_LENGTH",
    "BottleneckOutput",
    "CostEstimate",
    "KeyTokenEncoding",
    "NextLatentPredictor",
    "RegisterBottleneck",
    "bottleneck_overhead_flops",
    "key_token_keep_indices",
    "key_token_positions",
    "key_token_diversity_penalty",
    "key_token_scale_penalty",
    "key_token_std",
    "key_tokens_for_context",
    "layer_sequence_cost",
    "next_latent_prediction_loss",
    "position_features",
    "press_cost_report",
    "restore_key_token_sequence",
    "shuffle_key_tokens",
    "splice_key_tokens",
    "trivial_prediction_loss",
    "variance_covariance_penalty",
]

# ---------------------------------------------------------------------------
# Real DriveVA/Wan2.2-TI2V-5B deployment constants (2026-09-11 official runs).
# * trajectory-only inference sequence: 4 latent frames x 390 patch tokens
#   (15 x 26 grid at 480x832) + 9 trajectory tokens = 1,569 tokens;
# * compression domain: ``last_history`` = the final 390 video tokens, other
#   1,179 tokens protected;
# * capture/selector layer 15, hidden-sequence pruning through layer 29 ->
#   14 compressed layers.
DEFAULT_HISTORY_TOKENS = 390
DEFAULT_PROTECTED_TOKENS = 1179
DEFAULT_TOTAL_SEQUENCE = 1569
DEFAULT_COMPRESSED_LAYERS = 14
DEFAULT_SELECTOR_LAYER = 15
DEFAULT_HIDDEN_DIM = 3072
DEFAULT_FFN_DIM = 14336
DEFAULT_TEXT_CONTEXT_LENGTH = 512


def position_features(positions: torch.Tensor) -> torch.Tensor:
    """Second-order Fourier-style features of normalised token positions.

    ``positions`` is ``[..., P]`` with the repo's ``(t, r, c)`` convention
    (``r``/``c`` normalised to ``[0, 1]`` by the capture hook).  A purely linear
    position bias can only express monotone half-plane preferences; adding the
    squares and pairwise products makes a Gaussian-like (quadratic) logit
    representable, which is what "attend to this image region" needs.
    """
    if positions.ndim < 1 or positions.shape[-1] < 1:
        raise ValueError("positions must end in a positive feature dimension")
    p = positions.float()
    squared = p * p
    if p.shape[-1] < 2:
        return torch.cat([p, squared], dim=-1)
    cross = torch.stack(
        [p[..., i] * p[..., j] for i in range(p.shape[-1]) for j in range(i + 1, p.shape[-1])],
        dim=-1,
    )
    return torch.cat([p, squared, cross], dim=-1)


def _as_batch_condition(
    value: Optional[torch.Tensor | Sequence[float]],
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast the repo's scalar-ish condition vectors to ``[B, dim]``.

    ``TokenContext.metadata["selector_ego_state"]`` holds one ``[ego_dim]``
    list for the whole batch (that is what ``LearnedPlanningSelectorScorer``
    reads), while a caller may also pass a fully batched ``[B, dim]`` tensor.
    Both must work, and anything else must fail loudly.
    """
    if value is None:
        return torch.zeros(batch, dim, device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == dim:
        tensor = tensor.reshape(1, dim)
    if tensor.ndim != 2 or tensor.shape[1] != dim:
        raise ValueError(f"condition must be [dim]={dim} or [B,{dim}], got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and batch > 1:
        tensor = tensor.expand(batch, dim)
    if tensor.shape[0] != batch:
        raise ValueError(f"condition batch {tensor.shape[0]} != {batch}")
    return tensor


@dataclass
class BottleneckOutput:
    """Result of one bottleneck forward pass.

    ``key_tokens`` is what the downstream layers consume (``[B, K, D]``, the
    backbone hidden width, so it can be spliced straight into the residual
    stream).  ``attention`` is ``[B, heads, K, N]`` and is kept for diagnostics,
    for the positional read-out below and for the collapse/entropy instruments
    in the experiment plan.  It is *not* an importance score over tokens: it is
    a read-out cost, and no loss in this module ever supervises it.
    """

    key_tokens: torch.Tensor
    attention: torch.Tensor
    key_positions: torch.Tensor

    def metadata(self) -> dict:
        """Flat diagnostics dict, following the repo's metadata convention."""
        with torch.no_grad():
            attn = self.attention.float()
            entropy = -(attn.clamp_min(1e-9).log() * attn).sum(dim=-1).mean()
            top1 = attn.max(dim=-1).values.mean()
            # Fraction of source tokens that some query/head reads above the
            # uniform weight: a collapsing bottleneck leaves most of the frame
            # unread, which this number exposes.
            coverage = (attn.amax(dim=2) > (1.0 / attn.shape[-1])).float().mean()
        return {
            "register_bottleneck_key_tokens": int(self.key_tokens.shape[1]),
            "register_bottleneck_hidden_dim": int(self.key_tokens.shape[-1]),
            "register_bottleneck_attention_entropy": float(entropy.detach()),
            "register_bottleneck_attention_top1": float(top1.detach()),
            "register_bottleneck_source_coverage": float(coverage.detach()),
            "register_bottleneck_key_token_std": (
                float(key_token_std(self.key_tokens).detach())
                if self.key_tokens.shape[0] > 1
                else 0.0
            ),
        }


class RegisterBottleneck(nn.Module):
    """``N`` candidate history tokens -> ``K`` key tokens by cross-attention.

    The module is deliberately *not* an identity at initialisation: a zero
    output projection would make the first optimiser step unable to reach the
    source tokens at all (the gradient into the values is multiplied by the
    output projection), and the invariance/gradient tests here are the
    contract that keeps that property.
    """

    def __init__(
        self,
        *,
        num_key_tokens: int = 64,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        attn_dim: int = 1024,
        num_heads: int = 8,
        ego_dim: int = 2,
        command_dim: int = 3,
        dropout: float = 0.0,
        use_position_bias: bool = True,
        position_bias_dim: Optional[int] = None,
        key_bias_init: str = "zero",
        condition_dim: int = 256,
        cond_rank: int = 32,
        value_norm: bool = False,
    ) -> None:
        super().__init__()
        if int(num_key_tokens) <= 0:
            raise ValueError("num_key_tokens must be positive")
        if int(hidden_dim) <= 0 or int(attn_dim) <= 0:
            raise ValueError("hidden_dim and attn_dim must be positive")
        if int(attn_dim) % int(num_heads) != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if key_bias_init not in {"zero", "normal"}:
            raise ValueError("key_bias_init must be 'zero' or 'normal'")

        self.num_key_tokens = int(num_key_tokens)
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attn_dim // self.num_heads
        self.use_position_bias = bool(use_position_bias)
        self.value_norm = bool(value_norm)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        # Learned register/query bank (DrivoR-style): the queries carry no
        # content from the scene, they are the read-out pattern.
        self.queries = nn.Parameter(torch.randn(self.num_key_tokens, self.attn_dim) * 0.02)
        self.key_bias = nn.Parameter(torch.zeros(self.num_key_tokens, self.hidden_dim))
        if key_bias_init == "normal":
            nn.init.normal_(self.key_bias, std=0.02)

        self.kv_norm = nn.LayerNorm(self.hidden_dim)
        self.q_norm = nn.LayerNorm(self.attn_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(float(dropout))

        # Conditioning follows the repo's selector convention: ego state +
        # driving command, plus the diffusion phase.  The modulation is
        # low-rank (rank ``cond_rank`` coefficients per key token, shared
        # ``cond_rank -> attn_dim`` up-projection) because a dense
        # ``K * attn_dim`` modulation would cost ~34M parameters on its own.
        # The last layer of each MLP is zero-initialised so a fresh module is
        # condition agnostic and older checkpoints stay loadable (same trick as
        # ``DynamicTokenSelector.context_scoring``).
        self.ego_dim = int(ego_dim)
        self.command_dim = int(command_dim)
        self.cond_rank = int(cond_rank)
        if self.cond_rank <= 0:
            raise ValueError("cond_rank must be positive")
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.ego_dim + self.command_dim, int(condition_dim)),
            nn.GELU(),
            nn.Linear(int(condition_dim), self.num_key_tokens * self.cond_rank),
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(5, int(condition_dim)),
            nn.GELU(),
            nn.Linear(int(condition_dim), self.num_key_tokens * self.cond_rank),
        )
        self.condition_up = nn.Linear(self.cond_rank, self.attn_dim, bias=False)
        nn.init.zeros_(self.condition_mlp[-1].weight)
        nn.init.zeros_(self.condition_mlp[-1].bias)
        nn.init.zeros_(self.timestep_mlp[-1].weight)
        nn.init.zeros_(self.timestep_mlp[-1].bias)

        if self.use_position_bias:
            feat_dim = int(position_bias_dim) if position_bias_dim is not None else 3 * 3
            self.position_bias_dim = feat_dim
            # [heads, K, feat] read-out weights; zero init = content-only
            # attention at step 0, so permutation invariance is exact until the
            # optimiser decides spatial selectivity is worth having.
            self.position_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_key_tokens, feat_dim))
        else:
            self.position_bias_dim = 0
            self.register_parameter("position_bias", None)

    # -- helpers ---------------------------------------------------------
    def _condition_bias(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        ego_state: Optional[torch.Tensor],
        command: Optional[torch.Tensor],
        timestep: Optional[torch.Tensor | float],
    ) -> torch.Tensor:
        ego = _as_batch_condition(ego_state, batch, self.ego_dim, device, dtype)
        cmd = _as_batch_condition(command, batch, self.command_dim, device, dtype)
        bias = self.condition_mlp(torch.cat([ego, cmd], dim=-1))
        phase = (
            torch.zeros(batch, device=device, dtype=dtype)
            if timestep is None
            else torch.as_tensor(timestep, device=device, dtype=dtype).reshape(-1)
        )
        if phase.numel() == 1:
            phase = phase.expand(batch)
        if phase.numel() != batch:
            raise ValueError(f"timestep must be scalar or length {batch}, got {phase.numel()}")
        phase = phase / 1000.0
        time_features = torch.stack(
            [
                phase,
                torch.sin(math.pi * phase),
                torch.cos(math.pi * phase),
                torch.sin(2.0 * math.pi * phase),
                torch.cos(2.0 * math.pi * phase),
            ],
            dim=-1,
        )
        bias = bias + self.timestep_mlp(time_features)
        bias = bias.view(batch, self.num_key_tokens, self.cond_rank)
        return self.condition_up(bias)

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        ego_state: Optional[torch.Tensor] = None,
        command: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor | float] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BottleneckOutput:
        """Cross-attend ``K`` queries over ``N`` candidate history tokens.

        ``attention_mask`` is an optional boolean ``[B, N]`` (or ``[N]``) mask
        whose ``False`` entries are excluded from attention.  It exists so the
        registered leakage control (ablate the *next* latent's tokens at the
        input and re-measure the predictive loss) can be run with the same
        module instead of a second code path.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,D], got {tuple(tokens.shape)}")
        batch, num_tokens, dim = tokens.shape
        if int(dim) != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {int(dim)}")
        if num_tokens <= 0:
            raise ValueError("tokens must contain at least one candidate token")

        param_dtype = next(self.parameters()).dtype
        device = tokens.device
        x = tokens.to(dtype=param_dtype)

        # Keys are normalised for routing; values keep the residual-stream
        # magnitude unless ``value_norm`` is set (see the class docstring).
        k = self.k_proj(self.kv_norm(x)).view(batch, num_tokens, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)
        values = self.kv_norm(x) if self.value_norm else x
        v = self.v_proj(values).view(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + self._condition_bias(
            batch, device, param_dtype, ego_state, command, timestep
        )
        q = self.q_norm(queries).view(batch, self.num_key_tokens, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.use_position_bias:
            if positions is None:
                raise ValueError("positions are required when use_position_bias=True")
            pos = torch.as_tensor(positions, device=device, dtype=param_dtype)
            if pos.ndim == 2:
                pos = pos.unsqueeze(0).expand(batch, -1, -1)
            if pos.shape[0] != batch or pos.shape[1] != num_tokens:
                raise ValueError(f"positions must be [B,N,P] matching tokens, got {tuple(pos.shape)}")
            features = position_features(pos)
            if features.shape[-1] != self.position_bias_dim:
                raise ValueError(
                    f"position feature dim {features.shape[-1]} != configured {self.position_bias_dim}"
                )
            # [H,K,F] x [B,N,F] -> [B,H,K,N]
            logits = logits + torch.einsum("hkf,bnf->bhkn", self.position_bias, features)

        if attention_mask is not None:
            mask = torch.as_tensor(attention_mask, device=device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch, -1)
            if mask.shape != (batch, num_tokens):
                raise ValueError(
                    f"attention_mask must be [B,N]={ (batch, num_tokens) }, got {tuple(mask.shape)}"
                )
            logits = logits.masked_fill(~mask.bool().view(batch, 1, 1, num_tokens), float("-inf"))

        attention = F.softmax(logits, dim=-1)
        attention = self.attn_dropout(attention)
        read = torch.matmul(attention, v)  # [B,H,K,hd]
        read = read.transpose(1, 2).reshape(batch, self.num_key_tokens, self.attn_dim)

        key_tokens = self.out_proj(read) + self.key_bias
        key_positions = key_token_positions(attention, positions, batch=batch, device=device)
        return BottleneckOutput(key_tokens=key_tokens, attention=attention, key_positions=key_positions)

    def extra_repr(self) -> str:
        return (
            f"num_key_tokens={self.num_key_tokens}, hidden_dim={self.hidden_dim}, "
            f"attn_dim={self.attn_dim}, heads={self.num_heads}, "
            f"position_bias={self.use_position_bias}"
        )

    def describe(self) -> dict[str, Any]:
        """Plugin-style description, matching ``TokenScorer.describe``."""
        return {
            "name": "register_bottleneck",
            "mechanism": "learned_query_cross_attention",
            "num_key_tokens": self.num_key_tokens,
            "hidden_dim": self.hidden_dim,
            "attn_dim": self.attn_dim,
            "num_heads": self.num_heads,
            "use_position_bias": self.use_position_bias,
            "requires_importance_label": False,
            "trainable_parameters": int(sum(p.numel() for p in self.parameters())),
        }


def key_token_positions(
    attention: torch.Tensor,
    positions: Optional[torch.Tensor],
    *,
    batch: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Attention-weighted mean source position per key token (``[B, K, P]``).

    The compressed sequence needs RoPE frequencies for the K new tokens.  Using
    the read-out centroid keeps the key token where the content it summarises
    actually is, which is what the backbone's relative-position machinery
    expects; when no positions are supplied the centroid is identically zero.
    """
    if attention.ndim != 4:
        raise ValueError("attention must be [B,H,K,N]")
    b = int(attention.shape[0] if batch is None else batch)
    dev = attention.device if device is None else device
    if positions is None:
        return torch.zeros(b, attention.shape[2], 3, device=dev, dtype=attention.dtype)
    pos = torch.as_tensor(positions, device=dev, dtype=attention.dtype)
    if pos.ndim == 2:
        pos = pos.unsqueeze(0).expand(b, -1, -1)
    if pos.shape[0] != b or pos.shape[1] != attention.shape[-1]:
        raise ValueError("positions must match [B, N] of the attention map")
    weights = attention.mean(dim=1)  # [B,K,N]
    return torch.matmul(weights, pos)


class _PredictorBlock(nn.Module):
    """Cross-attention with an explicit *identity* path from slots to key tokens.

    Plain content-based cross-attention cannot route "slot ``s`` must read key
    token ``s``": key tokens carry no canonical identity, so with learned
    queries and sample-dependent keys the routing is permutation-ambiguous and
    the predictive loss stalls at the constant-predictor floor (measured while
    building this module).  Two additive logit biases fix that:

    * an **index bias** - the learned dot product of a per-slot embedding with
      a per-key-slot embedding bank - which makes the learned slot <-> register
      assignment expressible regardless of content;
    * an optional **geometric bias** from the (slot position, key position)
      pair, which is the inductive bias the real pipeline uses (key positions
      are the read-out centroids, target slots are the next latent's patch
      positions).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        *,
        num_slots: int,
        index_bank_size: int = 512,
        index_dim: int = 32,
        relation_hidden: int = 64,
        pos_dim: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.norm_q = nn.LayerNorm(self.hidden_dim)
        self.norm_kv = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.norm_mlp = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.index_bank_size = int(index_bank_size)
        if int(num_slots) > self.index_bank_size:
            raise ValueError("num_slots must not exceed index_bank_size")
        self.slot_index = nn.Parameter(torch.randn(int(num_slots), int(index_dim)) * 0.02)
        self.key_index = nn.Parameter(torch.randn(self.index_bank_size, int(index_dim)) * 0.02)
        self.head_gain = nn.Parameter(torch.ones(self.num_heads))

        self.pos_dim = int(pos_dim)
        self.relation = nn.Sequential(
            nn.Linear(4 * self.pos_dim, int(relation_hidden)),
            nn.GELU(),
            nn.Linear(int(relation_hidden), self.num_heads),
        )
        nn.init.zeros_(self.relation[-1].weight)
        nn.init.zeros_(self.relation[-1].bias)

    def _relation_features(
        self, slot_positions: torch.Tensor, key_positions: torch.Tensor
    ) -> torch.Tensor:
        slot = slot_positions[..., : self.pos_dim].unsqueeze(2)
        key = key_positions[..., : self.pos_dim].unsqueeze(1)
        delta = slot - key
        return torch.cat([slot.expand_as(delta), key.expand_as(delta), delta, delta * delta], dim=-1)

    def forward(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        *,
        slot_positions: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, num_slots, _ = slots.shape
        num_keys = keys.shape[1]
        if num_keys > self.index_bank_size:
            raise ValueError(
                f"predictor index bank holds {self.index_bank_size} key slots, got {num_keys}"
            )
        q = self.q_proj(self.norm_q(slots)).view(batch, num_slots, self.num_heads, self.head_dim)
        # Normalise for *routing* only: a LayerNorm on the value path makes the
        # read-out scale-invariant, which caps the predictive loss at the
        # constant-predictor floor for any target whose magnitude matters.
        k = self.k_proj(self.norm_kv(keys)).view(batch, num_keys, self.num_heads, self.head_dim)
        v = self.v_proj(keys).view(batch, num_keys, self.num_heads, self.head_dim)

        logits = torch.einsum("bshd,bkhd->bhsk", q, k) * self.scale
        # [S,K] identity affinity, one learned temperature per head.
        affinity = self.slot_index @ self.key_index[:num_keys].t()
        logits = logits + (affinity.unsqueeze(0) * self.head_gain.view(-1, 1, 1))
        if slot_positions is not None and key_positions is not None:
            features = self._relation_features(
                slot_positions.to(logits.dtype), key_positions.to(logits.dtype)
            )
            if features.shape[1] != num_slots or features.shape[2] != num_keys:
                raise ValueError("slot_positions/key_positions must match the slot/key counts")
            relation = self.relation(features).permute(0, 3, 1, 2)  # [B,H,S,K]
            logits = logits + relation

        attention = F.softmax(logits, dim=-1)
        read = torch.einsum("bhsk,bkhd->bshd", attention, v).reshape(batch, num_slots, self.hidden_dim)
        slots = slots + self.dropout(self.out_proj(read))
        return slots + self.mlp(self.norm_mlp(slots))


class NextLatentPredictor(nn.Module):
    """Predict the next latent's content from the K key tokens alone.

    ``S`` learned slot queries read the key tokens and regress a target vector
    per slot.  Slots may be given explicit positions (the next latent's patch
    grid) and the key tokens their read-out centroids, in which case routing is
    geometric; without positions routing is learned per (slot, register) index.
    The head is intentionally tiny and separate from the backbone: it is a
    *probe* on the bottleneck, so all of the predictive pressure has to pass
    through the K key tokens.
    """

    def __init__(
        self,
        *,
        key_dim: int = DEFAULT_HIDDEN_DIM,
        target_dim: int = 48,
        num_slots: int = 16,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        index_bank_size: int = 512,
        index_dim: int = 32,
        relation_hidden: int = 64,
        pos_dim: int = 3,
        output_norm: bool = False,
    ) -> None:
        super().__init__()
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_slots = int(num_slots)
        self.target_dim = int(target_dim)
        self.key_proj = nn.Linear(int(key_dim), int(hidden_dim), bias=False)
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)
        self.blocks = nn.ModuleList(
            [
                _PredictorBlock(
                    int(hidden_dim),
                    int(num_heads),
                    float(dropout),
                    num_slots=self.num_slots,
                    index_bank_size=int(index_bank_size),
                    index_dim=int(index_dim),
                    relation_hidden=int(relation_hidden),
                    pos_dim=int(pos_dim),
                )
                for _ in range(int(num_layers))
            ]
        )
        # No output LayerNorm by default: a normalised slot representation is
        # scale-invariant, so a *linear* next-latent map (the synthetic gate, and
        # any target whose magnitude carries information) becomes
        # unrepresentable and the loss stalls above the constant predictor.
        self.norm_out = nn.LayerNorm(int(hidden_dim)) if output_norm else nn.Identity()
        self.head = nn.Linear(int(hidden_dim), self.target_dim)

    def forward(
        self,
        key_tokens: torch.Tensor,
        *,
        key_positions: Optional[torch.Tensor] = None,
        slot_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_tokens.ndim != 3:
            raise ValueError(f"key_tokens must be [B,K,D], got {tuple(key_tokens.shape)}")
        param_dtype = next(self.parameters()).dtype
        keys = self.key_proj(key_tokens.to(dtype=param_dtype))
        slots = self.slot_queries.unsqueeze(0).expand(keys.shape[0], -1, -1)
        for block in self.blocks:
            slots = block(
                slots, keys, slot_positions=slot_positions, key_positions=key_positions
            )
        return self.head(self.norm_out(slots))


# ---------------------------------------------------------------------------
# Objectives.  Nothing here consumes an importance label.
# ---------------------------------------------------------------------------
def next_latent_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "normalized_mse",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Self-supervised predictive loss on the key tokens.

    ``prediction`` is ``[..., T]`` from ``NextLatentPredictor``; ``target`` is
    the *content of the next latent*, detached and expected to be data (a clean
    VAE latent) rather than a learned activation.  ``normalized_mse`` divides by
    the mean squared target norm, so the oracle constant predictor scores ~1.0
    and the number is comparable across scenes and timesteps; that is what the
    synthetic learnability gate in the tests uses as its floor.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"prediction {tuple(prediction.shape)} != target {tuple(target.shape)}")
    detached = target.detach().to(dtype=prediction.dtype, device=prediction.device)
    if mode == "normalized_mse":
        denom = detached.pow(2).sum(dim=-1).mean().clamp_min(float(eps))
        return (prediction - detached).pow(2).sum(dim=-1).mean() / denom
    if mode == "mse":
        return F.mse_loss(prediction, detached)
    if mode == "cosine":
        return (1.0 - F.cosine_similarity(prediction, detached, dim=-1)).mean()
    raise ValueError(f"unsupported predictive loss mode: {mode}")


def trivial_prediction_loss(target: torch.Tensor, *, mode: str = "normalized_mse") -> torch.Tensor:
    """Loss of the best *constant* predictor (the oracle batch mean).

    This is the floor the synthetic gate must beat by a wide margin: a
    bottleneck that only learns the marginal target distribution scores here.
    """
    if target.ndim < 2:
        raise ValueError("target must have at least two dimensions")
    mean = target.detach().mean(dim=0, keepdim=True).expand_as(target)
    return next_latent_prediction_loss(mean, target, mode=mode)


def key_token_std(key_tokens: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Mean per-dimension standard deviation over the batch (collapse alarm)."""
    if key_tokens.ndim != 3:
        raise ValueError("key_tokens must be [B,K,D]")
    if key_tokens.shape[0] < 2:
        raise ValueError("key_token_std needs at least two batch elements")
    return key_tokens.float().std(dim=0, unbiased=False).clamp_min(float(eps)).mean()


def key_token_diversity_penalty(key_tokens: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample squared off-diagonal cosine similarity of the K key tokens.

    Batch-free by construction, which matters here: the DriveVA DiT trains with
    a batch of one window per rank, so any regulariser that relies on
    across-scene statistics is unusable in the real recipe.  This term is the
    one that still works at batch 1 - it forbids the degenerate solution in
    which all K slots emit the same vector (and therefore carry the information
    of a single token).
    """
    if key_tokens.ndim != 3:
        raise ValueError("key_tokens must be [B,K,D]")
    if key_tokens.shape[1] < 2:
        return key_tokens.new_zeros(())
    flat = key_tokens.float()
    normed = flat / flat.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    gram = torch.matmul(normed, normed.transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    off_diagonal = gram * (1.0 - eye)
    per_sample = off_diagonal.pow(2).sum(dim=(1, 2)) / (gram.shape[-1] * (gram.shape[-1] - 1))
    return per_sample.mean()


def key_token_scale_penalty(
    key_tokens: torch.Tensor, reference_tokens: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Relative RMS mismatch between the key tokens and the tokens they replace.

    A bottleneck can lower the predictive loss by shrinking its output towards
    zero (the target mean).  Matching the RMS of the replaced candidate tokens
    is a cheap, batch-free instrument against that shortcut.
    """
    if key_tokens.ndim != 3 or reference_tokens.ndim != 3:
        raise ValueError("key_tokens and reference_tokens must be [B,N,D]")
    key_rms = key_tokens.float().pow(2).mean().sqrt().clamp_min(float(eps))
    ref_rms = reference_tokens.detach().float().pow(2).mean().sqrt().clamp_min(float(eps))
    return (key_rms - ref_rms).abs() / ref_rms


def variance_covariance_penalty(
    key_tokens: torch.Tensor,
    *,
    reference_tokens: Optional[torch.Tensor] = None,
    variance_target: float = 1.0,
    variance_weight: float = 0.0,
    covariance_weight: float = 0.0,
    diversity_weight: float = 0.1,
    scale_weight: float = 0.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Anti-collapse regulariser bundle for the key tokens.

    Three terms, in decreasing order of usefulness for *this* repo:

    ``diversity`` (default on)
        batch-free pairwise cosine penalty between the K slots - the failure
        mode "all registers collapse into one" is invisible to the trajectory
        loss but is exactly what makes K=128 pointless.
    ``variance`` / ``covariance`` (default off)
        VICReg-style hinge on the per-dimension standard deviation of the
        pooled ``[B, D]`` summary and a mean-squared off-diagonal correlation
        penalty.  Both are *batch* statistics: with the DiT's batch-of-one
        windows per rank they are meaningless, so they are opt-in and must only
        be enabled once the training loop accumulates real scene batches.
    ``scale`` (default off)
        optional RMS match against ``reference_tokens`` (the replaced candidate
        tokens), against the "shrink towards the target mean" shortcut.
    """
    if key_tokens.ndim != 3:
        raise ValueError("key_tokens must be [B,K,D]")
    total = key_tokens.new_zeros(())
    if float(diversity_weight) > 0.0:
        total = total + float(diversity_weight) * key_token_diversity_penalty(key_tokens)
    if float(scale_weight) > 0.0:
        if reference_tokens is None:
            raise ValueError("scale_weight > 0 requires reference_tokens")
        total = total + float(scale_weight) * key_token_scale_penalty(key_tokens, reference_tokens)
    if float(variance_weight) > 0.0 or float(covariance_weight) > 0.0:
        if key_tokens.shape[0] < 2:
            raise ValueError(
                "variance/covariance terms need at least two batch rows; "
                "leave them disabled in the batch-1-per-rank recipe"
            )
        pooled = key_tokens.float().mean(dim=1)
        std = torch.sqrt(pooled.var(dim=0, unbiased=False) + float(eps))
        if float(variance_weight) > 0.0:
            total = total + float(variance_weight) * F.relu(float(variance_target) - std).mean()
        if float(covariance_weight) > 0.0:
            centered = pooled - pooled.mean(dim=0, keepdim=True)
            denom = max(pooled.shape[0] - 1, 1)
            cov = (centered.t() @ centered) / denom
            scale = std.unsqueeze(0) * std.unsqueeze(1)
            corr = cov / scale.clamp_min(float(eps))
            off_diagonal = corr - torch.diag(torch.diagonal(corr))
            total = total + float(covariance_weight) * off_diagonal.pow(2).mean()
    return total


# ---------------------------------------------------------------------------
# Deployment helpers: the short-sequence contract.
# ---------------------------------------------------------------------------
def key_token_keep_indices(
    sequence_length: int,
    candidate_start: int,
    candidate_end: int,
    num_key_tokens: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Index map ``full sequence -> [protected prefix, K key slots, protected suffix]``.

    Returns a 1-D ``LongTensor`` of length ``sequence_length - (candidate_end -
    candidate_start) + num_key_tokens``.  The K key slots are emitted *in place*
    of the candidate block (the first K of the candidate positions), which is
    the layout the hidden-sequence controller already builds for selected
    tokens; the pipeline hook only has to overwrite those rows with the key
    tokens.
    """
    if not 0 <= candidate_start < candidate_end <= int(sequence_length):
        raise ValueError("candidate range must be a non-empty sub-range of the sequence")
    if int(num_key_tokens) <= 0:
        raise ValueError("num_key_tokens must be positive")
    prefix = torch.arange(0, int(candidate_start), device=device)
    slots = torch.arange(int(candidate_start), int(candidate_start) + int(num_key_tokens), device=device)
    suffix = torch.arange(int(candidate_end), int(sequence_length), device=device)
    return torch.cat([prefix, slots, suffix])


def splice_key_tokens(
    sequence: torch.Tensor,
    key_tokens: torch.Tensor,
    candidate_start: int,
    candidate_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the candidate block by the K key tokens.

    Returns ``(short_sequence, keep_indices)``.  ``short_sequence`` is
    ``[B, L - (cand_end - cand_start) + K, D]``; ``keep_indices`` are the
    *original* positions of its rows, so RoPE frequencies and the per-token
    timestep modulation can be gathered with the same index map (that is what
    ``HiddenSequencePersistenceController._gather_sequence`` does today).
    """
    if sequence.ndim != 3:
        raise ValueError("sequence must be [B,L,D]")
    if key_tokens.ndim != 3:
        raise ValueError("key_tokens must be [B,K,D]")
    if sequence.shape[0] != key_tokens.shape[0] or sequence.shape[2] != key_tokens.shape[2]:
        raise ValueError("sequence and key_tokens must share batch and hidden dimension")
    if not 0 <= candidate_start < candidate_end <= sequence.shape[1]:
        raise ValueError("candidate range must be a non-empty sub-range of the sequence")
    keep = key_token_keep_indices(
        sequence.shape[1], candidate_start, candidate_end, key_tokens.shape[1], device=sequence.device
    )
    short = torch.cat(
        [sequence[:, :candidate_start], key_tokens, sequence[:, candidate_end:]], dim=1
    )
    return short, keep


def restore_key_token_sequence(
    short_sequence: torch.Tensor,
    *,
    original_length: int,
    candidate_start: int,
    candidate_end: int,
    num_key_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Scatter the compressed sequence back to the full layout.

    Dropped candidate positions are filled with zeros, exactly like the current
    hidden-sequence controller: those positions are conditioned history, are not
    generation targets, and the surrounding denoising loop overwrites them with
    the clean history latent.
    """
    if short_sequence.ndim != 3:
        raise ValueError("short_sequence must be [B,L',D]")
    batch, length, dim = short_sequence.shape
    suffix_length = int(original_length) - int(candidate_end)
    inferred = length - int(candidate_start) - suffix_length
    if num_key_tokens is not None and int(num_key_tokens) != inferred:
        raise ValueError(
            f"declared num_key_tokens={int(num_key_tokens)} contradicts the compressed length {length}"
        )
    if inferred <= 0:
        raise ValueError("compressed sequence is too short for the declared layout")
    keep = key_token_keep_indices(
        int(original_length), int(candidate_start), int(candidate_end), inferred,
        device=short_sequence.device,
    )
    restored = short_sequence.new_zeros((batch, int(original_length), dim))
    restored.scatter_(1, keep.unsqueeze(-1).expand_as(short_sequence), short_sequence)
    return restored


def shuffle_key_tokens(key_tokens: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Permute key tokens across the batch (registered "is the planner using them" control).

    ``key_tokens`` is ``[B,K,D]``; the returned tensor keeps scene *b*'s own K
    tokens but takes them from a random other scene.  If the official-test PDM
    is unchanged under this control, the backbone is ignoring the bottleneck and
    any gain came from the LoRA capacity instead.
    """
    if key_tokens.ndim != 3:
        raise ValueError("key_tokens must be [B,K,D]")
    if key_tokens.shape[0] < 2:
        return key_tokens.clone()
    order = torch.randperm(key_tokens.shape[0], generator=generator, device="cpu").to(key_tokens.device)
    return key_tokens.index_select(0, order)


@dataclass
class KeyTokenEncoding:
    """Bottleneck output plus the layout facts a pipeline hook needs."""

    output: BottleneckOutput
    candidate_start: int
    candidate_end: int
    positions: torch.Tensor

    @property
    def key_tokens(self) -> torch.Tensor:
        return self.output.key_tokens

    @property
    def attention(self) -> torch.Tensor:
        return self.output.attention

    def metadata(self) -> dict:
        data = self.output.metadata()
        data.update(
            {
                "register_bottleneck_candidate_start": int(self.candidate_start),
                "register_bottleneck_candidate_end": int(self.candidate_end),
                "register_bottleneck_compressed_gain": int(
                    self.candidate_end - self.candidate_start - self.output.key_tokens.shape[1]
                ),
            }
        )
        return data


def last_history_positions(
    video_h: int,
    video_w: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[h*w, 3]`` normalised ``(t, r, c)`` positions, as the capture hook builds them."""
    if int(video_h) <= 0 or int(video_w) <= 0:
        raise ValueError("video_h and video_w must be positive")
    rows = torch.arange(int(video_h), device=device).repeat_interleave(int(video_w))
    cols = torch.arange(int(video_w), device=device).repeat(int(video_h))
    ones = torch.ones_like(rows)
    denom = torch.tensor(
        [1.0, max(int(video_h) - 1, 1), max(int(video_w) - 1, 1)], device=device, dtype=dtype
    )
    return torch.stack([ones, rows, cols], dim=-1).to(dtype) / denom


def key_tokens_for_context(
    module: RegisterBottleneck,
    ctx: Any,
    *,
    predictor: Optional[NextLatentPredictor] = None,
) -> tuple[KeyTokenEncoding, Optional[torch.Tensor]]:
    """Encode a repo ``TokenContext`` into key tokens (+ optional prediction).

    Uses the project's own conventions: candidates come from
    ``ctx.candidate_tokens()`` (the ``last_history`` domain), positions from
    ``layout.video_h``/``video_w``, and ego state / command / timestep from
    ``ctx.metadata`` exactly as ``LearnedPlanningSelectorScorer`` reads them.
    """
    layout = ctx.layout
    candidate = ctx.candidate_tokens()
    positions = last_history_positions(layout.video_h, layout.video_w, device=candidate.device,
                                       dtype=candidate.dtype)
    if candidate.shape[1] != positions.shape[0]:
        raise ValueError(
            f"candidate token count {candidate.shape[1]} != video_h*video_w {positions.shape[0]}"
        )
    metadata = getattr(ctx, "metadata", {}) or {}
    ego = metadata.get("selector_ego_state")
    command = metadata.get("selector_command")
    output = module(
        candidate,
        positions=positions,
        ego_state=None if ego is None else torch.as_tensor(ego, device=candidate.device).reshape(1, -1),
        command=None if command is None else torch.as_tensor(command, device=candidate.device).reshape(1, -1),
        timestep=getattr(ctx, "timestep", None),
    )
    prediction = None if predictor is None else predictor(output.key_tokens)
    start = int(layout.history_video.end - layout.tokens_per_latent)
    encoding = KeyTokenEncoding(
        output=output, candidate_start=start, candidate_end=int(layout.history_video.end),
        positions=positions,
    )
    return encoding, prediction


# ---------------------------------------------------------------------------
# Cost model (shape arithmetic).
# ---------------------------------------------------------------------------
@dataclass
class CostEstimate:
    """Multiply-accumulate FLOPs (2 per MAC) and K/V bytes for one layer or a stack."""

    layers: int
    sequence_length: int
    qkv_output_flops: float
    attention_flops: float
    ffn_flops: float
    text_cross_flops: float
    kv_bytes: int

    @property
    def total_flops(self) -> float:
        return (
            self.qkv_output_flops + self.attention_flops + self.ffn_flops + self.text_cross_flops
        )

    def as_dict(self) -> dict:
        return {
            "layers": int(self.layers),
            "sequence_length": int(self.sequence_length),
            "total_flops": float(self.total_flops),
            "attention_flops": float(self.attention_flops),
            "attention_share": float(self.attention_flops / max(self.total_flops, 1.0)),
            "kv_bytes": int(self.kv_bytes),
        }


def layer_sequence_cost(
    sequence_length: int,
    *,
    layers: int = DEFAULT_COMPRESSED_LAYERS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ffn_dim: int = DEFAULT_FFN_DIM,
    text_context_length: int = DEFAULT_TEXT_CONTEXT_LENGTH,
    dtype_bytes: int = 2,
) -> CostEstimate:
    """Sequence-dependent cost of ``layers`` Wan DiT blocks at a given length.

    Per layer and token: ``q,k,v,o`` projections (``8 L D^2``), attention
    scores + AV (``4 L^2 D``), FFN (``4 L D F``) and the text cross-attention
    Q/O plus scores (``4 L D^2 + 4 L S D``).  Set ``text_context_length=0`` to
    exclude the last term.  Only the *sequence-dependent* terms are counted;
    the modulation/AdaLN and norm costs are length-independent.
    """
    if int(sequence_length) <= 0 or int(layers) <= 0:
        raise ValueError("sequence_length and layers must be positive")
    length = int(sequence_length)
    d = int(hidden_dim)
    f = int(ffn_dim)
    s = int(text_context_length)
    qkv_output = float(8 * length * d * d) * int(layers)
    attention = float(4 * length * length * d) * int(layers)
    ffn = float(4 * length * d * f) * int(layers)
    text = float((4 * length * d * d) + (4 * length * s * d)) * int(layers)
    return CostEstimate(
        layers=int(layers),
        sequence_length=length,
        qkv_output_flops=qkv_output,
        attention_flops=attention,
        ffn_flops=ffn,
        text_cross_flops=text,
        kv_bytes=int(2 * length * d * int(dtype_bytes)) * int(layers),
    )


def bottleneck_overhead_flops(
    num_source_tokens: int,
    num_key_tokens: int,
    *,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    attn_dim: int = 1024,
    num_heads: int = 8,
) -> float:
    """One-shot cost of the cross-attention read-out at the capture layer."""
    n = int(num_source_tokens)
    k = int(num_key_tokens)
    d = int(hidden_dim)
    da = int(attn_dim)
    if min(n, k, d, da) <= 0:
        raise ValueError("dimensions must be positive")
    kv_proj = 2 * (2 * n * d * da)      # k + v projections of the source tokens
    q_proj = 2 * (2 * k * d * da)       # query conditioning / read-out path
    attn = 2 * (2 * k * n * da)         # scores + AV
    out_proj = 2 * k * da * d
    return float(kv_proj + q_proj + attn + out_proj)


def press_cost_report(
    *,
    num_source_tokens: int = DEFAULT_HISTORY_TOKENS,
    protected_tokens: int = DEFAULT_PROTECTED_TOKENS,
    key_token_options: Sequence[int] = (32, 64, 128),
    layers: int = DEFAULT_COMPRESSED_LAYERS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ffn_dim: int = DEFAULT_FFN_DIM,
    attn_dim: int = 1024,
    num_heads: int = 8,
    num_heads_full: int = 24,
    text_context_length: int = DEFAULT_TEXT_CONTEXT_LENGTH,
    dtype_bytes: int = 2,
    diffusion_steps: int = 3,
) -> dict:
    """Full cost table: baseline vs K key tokens over the compressed layers."""
    base_length = int(protected_tokens) + int(num_source_tokens)
    baseline = layer_sequence_cost(
        base_length, layers=layers, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
        text_context_length=text_context_length, dtype_bytes=dtype_bytes,
    )
    arms = []
    for k in key_token_options:
        length = int(protected_tokens) + int(k)
        cost = layer_sequence_cost(
            length, layers=layers, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
            text_context_length=text_context_length, dtype_bytes=dtype_bytes,
        )
        overhead = bottleneck_overhead_flops(
            num_source_tokens, k, hidden_dim=hidden_dim, attn_dim=attn_dim, num_heads=num_heads
        )
        saved = baseline.total_flops - cost.total_flops
        arms.append(
            {
                "key_tokens": int(k),
                "sequence_length": length,
                "sequence_ratio": float(length / base_length),
                "total_flops": float(cost.total_flops),
                "flops_saved_per_forward": float(saved),
                "flops_saved_fraction": float(saved / baseline.total_flops),
                "attention_flops_saved_fraction": float(
                    1.0 - cost.attention_flops / baseline.attention_flops
                ),
                "kv_bytes": int(cost.kv_bytes),
                "kv_bytes_saved": int(baseline.kv_bytes - cost.kv_bytes),
                "kv_bytes_ratio": float(cost.kv_bytes / baseline.kv_bytes),
                "bottleneck_overhead_flops": float(overhead),
                "bottleneck_overhead_share_of_saving": float(overhead / max(saved, 1.0)),
                "flops_saved_per_forward_all_steps": float(saved * int(diffusion_steps)),
            }
        )
    return {
        "baseline": {
            **baseline.as_dict(),
            "candidate_tokens": int(num_source_tokens),
            "protected_tokens": int(protected_tokens),
            "kv_bytes_per_layer": int(baseline.kv_bytes // max(int(layers), 1)),
        },
        "arms": arms,
        "notes": [
            "FLOPs are multiply-accumulate x2 and cover only sequence-dependent terms "
            "of the compressed layers; AdaLN/norm/rope are omitted.",
            "KV bytes are the per-forward K/V activations of the compressed layers at "
            f"{int(dtype_bytes)} bytes per element; this repo recomputes them for each "
            "of the 3 diffusion steps instead of caching across steps.",
            "The saving excludes layers 0..15, the VAE, the text encoder, the "
            "trajectory/video heads and feature preparation.",
        ],
    }


if __name__ == "__main__":  # pragma: no cover - manual cost inspection
    import json

    print(json.dumps(press_cost_report(), indent=2))

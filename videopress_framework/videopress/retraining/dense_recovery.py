"""Dense recovery decoder (plan section 12).

Route A keeps only original patch tokens, but the flow-matching video target is
still dense (``1560`` tokens at ``480x832``).  The selected tokens therefore have
to be scattered back onto the full grid before the Wan head runs.

The plan is explicit that this module must not take part in the main DiT
reasoning: it only converts the compact representation back into a dense video
flow prediction.  Trajectory-only inference can skip it entirely, so its cost
never enters the planning latency path.

Design notes:

* keys/values are the *sparse* hidden states plus their own position features,
  never a pooled global summary -- the whole point of Route A is that no new
  token is invented;
* queries are the full grid, so every dropped position is reconstructed from
  the kept neighbourhood it can actually attend to;
* the output projection is zero-initialised.  Starting from a DriveVA
  checkpoint this makes the decoder an identity-preserving perturbation at
  step 0 instead of injecting noise into the video FM loss.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class DenseRecoveryDecoder(nn.Module):
    """Cross-attention from full-grid queries to the selected tokens."""

    def __init__(
        self,
        dim: int = 3072,
        full_length: int = 1560,
        n_layers: int = 2,
        num_heads: int = 16,
        mlp_ratio: float = 2.0,
        position_dim: int = 64,
        freq_dim: int = 64,
        dropout: float = 0.0,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if int(dim) <= 0 or int(full_length) <= 0:
            raise ValueError("dim and full_length must be positive")
        if int(n_layers) <= 0:
            raise ValueError("n_layers must be positive")
        if int(dim) % int(num_heads):
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.full_length = int(full_length)
        self.n_layers = int(n_layers)

        hidden = max(int(dim * float(mlp_ratio)), int(dim))
        self.query_embed = nn.Parameter(
            torch.randn(self.full_length, self.dim) * (self.dim ** -0.5)
        )
        # Position features come in two flavours: the normalized (t, y, x)
        # coordinates the scorer already uses, and the flat grid index.  The
        # coordinate MLP generalises across the grid; the index embedding lets
        # the decoder special-case individual slots if it needs to.
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, int(position_dim)), nn.GELU(), nn.Linear(int(position_dim), self.dim)
        )
        self.index_embed = nn.Parameter(
            torch.randn(self.full_length, self.dim) * (self.dim ** -0.5)
        )
        self.key_coord_mlp = nn.Sequential(
            nn.Linear(3, int(position_dim)), nn.GELU(), nn.Linear(int(position_dim), self.dim)
        )
        self.key_index_embed = nn.Embedding(self.full_length, self.dim)

        self.input_norm = nn.LayerNorm(self.dim)
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(self.dim, int(num_heads), dropout=float(dropout), batch_first=True)
            for _ in range(self.n_layers)
        )
        self.attn_norm = nn.ModuleList(nn.LayerNorm(self.dim) for _ in range(self.n_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.dim, hidden), nn.GELU(), nn.Linear(hidden, self.dim)
            )
            for _ in range(self.n_layers)
        )
        self.ffn_norm = nn.ModuleList(nn.LayerNorm(self.dim) for _ in range(self.n_layers))
        self.output_norm = nn.LayerNorm(self.dim)
        self.output = nn.Linear(self.dim, self.dim)
        if zero_init_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        sparse_hidden: torch.Tensor,
        *,
        kept_indices: Optional[torch.Tensor] = None,
        sparse_positions: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return ``[B, full_length, D]`` dense hidden states.

        ``kept_indices`` addresses the full grid (``[K]`` or ``[B, K]``) and is
        required for the index embedding; ``sparse_positions`` /
        ``query_positions`` are the normalized ``(t, y, x)`` coordinates for the
        kept tokens and the full grid respectively.
        """
        if sparse_hidden.ndim != 3:
            raise ValueError("sparse_hidden must be [B,K,D]")
        # The backend runs in the model dtype (bf16 for DriveVA) while this
        # decoder is created in fp32.  Cast on entry rather than forcing every
        # caller to place the module in the right dtype: bf16 activations
        # against fp32 weights raise "mat1 and mat2 must have the same dtype".
        param_dtype = self.query_embed.dtype
        sparse_hidden = sparse_hidden.to(dtype=param_dtype)
        batch, n_kept, dim = sparse_hidden.shape
        if dim != self.dim:
            raise ValueError(f"sparse_hidden width {dim} != decoder dim {self.dim}")
        if batch_size is not None and int(batch_size) != batch:
            raise ValueError("batch_size does not match sparse_hidden")
        if kept_indices is not None:
            kept = torch.as_tensor(kept_indices, device=sparse_hidden.device).long()
            if kept.ndim == 1:
                kept = kept.unsqueeze(0).expand(batch, -1)
            if kept.shape != (batch, n_kept):
                raise ValueError("kept_indices must be [K] or [B,K] matching sparse_hidden")
            if int(kept.min().item()) < 0 or int(kept.max().item()) >= self.full_length:
                raise ValueError("kept_indices outside the full grid")
        else:
            if n_kept != self.full_length:
                raise ValueError(
                    "kept_indices is required when the kept count differs from full_length"
                )
            kept = (
                torch.arange(self.full_length, device=sparse_hidden.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )

        query_index = torch.arange(self.full_length, device=sparse_hidden.device)
        query = self.query_embed.unsqueeze(0).expand(batch, -1, -1) + self.index_embed.unsqueeze(0)
        if query_positions is not None:
            coords = torch.as_tensor(
                query_positions, device=sparse_hidden.device, dtype=param_dtype
            )
            if coords.ndim == 2:
                coords = coords.unsqueeze(0)
            if coords.shape[:2] != (batch, self.full_length):
                raise ValueError(
                    "query_positions must be [full_length,3] or [B,full_length,3]"
                )
            query = query + self.coord_mlp(coords)
        del query_index

        key = self.input_norm(sparse_hidden)
        key = key + self.key_index_embed(kept)
        if sparse_positions is not None:
            coords = torch.as_tensor(
                sparse_positions, device=sparse_hidden.device, dtype=param_dtype
            )
            if coords.ndim == 2:
                coords = coords.unsqueeze(0)
            if coords.shape[:2] != (batch, n_kept):
                raise ValueError(
                    "sparse_positions must be [K,3] or [B,K,3] matching sparse_hidden"
                )
            key = key + self.key_coord_mlp(coords)

        out = query
        for layer in range(self.n_layers):
            attended, _ = self.cross_attn[layer](out, key, key, need_weights=False)
            out = self.attn_norm[layer](out + attended)
            out = self.ffn_norm[layer](out + self.ffn[layer](out))
        return query + self.output(self.output_norm(out))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, full_length={self.full_length}, n_layers={self.n_layers}"

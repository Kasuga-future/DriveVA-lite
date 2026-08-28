"""Tensor shape helpers used by both synthetic tests and Wan hooks."""

from __future__ import annotations

from typing import Optional

import torch


def batch_gather_seq(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a per-batch sequence from ``[B,H,L,D]`` along dimension 2."""

    if x.ndim != 4:
        raise ValueError(f"x must have shape [B,H,L,D], got {tuple(x.shape)}")
    if indices.ndim == 1:
        indices = indices.unsqueeze(0).expand(x.shape[0], -1)
    if indices.ndim != 2 or indices.shape[0] != x.shape[0]:
        raise ValueError("indices must have shape [B,K]")
    if indices.dtype != torch.long:
        indices = indices.long()
    if indices.device != x.device:
        indices = indices.to(x.device)
    if indices.numel() and (indices.min() < 0 or indices.max() >= x.shape[2]):
        raise IndexError("sequence gather index is outside the input sequence")
    b, heads, _, dim = x.shape
    k = indices.shape[1]
    expanded = indices[:, None, :, None].expand(b, heads, k, dim)
    return torch.gather(x, dim=2, index=expanded)


def canonicalize_qkv(x: torch.Tensor, num_heads: Optional[int] = None) -> tuple[torch.Tensor, bool]:
    """Return Q/K/V in ``[B,H,L,D]`` form and whether input was flattened."""

    if x.ndim == 4:
        if num_heads is not None and x.shape[1] == num_heads:
            return x, False
        if num_heads is not None and x.shape[2] == num_heads:
            return x.transpose(1, 2), False
        # The framework canonical form is [B,H,L,D].  With no hint, retain it.
        return x, False
    if x.ndim != 3:
        raise ValueError(f"Q/K/V must have rank 3 or 4, got {x.ndim}")
    if num_heads is None or num_heads <= 0:
        raise ValueError("num_heads is required for flattened [B,L,D] Q/K/V")
    b, length, width = x.shape
    if width % num_heads:
        raise ValueError(f"hidden dimension {width} is not divisible by num_heads={num_heads}")
    return x.reshape(b, length, num_heads, width // num_heads).transpose(1, 2), True


def restore_qkv(x: torch.Tensor, flattened: bool) -> torch.Tensor:
    if not flattened:
        return x
    if x.ndim != 4:
        raise ValueError("canonical Q/K/V must have rank 4")
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])

"""Shape adapters for the flattened Q/K/V tensors used by Wan attention."""

from __future__ import annotations

import torch

from ..utils.tensor import canonicalize_qkv, restore_qkv


def canonicalize_wan_qkv(x: torch.Tensor, num_heads: int):
    return canonicalize_qkv(x, num_heads)


def restore_wan_qkv(x: torch.Tensor, flattened: bool):
    return restore_qkv(x, flattened)

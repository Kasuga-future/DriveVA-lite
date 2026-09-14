from __future__ import annotations

import torch


def ensure_finite(value: torch.Tensor, name: str = "tensor") -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def assert_protected_unchanged(before: torch.Tensor, after: torch.Tensor, protected_mask: torch.Tensor, atol: float = 1e-7) -> None:
    if before.shape != after.shape:
        raise AssertionError("protected-token check requires equal input/output shapes")
    protected = torch.where(protected_mask)[0]
    if protected.numel() == 0:
        return
    diff = (before.index_select(1, protected) - after.index_select(1, protected)).abs().max()
    if float(diff.item()) > atol:
        raise AssertionError(f"protected token invariant violated: max diff={float(diff.item())}")

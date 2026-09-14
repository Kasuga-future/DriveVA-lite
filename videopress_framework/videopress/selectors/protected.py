"""Safety wrapper that makes candidate-level protection non-negotiable."""

from __future__ import annotations

import torch

from ..core.registry import register_selector
from ..core.result import SelectionResult
from .base import TokenSelector


@register_selector("protected")
class ProtectedTokenSelector(TokenSelector):
    """Wrap another selector and force metadata-marked candidates to be kept.

    The mask may be candidate-local ``[B, N_candidate]`` or global
    ``[B, layout.total_length]`` and is read from ``TokenContext.metadata``.
    Protection replaces the lowest-ranked unprotected selections, preserving
    the base selector's K and therefore its latency budget.
    """

    name = "protected"

    def __init__(
        self,
        base_selector: TokenSelector,
        metadata_key: str = "critical_token_mask",
        required: bool = True,
    ) -> None:
        if not isinstance(base_selector, TokenSelector):
            raise TypeError("base_selector must implement TokenSelector")
        self.base_selector = base_selector
        self.metadata_key = str(metadata_key)
        self.required = bool(required)
        if not self.metadata_key:
            raise ValueError("metadata_key must not be empty")

    def _mask(self, scores, domain, ctx) -> torch.Tensor:
        if ctx is None:
            raise ValueError("protected selector requires a TokenContext")
        value = ctx.metadata.get(self.metadata_key)
        if value is None:
            if self.required:
                raise KeyError(
                    f"TokenContext.metadata lacks required {self.metadata_key!r}"
                )
            return torch.zeros_like(scores, dtype=torch.bool)
        mask = torch.as_tensor(value, device=scores.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and scores.shape[0] > 1:
            mask = mask.expand(scores.shape[0], -1)
        if mask.shape == scores.shape:
            return mask
        if mask.shape == (scores.shape[0], ctx.layout.total_length):
            return mask.index_select(1, domain.candidate_indices.to(scores.device))
        raise ValueError(
            f"{self.metadata_key} must be [B,N_candidate] or [B,total_length], "
            f"got {tuple(mask.shape)}"
        )

    def _enforce(self, base: SelectionResult, order, scores, domain, ctx):
        protected = self._mask(scores, domain, ctx)
        counts = protected.sum(dim=1)
        if bool((counts > base.K).any()):
            raise ValueError(
                "protected candidate count exceeds selected K="
                f"{base.K}: {counts.detach().cpu().tolist()}"
            )
        base_keep = torch.zeros_like(scores, dtype=torch.bool)
        if base.K:
            base_keep.scatter_(1, base.keep_candidate_indices.to(scores.device), True)
        replaced = []
        keep_rows = []
        for row in range(scores.shape[0]):
            keep = base_keep[row] | protected[row]
            surplus = int(keep.sum()) - int(base.K)
            if surplus > 0:
                # Reverse ranking is lowest first. Never evict protection.
                eviction = order[row].flip(0)
                eviction = eviction[keep[eviction] & ~protected[row, eviction]][:surplus]
                if eviction.numel() != surplus:
                    raise RuntimeError("could not make room for protected candidates")
                keep[eviction] = False
            keep_rows.append(order[row][keep[order[row]]])
            replaced.append(int((protected[row] & ~base_keep[row]).sum()))
        keep_local = torch.stack(keep_rows)
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        if keep_local.numel():
            keep_mask.scatter_(1, keep_local, True)
        all_local = torch.arange(domain.n_candidate, device=scores.device).expand_as(scores)
        drop_local = all_local[~keep_mask].reshape(scores.shape[0], domain.n_candidate - base.K)
        candidate = domain.candidate_indices.to(scores.device)
        metadata = dict(base.metadata)
        metadata.update(
            {
                "selector": self.name,
                "base_selector": self.base_selector.describe(),
                "protection_metadata_key": self.metadata_key,
                "protected_count_per_batch": counts.detach().cpu().tolist(),
                "protected_replacements_per_batch": replaced,
                "protection_budget_preserved": True,
            }
        )
        return SelectionResult(
            keep_candidate_indices=keep_local,
            drop_candidate_indices=drop_local,
            keep_global_indices=candidate[keep_local],
            K=base.K,
            metadata=metadata,
        )

    def select(self, scores, domain, K: int, ctx=None):
        order = scores.argsort(dim=-1, descending=True, stable=True)
        base = self.base_selector.select(scores, domain, K, ctx)
        return self._enforce(base, order, scores, domain, ctx)

    def select_from_order(self, order, scores, domain, K: int, ctx=None):
        base = self.base_selector.select_from_order(order, scores, domain, K, ctx)
        return self._enforce(base, order, scores, domain, ctx)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "base_selector": self.base_selector.describe(),
            "metadata_key": self.metadata_key,
            "required": self.required,
            "overflow_policy": "error",
            "budget_policy": "replace_lowest_unprotected",
        }

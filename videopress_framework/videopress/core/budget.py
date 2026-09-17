"""Unambiguous token budgets and reporting of both ratio conventions."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .domain import TokenDomain
from .layout import TokenLayout


@dataclass(frozen=True)
class TokenBudget:
    type: str
    value: float
    reference: str = "eligible"

    def __post_init__(self) -> None:
        kind = str(self.type).strip().lower()
        if kind not in {"absolute", "ratio"}:
            raise ValueError("TokenBudget.type must be 'absolute' or 'ratio'")
        if not math.isfinite(float(self.value)):
            raise ValueError("TokenBudget.value must be finite")
        if kind == "ratio" and not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("ratio budget must be within [0, 1]")
        if kind == "absolute" and float(self.value) < 0:
            raise ValueError("absolute budget cannot be negative")
        object.__setattr__(self, "type", kind)
        object.__setattr__(self, "reference", str(self.reference).strip().lower())


def _reference_count(reference: str, layout: TokenLayout, domain: TokenDomain) -> int:
    key = reference.strip().lower()
    if key in {"eligible", "candidate", "domain"}:
        return domain.n_candidate
    if key in {"history", "all_history"}:
        return layout.history_video.length
    if key in {"last_history"}:
        return layout.tokens_per_latent
    if key in {"video", "all_video"}:
        return layout.video.length
    if key == "future_video":
        return layout.future_video.length
    raise ValueError(f"Unknown budget reference: {reference}")


def resolve_budget(budget: TokenBudget, layout: TokenLayout, domain: TokenDomain) -> int:
    """Resolve a budget to an exact K; invalid K values fail loudly."""

    if not isinstance(budget, TokenBudget):
        raise TypeError("budget must be a TokenBudget")
    if budget.type == "absolute":
        value = float(budget.value)
        if not value.is_integer():
            raise ValueError("absolute token budget must be an integer")
        k = int(value)
    elif budget.reference in {"each_history", "per_history_latent"}:
        if domain.name not in {"history", "all_history"}:
            raise ValueError("reference=each_history requires domain=history/all_history")
        if domain.n_candidate != layout.history_video.length:
            raise ValueError("reference=each_history requires the complete history domain")
        per_latent = int(round(layout.tokens_per_latent * float(budget.value)))
        k = per_latent * layout.num_cond_latents
    elif budget.reference in {"each_future", "per_future_latent"}:
        if domain.name != "future_video":
            raise ValueError("reference=each_future requires domain=future_video")
        if domain.n_candidate != layout.future_video.length:
            raise ValueError("reference=each_future requires the complete future domain")
        num_future_latents = int(layout.video_f) - int(layout.num_cond_latents)
        per_latent = int(round(layout.tokens_per_latent * float(budget.value)))
        k = per_latent * num_future_latents
    else:
        k = int(round(_reference_count(budget.reference, layout, domain) * float(budget.value)))
    if k < 0 or k > domain.n_candidate:
        raise ValueError(
            f"resolved K={k} is outside eligible range [0, {domain.n_candidate}] "
            f"for domain={domain.name}"
        )
    return k


def budget_stats(layout: TokenLayout, domain: TokenDomain, kept: int) -> dict:
    if kept < 0 or kept > domain.n_candidate:
        raise ValueError("kept must be within the eligible domain")
    n_history = layout.history_video.length
    n_eligible = domain.n_candidate
    return {
        "n_history": n_history,
        "n_eligible": n_eligible,
        "n_kept": int(kept),
        "history_keep_ratio": (float(kept) / n_history) if n_history else None,
        "eligible_keep_ratio": (float(kept) / n_eligible) if n_eligible else None,
    }


def history_retention_stats(layout: TokenLayout, domain: TokenDomain, keep_global_indices) -> dict:
    """Report selected and effectively preserved tokens for every history latent.

    Protected history positions are effectively preserved even though they are
    not part of ``SelectionResult.K``.  Reporting both values avoids the old
    ambiguity where last-history 50% looked like total-history 25%.
    """

    import torch

    if not torch.is_tensor(keep_global_indices) or keep_global_indices.ndim != 2:
        raise ValueError("keep_global_indices must have shape [B,K]")
    device = keep_global_indices.device
    selected = torch.zeros(
        (keep_global_indices.shape[0], layout.total_length), dtype=torch.bool, device=device
    )
    if keep_global_indices.numel():
        selected.scatter_(1, keep_global_indices.long(), True)
    protected = domain.protected_mask.to(device).unsqueeze(0).expand_as(selected)
    effective = selected | protected
    selected_counts = []
    effective_counts = []
    for latent_index in range(layout.num_cond_latents):
        frame = layout.frame_range(latent_index)
        selected_counts.append(selected[:, frame.start : frame.end].sum(dim=1))
        effective_counts.append(effective[:, frame.start : frame.end].sum(dim=1))
    selected_matrix = torch.stack(selected_counts, dim=1) if selected_counts else selected.new_zeros((selected.shape[0], 0), dtype=torch.long)
    effective_matrix = torch.stack(effective_counts, dim=1) if effective_counts else selected.new_zeros((selected.shape[0], 0), dtype=torch.long)
    denominator = float(layout.tokens_per_latent)
    return {
        "history_latent_token_count": int(layout.tokens_per_latent),
        "selected_history_latent_counts": selected_matrix.detach().cpu().tolist(),
        "selected_history_latent_ratios": (selected_matrix.float() / denominator).detach().cpu().tolist(),
        "effective_history_latent_kept_counts": effective_matrix.detach().cpu().tolist(),
        "effective_history_latent_keep_ratios": (effective_matrix.float() / denominator).detach().cpu().tolist(),
        "effective_history_kept": effective_matrix.sum(dim=1).detach().cpu().tolist(),
        "effective_history_keep_ratio": (
            effective_matrix.sum(dim=1).float() / float(max(1, layout.history_video.length))
        ).detach().cpu().tolist(),
    }


def future_retention_stats(layout: TokenLayout, domain: TokenDomain, keep_global_indices) -> dict:
    """Report selected and effectively preserved tokens for every future latent.

    The accounting mirrors :func:`history_retention_stats`, but latent indices
    are the storage-order future frames ``num_cond_latents .. video_f - 1``.
    ``future_latent_0`` is always the nearest future latent.
    """

    import torch

    if not torch.is_tensor(keep_global_indices) or keep_global_indices.ndim != 2:
        raise ValueError("keep_global_indices must have shape [B,K]")
    device = keep_global_indices.device
    selected = torch.zeros(
        (keep_global_indices.shape[0], layout.total_length), dtype=torch.bool, device=device
    )
    if keep_global_indices.numel():
        selected.scatter_(1, keep_global_indices.long(), True)
    protected = domain.protected_mask.to(device).unsqueeze(0).expand_as(selected)
    effective = selected | protected
    num_future = int(layout.video_f) - int(layout.num_cond_latents)
    selected_counts = []
    effective_counts = []
    for future_index in range(num_future):
        frame = layout.frame_range(int(layout.num_cond_latents) + future_index)
        selected_counts.append(selected[:, frame.start : frame.end].sum(dim=1))
        effective_counts.append(effective[:, frame.start : frame.end].sum(dim=1))
    selected_matrix = (
        torch.stack(selected_counts, dim=1)
        if selected_counts
        else selected.new_zeros((selected.shape[0], 0), dtype=torch.long)
    )
    effective_matrix = (
        torch.stack(effective_counts, dim=1)
        if effective_counts
        else selected.new_zeros((selected.shape[0], 0), dtype=torch.long)
    )
    denominator = float(layout.tokens_per_latent)
    future_tokens = int(layout.future_video.length)
    return {
        "future_latent_token_count": int(layout.tokens_per_latent),
        "selected_future_latent_counts": selected_matrix.detach().cpu().tolist(),
        "selected_future_latent_ratios": (selected_matrix.float() / denominator)
        .detach()
        .cpu()
        .tolist(),
        "effective_future_latent_kept_counts": effective_matrix.detach().cpu().tolist(),
        "effective_future_latent_keep_ratios": (effective_matrix.float() / denominator)
        .detach()
        .cpu()
        .tolist(),
        "effective_future_kept": effective_matrix.sum(dim=1).detach().cpu().tolist(),
        "effective_future_keep_ratio": (
            effective_matrix.sum(dim=1).float() / float(max(1, future_tokens))
        )
        .detach()
        .cpu()
        .tolist(),
    }

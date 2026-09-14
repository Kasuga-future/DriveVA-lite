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

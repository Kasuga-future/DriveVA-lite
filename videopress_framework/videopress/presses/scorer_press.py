from __future__ import annotations

from typing import Any

from ..core.budget import budget_stats, history_retention_stats, resolve_budget
from ..core.persistence import CrossLayerPersistence
from ..core.registry import register_press
from ..core.result import CompressionResult
from ..core.runtime import InjectionPoint
from .base import BaseVideoPress


@register_press("scorer_press")
class ScorerPress(BaseVideoPress):
    """Compose scorer + selector + operator around one exact token budget."""

    name = "scorer_press"

    def __init__(
        self,
        scorer,
        selector,
        operator,
        budget,
        domain=None,
        injection_point: InjectionPoint | str = InjectionPoint.VIDEO_INPUT,
        random_scope: str | None = None,
        retention_policy: str | None = None,
        cross_layer_persistence=None,
    ):
        self.scorer = scorer
        self.selector = selector
        self.operator = operator
        self.budget = budget
        self.domain = domain
        self.injection_point = InjectionPoint.parse(injection_point)
        self.random_scope = None if random_scope is None else str(random_scope)
        self.retention_policy = None if retention_policy is None else str(retention_policy)
        self.cross_layer_persistence = CrossLayerPersistence.from_config(
            cross_layer_persistence
        )
        if self.cross_layer_persistence.enabled:
            source_layer = getattr(self.scorer, "layer", None)
            if self.injection_point is not InjectionPoint.SELF_ATTN_KV:
                raise ValueError(
                    "cross-layer persistence requires injection_point=self_attn_kv"
                )
            if source_layer is None or int(source_layer) < 0:
                raise ValueError(
                    "cross-layer persistence requires a non-negative scorer.layer"
                )
            # Rejects enabled=True with end_layer == scorer.layer instead of
            # silently downgrading it to a one-shot prune that still claims
            # `cross_layer_persistent: True` (audit 2026-09-12, BUG-13).
            self.cross_layer_persistence.validate_for(int(source_layer))
            if (
                self.cross_layer_persistence.mode == "hidden_sequence"
                and getattr(self.operator, "name", None) != "kv_prune"
            ):
                raise ValueError(
                    "hidden-sequence persistence currently requires operator=kv_prune"
                )

    def score(self, ctx):
        scores = self.scorer.score(ctx)
        if scores.ndim != 2 or scores.shape[0] != ctx.batch_size or scores.shape[1] != ctx.domain.n_candidate:
            raise ValueError("scorer must return [B, N_candidate]")
        return scores

    def select(self, ctx, scores, cached_ranking=None):
        K = resolve_budget(self.budget, ctx.layout, ctx.domain)
        if cached_ranking is None:
            return self.selector.select(scores, ctx.domain, K, ctx)
        return self.selector.select_from_order(cached_ranking, scores, ctx.domain, K, ctx)

    def ranking(self, scores):
        return scores.argsort(dim=-1, descending=True, stable=True)

    def apply_with_selection(self, ctx, scores, selection) -> CompressionResult:
        operator_result = self.operator.apply(ctx, selection)
        metadata = {
            "press": self.name,
            "scorer": self.scorer.describe() if hasattr(self.scorer, "describe") else type(self.scorer).__name__,
            "selector": self.selector.describe() if hasattr(self.selector, "describe") else type(self.selector).__name__,
            "operator": self.operator.describe() if hasattr(self.operator, "describe") else type(self.operator).__name__,
            "domain": ctx.domain.name,
            "injection_point": self.injection_point.value,
            "budget": {"type": self.budget.type, "value": self.budget.value, "reference": self.budget.reference},
            **budget_stats(ctx.layout, ctx.domain, selection.K),
            **history_retention_stats(ctx.layout, ctx.domain, selection.keep_global_indices),
            "selection_metadata": dict(selection.metadata),
            **operator_result.metadata,
        }
        if self.retention_policy is not None:
            metadata["retention_policy"] = self.retention_policy
        score_diagnostics = ctx.metadata.get("score_diagnostics")
        if isinstance(score_diagnostics, dict):
            metadata["score_diagnostics"] = dict(score_diagnostics)
        return CompressionResult(
            output=operator_result.output,
            scores=scores,
            selection=selection,
            metadata=metadata,
            mapping=operator_result.mapping,
            aux=operator_result.aux,
        )

    def apply_with_scores(self, ctx, scores, selection=None, cached_ranking=None) -> CompressionResult:
        if selection is None:
            selection = self.select(ctx, scores, cached_ranking=cached_ranking)
        return self.apply_with_selection(ctx, scores, selection)

    def apply(self, ctx) -> CompressionResult:
        return self.apply_with_scores(ctx, self.score(ctx))

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "scorer": self.scorer.describe() if hasattr(self.scorer, "describe") else type(self.scorer).__name__,
            "selector": self.selector.describe() if hasattr(self.selector, "describe") else type(self.selector).__name__,
            "operator": self.operator.describe() if hasattr(self.operator, "describe") else type(self.operator).__name__,
            "budget": {"type": self.budget.type, "value": self.budget.value, "reference": self.budget.reference},
            "domain": getattr(self.domain, "name", self.domain),
            "random_scope": self.random_scope,
            "retention_policy": self.retention_policy,
            "cross_layer_persistence": self.cross_layer_persistence.to_dict(),
        }

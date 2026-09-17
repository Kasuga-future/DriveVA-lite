"""Build a press from the YAML schema in ``press.md``."""

from __future__ import annotations

from typing import Any

from .core.budget import TokenBudget
from .core.domain import build_domain
from .core.registry import OPERATOR_REGISTRY, SCORER_REGISTRY, SELECTOR_REGISTRY, get_registered
from .core.runtime import InjectionPoint
from .core.retention import apply_history_retention_policy
from .operators import (
    HiddenPruneOperator,
    HiddenTokenMergeOperator,
    KVMergeOperator,
    KVPruneOperator,
    MeanReplaceOperator,
    ShuffleAllOperator,
    ShuffleDroppedOperator,
    ShuffleKeptOperator,
    ShuffleOperator,
    ZeroMaskOperator,
)
from .presses import (
    ComposedPress,
    NoPress,
    RegisterBottleneckPress,
    ScorerPress,
    SimilarityMergePress,
)
from .scorers import (
    ActionAttentionScorer,
    ActionAttentionVNormScorer,
    ActionAttentionVNormTemporalScorer,
    ActionContributionStabilityScorer,
    GradientInputScorer,
    GradientNormScorer,
    LearnedPlanningSelectorScorer,
    PlanningGradientInputScorer,
    RandomScorer,
    TokenNormScorer,
)
from .selectors import (
    AdaptiveMassSelector,
    AdaptiveSpatialMassSelector,
    FutureQuotaSelector,
    FutureThresholdSelector,
    HistoryGuidedFutureSelector,
    HistoryQuotaSelector,
    HistoryThresholdSelector,
    HistoryTopKSelector,
    ProtectedTokenSelector,
    ThresholdSelector,
    TopKSelector,
)


def _section(value: Any, default: dict | None = None) -> dict:
    if value is None:
        return dict(default or {})
    if isinstance(value, str):
        return {"name": value}
    if not isinstance(value, dict):
        raise TypeError(f"expected config mapping or name, got {type(value).__name__}")
    return dict(value)


def _register_builtin_aliases() -> None:
    # Importing the modules registers the canonical entries.  Aliases here are
    # intentionally explicit so configuration names remain stable.
    _ = (NoPress, ScorerPress, SimilarityMergePress, RegisterBottleneckPress, RandomScorer, TokenNormScorer,
         ActionAttentionScorer, ActionAttentionVNormScorer, ActionAttentionVNormTemporalScorer, ActionContributionStabilityScorer, GradientNormScorer,
         GradientInputScorer, PlanningGradientInputScorer, LearnedPlanningSelectorScorer, AdaptiveMassSelector, AdaptiveSpatialMassSelector, FutureQuotaSelector, FutureThresholdSelector, HistoryGuidedFutureSelector, HistoryQuotaSelector, HistoryThresholdSelector, HistoryTopKSelector, ProtectedTokenSelector, TopKSelector, ThresholdSelector, ZeroMaskOperator,
         MeanReplaceOperator, ShuffleOperator, ShuffleAllOperator,
         ShuffleDroppedOperator, ShuffleKeptOperator, KVPruneOperator, HiddenPruneOperator,
         HiddenTokenMergeOperator, KVMergeOperator, ComposedPress)


def build_scorer(config: Any, *, gradient_forward=None, gradient_objective=None):
    _register_builtin_aliases()
    section = _section(config)
    name = str(section.pop("name", "random")).lower()
    aliases = {"norm": "token_norm", "attention": "action_attention", "attention_vnorm": "action_attention_vnorm"}
    name = aliases.get(name, name)
    if name in {"action_attention", "action_attention_vnorm", "action_attention_vnorm_temporal", "action_contribution_stability"}:
        heads = section.pop("heads", None)
        if isinstance(heads, dict):
            section.setdefault("head_mode", heads.get("mode", "mean"))
            if "index" in heads:
                section.setdefault("head_index", heads["index"])
        action = section.pop("action", None)
        if isinstance(action, dict):
            section.setdefault("action_mode", action.get("mode", "mean"))
        value_norm = section.pop("value_norm", None)
        if isinstance(value_norm, dict):
            section.setdefault("value_norm", value_norm.get("enabled", False))
            if "head_mode" in value_norm:
                section.setdefault("value_norm_head_mode", value_norm["head_mode"])
            if "value_norm_head_mode" in value_norm:
                section.setdefault("value_norm_head_mode", value_norm["value_norm_head_mode"])
        if "layer" in section:
            section["layer"] = section["layer"]
    if name in {"gradient_norm", "gradient_input"}:
        section.setdefault("forward_fn", gradient_forward)
        section.setdefault("objective", gradient_objective)
    cls = get_registered(SCORER_REGISTRY, name)
    return cls(**section)


def build_selector(config: Any):
    _register_builtin_aliases()
    section = _section(config, {"name": "topk"})
    name = section.pop("name", "topk")
    if str(name).lower() == "protected":
        base = section.pop("base", section.pop("base_selector", {"name": "topk"}))
        return ProtectedTokenSelector(base_selector=build_selector(base), **section)
    return get_registered(SELECTOR_REGISTRY, name)(**section)


def build_operator(config: Any):
    _register_builtin_aliases()
    section = _section(config, {"name": "zero"})
    name = section.pop("name", "zero")
    return get_registered(OPERATOR_REGISTRY, name)(**section)


def build_budget(config: Any) -> TokenBudget:
    section = _section(config, {"type": "ratio", "value": 1.0, "reference": "eligible"})
    kind = section.pop("type", section.pop("kind", None))
    if kind is None:
        raise ValueError("budget needs type=absolute or ratio")
    return TokenBudget(type=kind, value=section.pop("value"), reference=section.pop("reference", "eligible"))


def build_press(config: Any, *, gradient_forward=None, gradient_objective=None):
    """Build a press without importing DriveVA internals."""

    _register_builtin_aliases()
    if isinstance(config, str):
        config = {"name": config}
    section = _section(config)
    name = str(section.pop("name", "scorer_press")).lower()
    retention_policy = section.get("retention_policy")
    if name in {"none", "noop", "full"}:
        if retention_policy is not None:
            raise ValueError("NoPress cannot implement a history retention policy")
        return NoPress()
    if name == "similarity_merge":
        if retention_policy is not None:
            raise ValueError("similarity_merge does not delete unselected tokens and cannot implement a retention policy")
        domain = section.get("domain", "last_history")
        return SimilarityMergePress(
            budget=build_budget(section.get("budget")),
            domain=domain,
            feature=section.get("feature", "tokens"),
            injection_point=section.get("injection_point", InjectionPoint.SELF_ATTN_KV),
            seed=int(section.get("seed", 0)),
        )
    if name in {"register_merge", "learnable_merge"}:
        if retention_policy is not None:
            raise ValueError("register_merge does not implement a retention policy")
        return RegisterBottleneckPress(
            budget=build_budget(section.get("budget")),
            domain=section.get("domain", "last_history"),
            num_key_tokens=int(section.get("num_key_tokens", 64)),
            hidden_dim=int(section.get("hidden_dim", 3072)),
            attn_dim=int(section.get("attn_dim", 1024)),
            num_heads=int(section.get("num_heads", 8)),
            dropout=float(section.get("dropout", 0.0)),
            use_position_bias=bool(section.get("use_position_bias", True)),
            value_norm=bool(section.get("value_norm", False)),
            checkpoint=section.get("checkpoint"),
            trainable=bool(section.get("trainable", True)),
            injection_point=section.get("injection_point", InjectionPoint.BLOCK_INPUT),
        )
    if name in {"composed", "compose"}:
        if retention_policy is not None:
            raise ValueError("apply history retention policies to each composed child explicitly")
        children = section.pop("presses", section.pop("children", None))
        if not children:
            raise ValueError("composed press requires a non-empty presses list")
        return ComposedPress(
            [build_press(child, gradient_forward=gradient_forward, gradient_objective=gradient_objective) for child in children]
        )
    # Convenience shorthand: {name: random, budget: ...}.
    if name not in {"scorer_press"}:
        scorer_section = section.pop("scorer", {})
        if isinstance(scorer_section, str):
            scorer_section = {"name": scorer_section}
        # A CLI method override replaces the configured scorer.  Do not pass
        # options belonging to the previous scorer (for example Random's seed
        # into TokenNorm) unless the names already agree.
        configured_name = str(scorer_section.get("name", "")).lower() if isinstance(scorer_section, dict) else ""
        if configured_name not in {name, aliases_for_scorer(name)}:
            scorer_section = {}
        scorer_config = {**scorer_section, "name": name}
        section["scorer"] = scorer_config
        name = "scorer_press"
    if retention_policy is not None:
        section = apply_history_retention_policy(section, str(retention_policy))
    retention_policy = section.pop("retention_policy", None)
    scorer = build_scorer(section.pop("scorer", {"name": "random"}), gradient_forward=gradient_forward, gradient_objective=gradient_objective)
    selector = build_selector(section.pop("selector", "topk"))
    operator = build_operator(section.pop("operator", "zero"))
    domain = section.pop("domain", "last_history")
    if isinstance(domain, dict):
        domain = domain.get("name", "last_history")
    budget = build_budget(section.pop("budget", None))
    injection_point = section.pop("injection_point", InjectionPoint.VIDEO_INPUT)
    random_scope = section.pop("random_scope", None)
    cross_layer_persistence = section.pop("cross_layer_persistence", None)
    return ScorerPress(
        scorer=scorer,
        selector=selector,
        operator=operator,
        budget=budget,
        domain=domain,
        injection_point=injection_point,
        random_scope=random_scope,
        retention_policy=retention_policy,
        cross_layer_persistence=cross_layer_persistence,
    )
def aliases_for_scorer(name: str) -> str:
    return {
        "norm": "token_norm",
        "attention": "action_attention",
        "attention_vnorm": "action_attention_vnorm",
    }.get(name, name)

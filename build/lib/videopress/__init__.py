"""DriveVA VideoTokenPress framework.

The package deliberately keeps token selection/compression independent from the
DriveVA evaluator.  A press is composed from a domain, scorer, selector,
operator and budget, and can be used with either causal masking or physical
KV compression.
"""

from .core.budget import TokenBudget, budget_stats, resolve_budget
from .core.context import TokenContext
from .core.domain import (
    AllVideoDomain,
    DomainBuilder,
    FutureVideoDomain,
    HistoryDomain,
    LastHistoryDomain,
    TokenDomain,
    build_domain,
)
from .core.layout import (
    TokenLayout,
    TokenRange,
    build_driveva_layout,
    decode_video_index,
    decode_video_index_checked,
    get_last_history_range,
)
from .core.result import CompressionResult, OperatorResult, SelectionResult, TokenMapping
from .core.runtime import CompressionEvent, CompressionEventKey, EvaluationMode, InjectionPoint, VideoPressRuntime

__all__ = [
    "AllVideoDomain",
    "CompressionResult",
    "CompressionEvent",
    "CompressionEventKey",
    "DomainBuilder",
    "EvaluationMode",
    "FutureVideoDomain",
    "HistoryDomain",
    "InjectionPoint",
    "LastHistoryDomain",
    "OperatorResult",
    "SelectionResult",
    "TokenBudget",
    "TokenContext",
    "TokenDomain",
    "TokenLayout",
    "TokenMapping",
    "TokenRange",
    "VideoPressRuntime",
    "budget_stats",
    "build_domain",
    "build_driveva_layout",
    "decode_video_index",
    "decode_video_index_checked",
    "get_last_history_range",
    "resolve_budget",
]

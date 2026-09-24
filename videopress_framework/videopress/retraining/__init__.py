"""Route A / Route B compression retraining (plan v2, sections 36-37).

This package is deliberately *not* a runtime Press hook.  The plan requires the
retrained compression to be a first-class ``nn.Module`` that lives on the
pipeline and is written into the checkpoint, so these modules own their forward
pass and their own parameters.

Implemented here:

``threshold_gate``
    Straight-through threshold gate with per-domain thresholds, a loose safety
    clamp, the Lagrangian sparsity penalty, and the sparsity curriculum.
``dynamic_selector``
    Planning-conditioned token scorer (token + action + timestep + position +
    history/future identity), plan section 10.
``dense_recovery``
    Query cross-attention that restores the dense video grid from the selected
    tokens so the original dense video flow-matching target still applies.
``distillation``
    Flow-matching and LN-matched hidden distillation against the frozen
    original DriveVA teacher.
``compression_stats``
    Per-scene/per-round runtime statistics and the "is K really dynamic?"
    analysis the plan demands.
``route_a``
    The assembled Route A forward pass (dense front-end -> threshold -> sparse
    backend -> trajectory head -> dense recovery -> Wan head).
``curriculum``
    A0-A4 stages, per-group learning rates, threshold jitter, and the
    compression-layer curriculum 18 -> 15 -> 12.
"""

from .compression_stats import (
    CompressionStatsRecorder,
    RoundRecord,
    threshold_sweep_report,
)
from .curriculum import (
    DEFAULT_LAYER_CURRICULUM,
    DEFAULT_SPARSITY_SWEEP,
    STAGE_A0,
    STAGE_A1,
    STAGE_A2,
    STAGE_A3,
    STAGE_A4,
    STAGES,
    RouterStageSchedule,
    StageSpec,
    apply_stage,
    build_optimizer,
    default_stage_specs,
    jitter_for_step,
)
from .dense_recovery import DenseRecoveryDecoder
from .distillation import (
    RouteALossOutput,
    RouteALossWeights,
    action_hidden_kd,
    compute_route_a_loss,
    layer_norm_mse,
    weighted_mse,
)
from .dynamic_selector import DynamicVideoTokenScorer, build_token_type_vector
from .route_a import (
    DRIVEVA_FUTURE_LATENTS,
    DRIVEVA_HISTORY_LATENTS,
    DRIVEVA_PATCH_H,
    DRIVEVA_PATCH_W,
    RouteAConfig,
    RouteADynamicSelect,
    RouteAForwardOutput,
    RouteALayoutSpec,
    build_driveva_video_positions,
    sync_keep_lengths,
)
from .threshold_gate import (
    DEFAULT_FUTURE_THRESHOLD,
    DEFAULT_HISTORY_THRESHOLD,
    STEThresholdGate,
    GateHealth,
    SafetyClampConfig,
    SparsityCurriculum,
    SparsityGuard,
    ThresholdGateOutput,
    binding_row_domain_counts,
    gate_health,
    jittered_thresholds,
    select_kept_indices,
    sparsity_loss,
)

__all__ = [
    "CompressionStatsRecorder",
    "RoundRecord",
    "threshold_sweep_report",
    "DEFAULT_LAYER_CURRICULUM",
    "DEFAULT_SPARSITY_SWEEP",
    "STAGE_A0",
    "STAGE_A1",
    "STAGE_A2",
    "STAGE_A3",
    "STAGE_A4",
    "STAGES",
    "RouterStageSchedule",
    "StageSpec",
    "apply_stage",
    "build_optimizer",
    "default_stage_specs",
    "jitter_for_step",
    "DenseRecoveryDecoder",
    "RouteALossOutput",
    "RouteALossWeights",
    "action_hidden_kd",
    "compute_route_a_loss",
    "layer_norm_mse",
    "weighted_mse",
    "DynamicVideoTokenScorer",
    "build_token_type_vector",
    "DRIVEVA_FUTURE_LATENTS",
    "DRIVEVA_HISTORY_LATENTS",
    "DRIVEVA_PATCH_H",
    "DRIVEVA_PATCH_W",
    "RouteAConfig",
    "RouteADynamicSelect",
    "RouteAForwardOutput",
    "RouteALayoutSpec",
    "build_driveva_video_positions",
    "sync_keep_lengths",
    "DEFAULT_FUTURE_THRESHOLD",
    "DEFAULT_HISTORY_THRESHOLD",
    "STEThresholdGate",
    "SafetyClampConfig",
    "SparsityCurriculum",
    "GateHealth",
    "SparsityGuard",
    "ThresholdGateOutput",
    "binding_row_domain_counts",
    "gate_health",
    "jittered_thresholds",
    "select_kept_indices",
    "sparsity_loss",
]

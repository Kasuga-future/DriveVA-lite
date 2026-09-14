"""Learnable register/query bottleneck (route: latent predictive bottleneck).

The canonical implementation lives in :mod:`videopress.press.register_bottleneck`
(the path requested by the 2026-09-11 design task).  This module re-exports it
inside the package where this repo keeps its press plugins (``videopress.presses``)
so callers can use either import path::

    from videopress.presses.register_bottleneck import RegisterBottleneck
    from videopress.press.register_bottleneck import RegisterBottleneck

No behaviour is added here; see the implementation module for the design notes
(motivation, gradient routing, deployment splicing and the cost model).
"""

from ..press.register_bottleneck import (  # noqa: F401
    DEFAULT_COMPRESSED_LAYERS,
    DEFAULT_FFN_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HISTORY_TOKENS,
    DEFAULT_PROTECTED_TOKENS,
    DEFAULT_SELECTOR_LAYER,
    DEFAULT_TEXT_CONTEXT_LENGTH,
    DEFAULT_TOTAL_SEQUENCE,
    BottleneckOutput,
    CostEstimate,
    KeyTokenEncoding,
    NextLatentPredictor,
    RegisterBottleneck,
    bottleneck_overhead_flops,
    key_token_diversity_penalty,
    key_token_keep_indices,
    key_token_positions,
    key_token_scale_penalty,
    key_token_std,
    key_tokens_for_context,
    last_history_positions,
    layer_sequence_cost,
    next_latent_prediction_loss,
    position_features,
    press_cost_report,
    restore_key_token_sequence,
    shuffle_key_tokens,
    splice_key_tokens,
    trivial_prediction_loss,
    variance_covariance_penalty,
)

__all__ = [
    "BottleneckOutput",
    "CostEstimate",
    "KeyTokenEncoding",
    "NextLatentPredictor",
    "RegisterBottleneck",
    "bottleneck_overhead_flops",
    "key_token_diversity_penalty",
    "key_token_keep_indices",
    "key_token_positions",
    "key_token_scale_penalty",
    "key_token_std",
    "key_tokens_for_context",
    "last_history_positions",
    "layer_sequence_cost",
    "next_latent_prediction_loss",
    "position_features",
    "press_cost_report",
    "restore_key_token_sequence",
    "shuffle_key_tokens",
    "splice_key_tokens",
    "trivial_prediction_loss",
    "variance_covariance_penalty",
    "DEFAULT_COMPRESSED_LAYERS",
    "DEFAULT_FFN_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_HISTORY_TOKENS",
    "DEFAULT_PROTECTED_TOKENS",
    "DEFAULT_SELECTOR_LAYER",
    "DEFAULT_TEXT_CONTEXT_LENGTH",
    "DEFAULT_TOTAL_SEQUENCE",
]

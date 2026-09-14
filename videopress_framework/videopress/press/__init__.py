"""Register-token latent bottleneck (route 2026-09-11, additive; no existing file changes).

Canonical implementation: :mod:`videopress.press.register_bottleneck`.
Press-plugin alias:      :mod:`videopress.presses.register_bottleneck`.
"""

from .register_bottleneck import (  # noqa: F401
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
from . import register_bottleneck as _module  # noqa: F401

__all__ = list(_module.__all__)

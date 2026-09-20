"""Token-level set-level oracle for future video token compression.

See :mod:`videopress.oracle.token_set` for the search library and
:mod:`videopress.oracle.metrics` for the trajectory objectives used to score a
candidate subset.
"""

from .metrics import (
    OBJECTIVES,
    combined_harm,
    load_target_trajectory,
    load_trajectory,
    objective_higher_is_better,
    objective_value,
    planning_harm,
    trajectory_displacement,
    zscore,
)
from .token_set import (
    GROUP_MODES,
    SearchStep,
    SetScorer,
    SetSearchResult,
    beam_search,
    build_token_groups,
    entry_to_flat,
    flat_from_latent_local,
    greedy_backward_elimination,
    greedy_forward_selection,
    latent_local_mask,
    mean_over_scenes,
    per_scene_oracle,
    random_search,
    random_token_masks,
    read_token_mask_json,
    write_token_mask_json,
)

__all__ = [
    "GROUP_MODES",
    "OBJECTIVES",
    "SearchStep",
    "SetScorer",
    "SetSearchResult",
    "beam_search",
    "build_token_groups",
    "combined_harm",
    "entry_to_flat",
    "flat_from_latent_local",
    "greedy_backward_elimination",
    "greedy_forward_selection",
    "latent_local_mask",
    "load_target_trajectory",
    "load_trajectory",
    "mean_over_scenes",
    "objective_higher_is_better",
    "objective_value",
    "per_scene_oracle",
    "planning_harm",
    "random_search",
    "random_token_masks",
    "read_token_mask_json",
    "trajectory_displacement",
    "write_token_mask_json",
    "zscore",
]

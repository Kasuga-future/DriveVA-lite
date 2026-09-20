"""Trajectory-space objectives for the future token set-level oracle.

The 2026-09-20 conclusion report asked to keep scoring the token-level oracle
with PDM **and** trajectory displacement **and** planning harm at the same time,
because a single PDM drop is a noisy, mostly-binary signal.  These helpers are
the trajectory side of that objective and are deliberately free of any torch /
framework dependency so they can be unit-tested on plain arrays.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "OBJECTIVES",
    "combined_harm",
    "load_target_trajectory",
    "load_trajectory",
    "objective_higher_is_better",
    "objective_value",
    "planning_harm",
    "trajectory_displacement",
    "zscore",
]


def _as_2d(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"trajectory must be 2-D, got shape {value.shape}")
    return value


def trajectory_displacement(baseline: np.ndarray, masked: np.ndarray) -> float:
    """Mean per-waypoint L2 between two ego-relative trajectories."""

    base = _as_2d(baseline)
    other = _as_2d(masked)
    steps = min(base.shape[0], other.shape[0])
    dims = min(base.shape[1], other.shape[1], 3)
    if steps <= 0 or dims <= 0:
        return math.nan
    return float(np.linalg.norm(base[:steps, :dims] - other[:steps, :dims], axis=-1).mean())


def planning_harm(
    baseline: np.ndarray,
    masked: np.ndarray,
    target: np.ndarray,
) -> float:
    """Change in ADE-to-ground-truth caused by masking (masked minus baseline)."""

    if target is None:
        return math.nan
    base_ade = trajectory_displacement(target, baseline)
    masked_ade = trajectory_displacement(target, masked)
    if not math.isfinite(base_ade) or not math.isfinite(masked_ade):
        return math.nan
    return float(masked_ade - base_ade)


def zscore(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    if not bool(finite.any()):
        return np.zeros_like(array)
    mean = float(array[finite].mean())
    std = float(array[finite].std())
    if std <= 1e-12:
        return np.zeros_like(array)
    result = np.zeros_like(array)
    result[finite] = (array[finite] - mean) / std
    return result


def combined_harm(
    pdm_harm: Sequence[float],
    traj_disp: Sequence[float],
    planning_harm_values: Sequence[float],
) -> np.ndarray:
    """Equal-weight z-score sum of the three harm signals (lower is better)."""

    components = [
        zscore(pdm_harm),
        zscore(traj_disp),
        zscore(planning_harm_values),
    ]
    return np.nansum(np.stack(components, axis=0), axis=0)


# Objective name -> (direction, requires_trajectories, requires_target)
OBJECTIVES: dict[str, dict[str, Any]] = {
    "pdm": {"higher_is_better": True, "needs_trajectory": False, "needs_target": False},
    "pdm_harm": {"higher_is_better": False, "needs_trajectory": False, "needs_target": False},
    "traj_disp": {"higher_is_better": False, "needs_trajectory": True, "needs_target": False},
    "planning_harm": {"higher_is_better": False, "needs_trajectory": True, "needs_target": True},
    "combined": {"higher_is_better": False, "needs_trajectory": True, "needs_target": True},
}


def objective_value(
    objective: str,
    metrics: Mapping[str, float],
) -> float:
    """Extract one scalar objective from an aggregated metric mapping.

    ``metrics`` must contain ``pdm``, ``pdm_harm``, ``traj_disp``,
    ``planning_harm`` and ``combined_harm`` (missing values become NaN).
    """

    name = str(objective).strip().lower()
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
    return float(metrics.get(name, math.nan))


def objective_higher_is_better(objective: str) -> bool:
    name = str(objective).strip().lower()
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)}")
    return bool(OBJECTIVES[name]["higher_is_better"])


def load_trajectory(method_dir: str | Path, scene_token: str) -> np.ndarray | None:
    """Load the dumped predicted trajectory for one scene (first rank file)."""

    matches = sorted(Path(method_dir).glob(f"trajectories/{scene_token}.rank*.npz"))
    if not matches:
        return None
    with np.load(matches[0]) as payload:
        return np.asarray(payload["traj"], dtype=np.float64)


def load_target_trajectory(
    method_dir: str | Path, scene_token: str
) -> np.ndarray | None:
    path = Path(method_dir) / "target_trajectories" / f"{scene_token}.npy"
    if not path.is_file():
        return None
    return np.asarray(np.load(path), dtype=np.float64)

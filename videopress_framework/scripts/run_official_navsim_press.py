#!/usr/bin/env python3
"""Run VideoTokenPress through the official DriveVA/NAVSIM evaluation path.

This runner deliberately keeps the official evaluator as the owner of scene
loading, feature construction, Wan inference, trajectory conversion and PDM
scoring.  It supplies an externally constructed official pipeline only so a
runtime-only press can be installed on that pipeline instance.  No file under
``diffsynth`` or ``examples`` is modified by this integration.

The official evaluator's causal attention and gradient methods need a probe
forward because their scores are not available at the pre-concatenation
VIDEO_INPUT boundary.  The runner performs that probe, stores the frozen
ranking in ``ScoreCache``, and then lets the normal intervention forward use
the cached ranking.  Other methods use one official forward per scene.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
POC_TEST_DERIVED_STATUS = ["POC_ONLY", "TEST_DERIVED", "NOT_FOR_OFFICIAL_REPORTING"]

# ---------------------------------------------------------------------------
# Evaluation protocols.
#
# Absolute PDM is only comparable to published NAVSIM numbers on the primary
# protocol (`navtest-7876`).  The project previously reported the 1,920-scene
# split-audit subset, whose harder scene composition lowers every absolute PDM by
# about one point while leaving paired deltas unchanged.  See
# `reports/eval_protocol_baseline_discrepancy_20260911.md`.
#
#   navtest-7876 : 7,876-scene navtest enumeration (navtest.yaml, 136 drive logs,
#                  metric_cache_full).  Paper-comparable AND leak-free: its drive
#                  logs are disjoint from the 3,768-scene training manifest.
#   split-test-1920 : 1,920-scene split-audit subset built from route-repaired
#                  metadata (147 source files from the same navtest pool, but a
#                  different window enumeration).  Auditable sub-protocol.
# ---------------------------------------------------------------------------
_NVME = Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1")

EVAL_PROTOCOLS: dict[str, dict[str, Any]] = {
    "navtest-7876": {
        "description": "7,876-scene navtest enumeration; paper-comparable primary protocol",
        "expected_scenes": 7876,
        "baselines": {
            "no_press_pdm": 0.9098390680735604,
            "learned_threshold40_pdm": 0.9096916412430458,
        },
        "navsim_log_path": _NVME / "openscene-v1.1/meta_datas/test",
        "sensor_blobs_path": _NVME / "openscene-v1.1/sensor_blobs/test",
        "metric_cache_path": _NVME / "metric_cache_full",
        "scene_filter_yaml": PROJECT_ROOT
        / "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml",
    },
    "split-test-1920": {
        "description": "1,920-scene split-audit subset; harder composition, paired deltas only",
        "expected_scenes": 1920,
        "baselines": {
            "no_press_pdm": 0.8998275028270585,
            "learned_threshold40_pdm": 0.8989701857427245,
        },
        "navsim_log_path": FRAMEWORK_ROOT / "outputs/navsim_official_test_repaired/metadata",
        "sensor_blobs_path": _NVME / "openscene-v1.1/sensor_blobs/test",
        "metric_cache_path": _NVME / "metric_cache_split_test",
        "scene_filter_yaml": FRAMEWORK_ROOT
        / "outputs/navsim_official_test_repaired/official_test_repaired_1920_scene_filter.yaml",
    },
}

PRIMARY_EVAL_PROTOCOL = "navtest-7876"

# Fixed by the backbone: the DriveVA history latent is a 60x104 patch grid at
# 480x832, so one latent contributes 390 candidate tokens.
_HISTORY_TOKENS_PER_LATENT = 390

_PATH_KEYS = ("navsim_log_path", "sensor_blobs_path", "metric_cache_path", "scene_filter_yaml")


def resolve_eval_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Fill unspecified data paths from a protocol preset and label the run.

    Priority rules (documented so absolute numbers stay self-describing):

    * every path left as ``None`` is taken from the requested preset (``auto``
      means the primary protocol);
    * an explicitly supplied path always wins, but if it contradicts the preset
      the label is suffixed with ``+overridden`` so the number cannot be quoted
      as a clean protocol result;
    * if no preset matches the supplied paths, the label becomes ``custom``.
    """
    requested = str(getattr(args, "eval_protocol", "auto") or "auto")
    if requested not in ("auto", "custom") and requested not in EVAL_PROTOCOLS:
        raise ValueError(
            f"unknown eval protocol {requested!r}; choose from "
            f"{sorted(EVAL_PROTOCOLS)} or 'custom'"
        )
    supplied = {key: getattr(args, key, None) for key in _PATH_KEYS}
    if requested == "custom" or (requested == "auto" and all(value is not None for value in supplied.values())):
        # Wholly caller-specified data: try to recognise it, otherwise custom.
        label = "custom"
        for name, preset in EVAL_PROTOCOLS.items():
            if all(
                Path(supplied[key]).resolve() == Path(preset[key]).resolve()
                for key in _PATH_KEYS
            ):
                label = name
                break
        resolved = {key: Path(supplied[key]) for key in _PATH_KEYS}
        return {"label": label, "preset": EVAL_PROTOCOLS.get(label), "overridden": False, **resolved}

    name = requested
    if name == "auto":
        if any(value is not None for value in supplied.values()):
            # Partial specification: complete it from the primary protocol.
            name = PRIMARY_EVAL_PROTOCOL
        else:
            name = PRIMARY_EVAL_PROTOCOL
    preset = EVAL_PROTOCOLS[name]
    overridden = False
    resolved: dict[str, Any] = {}
    for key in _PATH_KEYS:
        if supplied[key] is None:
            resolved[key] = Path(preset[key])
        else:
            resolved[key] = Path(supplied[key])
            if Path(supplied[key]).resolve() != Path(preset[key]).resolve():
                overridden = True
    return {
        "label": f"{name}+overridden" if overridden else name,
        "preset": preset,
        "overridden": overridden,
        **resolved,
    }
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from evaluation.artifacts import environment_snapshot, jsonable
from evaluation.statistics import aggregate_suite, write_suite_tables
from evaluation.visualization import generate_suite_visualizations
from scripts.run_full_compression_suite import method_specs, persistent_attention_vnorm_spec
from videopress.adapters.driveva import DriveVAAdapter
from videopress.adapters.scene_boundary import install_scene_boundary_guard, window_is_single_scene
from videopress.core.context import TokenContext
from videopress.core.domain import build_domain
from videopress.core.layout import TokenLayout
from videopress.core.retention import HISTORY_RETENTION_POLICIES, apply_history_retention_policy
from videopress.core.runtime import InjectionPoint, VideoPressRuntime
from videopress.factory import build_press
from videopress.probes.score_cache import ScoreCache
from videopress.scorers.planning_gradient import (
    OBJECTIVE_TYPE as PLANNING_OBJECTIVE_TYPE,
    SCORE_REDUCTION as PLANNING_SCORE_REDUCTION,
    original_gradient_input_reduction,
    trajectory_projection_objective,
)
from videopress.utils.tensor import canonicalize_qkv


def _load_official_eval_module():
    """Load the repository-local official evaluator by absolute path."""

    path = PROJECT_ROOT / "examples" / "wanvideo" / "driveva_infer" / "eval_navsim_pdm.py"
    spec = importlib.util.spec_from_file_location("driveva_lite_official_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_official_data_environment(repo_root: Path) -> dict[str, str]:
    """Force the official process to use this DriveVA data mount.

    The host may have another project exporting ``OPENSCENE_DATA_ROOT`` or
    ``NUPLAN_MAPS_ROOT``.  NAVSIM reads these values at module-import time, so
    inheriting them can make a correct DriveVA run fail before inference with a
    misleading missing-map error.  The evaluator arguments remain the source
    of truth for log/sensor/cache paths; these variables align the libraries
    that construct maps and scenarios with the same repository.
    """

    repo_root = repo_root.expanduser().resolve()
    openscene_root = repo_root / "data" / "navsim_v1.1" / "openscene-v1.1"
    map_candidates = (
        repo_root / "data" / "nuplan" / "nuplan-maps-v1.0",
        repo_root / "data" / "nuplan" / "maps",
    )
    map_root = next(
        (candidate for candidate in map_candidates if (candidate / "nuplan-maps-v1.0.json").exists()),
        None,
    )
    if not openscene_root.exists():
        raise FileNotFoundError(f"DriveVA OPENSCENE_DATA_ROOT does not exist: {openscene_root}")
    if map_root is None:
        raise FileNotFoundError(
            "DriveVA NUPLAN_MAPS_ROOT does not contain nuplan-maps-v1.0.json; tried:\n"
            + "\n".join(str(candidate) for candidate in map_candidates)
        )
    values = {
        "OPENSCENE_DATA_ROOT": str(openscene_root),
        "NUPLAN_MAPS_ROOT": str(map_root),
        "NUPLAN_DATA_ROOT": str(repo_root / "data" / "nuplan"),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def _optional_path(value: str | None, default: Path) -> str:
    return str(Path(value).expanduser().resolve() if value else default.resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete VideoTokenPress matrix with official DriveVA/NAVSIM PDM."
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--navsim-log-path",
        type=Path,
        default=None,
        help="defaults to the primary eval protocol (navtest-7876)",
    )
    parser.add_argument(
        "--sensor-blobs-path",
        type=Path,
        default=None,
        help="defaults to the primary eval protocol (navtest-7876)",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        default=None,
        help="defaults to the primary eval protocol (navtest-7876)",
    )
    parser.add_argument(
        "--full-ckpt",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/pdms90_9.safetensors",
    )
    parser.add_argument("--local-model-path", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--eval-protocol",
        type=str,
        default="auto",
        choices=["auto", "custom", *sorted(EVAL_PROTOCOLS)],
        help=(
            "data protocol preset; 'auto' (default) uses the primary paper-comparable "
            f"protocol {PRIMARY_EVAL_PROTOCOL!r} unless every data path is given explicitly"
        ),
    )
    parser.add_argument(
        "--scene-filter-yaml",
        type=Path,
        default=None,
        help="defaults to the primary eval protocol (navtest-7876)",
    )
    parser.add_argument(
        "--allow-missing-route",
        action="store_true",
        help="Evaluate explicitly selected scenes even when the current frame has no route roadblocks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FRAMEWORK_ROOT / "outputs/official_navsim_full",
    )
    parser.add_argument(
        "--score-cache-root",
        type=Path,
        default=None,
        help="Optional root for frozen probe score caches; useful for large runs on NVMe.",
    )
    parser.add_argument("--methods", default=None, help="comma-separated method names; default is the complete compatible matrix")
    parser.add_argument(
        "--dynamic-selector-checkpoint",
        type=Path,
        default=None,
        help="run the learned layer-15 selector matrix using this distilled checkpoint",
    )
    parser.add_argument(
        "--c4-replica-triad-checkpoint",
        type=Path,
        default=None,
        help=(
            "run exactly three frozen C4 arms: physical_no_press, the original "
            "layer-15 attention-vnorm adaptive-balanced baseline, and the learned "
            "two-history-latent [0.05,0.40] threshold selector from this checkpoint"
        ),
    )
    parser.add_argument(
        "--dynamic-selector-suite",
        choices=("legacy", "signed"),
        default="legacy",
        help="legacy magnitude-teacher matrix or signed-teacher risk-gated matrix",
    )
    parser.add_argument("--signed-selector-threshold", type=float, default=0.5)
    parser.add_argument("--signed-selector-abstain-margin", type=float, default=0.03)
    parser.add_argument("--signed-selector-min-keep-ratio", type=float, default=0.375)
    parser.add_argument("--signed-selector-min-drop-ratio", type=float, default=0.05)
    parser.add_argument(
        "--promising-full-matrix",
        action="store_true",
        help=(
            "run the layer-16 hidden-sequence full-test matrix: baseline, fixed "
            "37.5/50 percent, balanced adaptive K, and cautious adaptive K"
        ),
    )
    parser.add_argument(
        "--breakthrough-full-matrix",
        action="store_true",
        help=(
            "run the next-stage hidden-sequence matrix: layer-15 adaptive Top-K "
            "and layer-15/16 adaptive spatial-coverage selection"
        ),
    )
    parser.add_argument(
        "--pre-dit-token-matrix",
        action="store_true",
        help=(
            "run a true block-0-input history-token pruning pilot: no-press, "
            "equal-budget random controls, fixed TokenNorm, and dynamic "
            "TokenNorm selection"
        ),
    )
    parser.add_argument(
        "--pre-dit-token-matrix-profile",
        choices=("main", "conservative"),
        default="main",
        help=(
            "pre-DiT pilot profile: 'main' tests 50/75 percent plus adaptive K; "
            "'conservative' isolates the 90/95 percent quality cliff"
        ),
    )
    parser.add_argument(
        "--pre-dit-token-methods",
        default=None,
        help=(
            "optional comma-separated subset of the selected pre-DiT matrix; "
            "intended for independent multi-GPU shards"
        ),
    )
    parser.add_argument(
        "--pre-dit-merge-matrix",
        action="store_true",
        help=(
            "pre-DiT merge-vs-prune matrix at an identical output sequence "
            "length: no-press, random top-K prune, random-grouping merge and "
            "similarity merge all emit the same number of tokens"
        ),
    )
    parser.add_argument(
        "--pre-dit-merge-keep-ratio",
        type=float,
        default=0.5,
        help="candidate keep ratio shared by every arm of --pre-dit-merge-matrix",
    )
    parser.add_argument(
        "--pre-dit-register-checkpoint",
        type=Path,
        default=None,
        help=(
            "optional RegisterBottleneck weights for the learnable-merge arm of "
            "--pre-dit-merge-matrix; omitted means a fresh (untrained) module"
        ),
    )
    parser.add_argument(
        "--pre-dit-register-lora-checkpoint",
        type=Path,
        default=None,
        help=(
            "optional LoRA weights (same checkpoint as the bottleneck) applied "
            "to pipe.dit before the learnable-merge arm runs"
        ),
    )
    parser.add_argument("--pre-dit-register-lora-rank", type=int, default=32)
    parser.add_argument(
        "--pre-dit-register-lora-target-modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
    )
    parser.add_argument(
        "--pre-dit-learned-checkpoint",
        type=Path,
        default=None,
        help=(
            "add a pre-DiT arm that scores block-0 input with a distilled "
            "planning selector checkpoint, plus an equal-budget random control"
        ),
    )
    parser.add_argument(
        "--pre-dit-learned-domain",
        choices=("last_history", "history"),
        default="last_history",
        help=(
            "candidate domain for the learned pre-DiT arm; last_history matches "
            "the online teacher, which supervises only the newest history latent"
        ),
    )
    parser.add_argument(
        "--pre-dit-learned-threshold",
        type=float,
        default=0.4,
        help="absolute keep threshold applied to the learned selector scores",
    )
    parser.add_argument(
        "--pre-dit-learned-random-keep-ratio",
        type=float,
        default=0.5,
        help="budget of the matched random control for the learned pre-DiT arm",
    )
    parser.add_argument(
        "--pre-dit-learned-feature-mode",
        choices=("all", "condition_position_time"),
        default="all",
        help="must match the checkpoint's training feature mode",
    )
    parser.add_argument(
        "--temporal-motion-matrix",
        action="store_true",
        help=(
            "run layer-15 spatial adaptive ablations blending attention-vnorm "
            "with cross-history token change"
        ),
    )
    parser.add_argument(
        "--persistent-layer-sweep",
        default=None,
        help=(
            "run physical attention-vnorm persistent K/V pruning for each requested "
            "start layer; accepts 'all', '0-29', or comma-separated values/ranges"
        ),
    )
    parser.add_argument(
        "--persistent-end-layer",
        type=int,
        default=None,
        help="optional inclusive final layer for persistent reuse; default is the last DiT layer",
    )
    parser.add_argument(
        "--persistent-oneshot",
        action="store_true",
        help=(
            "disable downstream reuse and prune K/V only at the source layer; "
            "cannot be combined with --persistent-end-layer or hidden_sequence"
        ),
    )
    parser.add_argument(
        "--persistent-mode",
        choices=("kv_only", "hidden_sequence"),
        default="kv_only",
        help=(
            "kv_only gathers fresh K/V at every selected layer; hidden_sequence "
            "physically shortens the residual stream after the source layer"
        ),
    )
    parser.add_argument(
        "--persistent-keep-ratio",
        type=float,
        default=0.5,
        help="eligible-token keep ratio used by persistent layer sweeps",
    )
    parser.add_argument(
        "--persistent-scorer",
        choices=(
            "action_attention_vnorm",
            "action_attention_vnorm_temporal",
            "action_contribution_stability",
            "learned_planning_selector",
            "random",
        ),
        default="action_attention_vnorm",
        help="source-layer importance scorer used by persistent sweeps",
    )
    parser.add_argument(
        "--persistent-learned-checkpoint",
        type=Path,
        default=None,
        help=(
            "with --persistent-scorer learned_planning_selector: distilled selector "
            "checkpoint used at every sweep layer. Reproduces the deployed "
            "learned threshold-0.4 arm (layer 15) and extends it to deeper start "
            "layers, so the compression start point can be moved without changing "
            "anything else about the selector."
        ),
    )
    parser.add_argument(
        "--persistent-feature-layer",
        type=int,
        default=None,
        help=(
            "feature-read layer for learned_planning_selector; the sweep layer "
            "remains the compression/persistence source (fixes BUG-12)"
        ),
    )
    parser.add_argument(
        "--persistent-future-position-mode",
        choices=("storage", "history_compatible"),
        default="storage",
        help=(
            "temporal position feature for learned_planning_selector on future "
            "domains: real storage index (2,3) or history-compatible range (0,1)"
        ),
    )
    parser.add_argument(
        "--persistent-selector",
        choices=(
            "topk",
            "threshold",
            "history_threshold",
            "history_quota",
            "future_threshold",
            "future_quota",
            "adaptive_mass",
            "adaptive_spatial_mass",
        ),
        default="topk",
        help=(
            "fixed Top-K, absolute-probability threshold, or risk-gated dynamic K. "
            "'threshold' is the deployed learned threshold-0.4 selection rule; pair "
            "it with --persistent-keep-ratio 1.0 so the threshold, not a ratio, "
            "decides how many tokens are kept."
        ),
    )
    parser.add_argument(
        "--persistent-threshold",
        type=float,
        default=0.4,
        help=(
            "absolute keep-probability threshold for --persistent-selector threshold; "
            "0.4 is the deployed learned threshold-0.4 rule"
        ),
    )
    parser.add_argument(
        "--per-latent-thresholds",
        default=None,
        help=(
            "comma-separated thresholds in [oldest,newest] storage order; "
            "required by --persistent-selector history_threshold"
        ),
    )
    parser.add_argument(
        "--per-latent-keep-ratios",
        default=None,
        help=(
            "comma-separated exact keep ratios in [oldest,newest] order; "
            "required by --persistent-selector history_quota"
        ),
    )
    parser.add_argument(
        "--per-future-latent-thresholds",
        default=None,
        help=(
            "comma-separated thresholds in [near,far] storage order; "
            "required by --persistent-selector future_threshold"
        ),
    )
    parser.add_argument(
        "--per-future-latent-keep-ratios",
        default=None,
        help=(
            "comma-separated exact keep ratios in [near,far] storage order; "
            "required by --persistent-selector future_quota"
        ),
    )
    parser.add_argument(
        "--adaptive-ratios",
        default="0.375,0.5,1.0",
        help="comma-separated dynamic keep tiers; must end in 1.0",
    )
    parser.add_argument(
        "--adaptive-mass-thresholds",
        default="0.60,0.68",
        help="cumulative-mass thresholds for compressed adaptive tiers",
    )
    parser.add_argument(
        "--adaptive-gap-thresholds",
        default="0.04,0.025",
        help="local boundary-gap thresholds for compressed adaptive tiers",
    )
    parser.add_argument("--adaptive-gap-window", type=int, default=8)
    parser.add_argument("--contribution-observation-start-layer", type=int, default=None)
    parser.add_argument("--contribution-redundancy-weight", type=float, default=0.15)
    parser.add_argument("--contribution-stability-weight", type=float, default=0.25)
    parser.add_argument(
        "--persistent-skip-baseline",
        action="store_true",
        help=(
            "omit physical_no_press from a persistent layer sweep when an exactly "
            "matched baseline has already been evaluated"
        ),
    )
    parser.add_argument(
        "--domain",
        default="last_history",
        choices=(
            "last_history",
            "history",
            "all_history",
            "future_video",
            "future_latent_0",
            "future_latent_1",
        ),
        help=(
            "token selection domain; future_video covers both future latents "
            "(near,far storage order)"
        ),
    )
    parser.add_argument(
        "--history-guided-future-mapping",
        choices=(
            "same_latent",
            "reverse_latent",
            "nearest_history",
            "oldest_history",
            "union_history",
            "intersection_history",
            "majority_history",
        ),
        default=None,
        help=(
            "enable history-mask-to-future compression; selects future tokens "
            "at the local positions selected by the history thresholds"
        ),
    )
    parser.add_argument(
        "--history-guided-layer",
        type=int,
        default=15,
        help="source/compression layer for --history-guided-future-mapping",
    )
    parser.add_argument(
        "--history-guided-feature-layer",
        type=int,
        default=None,
        help="optional learned feature-read layer for history-guided compression",
    )
    parser.add_argument(
        "--history-guided-thresholds",
        default="0.05,0.40",
        help="oldest,newest history thresholds for history-guided future mapping",
    )
    parser.add_argument(
        "--retention-policy",
        choices=tuple(HISTORY_RETENTION_POLICIES),
        default=None,
        help=(
            "apply one of the six two-history-latent deletion policies; "
            "similarity merge is excluded because it cannot honor exact quotas"
        ),
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=20260828)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=3)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "override the diffusion sampling seed passed to the official evaluator "
            "(equivalent to --seed but named for best-of-N sampling); when unset the "
            "historical --seed value is used so existing arms stay bit-identical"
        ),
    )
    parser.add_argument(
        "--dump-trajectories",
        action="store_true",
        help=(
            "write one .npz per evaluated scene into <method_dir>/trajectories/ "
            "containing the raw ego-relative predicted trajectory (float32, metres) "
            "returned by the diffusion planner; used for best-of-N oracle analysis"
        ),
    )
    parser.add_argument(
        "--dump-history-tokens",
        action="store_true",
        help=(
            "write the DiT hidden states of the candidate history latent into "
            "<method_dir>/history_tokens/. Run the SAME scene twice with different "
            "--sample-seed values and compare: the clean history is unchanged while "
            "the noised future differs, so bit-identical history tokens prove the "
            "next latent does not leak into the candidate representation at this "
            "layer (P0-2 leakage control)"
        ),
    )
    parser.add_argument(
        "--selector-layer",
        type=int,
        default=15,
        help=(
            "DiT block whose residual stream is read by --dump-history-tokens "
            "(must match the layer a learned selector reads)"
        ),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-history-frames", type=int, default=5)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--model-future-frames", type=int, default=8)
    parser.add_argument("--target-fps", type=int, default=2)
    parser.add_argument("--pdm-num-poses", type=int, default=40)
    parser.add_argument("--pdm-interval-length", type=float, default=0.1)
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument(
        "--infer-video",
        action="store_true",
        help=(
            "decode the model-generated future video (slow); pair with "
            "--save-viz to render GT-vs-pred camera videos. Default keeps "
            "trajectory-only mode (no video decode)."
        ),
    )
    parser.add_argument("--viz-total-tokens", type=int, default=100)
    parser.add_argument("--viz-max-tokens", type=int, default=20)
    parser.add_argument("--enable-nuscenes-metrics", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument(
        "--poc-test-derived",
        action="store_true",
        help="mark every emitted artifact as POC_ONLY / TEST_DERIVED / NOT_FOR_OFFICIAL_REPORTING",
    )
    parser.add_argument(
        "--force-full-scene-set",
        action="store_true",
        help="fail if the official metric-cache intersection is smaller than the scene-filter token set",
    )
    parser.add_argument(
        "--gradient-debug-compare",
        action="store_true",
        help="also compute framework Gradient x Input diagnostics for the planning method",
    )
    return parser.parse_args(argv)


def parse_layer_sweep(spec: str, *, num_layers: int = 30) -> list[int]:
    """Parse an inclusive layer list such as ``all``, ``0-29`` or ``8,12-16``."""

    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    text = str(spec).strip().lower()
    if not text:
        raise ValueError("persistent layer sweep cannot be empty")
    if text == "all":
        return list(range(num_layers))
    layers: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"invalid persistent layer sweep: {spec}")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending layer range is not allowed: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= num_layers)
    if invalid:
        raise ValueError(
            f"persistent layers outside [0, {num_layers - 1}]: {invalid}"
        )
    return sorted(layers)


def _parse_float_list(spec: str, name: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated float list") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def _pre_dit_learned_specs(args, round_seed, pre_dit_spec, baseline) -> list[dict[str, Any]]:
    """Optional learned-selector arms for the pre-DiT matrix.

    The online teacher supervises only the newest conditioned history latent
    (``--selector-counterfactual-latent-index 0``), so a checkpoint trained that
    way must be deployed with ``domain=last_history``.  Scoring the older latent
    as well (``domain=history``) reads out-of-distribution features that were
    never supervised, which is exactly how the TokenNorm pilot acquired its
    "never drop the newest latent" bias.
    """

    checkpoint = getattr(args, "pre_dit_learned_checkpoint", None)
    if checkpoint is None:
        return []
    domain = str(getattr(args, "pre_dit_learned_domain", "last_history")).strip().lower()
    if domain not in {"last_history", "history"}:
        raise ValueError("--pre-dit-learned-domain must be last_history or history")
    threshold = float(getattr(args, "pre_dit_learned_threshold", 0.4))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--pre-dit-learned-threshold must be within [0, 1]")
    random_keep = float(getattr(args, "pre_dit_learned_random_keep_ratio", 0.5))
    if not 0.0 < random_keep <= 1.0:
        raise ValueError("--pre-dit-learned-random-keep-ratio must be within (0, 1]")
    scorer_options = {
        "checkpoint": str(checkpoint),
        # The pre-DiT press reads the block-0 input directly, so the scorer's
        # feature layer is the same depth as its compression layer.
        "layer": 0,
        "feature_layer": 0,
        "feature_mode": str(getattr(args, "pre_dit_learned_feature_mode", "all")),
    }
    return [
        pre_dit_spec(
            f"physical_pre_dit_learned_{domain}_threshold_{threshold:g}",
            "learned_planning_selector",
            1.0,
            selector={"name": "threshold", "threshold": threshold},
            domain=domain,
            scorer_options=scorer_options,
        ),
        pre_dit_spec(
            f"physical_pre_dit_random_{domain}_keep{int(round(random_keep * 100)):02d}",
            "random",
            random_keep,
            seed=round_seed + 107,
            domain=domain,
        ),
    ]


def _method_specs_for_run(args: argparse.Namespace, round_seed: int) -> list[dict[str, Any]]:
    domain_for_policy = str(getattr(args, "domain", "last_history"))
    if getattr(args, "retention_policy", None) is not None and domain_for_policy.startswith(
        "future"
    ):
        raise ValueError(
            "--retention-policy is history-only and cannot be applied to "
            f"future domain {domain_for_policy!r}"
        )
    if getattr(args, "pre_dit_merge_matrix", False):
        conflicts = (
            getattr(args, "dynamic_selector_checkpoint", None),
            getattr(args, "c4_replica_triad_checkpoint", None),
            getattr(args, "persistent_layer_sweep", None),
            getattr(args, "promising_full_matrix", False),
            getattr(args, "breakthrough_full_matrix", False),
            getattr(args, "temporal_motion_matrix", False),
            getattr(args, "pre_dit_token_matrix", False),
            getattr(args, "methods", None),
        )
        if any(conflicts):
            raise ValueError(
                "--pre-dit-merge-matrix cannot be combined with another method matrix"
            )
        if args.domain not in {"history", "all_history", "last_history"}:
            raise ValueError("--pre-dit-merge-matrix requires a history domain")
        if args.retention_policy is not None:
            raise ValueError("--pre-dit-merge-matrix cannot use a retention policy")
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="history")
            if spec["name"] == "physical_no_press"
        )
        keep_ratio = float(getattr(args, "pre_dit_merge_keep_ratio", 0.5))
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--pre-dit-merge-keep-ratio must be within (0, 1]")
        domain = str(args.domain)
        tag = f"{domain}_keep{int(round(keep_ratio * 100)):02d}"

        def merge_spec(name, feature, seed=0):
            config = {
                "name": "similarity_merge",
                "injection_point": "block_input",
                "domain": domain,
                "feature": feature,
                "budget": {
                    "type": "ratio",
                    "value": keep_ratio,
                    "reference": "eligible",
                },
            }
            if feature == "random":
                config["seed"] = int(seed)
            return {"name": name, "mode": "physical", "press": config}

        def prune_spec(name, scorer, seed=0):
            config = {
                "name": "scorer_press",
                "injection_point": "block_input",
                "domain": domain,
                "scorer": {"name": scorer},
                "selector": {"name": "topk"},
                "operator": {"name": "hidden_prune"},
                "budget": {
                    "type": "ratio",
                    "value": keep_ratio,
                    "reference": "eligible",
                },
            }
            if scorer == "random":
                config["scorer"].update({"seed": int(seed), "scope": "scene_step"})
            return {"name": name, "mode": "physical", "press": config}

        # Both operators emit exactly `protected + round(ratio * n_candidate)`
        # tokens, so every arm runs the identical DiT sequence length: the
        # comparison isolates information retention, not compute.
        def register_spec(name):
            n_candidate = 390 if domain == "last_history" else 780
            config = {
                "name": "register_merge",
                "injection_point": "block_input",
                "domain": domain,
                "num_key_tokens": int(round(keep_ratio * n_candidate)),
                "budget": {
                    "type": "ratio",
                    "value": keep_ratio,
                    "reference": "eligible",
                },
            }
            checkpoint = getattr(args, "pre_dit_register_checkpoint", None)
            if checkpoint:
                config["checkpoint"] = str(checkpoint)
            return {"name": name, "mode": "physical", "press": config}

        # The combined training checkpoint carries both LoRA and bottleneck
        # weights; apply the same LoRA config to the eval pipe.
        if getattr(args, "pre_dit_register_lora_checkpoint", None):
            if not getattr(args, "pre_dit_register_checkpoint", None):
                args.pre_dit_register_checkpoint = args.pre_dit_register_lora_checkpoint

        specs = [
            baseline,
            prune_spec(f"physical_pre_dit_prune_random_{tag}", "random", round_seed + 111),
            merge_spec(f"physical_pre_dit_merge_random_{tag}", "random", round_seed + 112),
            merge_spec(f"physical_pre_dit_merge_similarity_{tag}", "tokens"),
            merge_spec(f"physical_pre_dit_merge_kmeans_{tag}", "kmeans"),
            # Reference grouping kept so a fast-grouping change can be isolated
            # from a merge-vs-prune change instead of confounded with it.
            merge_spec(f"physical_pre_dit_merge_greedy_{tag}", "greedy"),
        ]
        if domain == "last_history":
            specs.append(register_spec(f"physical_pre_dit_register_{tag}"))
        requested_text = getattr(args, "pre_dit_token_methods", None)
        if requested_text:
            requested = [item.strip() for item in requested_text.split(",") if item.strip()]
            available = {spec["name"]: spec for spec in specs}
            unknown = [name for name in requested if name not in available]
            if unknown:
                raise ValueError(
                    f"unknown --pre-dit-token-methods {unknown}; available={list(available)}"
                )
            return [available[name] for name in requested]
        return specs

    if getattr(args, "pre_dit_token_matrix", False):
        conflicts = (
            getattr(args, "dynamic_selector_checkpoint", None),
            getattr(args, "c4_replica_triad_checkpoint", None),
            getattr(args, "persistent_layer_sweep", None),
            getattr(args, "promising_full_matrix", False),
            getattr(args, "breakthrough_full_matrix", False),
            getattr(args, "temporal_motion_matrix", False),
            getattr(args, "methods", None),
        )
        if any(conflicts):
            raise ValueError(
                "--pre-dit-token-matrix cannot be combined with another method matrix"
            )
        learned_checkpoint = getattr(args, "pre_dit_learned_checkpoint", None)
        allowed_domains = {"history", "all_history"}
        if learned_checkpoint is not None:
            allowed_domains.add("last_history")
        if args.domain not in allowed_domains or args.retention_policy is not None:
            raise ValueError(
                "--pre-dit-token-matrix requires --domain history (or last_history "
                "with --pre-dit-learned-checkpoint) and no retention policy"
            )
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="history")
            if spec["name"] == "physical_no_press"
        )

        def pre_dit_spec(
            name, scorer, keep_ratio, *, selector=None, seed=0, domain=None, scorer_options=None
        ):
            config = {
                "name": "scorer_press",
                "injection_point": "block_input",
                "domain": str(domain or "history"),
                "scorer": dict(scorer_options or {}, name=scorer),
                "selector": dict(selector or {"name": "topk"}),
                "operator": {"name": "hidden_prune"},
                "budget": {
                    "type": "ratio",
                    "value": float(keep_ratio),
                    "reference": "eligible",
                },
            }
            if scorer == "random":
                config["scorer"].update({"seed": int(seed), "scope": "scene_step"})
            return {"name": name, "mode": "physical", "press": config}

        def select_pre_dit_methods(specs):
            requested_text = getattr(args, "pre_dit_token_methods", None)
            if not requested_text:
                return specs
            requested = [item.strip() for item in requested_text.split(",") if item.strip()]
            available = {spec["name"]: spec for spec in specs}
            unknown = [name for name in requested if name not in available]
            if unknown:
                raise ValueError(
                    f"unknown --pre-dit-token-methods {unknown}; "
                    f"available={list(available)}"
                )
            return [available[name] for name in requested]

        if getattr(args, "pre_dit_token_matrix_profile", "main") == "conservative":
            return select_pre_dit_methods([
                baseline,
                pre_dit_spec(
                    "physical_pre_dit_random_history_keep95",
                    "random",
                    0.95,
                    seed=round_seed + 103,
                ),
                pre_dit_spec(
                    "physical_pre_dit_token_norm_history_keep95",
                    "token_norm",
                    0.95,
                ),
                pre_dit_spec(
                    "physical_pre_dit_random_history_keep90",
                    "random",
                    0.90,
                    seed=round_seed + 104,
                ),
                pre_dit_spec(
                    "physical_pre_dit_token_norm_history_keep90",
                    "token_norm",
                    0.90,
                ),
            ] + _pre_dit_learned_specs(args, round_seed, pre_dit_spec, baseline))

        return select_pre_dit_methods([
            baseline,
            pre_dit_spec(
                "physical_pre_dit_random_history_keep75",
                "random",
                0.75,
                seed=round_seed + 101,
            ),
            pre_dit_spec(
                "physical_pre_dit_token_norm_history_keep75",
                "token_norm",
                0.75,
            ),
            pre_dit_spec(
                "physical_pre_dit_random_history_keep50",
                "random",
                0.50,
                seed=round_seed + 102,
            ),
            pre_dit_spec(
                "physical_pre_dit_token_norm_history_keep50",
                "token_norm",
                0.50,
            ),
            pre_dit_spec(
                "physical_pre_dit_token_norm_history_dynamic",
                "token_norm",
                1.0,
                selector={
                    "name": "adaptive_mass",
                    "ratios": [0.5, 0.75, 1.0],
                    "mass_thresholds": [0.60, 0.80],
                    "gap_thresholds": [0.02, 0.01],
                    "gap_window": 8,
                },
            ),
        ] + _pre_dit_learned_specs(args, round_seed, pre_dit_spec, baseline))
    guided_mapping = getattr(args, "history_guided_future_mapping", None)
    if guided_mapping is not None:
        if getattr(args, "methods", None):
            raise ValueError(
                "--history-guided-future-mapping cannot be combined with --methods"
            )
        if args.retention_policy is not None:
            raise ValueError(
                "--history-guided-future-mapping is not compatible with "
                "--retention-policy"
            )
        learned_checkpoint = (
            getattr(args, "persistent_learned_checkpoint", None)
            or getattr(args, "dynamic_selector_checkpoint", None)
        )
        if learned_checkpoint is None:
            raise ValueError(
                "--history-guided-future-mapping requires "
                "--persistent-learned-checkpoint"
            )
        layer = int(getattr(args, "history_guided_layer", 15))
        if layer < 0 or layer >= 30:
            raise ValueError("--history-guided-layer must be within [0, 29]")
        feature_layer = getattr(args, "history_guided_feature_layer", None)
        if feature_layer is None:
            feature_layer = getattr(args, "persistent_feature_layer", None)
        feature_layer = int(layer if feature_layer is None else feature_layer)
        if feature_layer < 0 or feature_layer > layer:
            raise ValueError(
                "--history-guided-feature-layer must be in [0, --history-guided-layer]"
            )
        thresholds = _parse_float_list(
            getattr(args, "history_guided_thresholds", "0.05,0.40"),
            "--history-guided-thresholds",
        )
        if len(thresholds) != 2 or any(
            value < 0.0 or value > 1.0 for value in thresholds
        ):
            raise ValueError(
                "--history-guided-thresholds needs two values within [0, 1] "
                "in oldest,newest order"
            )
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="all_video")
            if spec["name"] == "physical_no_press"
        )
        guided_name = (
            "physical_history_guided_future_"
            f"{str(guided_mapping).lower()}_hidden_persistent_layer_{layer:02d}"
        )
        press = {
            "name": "scorer_press",
            "injection_point": "self_attn_kv",
            "domain": "all_video",
            "scorer": {
                "name": "learned_planning_selector",
                "layer": layer,
                "feature_layer": feature_layer,
                "checkpoint": str(Path(learned_checkpoint).expanduser().resolve()),
                "all_video_history_only": True,
            },
            "selector": {
                "name": "history_guided_future",
                "thresholds": thresholds,
                "future_mapping": str(guided_mapping).lower(),
            },
            "operator": {"name": "kv_prune"},
            "budget": {"type": "ratio", "value": 1.0, "reference": "eligible"},
            "cross_layer_persistence": {
                "enabled": True,
                "end_layer": None,
                "mode": "hidden_sequence",
            },
        }
        specs = [] if getattr(args, "persistent_skip_baseline", False) else [baseline]
        specs.append({"name": guided_name, "mode": "physical", "press": press})
        return specs

    c4_checkpoint = getattr(args, "c4_replica_triad_checkpoint", None)
    if c4_checkpoint is not None:
        conflicts = (
            getattr(args, "dynamic_selector_checkpoint", None),
            getattr(args, "persistent_layer_sweep", None),
            getattr(args, "promising_full_matrix", False),
            getattr(args, "breakthrough_full_matrix", False),
            getattr(args, "temporal_motion_matrix", False),
            getattr(args, "methods", None),
        )
        if any(conflicts):
            raise ValueError(
                "--c4-replica-triad-checkpoint cannot be combined with another method matrix"
            )
        checkpoint = str(Path(c4_checkpoint).expanduser().resolve())
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="last_history")
            if spec["name"] == "physical_no_press"
        )
        attention_baseline = persistent_attention_vnorm_spec(
            start_layer=15,
            domain="last_history",
            persistence_mode="hidden_sequence",
            scorer_name="action_attention_vnorm",
            name="physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
            selector_config={
                "name": "adaptive_mass",
                "ratios": [0.375, 0.5, 1.0],
                "mass_thresholds": [0.60, 0.68],
                "gap_thresholds": [0.04, 0.025],
                "gap_window": 8,
            },
        )
        dynamic = persistent_attention_vnorm_spec(
            start_layer=15,
            domain="history",
            keep_ratio=1.0,
            persistence_mode="hidden_sequence",
            scorer_name="learned_planning_selector",
            scorer_options={"checkpoint": checkpoint, "feature_layer": 15},
            selector_config={"name": "history_threshold", "thresholds": [0.05, 0.40]},
            name="physical_learned_planning_selector_history_threshold_hidden_persistent_layer_15",
        )
        return [baseline, attention_baseline, dynamic]
    if getattr(args, "dynamic_selector_checkpoint", None) is not None:
        if any(
            (
                args.persistent_layer_sweep,
                args.promising_full_matrix,
                args.breakthrough_full_matrix,
                args.temporal_motion_matrix,
            )
        ):
            raise ValueError("--dynamic-selector-checkpoint cannot be combined with another method matrix")
        checkpoint = str(args.dynamic_selector_checkpoint.expanduser().resolve())
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="last_history")
            if spec["name"] == "physical_no_press"
        )
        common = {
            "start_layer": 15,
            "domain": "last_history",
            "persistence_mode": "hidden_sequence",
            "scorer_name": "learned_planning_selector",
            "scorer_options": {"checkpoint": checkpoint},
        }
        attention_baseline = persistent_attention_vnorm_spec(
            start_layer=15,
            domain="last_history",
            persistence_mode="hidden_sequence",
            scorer_name="action_attention_vnorm",
            name="physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
            selector_config={
                "name": "adaptive_mass",
                "ratios": [0.375, 0.5, 1.0],
                "mass_thresholds": [0.60, 0.68],
                "gap_thresholds": [0.04, 0.025],
                "gap_window": 8,
            },
        )
        fixed375 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=0.375,
            name="physical_learned_teacher_hidden_fixed375_layer_15",
        )
        fixed50 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=0.5,
            name="physical_learned_teacher_hidden_fixed50_layer_15",
        )
        adaptive = persistent_attention_vnorm_spec(
            **common,
            name="physical_learned_teacher_hidden_adaptive_layer_15",
            selector_config={
                "name": "adaptive_mass",
                "ratios": [0.375, 0.5, 1.0],
                "mass_thresholds": [0.60, 0.68],
                "gap_thresholds": [0.04, 0.025],
                "gap_window": 8,
            },
        )
        threshold50 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=1.0,
            name="physical_learned_teacher_hidden_threshold50_layer_15",
            selector_config={"name": "threshold", "threshold": 0.5},
        )
        threshold40 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=1.0,
            name="physical_learned_teacher_hidden_threshold40_layer_15",
            selector_config={"name": "threshold", "threshold": 0.4},
        )
        threshold30 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=1.0,
            name="physical_learned_teacher_hidden_threshold30_layer_15",
            selector_config={"name": "threshold", "threshold": 0.3},
        )
        if getattr(args, "dynamic_selector_suite", "legacy") == "signed":
            signed_threshold = float(getattr(args, "signed_selector_threshold", 0.5))
            signed_fixed_threshold = persistent_attention_vnorm_spec(
                **common,
                keep_ratio=1.0,
                name="physical_signed_teacher_hidden_threshold_layer_15",
                selector_config={"name": "threshold", "threshold": signed_threshold},
            )
            signed_risk = persistent_attention_vnorm_spec(
                **common,
                keep_ratio=1.0,
                name="physical_signed_teacher_hidden_risk_gated_layer_15",
                selector_config={
                    "name": "signed_risk",
                    "threshold": signed_threshold,
                    "abstain_margin": float(
                        getattr(args, "signed_selector_abstain_margin", 0.03)
                    ),
                    "min_keep_ratio": float(
                        getattr(args, "signed_selector_min_keep_ratio", 0.375)
                    ),
                    "min_drop_ratio": float(
                        getattr(args, "signed_selector_min_drop_ratio", 0.05)
                    ),
                },
            )
            return [
                baseline,
                attention_baseline,
                fixed375,
                signed_fixed_threshold,
                signed_risk,
            ]
        return [
            baseline,
            attention_baseline,
            fixed375,
            fixed50,
            adaptive,
            threshold50,
            threshold40,
            threshold30,
        ]
    if getattr(args, "temporal_motion_matrix", False):
        if getattr(args, "promising_full_matrix", False) or getattr(
            args, "breakthrough_full_matrix", False
        ):
            raise ValueError("choose only one fixed matrix option")
        if args.persistent_layer_sweep is not None or args.methods:
            raise ValueError(
                "--temporal-motion-matrix cannot be combined with --persistent-layer-sweep or --methods"
            )
        if args.retention_policy is not None or args.domain != "last_history":
            raise ValueError(
                "--temporal-motion-matrix requires domain=last_history and no retention policy"
            )
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="last_history")
            if spec["name"] == "physical_no_press"
        )
        selector = {
            "name": "adaptive_spatial_mass",
            "ratios": [0.375, 0.5, 1.0],
            "mass_thresholds": [0.60, 0.68],
            "gap_thresholds": [0.04, 0.025],
            "gap_window": 8,
            "tile_h": 3,
            "tile_w": 4,
        }
        common = {
            "domain": "last_history",
            "keep_ratio": 0.5,
            "persistence_mode": "hidden_sequence",
            "selector_config": selector,
        }
        attention = persistent_attention_vnorm_spec(
            15,
            **common,
            scorer_name="action_attention_vnorm",
            name="physical_attention_vnorm_hidden_adaptive_spatial_layer_15",
        )
        temporal25 = persistent_attention_vnorm_spec(
            15,
            **common,
            scorer_name="action_attention_vnorm_temporal",
            scorer_options={"temporal_weight": 0.25},
            name="physical_attention_vnorm_temporal25_hidden_adaptive_spatial_layer_15",
        )
        temporal50 = persistent_attention_vnorm_spec(
            15,
            **common,
            scorer_name="action_attention_vnorm_temporal",
            scorer_options={"temporal_weight": 0.50},
            name="physical_attention_vnorm_temporal50_hidden_adaptive_spatial_layer_15",
        )
        return [baseline, attention, temporal25, temporal50]
    if getattr(args, "breakthrough_full_matrix", False):
        if getattr(args, "promising_full_matrix", False):
            raise ValueError("choose only one fixed full-matrix option")
        if args.persistent_layer_sweep is not None:
            raise ValueError(
                "--breakthrough-full-matrix cannot be combined with --persistent-layer-sweep"
            )
        if args.retention_policy is not None or args.domain != "last_history":
            raise ValueError(
                "--breakthrough-full-matrix requires domain=last_history and no retention policy"
            )
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="last_history")
            if spec["name"] == "physical_no_press"
        )
        balanced = {
            "name": "adaptive_mass",
            "ratios": [0.375, 0.5, 1.0],
            "mass_thresholds": [0.60, 0.68],
            "gap_thresholds": [0.04, 0.025],
            "gap_window": 8,
        }
        spatial = {
            **balanced,
            "name": "adaptive_spatial_mass",
            "tile_h": 3,
            "tile_w": 4,
        }
        common = {
            "domain": "last_history",
            "persistence_mode": "hidden_sequence",
            "scorer_name": "action_attention_vnorm",
        }
        layer15 = persistent_attention_vnorm_spec(
            15,
            **common,
            name="physical_attention_vnorm_hidden_adaptive_balanced_layer_15",
            selector_config=balanced,
        )
        spatial15 = persistent_attention_vnorm_spec(
            15,
            **common,
            name="physical_attention_vnorm_hidden_adaptive_spatial_layer_15",
            selector_config=spatial,
        )
        spatial16 = persistent_attention_vnorm_spec(
            16,
            **common,
            name="physical_attention_vnorm_hidden_adaptive_spatial_layer_16",
            selector_config=spatial,
        )
        return [baseline, layer15, spatial15, spatial16]
    if getattr(args, "promising_full_matrix", False):
        if args.persistent_layer_sweep is not None or args.methods:
            raise ValueError(
                "--promising-full-matrix cannot be combined with --persistent-layer-sweep or --methods"
            )
        if args.retention_policy is not None or args.domain != "last_history":
            raise ValueError(
                "--promising-full-matrix requires domain=last_history and no retention policy"
            )
        baseline = next(
            spec
            for spec in method_specs(round_seed, domain="last_history")
            if spec["name"] == "physical_no_press"
        )
        common = {
            "start_layer": 16,
            "domain": "last_history",
            "persistence_mode": "hidden_sequence",
            "scorer_name": "action_attention_vnorm",
        }
        fixed375 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=0.375,
            name="physical_attention_vnorm_hidden_fixed375_layer_16",
        )
        fixed50 = persistent_attention_vnorm_spec(
            **common,
            keep_ratio=0.5,
            name="physical_attention_vnorm_hidden_fixed50_layer_16",
        )
        balanced = persistent_attention_vnorm_spec(
            **common,
            name="physical_attention_vnorm_hidden_adaptive_balanced_layer_16",
            selector_config={
                "name": "adaptive_mass",
                "ratios": [0.375, 0.5, 1.0],
                "mass_thresholds": [0.60, 0.68],
                "gap_thresholds": [0.04, 0.025],
                "gap_window": 8,
            },
        )
        cautious = persistent_attention_vnorm_spec(
            **common,
            name="physical_attention_vnorm_hidden_adaptive_cautious_layer_16",
            selector_config={
                "name": "adaptive_mass",
                "ratios": [0.375, 0.5, 1.0],
                "mass_thresholds": [0.60, 0.68],
                "gap_thresholds": [0.06, 0.04],
                "gap_window": 8,
            },
        )
        return [baseline, fixed375, fixed50, balanced, cautious]
    if args.persistent_layer_sweep is None:
        if getattr(args, "persistent_skip_baseline", False):
            raise ValueError("--persistent-skip-baseline requires --persistent-layer-sweep")
        return method_specs(
            round_seed,
            domain=args.domain,
            retention_policy=args.retention_policy,
        )
    if args.methods:
        raise ValueError("--methods cannot be combined with --persistent-layer-sweep")
    layers = parse_layer_sweep(args.persistent_layer_sweep)
    if not 0.0 <= float(args.persistent_keep_ratio) <= 1.0:
        raise ValueError("--persistent-keep-ratio must be within [0, 1]")
    if args.persistent_end_layer is not None:
        end_layer = int(args.persistent_end_layer)
        if end_layer < 0 or end_layer >= 30:
            raise ValueError("--persistent-end-layer must be within [0, 29]")
        if any(layer > end_layer for layer in layers):
            raise ValueError("persistent sweep start layer cannot exceed --persistent-end-layer")
    one_shot = bool(getattr(args, "persistent_oneshot", False))
    if one_shot and args.persistent_end_layer is not None:
        raise ValueError("--persistent-oneshot cannot be combined with --persistent-end-layer")
    if one_shot and args.persistent_mode != "kv_only":
        raise ValueError("--persistent-oneshot only supports --persistent-mode kv_only")
    baseline = next(
        spec
        for spec in method_specs(round_seed, domain=args.domain)
        if spec["name"] == "physical_no_press"
    )
    specs = [] if getattr(args, "persistent_skip_baseline", False) else [baseline]
    scorer_name = getattr(args, "persistent_scorer", "action_attention_vnorm")
    selector_name = getattr(args, "persistent_selector", "topk")
    selector_config: dict[str, Any] = {"name": selector_name}
    resolved_keep_ratio = float(args.persistent_keep_ratio)
    if selector_name == "threshold":
        # The deployed learned arm selects with an absolute probability
        # threshold (0.4), so the sweep must carry the same field or the
        # layer-15 arm would not reproduce the published baseline.
        selector_config["threshold"] = float(
            getattr(args, "persistent_threshold", 0.4)
        )
    if selector_name == "history_threshold":
        if args.domain not in {"history", "all_history"}:
            raise ValueError("history_threshold requires --domain history/all_history")
        raw_thresholds = getattr(args, "per_latent_thresholds", None)
        if raw_thresholds is None:
            raise ValueError(
                "history_threshold requires --per-latent-thresholds"
            )
        thresholds = _parse_float_list(
            raw_thresholds, "--per-latent-thresholds"
        )
        if len(thresholds) != 2 or any(
            value < 0.0 or value > 1.0 for value in thresholds
        ):
            raise ValueError(
                "--per-latent-thresholds needs two values within [0, 1]"
            )
        selector_config["thresholds"] = thresholds
        resolved_keep_ratio = 1.0
    if selector_name == "history_quota":
        if args.domain not in {"history", "all_history"}:
            raise ValueError("history_quota requires --domain history/all_history")
        raw_ratios = getattr(args, "per_latent_keep_ratios", None)
        if raw_ratios is None:
            raise ValueError("history_quota requires --per-latent-keep-ratios")
        ratios = _parse_float_list(raw_ratios, "--per-latent-keep-ratios")
        if len(ratios) != 2 or any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError(
                "--per-latent-keep-ratios needs two values within [0, 1]"
            )
        selector_config["ratios"] = ratios
        resolved_keep_ratio = sum(ratios) / len(ratios)
    if selector_name == "future_threshold":
        if args.domain != "future_video":
            raise ValueError(
                "future_threshold requires --domain future_video "
                "(both future latents in [near,far] order)"
            )
        raw_thresholds = getattr(args, "per_future_latent_thresholds", None)
        if raw_thresholds is None:
            raise ValueError(
                "future_threshold requires --per-future-latent-thresholds"
            )
        thresholds = _parse_float_list(
            raw_thresholds, "--per-future-latent-thresholds"
        )
        if len(thresholds) != 2 or any(
            value < 0.0 or value > 1.0 for value in thresholds
        ):
            raise ValueError(
                "--per-future-latent-thresholds needs two values within [0, 1]"
            )
        selector_config["thresholds"] = thresholds
        resolved_keep_ratio = 1.0
    if selector_name == "future_quota":
        if args.domain != "future_video":
            raise ValueError(
                "future_quota requires --domain future_video "
                "(both future latents in [near,far] order)"
            )
        raw_ratios = getattr(args, "per_future_latent_keep_ratios", None)
        if raw_ratios is None:
            raise ValueError(
                "future_quota requires --per-future-latent-keep-ratios"
            )
        ratios = _parse_float_list(
            raw_ratios, "--per-future-latent-keep-ratios"
        )
        if len(ratios) != 2 or any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError(
                "--per-future-latent-keep-ratios needs two values within [0, 1]"
            )
        selector_config["ratios"] = ratios
        resolved_keep_ratio = sum(ratios) / len(ratios)
    if selector_name in {
        "history_threshold",
        "history_quota",
        "future_threshold",
        "future_quota",
    } and args.retention_policy is not None:
        raise ValueError(
            "per-latent selector options cannot be combined with --retention-policy"
        )
    if selector_name in {"adaptive_mass", "adaptive_spatial_mass"}:
        selector_config.update(
            {
                "ratios": _parse_float_list(
                    getattr(args, "adaptive_ratios", "0.375,0.5,1.0"),
                    "--adaptive-ratios",
                ),
                "mass_thresholds": _parse_float_list(
                    getattr(args, "adaptive_mass_thresholds", "0.60,0.68"),
                    "--adaptive-mass-thresholds",
                ),
                "gap_thresholds": _parse_float_list(
                    getattr(args, "adaptive_gap_thresholds", "0.04,0.025"),
                    "--adaptive-gap-thresholds",
                ),
                "gap_window": int(getattr(args, "adaptive_gap_window", 8)),
            }
        )
    scorer_options: dict[str, Any] = {}
    if scorer_name == "random":
        scorer_options.update(
            {
                "seed": int(round_seed) + 101,
                "scope": "scene",
            }
        )
    feature_layer = getattr(args, "persistent_feature_layer", None)
    if feature_layer is not None:
        feature_layer = int(feature_layer)
        if scorer_name != "learned_planning_selector":
            raise ValueError(
                "--persistent-feature-layer requires --persistent-scorer "
                "learned_planning_selector"
            )
        if feature_layer < 0 or feature_layer >= 30:
            raise ValueError("--persistent-feature-layer must be within [0, 29]")
        if any(feature_layer > layer for layer in layers):
            raise ValueError(
                "--persistent-feature-layer cannot follow a sweep source layer"
            )
        scorer_options["feature_layer"] = feature_layer
    if scorer_name == "learned_planning_selector":
        # Deployed-selector arm: the sweep layer is written into the scorer
        # config by persistent_attention_vnorm_spec, and `scorer.layer` is the
        # single source of truth for BOTH the feature-read layer and the
        # cross-layer persistence source layer.  Putting the trained checkpoint
        # here (rather than in --dynamic-selector-checkpoint) is what makes the
        # arm differ from the layer-15 baseline in exactly one field.
        learned_checkpoint = getattr(args, "persistent_learned_checkpoint", None)
        if learned_checkpoint is None:
            raise ValueError(
                "--persistent-scorer learned_planning_selector requires "
                "--persistent-learned-checkpoint"
            )
        scorer_options["checkpoint"] = str(Path(learned_checkpoint).expanduser().resolve())
        if str(getattr(args, "domain", "")).startswith("future"):
            scorer_options["future_position_mode"] = str(
                getattr(args, "persistent_future_position_mode", "storage")
            )
    if scorer_name == "action_contribution_stability":
        observation_start = getattr(
            args, "contribution_observation_start_layer", None
        )
        if observation_start is not None:
            scorer_options["observation_start_layer"] = int(observation_start)
        scorer_options.update(
            {
                "redundancy_weight": float(
                    getattr(args, "contribution_redundancy_weight", 0.15)
                ),
                "stability_weight": float(
                    getattr(args, "contribution_stability_weight", 0.25)
                ),
            }
        )
    for layer in layers:
        spec = persistent_attention_vnorm_spec(
            layer,
            domain=args.domain,
            keep_ratio=resolved_keep_ratio,
            end_layer=args.persistent_end_layer,
            persistence_mode=args.persistent_mode,
            persistence_enabled=not one_shot,
            scorer_name=scorer_name,
            selector_config=selector_config,
            scorer_options=scorer_options,
        )
        if selector_name in {"history_quota", "future_quota"}:
            spec["press"]["budget"] = {
                "type": "absolute",
                "value": sum(
                    round(_HISTORY_TOKENS_PER_LATENT * ratio)
                    for ratio in selector_config["ratios"]
                ),
                "reference": "eligible",
            }
        if args.retention_policy is not None:
            spec["press"] = apply_history_retention_policy(
                spec["press"], args.retention_policy
            )
        specs.append(spec)
    return specs


def _latest_official_csv(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("pdm_score_*.csv"))
    if not paths:
        raise FileNotFoundError(f"official evaluator wrote no PDM CSV in {run_dir}")
    return paths[-1]


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep event records compact while retaining all reporting fields."""

    keys = {
        "press",
        "scorer",
        "selector",
        "operator",
        "domain",
        "injection_point",
        "n_history",
        "n_eligible",
        "n_candidate",
        "candidate_start",
        "candidate_end",
        "n_kept",
        "history_keep_ratio",
        "eligible_keep_ratio",
        "retention_policy",
        "selection_metadata",
        "score_diagnostics",
        "history_latent_token_count",
        "selected_history_latent_counts",
        "selected_history_latent_ratios",
        "effective_history_latent_kept_counts",
        "effective_history_latent_keep_ratios",
        "effective_history_kept",
        "effective_history_keep_ratio",
        "future_latent_token_count",
        "selected_future_latent_counts",
        "selected_future_latent_ratios",
        "effective_future_latent_kept_counts",
        "effective_future_latent_keep_ratios",
        "effective_future_kept",
        "effective_future_keep_ratio",
        "protected_count",
        "selected_count",
        "q_length",
        "k_length_before",
        "k_length_after",
        "v_length_after",
        "theoretical_attn_ratio",
        "post_rope",
        "video_only_layout",
        "feature",
        "configured_domain",
        "resolved_domain",
        "domain_override",
        "scene_token",
        "scene_tokens",
        "segment_scene_token",
        "segment_scene_name",
        "segment_frame_idx_start",
        "segment_frame_idx_end",
        "segment_frame_count",
        "segment_frame_idx_contiguous",
        "selection_candidate_valid",
        "selection_candidate_unique",
        "selected_global_count",
        "selected_global_min",
        "selected_global_max",
        "cross_layer_persistent",
        "cross_layer_persistence_mode",
        "persistent_selection_reused",
        "selection_source_layer",
        "selection_applied_layer",
        "hidden_sequence_length_before",
        "hidden_sequence_length_after",
        "hidden_sequence_ratio",
        "hidden_sequence_first_compressed_layer",
        "hidden_sequence_last_compressed_layer",
        "hidden_sequence_compressed_layer_count",
    }
    return {key: jsonable(metadata[key]) for key in keys if key in metadata}


def _runtime_event_summary(runtime: VideoPressRuntime) -> dict[str, Any]:
    events = list(runtime.events)
    metadata = [_metadata_subset(event.result.metadata or {}) for event in events]
    last = metadata[-1] if metadata else {}

    source_events = [
        (event, item)
        for event, item in zip(events, metadata)
        if item.get("n_kept") is not None
        and item.get("persistent_selection_reused") is not True
    ]
    kept_values = [float(item["n_kept"]) for _, item in source_events]

    def finite_mean(values) -> float | None:
        values = [float(value) for value in values if value is not None]
        return float(sum(values) / len(values)) if values else None

    def first_scalar(value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def mean_first_batch_vector(key: str) -> list[float] | None:
        vectors: list[list[float]] = []
        for _, item in source_events:
            value = item.get(key)
            if not isinstance(value, list) or not value:
                continue
            value = value[0] if isinstance(value[0], list) else value
            try:
                vector = [float(component) for component in value]
            except (TypeError, ValueError):
                continue
            if vectors and len(vector) != len(vectors[0]):
                raise ValueError(f"inconsistent {key} width across diffusion steps")
            vectors.append(vector)
        if not vectors:
            return None
        return [float(sum(column) / len(column)) for column in zip(*vectors)]

    kept_histogram: dict[str, int] = {}
    for value in kept_values:
        label = str(int(value))
        kept_histogram[label] = kept_histogram.get(label, 0) + 1

    def distinct(key: str) -> list[Any]:
        values = []
        for item in metadata:
            value = item.get(key)
            if value not in values:
                values.append(value)
        return values

    return {
        "event_count": len(events),
        "layer_count": len({event.key.layer_idx for event in events}),
        "diffusion_rank_count": len({event.key.diffusion_rank for event in events}),
        "selector_latency_ms": float(runtime.selector_latency_ms),
        "last": last,
        "invalid_selection_count": sum(
            1 for item in metadata if item.get("selection_candidate_valid") is False
        ),
        "noncontiguous_segment_count": sum(
            1 for item in metadata if item.get("segment_frame_idx_contiguous") is False
        ),
        "distinct_segment_scene_tokens": distinct("segment_scene_token"),
        "distinct_segment_scene_names": distinct("segment_scene_name"),
        "distinct_k_length_after": distinct("k_length_after"),
        "distinct_v_length_after": distinct("v_length_after"),
        "distinct_theoretical_attn_ratio": distinct("theoretical_attn_ratio"),
        "distinct_n_kept": distinct("n_kept"),
        "n_kept_mean": finite_mean(kept_values),
        "n_kept_min": min(kept_values) if kept_values else None,
        "n_kept_max": max(kept_values) if kept_values else None,
        "n_kept_histogram": kept_histogram,
        "eligible_keep_ratio_mean": finite_mean(
            item.get("eligible_keep_ratio") for _, item in source_events
        ),
        "history_keep_ratio_mean": finite_mean(
            item.get("history_keep_ratio") for _, item in source_events
        ),
        "effective_history_keep_ratio_mean": finite_mean(
            first_scalar(item.get("effective_history_keep_ratio"))
            for _, item in source_events
        ),
        "selected_history_latent_counts_mean": mean_first_batch_vector(
            "selected_history_latent_counts"
        ),
        "selected_history_latent_ratios_mean": mean_first_batch_vector(
            "selected_history_latent_ratios"
        ),
        "effective_history_latent_kept_counts_mean": mean_first_batch_vector(
            "effective_history_latent_kept_counts"
        ),
        "effective_history_latent_keep_ratios_mean": mean_first_batch_vector(
            "effective_history_latent_keep_ratios"
        ),
        "k_length_after_mean": finite_mean(
            item.get("k_length_after") for _, item in source_events
        ),
        "v_length_after_mean": finite_mean(
            item.get("v_length_after") for _, item in source_events
        ),
        "theoretical_attn_ratio_mean": finite_mean(
            item.get("theoretical_attn_ratio") for _, item in source_events
        ),
        "hidden_sequence_length_after_mean": finite_mean(
            item.get("hidden_sequence_length_after") for _, item in source_events
        ),
        "hidden_sequence_ratio_mean": finite_mean(
            item.get("hidden_sequence_ratio") for _, item in source_events
        ),
        "selection_snapshots": [
            {
                "diffusion_rank": event.key.diffusion_rank,
                "n_kept": item.get("n_kept"),
                "selection_metadata": item.get("selection_metadata"),
                "score_diagnostics": item.get("score_diagnostics"),
            }
            for event, item in source_events
        ],
        "distinct_selection_source_layers": distinct("selection_source_layer"),
        "distinct_persistence_modes": distinct("cross_layer_persistence_mode"),
        "distinct_hidden_sequence_lengths": distinct("hidden_sequence_length_after"),
        "distinct_hidden_sequence_ratios": distinct("hidden_sequence_ratio"),
        "distinct_hidden_compressed_layer_counts": distinct(
            "hidden_sequence_compressed_layer_count"
        ),
        "persistent_reuse_event_count": sum(
            1 for item in metadata if item.get("persistent_selection_reused") is True
        ),
    }


class _RunState:
    def __init__(
        self,
        runtime: VideoPressRuntime,
        method_dir: Path,
        method_name: str,
        artifact_status: list[str] | None = None,
    ):
        self.runtime = runtime
        self.method_dir = method_dir
        self.method_name = method_name
        self.artifact_status = list(artifact_status or [])
        self.current_token: str | None = None
        self.segment_scene_token: str | None = None
        self.segment_scene_name: str | None = None
        self.segment_frame_idx_start: int | None = None
        self.segment_frame_idx_end: int | None = None
        self.segment_frame_count: int = 0
        self.segment_frame_idx_contiguous: bool = False
        self.event_path = method_dir / f"press_events.rank{_rank()}.jsonl"
        self.probe_count = 0
        self.probe_details: list[dict[str, Any]] = []

    def bind_scene(self, loader, token: str) -> None:
        """Bind the anchor token and its raw same-segment window atomically."""

        token = str(token)
        frames = getattr(loader, "scene_frames_dicts", {}).get(token)
        if not frames:
            raise RuntimeError(f"official scene loader has no frame window for token={token}")
        if not window_is_single_scene(frames):
            scene_tokens = sorted({str(frame.get("scene_token")) for frame in frames})
            scene_names = sorted({str(frame.get("scene_name")) for frame in frames})
            raise RuntimeError(
                f"scene window crosses a segment boundary token={token} "
                f"scene_tokens={scene_tokens} scene_names={scene_names}"
            )
        frame_indices = [int(frame["frame_idx"]) for frame in frames]
        self.current_token = token
        self.segment_scene_token = str(frames[0].get("scene_token"))
        self.segment_scene_name = str(frames[0].get("scene_name"))
        self.segment_frame_idx_start = frame_indices[0]
        self.segment_frame_idx_end = frame_indices[-1]
        self.segment_frame_count = len(frames)
        self.segment_frame_idx_contiguous = all(
            right - left == 1 for left, right in zip(frame_indices, frame_indices[1:])
        )

    def segment_metadata(self) -> dict[str, Any]:
        return {
            "segment_scene_token": self.segment_scene_token,
            "segment_scene_name": self.segment_scene_name,
            "segment_frame_idx_start": self.segment_frame_idx_start,
            "segment_frame_idx_end": self.segment_frame_idx_end,
            "segment_frame_count": self.segment_frame_count,
            "segment_frame_idx_contiguous": self.segment_frame_idx_contiguous,
        }

    def sample(self):
        token = self.current_token or "unknown"
        return SimpleNamespace(
            scene_token=token,
            metadata={
                "scene_tokens": [token],
                "method": self.method_name,
                **self.segment_metadata(),
            },
        )

    def write_event(self, *, valid: bool, error: str | None = None) -> None:
        summary = _runtime_event_summary(self.runtime)
        row = {
            "artifact_status": self.artifact_status,
            "scene_token": self.current_token or "unknown",
            "method": self.method_name,
            "valid": bool(valid),
            "error": error,
            "probe_count": self.probe_count,
            "probe_details": list(self.probe_details),
            "segment": self.segment_metadata(),
            "runtime": summary,
        }
        self.method_dir.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        self.probe_details = []


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _device_from_dist(official_eval, dist_info: dict[str, int]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("official VideoTokenPress evaluation requires CUDA")
    torch.cuda.set_device(int(dist_info["local_rank"]))
    return torch.device(f"cuda:{int(dist_info['local_rank'])}")


def _build_official_pipeline(official_eval, args: argparse.Namespace, device: torch.device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    model_configs = [
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="Wan2.2_VAE.pth",
            offload_device=str(device),
            local_model_path=str(args.local_model_path),
            skip_download=True,
        ),
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=model_configs,
        use_trajectory=True,
    )
    pipe.eval()

    if not args.full_ckpt.is_file():
        raise FileNotFoundError(args.full_ckpt)
    if _rank() == 0:
        print(f"[official-press] loading checkpoint: {args.full_ckpt}", flush=True)
    state_dict = official_eval._normalize_checkpoint_keys(official_eval.load_state_dict(str(args.full_ckpt)))
    missing, unexpected = pipe.load_state_dict(state_dict, strict=False)
    if _rank() == 0:
        print(
            f"[official-press] checkpoint loaded missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    lora_ckpt = getattr(args, "pre_dit_register_lora_checkpoint", None)
    if lora_ckpt:
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file

        targets = [
            name.strip()
            for name in str(args.pre_dit_register_lora_target_modules).split(",")
            if name.strip()
        ]
        rank = int(args.pre_dit_register_lora_rank)
        lora_config = LoraConfig(r=rank, lora_alpha=rank, target_modules=targets)
        pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)
        # A training checkpoint mixes three prefixes:
        #   blocks.*            LoRA (relative to pipe.dit)
        #   pipe.trajectory_*   trained trajectory encoder/head (relative to pipe)
        #   learnable_merge.*   bottleneck (loaded by the press)
        combined = {}
        for key, value in load_file(str(lora_ckpt)).items():
            key = str(key)
            if key.startswith("learnable_merge."):
                continue
            if key.startswith("pipe."):
                combined[key[len("pipe."):]] = value
            else:
                combined[f"dit.{key}"] = value
        lora_missing, lora_unexpected = pipe.load_state_dict(combined, strict=False)
        if _rank() == 0:
            print(
                f"[official-press] LoRA+trajectory loaded rank={rank} targets={targets} "
                f"missing={len(lora_missing)} unexpected={len(lora_unexpected)}",
                flush=True,
            )
    return pipe


def _layout_from_model_call(kwargs: dict[str, Any], adapter: DriveVAAdapter) -> TokenLayout | None:
    latents = kwargs.get("latents")
    dit = kwargs.get("dit")
    if not torch.is_tensor(latents) or latents.ndim != 5 or dit is None:
        return None
    patch_size = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        return None
    longcat = kwargs.get("longcat_latents")
    traj = kwargs.get("traj_tokens")
    num_cond = int(longcat.shape[2]) if torch.is_tensor(longcat) and longcat.ndim >= 3 else 0
    traj_len = int(traj.shape[1]) if torch.is_tensor(traj) and traj.ndim >= 2 else 0
    prefix_len = int(kwargs.get("traj_prefix_len", 0) or 0)
    return adapter.build_layout(
        int(latents.shape[2]) // patch_size[0],
        int(latents.shape[3]) // patch_size[1],
        int(latents.shape[4]) // patch_size[2],
        num_cond,
        traj_len,
        prefix_len,
    )


def _cache_probe_scores(
    *,
    runtime: VideoPressRuntime,
    press: Any,
    scene_token: str,
    captures: dict[int, dict[str, Any]],
    cache: ScoreCache,
    adapter: DriveVAAdapter,
    kind: str,
) -> list[dict[str, Any]]:
    scorer = getattr(press, "scorer", None)
    if scorer is None:
        raise RuntimeError(f"{kind} probe requires a scorer")
    details = []
    for diffusion_rank in sorted(captures):
        capture = captures[diffusion_rank]
        q_raw = capture.get("q")
        k_raw = capture.get("k")
        v_raw = capture.get("v")
        layout = capture.get("layout")
        if q_raw is None or k_raw is None or layout is None:
            raise RuntimeError(
                f"{kind} probe did not capture q/k/layout for scene={scene_token} rank={diffusion_rank}"
            )
        q, _ = canonicalize_qkv(q_raw, int(capture.get("num_heads") or 0))
        k, _ = canonicalize_qkv(k_raw, int(capture.get("num_heads") or 0))
        v = None
        if v_raw is not None:
            v, _ = canonicalize_qkv(v_raw, int(capture.get("num_heads") or 0))
        if q.shape[2] != layout.total_length or k.shape[2] != layout.total_length:
            raise RuntimeError(
                f"probe layout/attention length mismatch: layout={layout.total_length} q={q.shape[2]} k={k.shape[2]}"
            )
        tokens = torch.zeros(
            (q.shape[0], layout.total_length, q.shape[-1]),
            device=q.device,
            dtype=q.dtype,
        )
        domain_name = getattr(press, "domain", None) or "last_history"
        domain = build_domain(domain_name, layout, tokens.device)
        context = TokenContext(
            tokens=tokens,
            layout=layout,
            domain=domain,
            scene_token=scene_token,
            diffusion_rank=int(diffusion_rank),
            q=q,
            k=k,
            v=v,
            metadata={"scene_tokens": [scene_token], "probe": kind},
        )
        scores = press.score(context)
        ranking = press.ranking(scores)
        key = runtime.score_key(context)
        cache.save(
            key,
            scores,
            ranking=ranking,
            metadata={
                "probe": kind,
                "scene_token": scene_token,
                "diffusion_rank": int(diffusion_rank),
                "layer": getattr(scorer, "layer", None),
                "layout": layout.to_dict(),
            },
        )
        details.append(
            {
                "kind": kind,
                "diffusion_rank": int(diffusion_rank),
                "layer": getattr(scorer, "layer", None),
                "n_candidate": int(domain.n_candidate),
                "score_cache": str(cache.path_for(key)),
            }
        )
    if not details:
        raise RuntimeError(f"{kind} probe captured no attention/model calls for scene={scene_token}")
    return details


def _run_attention_probe(
    *,
    pipe: Any,
    state: _RunState,
    args: argparse.Namespace,
    adapter: DriveVAAdapter,
    invoke_kwargs: dict[str, Any],
) -> None:
    """Run one uncompressed official forward and cache post-RoPE attention scores."""

    press = state.runtime.press
    scorer = getattr(press, "scorer", None)
    target_layer = getattr(scorer, "layer", None)
    if target_layer is None:
        target_layer = 15
    captures: dict[int, dict[str, Any]] = {}
    original_model_fn = pipe.model_fn
    attention_restores: list[tuple[Any, Any]] = []

    def model_fn_probe(*call_args, **call_kwargs):
        layout = _layout_from_model_call(call_kwargs, adapter)
        timestep = call_kwargs.get("timestep")
        rank = int(timestep.reshape(-1)[0].item()) if torch.is_tensor(timestep) and timestep.numel() else None
        if rank is not None:
            captures.setdefault(rank, {})["layout"] = layout
        return original_model_fn(*call_args, **call_kwargs)

    pipe.model_fn = model_fn_probe
    try:
        for model_name in ("dit", "dit2"):
            model = getattr(pipe, model_name, None)
            blocks = getattr(model, "blocks", None) if model is not None else None
            if blocks is None or not (0 <= int(target_layer) < len(blocks)):
                continue
            attention = getattr(blocks[int(target_layer)], "self_attn", None)
            module = getattr(attention, "attn", None) if attention is not None else None
            if module is None:
                continue
            original_forward = module.forward

            def wrapped_forward(_module, q, k, v, *, _original=original_forward, _attention=attention):
                timestep = getattr(pipe, "_videopress_probe_timestep", None)
                rank = int(timestep) if timestep is not None else None
                if rank is not None:
                    payload = captures.setdefault(rank, {})
                    payload["q"] = q.detach().clone()
                    payload["k"] = k.detach().clone()
                    payload["v"] = v.detach().clone()
                    payload["num_heads"] = int(getattr(_attention, "num_heads", 0))
                    payload["model_name"] = model_name
                return _original(q, k, v)

            # The model_fn wrapper identifies the active timestep before the
            # attention hook is reached.  It is attached below per call.
            def wrapped_model_fn_with_timestep(*call_args, _base=model_fn_probe, **call_kwargs):
                timestep = call_kwargs.get("timestep")
                pipe._videopress_probe_timestep = (
                    int(timestep.reshape(-1)[0].item())
                    if torch.is_tensor(timestep) and timestep.numel()
                    else None
                )
                try:
                    return _base(*call_args, **call_kwargs)
                finally:
                    pipe._videopress_probe_timestep = None

            module.forward = types.MethodType(wrapped_forward, module)
            attention_restores.append((module, original_forward))

        # Replace the model function after all hooks are installed.  The
        # current timestep wrapper remains transparent to the official pipe.
        pipe.model_fn = wrapped_model_fn_with_timestep
        pipe(**invoke_kwargs)
    finally:
        for module, original_forward in attention_restores:
            module.forward = original_forward
        pipe.model_fn = original_model_fn
        if hasattr(pipe, "_videopress_probe_timestep"):
            delattr(pipe, "_videopress_probe_timestep")

    details = _cache_probe_scores(
        runtime=state.runtime,
        press=press,
        scene_token=str(state.current_token),
        captures=captures,
        cache=state.runtime.score_cache,
        adapter=adapter,
        kind="attention",
    )
    state.probe_count += 1
    state.probe_details.extend(details)


def _gradient_scores(
    scorer: Any,
    gradients: torch.Tensor,
    tokens: torch.Tensor,
    layout: TokenLayout,
    domain_name: str,
) -> torch.Tensor:
    candidate = build_domain(domain_name, layout, gradients.device).candidate_indices
    grad_candidate = gradients.index_select(1, candidate).float()
    token_candidate = tokens.index_select(1, candidate).float()
    if getattr(scorer, "name", "") == "gradient_input":
        attribution = grad_candidate * token_candidate
        if str(getattr(scorer, "reduction", "l2")) == "abs_sum":
            return attribution.abs().sum(dim=-1)
        return torch.linalg.vector_norm(attribution, dim=-1)
    return torch.linalg.vector_norm(grad_candidate, dim=-1)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float("nan") if float(denominator) == 0.0 else float(((left * right).sum() / denominator).item())


def _selection_diagnostics(
    framework: torch.Tensor,
    planning: torch.Tensor,
    k: int,
    framework_selected: torch.Tensor | None = None,
    planning_selected: torch.Tensor | None = None,
) -> dict[str, Any]:
    def ranks(values: torch.Tensor) -> torch.Tensor:
        order = values.argsort(stable=True)
        output = torch.empty(values.numel(), device=values.device, dtype=torch.float32)
        output[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
        return output

    framework_selected = (
        framework.argsort(descending=True, stable=True)[:k]
        if framework_selected is None
        else framework_selected.reshape(-1)
    )
    planning_selected = (
        planning.argsort(descending=True, stable=True)[:k]
        if planning_selected is None
        else planning_selected.reshape(-1)
    )
    framework_set = set(framework_selected.detach().cpu().tolist())
    planning_set = set(planning_selected.detach().cpu().tolist())
    intersection = len(framework_set & planning_set)
    union = len(framework_set | planning_set)
    return {
        "pearson": _pearson(framework, planning),
        "spearman": _pearson(ranks(framework), ranks(planning)),
        "topk_intersection": intersection,
        "topk_overlap_ratio": float(intersection / k) if k else 1.0,
        "topk_jaccard": float(intersection / union) if union else 1.0,
        "masks_different": framework_set != planning_set,
    }


def _run_gradient_probe(
    *,
    pipe: Any,
    state: _RunState,
    args: argparse.Namespace,
    adapter: DriveVAAdapter,
    invoke_kwargs: dict[str, Any],
) -> None:
    """Probe trajectory-head sensitivity to patchified history tokens."""

    press = state.runtime.press
    scorer = getattr(press, "scorer", None)
    if scorer is None:
        raise RuntimeError("gradient probe requires a scorer")
    domain_name = str(getattr(press, "domain", None) or "last_history")
    captures: dict[int, dict[str, Any]] = {}
    original_model_fn = pipe.model_fn

    def model_fn_probe(*call_args, **call_kwargs):
        dit = call_kwargs.get("dit")
        if dit is None or not hasattr(dit, "patchify"):
            return original_model_fn(*call_args, **call_kwargs)
        layout = _layout_from_model_call(call_kwargs, adapter)
        capture: dict[str, Any] = {"layout": layout}
        original_patchify = dit.patchify

        def wrapped_patchify(_model, x, *, _original=original_patchify):
            patched = _original(x)
            if not torch.is_tensor(patched) or patched.ndim != 5:
                raise RuntimeError("gradient probe expected patchify() -> [B,C,F,H,W]")
            tokens = patched.permute(0, 2, 3, 4, 1).reshape(
                patched.shape[0], -1, patched.shape[1]
            )
            # Use the exact tensor returned by the official patchify call as
            # the autograd target. A separately-created flattened view is not
            # guaranteed to be the tensor recorded by the model graph.
            capture["patched"] = patched
            capture["tokens"] = tokens
            return patched

        dit.patchify = types.MethodType(wrapped_patchify, dit)
        try:
            with torch.enable_grad():
                output = original_model_fn(*call_args, **call_kwargs)
                if not isinstance(output, dict) or output.get("traj") is None:
                    raise RuntimeError("gradient probe requires the official trajectory output")
                trajectory = output["traj"]
                patched = capture.get("patched")
                tokens = capture.get("tokens")
                if patched is None or tokens is None or layout is None:
                    raise RuntimeError("gradient probe did not capture video tokens/layout")
                planning_method = getattr(scorer, "name", "") == "planning_gradient_input"
                debug_compare = bool(planning_method and args.gradient_debug_compare)
                framework_objective = trajectory.float().pow(2).mean()
                trajectory_points = None
                planning_objective = None
                if planning_method:
                    planning_objective, trajectory_points = trajectory_projection_objective(
                        trajectory, int(call_kwargs.get("traj_prefix_len", 0) or 0)
                    )
                objective = planning_objective if planning_method else framework_objective
                gradients = torch.autograd.grad(
                    objective,
                    patched,
                    retain_graph=debug_compare,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                gradient_tokens = gradients.permute(0, 2, 3, 4, 1).reshape(
                    gradients.shape[0], -1, gradients.shape[1]
                )
                timestep = call_kwargs.get("timestep")
                if not torch.is_tensor(timestep) or not timestep.numel():
                    raise RuntimeError("gradient probe did not receive a timestep")
                rank = int(timestep.reshape(-1)[0].item())
                candidate = build_domain(domain_name, layout, gradient_tokens.device).candidate_indices
                candidate_gradients = gradient_tokens.index_select(1, candidate)
                candidate_tokens = tokens.index_select(1, candidate)
                scores = (
                    original_gradient_input_reduction(candidate_gradients, candidate_tokens)
                    if planning_method
                    else _gradient_scores(scorer, gradient_tokens, tokens, layout, domain_name)
                )
                payload = {
                    "layout": layout,
                    "scores": scores.detach(),
                    "candidate_tokens": candidate_tokens.detach(),
                    "n_candidate": int(candidate.numel()),
                    "gradient_target_shape": tuple(patched.shape),
                    "candidate_gradient_shape": tuple(candidate_gradients.shape),
                    "trajectory_shape": tuple(trajectory.shape),
                    "trajectory_points_shape": tuple(trajectory_points.shape) if trajectory_points is not None else None,
                    "objective_value": float(objective.detach().float().item()),
                }
                if debug_compare:
                    framework_gradients = torch.autograd.grad(
                        framework_objective, patched, retain_graph=False, create_graph=False, allow_unused=False
                    )[0]
                    framework_tokens = framework_gradients.permute(0, 2, 3, 4, 1).reshape(
                        framework_gradients.shape[0], -1, framework_gradients.shape[1]
                    )
                    payload["framework_scores"] = _gradient_scores(
                        SimpleNamespace(name="gradient_input", reduction="l2"),
                        framework_tokens,
                        tokens,
                        layout,
                        domain_name,
                    ).detach()
                captures[rank] = payload
                return output
        finally:
            dit.patchify = original_patchify

    pipe.model_fn = model_fn_probe
    try:
        pipe(**invoke_kwargs)
    finally:
        pipe.model_fn = original_model_fn

    cache = state.runtime.score_cache
    details = []
    for diffusion_rank in sorted(captures):
        capture = captures[diffusion_rank]
        scores = capture["scores"]
        layout = capture["layout"]
        tokens = torch.zeros(
            (scores.shape[0], layout.total_length, 1),
            device=scores.device,
            dtype=scores.dtype,
        )
        # A zero-token context is sufficient to construct the stable cache
        # key.  The actual ranking was computed from the official gradient.
        context = TokenContext(
            tokens=tokens,
            layout=layout,
            domain=build_domain(domain_name, layout, tokens.device),
            scene_token=str(state.current_token),
            diffusion_rank=int(diffusion_rank),
            metadata={"scene_tokens": [str(state.current_token)], "probe": "gradient"},
        )
        key = state.runtime.score_key(context)
        ranking = scores.argsort(dim=-1, descending=True, stable=True)
        selection = press.select(context, scores, cached_ranking=ranking)
        k = int(selection.K)
        selected_global = selection.keep_global_indices
        candidate_keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        if selection.keep_candidate_indices.numel():
            candidate_keep_mask.scatter_(1, selection.keep_candidate_indices, True)
        candidate_positions = torch.tensor(
            [
                (int(global_index) // layout.tokens_per_latent,)
                + divmod(int(global_index) % layout.tokens_per_latent, layout.video_w)
                for global_index in context.domain.candidate_indices.detach().cpu().tolist()
            ],
            dtype=torch.int16,
        )
        keep_mask = torch.zeros((scores.shape[0], layout.total_length), dtype=torch.bool, device=scores.device)
        keep_mask.scatter_(1, selected_global, True)
        effective_keep_mask = keep_mask | context.domain.protected_mask.unsqueeze(0)
        planning_method = getattr(scorer, "name", "") == "planning_gradient_input"
        artifact_metadata = {
            "artifact_status": list(state.artifact_status),
            "method_name": getattr(scorer, "name", "gradient_input"),
            "anchor_id": str(state.current_token),
            "diffusion_timestep": int(diffusion_rank),
            "domain": domain_name,
            "candidate_count": int(context.domain.n_candidate),
            "K": k,
            "retention_policy": getattr(press, "retention_policy", None),
            "selection_metadata": dict(selection.metadata),
            "objective_type": PLANNING_OBJECTIVE_TYPE if planning_method else "mean_traj_noise_prediction_squared",
            "objective_value": capture["objective_value"],
            "score_reduction": PLANNING_SCORE_REDUCTION if planning_method else getattr(scorer, "reduction", "l2"),
            "trajectory_tensor_shape": capture["trajectory_shape"],
            "trajectory_points_shape": capture["trajectory_points_shape"],
            "gradient_target_shape": capture["gradient_target_shape"],
            "candidate_gradient_shape": capture["candidate_gradient_shape"],
        }
        if "framework_scores" in capture:
            framework_scores = capture["framework_scores"]
            framework_ranking = framework_scores.argsort(dim=-1, descending=True, stable=True)
            framework_selection = press.select(
                context, framework_scores, cached_ranking=framework_ranking
            )
            artifact_metadata["framework_comparison"] = _selection_diagnostics(
                framework_scores[0],
                scores[0],
                k,
                framework_selection.keep_global_indices[0],
                selected_global[0],
            )
        if planning_method:
            artifact_dir = state.method_dir / "planning_gradient_artifacts" / str(state.current_token)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **artifact_metadata,
                    "scores": scores.detach().cpu(),
                    "candidate_tokens": capture["candidate_tokens"].detach().cpu(),
                    "candidate_positions": candidate_positions,
                    "rank_local_descending": ranking.detach().cpu(),
                    "topk_mask_candidate": candidate_keep_mask.detach().cpu(),
                    "selected_indices": selected_global.detach().cpu(),
                    "mask": keep_mask.detach().cpu(),
                    "effective_keep_mask": effective_keep_mask.detach().cpu(),
                    "mask_semantics": (
                        "mask=True means selected; only mask=False positions inside domain are removed; "
                        "effective_keep_mask includes protected positions outside domain"
                    ),
                },
                artifact_dir / f"timestep_{int(diffusion_rank)}.pt",
            )
            print("[planning-gradient-debug] " + json.dumps(jsonable({
                **artifact_metadata,
                "score_min": float(scores.min().item()),
                "score_max": float(scores.max().item()),
                "score_mean": float(scores.mean().item()),
                "score_std": float(scores.std().item()),
                "score_has_nan": bool(torch.isnan(scores).any().item()),
                "score_has_inf": bool(torch.isinf(scores).any().item()),
                "selected_indices": selected_global[0].detach().cpu().tolist(),
            }), sort_keys=True), flush=True)
        cache.save(
            key,
            scores,
            ranking=ranking,
            metadata={
                "probe": "original_trajectory_projection_gradient_input" if planning_method else "gradient_trajectory_l2_zero_target",
                "scene_token": str(state.current_token),
                "diffusion_rank": int(diffusion_rank),
                "layout": layout.to_dict(),
                **artifact_metadata,
            },
        )
        details.append(
            {
                "kind": "gradient",
                "diffusion_rank": int(diffusion_rank),
                "score_cache": str(cache.path_for(key)),
                "objective": artifact_metadata["objective_type"],
            }
        )
    if not details:
        raise RuntimeError(f"gradient probe captured no calls for scene={state.current_token}")
    state.probe_count += 1
    state.probe_details.extend(details)


def _dump_predicted_trajectory(
    state: "_RunState", result: Any, args: argparse.Namespace
) -> None:
    """Persist the raw ego-relative predicted trajectory of one scene.

    ``physical_no_press`` returns ``(video, traj_pred, vel)`` from the pipeline,
    where ``traj_pred[0]`` is exactly the array the official evaluator feeds
    into NAVSIM as ``Trajectory.poses`` (metres, ego-relative).  Best-of-N
    oracle analysis needs the samples themselves, not only the derived PDM
    scalars, so that trajectory-level disagreement can be measured.
    """

    if not bool(getattr(args, "dump_trajectories", False)):
        return
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return
    traj = result[1]
    if not torch.is_tensor(traj):
        return
    token = str(state.current_token or "unknown")
    if token == "unknown":
        raise RuntimeError("trajectory dump requested without an active scene token")
    out_dir = state.method_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{token}.rank{_rank()}.npz"
    if out_path.exists():
        # Silent overwrite would merge two different sampling seeds into one
        # artifact; refuse instead (defect-J style guard).
        raise RuntimeError(f"trajectory dump already exists, refusing to overwrite: {out_path}")
    array = traj.detach().to(dtype=torch.float32).cpu().numpy()
    np.savez_compressed(
        out_path,
        traj=array,
        scene_token=np.asarray(token),
        sample_seed=np.asarray(int(getattr(args, "seed", 0))),
        method=np.asarray(str(state.method_name)),
    )


def _dump_history_tokens(
    state: "_RunState", result: Any, args: argparse.Namespace
) -> None:
    """Persist the DiT hidden states of the candidate history latent (P0-2 leak probe).

    Why this exists.  The register-bottleneck route (NTR-style) predicts the NEXT
    latent's content from the candidate history tokens read at the selector layer.
    Its pre-registered strongest counter-argument is leakage: the next latent's
    token sits in the SAME non-causal forward pass, so if the history tokens at
    that layer already carry the target's content, the predictive objective
    degenerates into copying and any apparent gain is an artefact.

    The discriminating test needs no training and no new model code: run the same
    scene twice with DIFFERENT diffusion sampling seeds and compare the captured
    history tokens.  Different seeds give a different noised future (the target),
    while the clean conditioned history is unchanged.  Therefore

        * history tokens bit-identical across seeds  -> the future does not leak
          back into the candidate representation at this layer;
        * history tokens differ                       -> there IS a forward-pass
          leak, and the P0-2 objective must be redesigned before any GPU budget
          is spent on it.

    The dump is per-scene and refuses to overwrite, so two sampler seeds cannot be
    silently merged into one artifact (same guard style as the trajectory dump).
    """

    if not bool(getattr(args, "dump_history_tokens", False)):
        return
    capture = getattr(state, "history_capture", None)
    tokens = getattr(capture, "tokens", None)
    if not torch.is_tensor(tokens):
        return
    token = str(state.current_token or "unknown")
    if token == "unknown":
        raise RuntimeError("history-token dump requested without an active scene token")
    out_dir = state.method_dir / "history_tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{token}.rank{_rank()}.npz"
    if out_path.exists():
        raise RuntimeError(
            f"history-token dump already exists, refusing to overwrite: {out_path}"
        )
    array = tokens.detach().to(dtype=torch.float32).cpu().numpy()
    np.savez_compressed(
        out_path,
        history_tokens=array,
        scene_token=np.asarray(token),
        sample_seed=np.asarray(int(getattr(args, "seed", 0))),
        method=np.asarray(str(state.method_name)),
    )


class _HistoryTokenCapture:
    """Capture the candidate history latent's residual stream at one DiT block.

    P0-2's registered leakage control needs the SAME tensor the selector would
    read: the hidden states of the candidate 390 history tokens entering the
    selector layer.  The pipeline exposes that as
    ``dit._tokenpress_pre_block_hidden`` during the block loop, but clears it
    before ``model_fn`` returns (wan_video_new.py:2040), so a module-level hook
    cannot see it.  A hook on the block itself can: the block is invoked as
    ``block(x, context, t_mod, freqs)``, so ``inputs[0]`` is exactly that
    residual stream before the block runs.

    Installation is deliberately scoped to the evaluation process: the hook is
    idempotent, holds no state between scenes, and is removed by
    :meth:`remove`.
    """

    def __init__(self, model: Any, layer: int) -> None:
        self.model = model
        self.layer = int(layer)
        self.tokens: Any = None
        self.handle = None
        self._range: tuple[int, int] | None = None

    def attach(self, num_cond_latents: int, tokens_per_latent: int) -> None:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None or not 0 <= self.layer < len(blocks):
            raise ValueError(f"cannot capture history tokens at layer {self.layer}")
        # The candidate (newest) history latent is the LAST conditioned latent,
        # i.e. tokens [ (num_cond-1)*T, num_cond*T ) of the residual stream.
        self._range = (
            int(num_cond_latents - 1) * int(tokens_per_latent),
            int(num_cond_latents) * int(tokens_per_latent),
        )
        self.handle = blocks[self.layer].register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self.tokens = None
        if self._range is None or not inputs:
            return None
        x = inputs[0]
        if torch.is_tensor(x) and x.ndim == 3:
            start, end = self._range
            if x.shape[1] >= end:
                self.tokens = x[:, start:end].detach()
        return None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class _PipelineProxy:
    """Delegate every official pipeline attribute but intercept scene calls."""

    def __init__(self, pipe: Any, state: _RunState, adapter: DriveVAAdapter, args: argparse.Namespace):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_args", args)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_pipe"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_pipe"), name, value)

    def eval(self):
        self._pipe.eval()
        return self

    def __call__(self, *call_args, **call_kwargs):
        state = self._state
        runtime = state.runtime
        if not state.current_token or state.current_token == "unknown":
            raise RuntimeError(
                "official pipeline call has no active scene token; refusing to reuse a stale scene binding"
            )
        sample = state.sample()
        ego_vel = call_kwargs.get("ego_vel")
        if ego_vel is not None:
            sample.metadata["selector_ego_state"] = (
                torch.as_tensor(ego_vel).detach().cpu().flatten()[:2].tolist()
            )
        prompt = str(call_kwargs.get("prompt", "")).lower()
        if "left" in prompt:
            sample.metadata["selector_command"] = [1.0, 0.0, 0.0]
        elif "right" in prompt:
            sample.metadata["selector_command"] = [0.0, 0.0, 1.0]
        else:
            sample.metadata["selector_command"] = [0.0, 1.0, 0.0]
        runtime.begin_sample(sample)
        capture = getattr(state, "history_capture", None)
        if capture is not None:
            # Clear before every scene so a missed capture cannot be mistaken for
            # the previous scene's tokens.
            capture.tokens = None
        press = runtime.press
        scorer = getattr(press, "scorer", None) if press is not None else None
        point = InjectionPoint.parse(getattr(press, "injection_point", InjectionPoint.VIDEO_INPUT)) if press else None
        requires_probe = bool(getattr(scorer, "requires_probe", False)) or (
            point is InjectionPoint.VIDEO_INPUT
            and str(getattr(scorer, "probe_mode", "none")) in {"online", "ProbeMode.ONLINE"}
        )
        try:
            if requires_probe:
                if runtime.score_cache is None:
                    raise RuntimeError("probe method requires a ScoreCache")
                runtime.remove(self._pipe)
                invoke_kwargs = dict(call_kwargs)
                if call_args:
                    raise RuntimeError("official VideoTokenPress proxy only supports keyword pipeline calls")
                if bool(getattr(scorer, "requires_probe", False)):
                    _run_gradient_probe(
                        pipe=self._pipe,
                        state=state,
                        args=self._args,
                        adapter=self._adapter,
                        invoke_kwargs=invoke_kwargs,
                    )
                else:
                    _run_attention_probe(
                        pipe=self._pipe,
                        state=state,
                        args=self._args,
                        adapter=self._adapter,
                        invoke_kwargs=invoke_kwargs,
                    )
                runtime.install(self._pipe)
                runtime.begin_sample(sample)
            result = self._pipe(*call_args, **call_kwargs)
            _dump_predicted_trajectory(state, result, self._args)
            _dump_history_tokens(state, result, self._args)
            state.write_event(valid=True)
            return result
        except Exception as exc:
            state.write_event(valid=False, error=str(exc))
            raise


def _patch_official_scene_hooks(official_eval, state_box: dict[str, _RunState]) -> None:
    original_loader_builder = official_eval._build_scene_loader
    original_scene_builder = official_eval._build_scene_without_print

    def build_scene_loader(*call_args, **call_kwargs):
        loader = original_loader_builder(*call_args, **call_kwargs)
        original_get = loader.get_agent_input_from_token

        def get_agent_input_from_token(token):
            state = state_box["state"]
            state.bind_scene(loader, str(token))
            return original_get(token)

        loader.get_agent_input_from_token = get_agent_input_from_token
        return loader

    def build_scene_without_print(scene_loader, scene_cls, token):
        state_box["state"].bind_scene(scene_loader, str(token))
        return original_scene_builder(scene_loader, scene_cls, token)

    official_eval._build_scene_loader = build_scene_loader
    official_eval._build_scene_without_print = build_scene_without_print


def _official_args(args: argparse.Namespace, output_dir: Path, official_eval) -> argparse.Namespace:
    argv = [
        "--repo_root",
        str(args.repo_root.resolve()),
        "--navsim_log_path",
        str(args.navsim_log_path.resolve()),
        "--sensor_blobs_path",
        str(args.sensor_blobs_path.resolve()),
        "--metric_cache_path",
        str(args.metric_cache_path.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--scene_filter_yaml",
        str(args.scene_filter_yaml.resolve()),
        "--scene_filter_yaml_filter_only",
        "1",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_history_frames",
        str(args.num_history_frames),
        "--num_future_frames",
        str(args.num_future_frames),
        "--model_future_frames",
        str(args.model_future_frames),
        "--target_fps",
        str(args.target_fps),
        "--pdm_num_poses",
        str(args.pdm_num_poses),
        "--pdm_interval_length",
        str(args.pdm_interval_length),
        "--local_model_path",
        str(args.local_model_path.resolve()),
        "--full_ckpt",
        str(args.full_ckpt.resolve()),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--cfg_scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--no_show_eval_progress",
        "--no_print_errors",
        "--no_print_alignment_params",
    ]
    if not bool(getattr(args, "infer_video", False)):
        argv.append("--infer_trajectory_only")
    if args.max_eval_tokens is not None:
        argv.extend(["--max_eval_tokens", str(args.max_eval_tokens)])
    if args.save_viz:
        argv.extend([
            "--save_viz",
            "--viz_total_tokens",
            str(args.viz_total_tokens),
            "--viz_max_tokens",
            str(args.viz_max_tokens),
        ])
    if args.enable_nuscenes_metrics:
        argv.append("--enable_nuscenes_metrics")
    official_args = official_eval.parse_args(argv)
    if args.allow_missing_route:
        # In this mode the supplied YAML is a complete protocol description,
        # including has_route=false, rather than a tokens-only overlay.
        official_args.scene_filter_yaml_filter_only = False
    return official_args


def _join_official_records(method_dir: Path, method_spec: dict[str, Any], round_index: int) -> dict[str, Any]:
    csv_path = _latest_official_csv(method_dir)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        official_rows = list(csv.DictReader(handle))
    official_rows = [row for row in official_rows if str(row.get("token", "")) != "average"]

    events: dict[str, dict[str, Any]] = {}
    for event_path in sorted(method_dir.glob("press_events.rank*.jsonl")):
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            events[str(item.get("scene_token"))] = item

    records = []
    for row in official_rows:
        token = str(row.get("token"))
        event = events.get(token, {})
        runtime = event.get("runtime", {}) if isinstance(event, dict) else {}
        last = runtime.get("last", {}) if isinstance(runtime, dict) else {}
        pdm = _finite(row.get("pdm_score"), default=_finite(row.get("score")))
        valid = _bool_value(row.get("valid"))
        if "valid" not in row:
            valid = bool(event.get("valid", True))
        record = {
            "scene_token": token,
            "valid": valid,
            "pdm": pdm,
            "trajectory_l2": float("nan"),
            "endpoint_l2": float("nan"),
            "latency_ms": _finite(row.get("infer_time_ms")),
            "model_latency_ms": _finite(row.get("infer_time_ms")),
            "e2e_latency_ms": _finite(row.get("infer_time_ms")),
            "selector_latency_ms": _finite(runtime.get("selector_latency_ms")),
            "peak_memory_mb": _finite(row.get("gpu_mem_alloc_peak_mb")),
            # Promote compression fields from the compact event metadata so
            # evaluation.statistics can aggregate them without knowing the
            # runner's event-file schema.
            "eligible_keep_ratio": _finite(
                runtime.get("eligible_keep_ratio_mean", last.get("eligible_keep_ratio"))
            ),
            "history_keep_ratio": _finite(
                runtime.get("history_keep_ratio_mean", last.get("history_keep_ratio"))
            ),
            "effective_history_keep_ratio": _finite(
                runtime.get(
                    "effective_history_keep_ratio_mean",
                    (last.get("effective_history_keep_ratio") or [float("nan")])[0]
                    if isinstance(last.get("effective_history_keep_ratio"), list)
                    else last.get("effective_history_keep_ratio"),
                )
            ),
            "retention_policy": last.get("retention_policy"),
            # Physical KV operators expose the eligible domain directly; use
            # it as the candidate count when they do not emit a separate
            # n_candidate field.
            "n_candidate": _finite(last.get("n_candidate", last.get("n_eligible"))),
            "K": _finite(runtime.get("n_kept_mean", last.get("n_kept"))),
            "press_name": method_spec["name"],
            "scorer": method_spec["press"].get("scorer", {}).get("name")
            if isinstance(method_spec["press"].get("scorer"), dict)
            else None,
            "operator": method_spec["press"].get("operator", {}).get("name")
            if isinstance(method_spec["press"].get("operator"), dict)
            else None,
            "domain": method_spec["press"].get("domain"),
            "metadata": {
                **last,
                "dynamic_n_kept_mean": runtime.get("n_kept_mean"),
                "dynamic_n_kept_min": runtime.get("n_kept_min"),
                "dynamic_n_kept_max": runtime.get("n_kept_max"),
                "dynamic_n_kept_histogram": runtime.get("n_kept_histogram", {}),
                "selected_history_latent_counts_mean_across_steps": runtime.get(
                    "selected_history_latent_counts_mean"
                ),
                "selected_history_latent_ratios_mean_across_steps": runtime.get(
                    "selected_history_latent_ratios_mean"
                ),
                "effective_history_latent_kept_counts_mean_across_steps": runtime.get(
                    "effective_history_latent_kept_counts_mean"
                ),
                "effective_history_latent_keep_ratios_mean_across_steps": runtime.get(
                    "effective_history_latent_keep_ratios_mean"
                ),
                "selection_snapshots": runtime.get("selection_snapshots", []),
                "hidden_sequence_length_after_mean_across_steps": runtime.get(
                    "hidden_sequence_length_after_mean"
                ),
                "hidden_sequence_ratio_mean_across_steps": runtime.get(
                    "hidden_sequence_ratio_mean"
                ),
                "k_length_after_mean_across_steps": runtime.get("k_length_after_mean"),
                "v_length_after_mean_across_steps": runtime.get("v_length_after_mean"),
                "theoretical_attn_ratio_mean_across_steps": runtime.get(
                    "theoretical_attn_ratio_mean"
                ),
                "runtime_event_count": runtime.get("event_count", 0),
                "probe_count": event.get("probe_count", 0),
                "probe_details": event.get("probe_details", []),
            },
            "official": {
                key: jsonable(value)
                for key, value in row.items()
                if key not in {"token", "valid", "rank"}
            },
        }
        records.append(record)

    (method_dir / "records.jsonl").write_text(
        "".join(json.dumps(jsonable(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    press_config = deepcopy(method_spec["press"])
    summary = {
        "method": method_spec["name"],
        "round": int(round_index),
        "mode": method_spec["mode"],
        "backend": "official_navsim",
        "press": press_config,
        "official_csv": str(csv_path),
        "n_scenes": len(records),
        "valid_scenes": sum(bool(record["valid"]) for record in records),
        "probe_protocol": "frozen_score_cache" if any(record["metadata"].get("probe_count", 0) for record in records) else "none",
    }
    (method_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def is_evaluation_truncated(args: argparse.Namespace) -> bool:
    """True when the run evaluated fewer scenes than its protocol declares.

    ``--max-eval-tokens N`` slices the official scene list to its first ``N``
    entries, so it is only a real truncation when ``N`` is smaller than the
    protocol's expected scene count.  A custom protocol has no declared size, so
    any explicit cap counts as truncation there (audit 2026-09-12, BUG-3).
    """
    max_tokens = getattr(args, "max_eval_tokens", None)
    if max_tokens is None:
        return False
    expected = getattr(args, "eval_protocol_expected_scenes", None)
    if expected is None:
        return True
    return int(max_tokens) < int(expected)


def evaluation_scope(args: argparse.Namespace) -> dict[str, Any]:
    """Scope flags that change WHAT was evaluated, persisted in every artifact.

    Without these a ``--max-eval-tokens 4`` smoke run is indistinguishable from a
    full-protocol run: it still carries ``eval_protocol=navtest-7876`` and
    ``expected_scenes=7876``, and ``coverage`` reports the UNTRUNCATED filter
    (audit 2026-09-12, BUG-3).  One helper feeds both ``config.json`` and
    ``suite_manifest.json`` so the two can never disagree.
    """
    return {
        "max_eval_tokens": getattr(args, "max_eval_tokens", None),
        "force_full_scene_set": bool(getattr(args, "force_full_scene_set", False)),
        "enable_nuscenes_metrics": bool(
            getattr(args, "enable_nuscenes_metrics", False)
        ),
        "expected_scenes": getattr(args, "eval_protocol_expected_scenes", None),
        "is_truncated": is_evaluation_truncated(args),
    }


def enforce_evaluation_scope(args: argparse.Namespace, summary: dict[str, Any]) -> int:
    """Return the process exit code for a finished suite.

    Two failure modes used to exit 0 while still carrying the primary-protocol
    label, so a smoke or partially-failed run could be quoted as a full
    ``navtest-7876`` number (audit 2026-09-12, BUG-2/BUG-3):

    * a truncated run (``--max-eval-tokens N`` with ``N < expected_scenes``);
    * a run whose valid scene count does not equal ``expected_scenes``.

    Both now return 1.  The artifacts are written before this runs, so the data
    is still available -- only the exit status says "not citable".
    """
    label = getattr(args, "eval_protocol_label", "custom")
    expected = getattr(args, "eval_protocol_expected_scenes", None)
    max_tokens = getattr(args, "max_eval_tokens", None)
    if is_evaluation_truncated(args):
        print(
            "[scope][FAIL] run is truncated: --max-eval-tokens "
            f"{max_tokens} < expected_scenes {expected} ({label}). "
            "Its PDM must NOT be quoted against the protocol baselines "
            "(recorded under evaluation_scope.is_truncated).",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if expected is None:
        return 0
    offenders = []
    for row in summary.get("method_rows", []):
        n_valid = int(row.get("valid_scenes") or 0)
        if n_valid != int(expected):
            offenders.append(
                f"{row.get('method')}: valid={n_valid} expected={int(expected)}"
            )
    if offenders:
        print(
            "[scope][FAIL] run is labelled "
            f"{label}/{int(expected)} scenes but did not produce that many valid "
            "records:\n  " + "\n  ".join(offenders),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"[scope][ok] every method produced {int(expected)} valid records ({label})",
        flush=True,
    )
    return 0


def method_config_dict(
    args: argparse.Namespace, method_spec: dict[str, Any], round_index: int
) -> dict[str, Any]:
    """The complete, self-describing config for one method directory."""
    return {
        "artifact_status": POC_TEST_DERIVED_STATUS if args.poc_test_derived else [],
        "retention_policy": args.retention_policy,
        "backend": "official_navsim",
        "round": round_index,
        "method": method_spec["name"],
        "mode": method_spec["mode"],
        "press": method_spec["press"],
        "data": {
            "repo_root": str(args.repo_root.resolve()),
            "navsim_log_path": str(args.navsim_log_path.resolve()),
            "sensor_blobs_path": str(args.sensor_blobs_path.resolve()),
            "metric_cache_path": str(args.metric_cache_path.resolve()),
            "scene_filter_yaml": str(args.scene_filter_yaml.resolve()),
        },
        "score_cache_root": str(args.score_cache_root.resolve()) if args.score_cache_root else None,
        "eval_protocol": {
            "label": getattr(args, "eval_protocol_label", "custom"),
            "expected_scenes": getattr(args, "eval_protocol_expected_scenes", None),
            "baselines": getattr(args, "eval_protocol_baselines", {}),
        },
        "model": {
            "full_ckpt": str(args.full_ckpt.resolve()),
            "local_model_path": str(args.local_model_path.resolve()),
            "num_inference_steps": args.num_inference_steps,
            "model_future_frames": args.model_future_frames,
            "sampling_seed": int(args.seed),
            "sample_seed_override": args.sample_seed,
        },
        "dump_trajectories": bool(args.dump_trajectories),
        # Scope flags that change WHAT was evaluated, persisted so a run is
        # self-describing.  Without these a `--max-eval-tokens 4` smoke run is
        # indistinguishable from a full-protocol run in every artifact: it still
        # carries eval_protocol=navtest-7876 and expected_scenes=7876, and
        # `coverage` reports the UNTRUNCATED filter (audit 2026-09-12, BUG-3).
        "evaluation_scope": evaluation_scope(args),
    }


def _write_method_config(
    method_dir: Path, args: argparse.Namespace, method_spec: dict[str, Any], round_index: int
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    config = method_config_dict(args, method_spec, round_index)
    (method_dir / "config.json").write_text(
        json.dumps(jsonable(config), indent=2, allow_nan=False), encoding="utf-8"
    )
    (method_dir / "environment.json").write_text(
        json.dumps(jsonable(environment_snapshot(args.repo_root)), indent=2), encoding="utf-8"
    )


def apply_eval_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve data paths from the protocol preset and write them back to args."""
    resolved = resolve_eval_protocol(args)
    for key in _PATH_KEYS:
        setattr(args, key, resolved[key])
    args.eval_protocol_label = resolved["label"]
    args.eval_protocol_expected_scenes = (
        None if resolved["preset"] is None else int(resolved["preset"]["expected_scenes"])
    )
    args.eval_protocol_baselines = (
        {} if resolved["preset"] is None else dict(resolved["preset"]["baselines"])
    )
    return resolved


def _resolve_output_root(path: Path, allow_existing: bool) -> Path:
    path = path if path.is_absolute() else FRAMEWORK_ROOT / path
    if allow_existing:
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.exists():
        path.mkdir(parents=True)
        return path
    index = 1
    while True:
        candidate = path.parent / f"{path.name}_rerun{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "suite_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, allow_nan=False), encoding="utf-8"
    )


def _check_official_scene_coverage(official_eval, official_args: argparse.Namespace, args: argparse.Namespace) -> dict[str, int]:
    official_eval._ensure_navsim_importable(Path(official_args.repo_root))
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader

    overrides = official_eval._load_scene_filter_yaml(
        official_args.scene_filter_yaml,
        filter_only=official_args.scene_filter_yaml_filter_only,
    )
    scene_filter = SceneFilter(
        num_history_frames=official_args.num_history_frames,
        num_future_frames=official_args.num_future_frames,
        frame_interval=official_args.frame_interval,
        has_route=not args.allow_missing_route,
        max_scenes=official_args.max_scenes,
        log_names=None,
    )
    for key, value in overrides.items():
        setattr(scene_filter, key, value)
    from navsim.common.dataloader import SceneLoader

    loader = SceneLoader(
        data_path=Path(official_args.navsim_log_path),
        sensor_blobs_path=Path(official_args.sensor_blobs_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=True),
        load_image_path=True,
    )
    metric_cache_loader = MetricCacheLoader(Path(official_args.metric_cache_path))
    scene_tokens = set(loader.tokens)
    cache_tokens = set(metric_cache_loader.tokens)
    intersection = scene_tokens & cache_tokens
    counts = {
        "scene_filter_tokens": len(scene_tokens),
        "metric_cache_tokens": len(cache_tokens),
        "intersection_tokens": len(intersection),
        "missing_metric_cache_tokens": len(scene_tokens - cache_tokens),
    }
    if _rank() == 0:
        print("[official-press] coverage:", json.dumps(counts, sort_keys=True), flush=True)
    if args.force_full_scene_set and counts["missing_metric_cache_tokens"]:
        raise RuntimeError(
            "official metric cache is incomplete; build it with the official NAVSIM metric-caching command "
            f"before --force-full-scene-set (missing={counts['missing_metric_cache_tokens']})"
        )
    return counts


def run(args: argparse.Namespace) -> int:
    protocol = apply_eval_protocol(args)
    print(
        f"[protocol] eval_protocol={protocol['label']} "
        f"expected_scenes={args.eval_protocol_expected_scenes} "
        f"navsim_log_path={args.navsim_log_path}",
        flush=True,
    )
    if protocol["label"].endswith("+overridden"):
        print(
            "[protocol][warn] data paths override the preset; absolute PDM from this "
            "run must NOT be quoted as a clean protocol result",
            flush=True,
        )
    if protocol["label"] == "custom":
        print(
            "[protocol][warn] unrecognised data paths: absolute PDM is only comparable "
            "within this run",
            flush=True,
        )
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.max_eval_tokens is not None and args.max_eval_tokens < 1:
        raise ValueError("--max-eval-tokens must be >= 1 when set")
    # Best-of-N sampling seed.  The diffusion sampler is deterministic given a
    # seed (``generate_noise`` builds a ``torch.Generator`` per call), so the
    # only supported way to obtain a second, independent sample of the *same*
    # method is to change the seed that the official evaluator passes into the
    # pipeline.  ``--sample-seed`` therefore rewrites ``args.seed`` before any
    # method spec, manifest or evaluator argument is built.  ``--seed-base``
    # (round/method-spec seed) is deliberately left untouched: for the
    # deterministic ``physical_no_press`` arm it only names the method matrix.
    if getattr(args, "sample_seed", None) is not None:
        args.seed = int(args.sample_seed)
    if args.retention_policy is not None:
        num_history_latents = 1 + (int(args.num_history_frames) - 1) // 4
        if num_history_latents != 2:
            raise ValueError(
                "the six retention policies require exactly two VAE history latents; "
                f"num_history_frames={args.num_history_frames} produces {num_history_latents}"
            )
    for path in (
        args.navsim_log_path,
        args.sensor_blobs_path,
        args.metric_cache_path,
        args.full_ckpt,
        args.local_model_path,
        args.scene_filter_yaml,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    data_environment = _configure_official_data_environment(args.repo_root)
    official_eval = _load_official_eval_module()
    # The guard wraps navsim.common.dataloader, so the official evaluator's
    # normal repository-local import setup must run before installing it.
    official_eval._ensure_navsim_importable(args.repo_root.resolve())
    scene_boundary_guard = install_scene_boundary_guard()
    dist_info = official_eval._init_distributed()
    device = _device_from_dist(official_eval, dist_info)
    # Resolve the output directory once.  In a torchrun job every rank enters
    # this function, so resolving independently would make rank 0 choose
    # ``*_rerun01`` while rank 1 observes that directory and chooses
    # ``*_rerun02``.  That splits event journals and official CSV inputs and
    # makes the resulting suite impossible to audit.  Broadcast rank 0's
    # resolved path before any rank creates method outputs.
    requested_output_root = (
        args.output_root if args.output_root.is_absolute() else FRAMEWORK_ROOT / args.output_root
    )
    if int(dist_info["world_size"]) > 1 and torch.distributed.is_initialized():
        root_payload = [
            str(_resolve_output_root(requested_output_root, args.allow_existing_output))
            if int(dist_info["rank"]) == 0
            else None
        ]
        torch.distributed.broadcast_object_list(root_payload, src=0)
        root = Path(str(root_payload[0]))
    else:
        root = _resolve_output_root(requested_output_root, args.allow_existing_output)
    state_box: dict[str, _RunState] = {}
    _patch_official_scene_hooks(official_eval, state_box)

    all_specs = _method_specs_for_run(args, args.seed_base)
    selected = None if not args.methods else {name.strip() for name in args.methods.split(",") if name.strip()}
    if selected is not None:
        known = {spec["name"] for spec in all_specs}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"--methods contains unknown methods for this matrix: {unknown}")
    specs = [spec for spec in all_specs if selected is None or spec["name"] in selected]
    if not specs:
        raise ValueError("--methods selected no known method")

    manifest: dict[str, Any] = {
        "artifact_status": POC_TEST_DERIVED_STATUS if args.poc_test_derived else [],
        "suite_name": "official_navsim_videopress",
        "version": 1,
        "backend": "official_navsim",
        "repo_root": str(args.repo_root.resolve()),
        "device": str(device),
        "world_size": int(dist_info["world_size"]),
        "rounds": int(args.rounds),
        "sampling_seed": int(args.seed),
        "sample_seed_override": args.sample_seed,
        "seed_base": int(args.seed_base),
        "dump_trajectories": bool(args.dump_trajectories),
        "retention_policy": args.retention_policy,
        "persistent_layer_sweep": (
            parse_layer_sweep(args.persistent_layer_sweep)
            if args.persistent_layer_sweep is not None
            else None
        ),
        "persistent_end_layer": args.persistent_end_layer,
        "persistent_mode": args.persistent_mode,
        "persistent_keep_ratio": args.persistent_keep_ratio,
        "scene_boundary_guard": scene_boundary_guard,
        "data_environment": data_environment,
        "scene_filter": {
            "num_history_frames": int(args.num_history_frames),
            "num_future_frames": int(args.num_future_frames),
            "frame_interval": 1,
            "has_route": not args.allow_missing_route,
            "yaml_filter_only": not args.allow_missing_route,
            "window_length": int(args.num_history_frames + args.num_future_frames),
        },
        "methods": [spec["name"] for spec in specs],
        "data": {
            "navsim_log_path": str(args.navsim_log_path.resolve()),
            "sensor_blobs_path": str(args.sensor_blobs_path.resolve()),
            "metric_cache_path": str(args.metric_cache_path.resolve()),
            "scene_filter_yaml": str(args.scene_filter_yaml.resolve()),
        },
        "score_cache_root": str(args.score_cache_root.resolve()) if args.score_cache_root else None,
        "eval_protocol": {
            "label": getattr(args, "eval_protocol_label", "custom"),
            "expected_scenes": getattr(args, "eval_protocol_expected_scenes", None),
            "baselines": getattr(args, "eval_protocol_baselines", {}),
            "note": (
                "absolute PDM is only comparable to published NAVSIM numbers on "
                "navtest-7876; pair deltas are protocol invariant"
            ),
        },
        # Same mapping as config.json (single source of truth in
        # `evaluation_scope`), so a truncated run cannot look complete in the
        # manifest either (audit 2026-09-12, BUG-3).
        "evaluation_scope": evaluation_scope(args),
        "runs": [],
    }
    if _rank() == 0:
        _write_manifest(root, manifest)

    # Build the model once and reuse its official weights for every method.
    pipe = _build_official_pipeline(official_eval, args, device)

    # P0-2 leakage instrument: attach once, before any method runs.  The layout is
    # fixed for this backbone (2 conditioned history latents x 390 tokens at
    # 480x832), so the candidate range is [390, 780).  It cannot be derived lazily
    # inside the call path because `runtime.layout` is only populated for real
    # presses, while the leakage control must also run under `press=noop`.
    history_capture = None
    if bool(getattr(args, "dump_history_tokens", False)):
        dit = getattr(pipe, "dit", None)
        if dit is None:
            raise RuntimeError("--dump-history-tokens requires pipe.dit")
        num_cond_latents = int(getattr(args, "num_history_frames", 5)) - 3
        history_capture = _HistoryTokenCapture(
            dit, int(getattr(args, "selector_layer", 15))
        )
        history_capture.attach(num_cond_latents, _HISTORY_TOKENS_PER_LATENT)
        print(
            f"[leak-probe] capturing layer {history_capture.layer} residual stream "
            f"of {num_cond_latents} x {_HISTORY_TOKENS_PER_LATENT} candidate tokens",
            flush=True,
        )

    for round_index in range(1, args.rounds + 1):
        round_seed = int(args.seed_base + round_index - 1)
        round_specs = _method_specs_for_run(args, round_seed)
        by_name = {spec["name"]: spec for spec in round_specs}
        for selected_spec in specs:
            spec = by_name[selected_spec["name"]]
            method_dir = root / f"round{round_index:02d}" / spec["name"]
            if _rank() == 0:
                print(
                    f"[official-press] start round={round_index} method={spec['name']} mode={spec['mode']}",
                    flush=True,
                )
            _write_method_config(method_dir, args, spec, round_index)
            score_cache = None
            scorer = spec["press"].get("scorer") if isinstance(spec["press"], dict) else None
            if isinstance(scorer, dict) and (
                scorer.get("name") in {"action_attention", "action_attention_vnorm", "gradient_norm", "gradient_input", "planning_gradient_input"}
            ):
                if args.score_cache_root is None:
                    score_cache_dir = method_dir / "score_cache"
                else:
                    score_cache_dir = (
                        args.score_cache_root.resolve()
                        / f"round{round_index:02d}"
                        / spec["name"]
                        / "score_cache"
                    )
                score_cache = ScoreCache(score_cache_dir)
            press = build_press(spec["press"])
            adapter = DriveVAAdapter()
            runtime = VideoPressRuntime(
                press=press,
                mode=spec["mode"],
                adapter=adapter,
                score_cache=score_cache,
            )
            state = _RunState(
                runtime,
                method_dir,
                spec["name"],
                POC_TEST_DERIVED_STATUS if args.poc_test_derived else None,
            )
            state.history_capture = history_capture
            state_box["state"] = state
            proxy = _PipelineProxy(pipe, state, adapter, args)
            runtime.install(pipe)
            official_args = _official_args(args, method_dir, official_eval)
            if round_index == 1 and selected_spec is specs[0]:
                coverage = _check_official_scene_coverage(official_eval, official_args, args)
                if _rank() == 0:
                    manifest["coverage"] = coverage
                    _write_manifest(root, manifest)
            try:
                official_eval.run_eval(official_args, external_pipe=proxy)
            finally:
                runtime.remove(pipe)
            if _rank() == 0:
                summary = _join_official_records(method_dir, spec, round_index)
                manifest["runs"].append(
                    {
                        "method": spec["name"],
                        "round": round_index,
                        "protocol": spec["mode"],
                        "backend": "official_navsim",
                        "output_dir": str(method_dir.relative_to(root)),
                        "official_csv": summary["official_csv"],
                        "n_scenes": summary["n_scenes"],
                        "valid_scenes": summary["valid_scenes"],
                    }
                )
                _write_manifest(root, manifest)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

    if _rank() == 0:
        summary = aggregate_suite(root)
        tables = write_suite_tables(summary, root / "statistics")
        plots = {} if args.skip_plots else generate_suite_visualizations(summary, root / "visualizations")
        report = {
            "suite_root": str(root.resolve()),
            "backend": "official_navsim",
            "eval_protocol": manifest.get("eval_protocol"),
            "run_count": len(summary["run_rows"]),
            "method_count": len(summary["method_rows"]),
            "coverage": manifest.get("coverage"),
            "tables": tables,
            "plots": plots,
            "method_rows": summary["method_rows"],
        }
        (root / "suite_summary.json").write_text(
            json.dumps(jsonable(report), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(jsonable(report), indent=2, allow_nan=False), flush=True)

        # ---- scene-count / truncation gate -----------------------------------
        # A truncated or partially-failed run used to exit 0 while still carrying
        # `eval_protocol=navtest-7876` and `expected_scenes=7876`, so a 4-scene
        # smoke result could be quoted as a full-protocol number (audit
        # 2026-09-12, BUG-2/BUG-3).  Refuse to report success unless the run was
        # untruncated AND every expected scene produced a VALID record.
        return enforce_evaluation_scope(args, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

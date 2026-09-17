#!/usr/bin/env python3
"""Run the complete synthetic VideoTokenPress compression suite.

This is an end-to-end framework test, not an official NAVSIM benchmark.  It
uses the same evaluator/artifact path as the real backend and covers every
compression family currently exposed by the independent framework:

* causal input interventions: zero, mean replacement and the three shuffle
  variants, combined with Random/TokenNorm/Attention/Gradient scorers;
* physical post-RoPE interventions: KV prune and similarity-based KV merge;
* NoPress controls for both protocol paths.

Each round regenerates the same-shaped cohort with a different seed.  The
output is self-contained and can be re-aggregated later with
``scripts/aggregate_suite.py`` (or the Python API in ``evaluation.statistics``).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from evaluation.artifacts import jsonable
from evaluation.evaluator import Evaluator
from evaluation.statistics import aggregate_suite, write_suite_tables
from evaluation.visualization import generate_suite_visualizations
from scripts.evaluate_press import make_synthetic_cohort, synthetic_metrics, synthetic_predict
from videopress.factory import build_press
from videopress.objectives import TrajectoryObjective
from videopress.core.retention import (
    HISTORY_RETENTION_POLICIES,
    apply_history_retention_policy,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run all VideoTokenPress compression families for several rounds")
    parser.add_argument("--rounds", type=int, default=3, help="number of independently seeded rounds")
    parser.add_argument("--max-scenes", type=int, default=4, help="scenes evaluated per method and round")
    parser.add_argument("--seed-base", type=int, default=20260828)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FRAMEWORK_ROOT / "outputs" / "full_compression_suite",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="optional comma-separated method names; default runs the complete suite",
    )
    parser.add_argument("--skip-plots", action="store_true", help="write tables but do not import matplotlib")
    parser.add_argument(
        "--retention-policy",
        choices=tuple(HISTORY_RETENTION_POLICIES),
        default=None,
        help="apply one of the six two-history-latent deletion policies",
    )
    return parser.parse_args(argv)


def _press_config(
    *,
    mode: str,
    scorer: str | None = None,
    operator: str | None = None,
    scorer_options: dict[str, Any] | None = None,
    operator_options: dict[str, Any] | None = None,
    seed: int = 0,
    scope: str = "scene",
    name: str = "scorer_press",
    domain: str = "last_history",
    cross_layer_persistence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if name == "noop":
        return {"name": "noop"}
    if name == "similarity_merge":
        return {
            "name": name,
            "injection_point": "self_attn_kv",
            "domain": domain,
            "feature": "tokens",
            "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
        }
    scorer_section = {"name": scorer}
    if scorer == "random":
        scorer_section.update({"seed": seed, "scope": scope})
    if scorer_options:
        scorer_section.update(scorer_options)
    operator_section = {"name": operator}
    if operator_options:
        operator_section.update(operator_options)
    config = {
        "name": name,
        "injection_point": "video_input" if mode == "causal" else "self_attn_kv",
        "domain": domain,
        "scorer": scorer_section,
        "selector": {"name": "topk"},
        "operator": operator_section,
        "budget": {"type": "ratio", "value": 0.5, "reference": "eligible"},
    }
    if cross_layer_persistence is not None:
        config["cross_layer_persistence"] = dict(cross_layer_persistence)
    return config


def persistent_attention_vnorm_spec(
    start_layer: int,
    *,
    domain: str = "last_history",
    keep_ratio: float = 0.5,
    end_layer: int | None = None,
    persistence_mode: str = "kv_only",
    persistence_enabled: bool = True,
    name: str | None = None,
    scorer_name: str = "action_attention_vnorm",
    selector_config: dict[str, Any] | None = None,
    scorer_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one physical action-aware persistent or one-shot method."""

    start_layer = int(start_layer)
    if start_layer < 0:
        raise ValueError("persistent start_layer must be non-negative")
    if not 0.0 <= float(keep_ratio) <= 1.0:
        raise ValueError("persistent keep_ratio must be within [0, 1]")
    if end_layer is not None and int(end_layer) < start_layer:
        raise ValueError("persistent end_layer cannot precede start_layer")
    persistence_mode = str(persistence_mode).strip().lower()
    if persistence_mode not in {"kv_only", "hidden_sequence"}:
        raise ValueError("persistence_mode must be kv_only or hidden_sequence")
    persistence_enabled = bool(persistence_enabled)
    if not persistence_enabled:
        if end_layer is not None:
            raise ValueError("one-shot pruning does not accept end_layer")
        if persistence_mode != "kv_only":
            raise ValueError("one-shot pruning only supports persistence_mode=kv_only")
    mode_label = "kv" if persistence_mode == "kv_only" else "hidden"
    scorer_name = str(scorer_name).strip().lower()
    scorer_label = {
        "action_attention_vnorm": "attention_vnorm",
        "action_attention_vnorm_temporal": "attention_vnorm_temporal",
        "action_contribution_stability": "contribution_stability",
    }.get(scorer_name, scorer_name)
    requested_scorer_options = dict(scorer_options or {})
    feature_layer = requested_scorer_options.get("feature_layer")
    if feature_layer is not None and int(feature_layer) != start_layer:
        scorer_label = f"{scorer_label}_feature_{int(feature_layer):02d}"
    selector_config = dict(selector_config or {"name": "topk"})
    selector_label = {
        "adaptive_mass": "_adaptive",
        "history_threshold": "_history_threshold",
        "history_quota": "_history_quota",
        "future_threshold": "_future_threshold",
        "future_quota": "_future_quota",
    }.get(selector_config.get("name"), "")
    persistence_label = "persistent" if persistence_enabled else "one_shot"
    method_name = name or (
        f"physical_{scorer_label}{selector_label}_{mode_label}_{persistence_label}_layer_{start_layer:02d}"
    )
    resolved_scorer_options = {
        "layer": start_layer,
        **requested_scorer_options,
    }
    if scorer_name != "random":
        # Attention-style and learned scorers accept the compatibility field;
        # RandomScorer does not and must not receive it.
        resolved_scorer_options.setdefault("action_mode", "mean")
    if scorer_name in {"action_attention_vnorm", "action_attention_vnorm_temporal"}:
        resolved_scorer_options.setdefault("head_mode", "mean")
    press = _press_config(
        mode="physical",
        scorer=scorer_name,
        operator="kv_prune",
        domain=domain,
        scorer_options=resolved_scorer_options,
        cross_layer_persistence={
            "enabled": persistence_enabled,
            "end_layer": end_layer,
            "mode": persistence_mode,
        },
    )
    press["selector"] = selector_config
    press["budget"] = {
        "type": "ratio",
        # Adaptive selection needs the full candidate count as its safe
        # fallback; fixed Top-K continues to use keep_ratio exactly.
        "value": 1.0
        if selector_config.get("name")
        in {"adaptive_mass", "history_threshold", "future_threshold"}
        else float(keep_ratio),
        "reference": "eligible",
    }
    return {"name": method_name, "mode": "physical", "press": press}


def method_specs(
    round_seed: int,
    domain: str = "last_history",
    retention_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Return the method matrix for one round.

    Scorer seeds are round-specific for Random controls.  The data seed and
    the method seed are intentionally separate so the comparisons remain
    paired within a round.
    """

    causal = [
        {"name": "causal_no_press", "mode": "causal", "press": _press_config(mode="causal", name="noop")},
        {
            "name": "causal_random_zero",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="random", operator="zero", seed=round_seed + 1),
        },
        {
            "name": "causal_random_mean",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="random", operator="mean", seed=round_seed + 2),
        },
        {
            "name": "causal_random_shuffle_all",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="random", operator="shuffle_all", seed=round_seed + 3),
        },
        {
            "name": "causal_random_shuffle_drop",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="random", operator="shuffle_drop", seed=round_seed + 4),
        },
        {
            "name": "causal_random_shuffle_keep",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="random", operator="shuffle_keep", seed=round_seed + 5),
        },
        {
            "name": "causal_token_norm_zero",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="token_norm", operator="zero"),
        },
        {
            "name": "causal_token_norm_mean",
            "mode": "causal",
            "press": _press_config(mode="causal", scorer="token_norm", operator="mean"),
        },
        {
            "name": "causal_attention_zero",
            "mode": "causal",
            "press": _press_config(
                mode="causal",
                scorer="action_attention",
                operator="zero",
                scorer_options={"layer": 15, "head_mode": "mean", "action_mode": "mean"},
            ),
        },
        {
            "name": "causal_attention_vnorm_zero",
            "mode": "causal",
            "press": _press_config(
                mode="causal",
                scorer="action_attention_vnorm",
                operator="zero",
                scorer_options={"layer": 15, "head_mode": "mean", "action_mode": "mean"},
            ),
        },
        {
            "name": "causal_gradient_norm_zero",
            "mode": "causal",
            "gradient": True,
            "press": _press_config(
                mode="causal",
                scorer="gradient_norm",
                operator="zero",
            ),
        },
        {
            "name": "causal_gradient_input_zero",
            "mode": "causal",
            "gradient": True,
            "press": _press_config(
                mode="causal",
                scorer="gradient_input",
                operator="zero",
                scorer_options={"reduction": "l2"},
            ),
        },
        {
            "name": "causal_planning_gradient_input_zero",
            "mode": "causal",
            "gradient": True,
            "press": _press_config(
                mode="causal",
                scorer="planning_gradient_input",
                operator="zero",
            ),
        },
    ]
    physical = [
        {"name": "physical_no_press", "mode": "physical", "press": _press_config(mode="physical", name="noop")},
        {
            "name": "physical_random_kv_prune",
            "mode": "physical",
            "press": _press_config(mode="physical", scorer="random", operator="kv_prune", seed=round_seed + 11),
        },
        {
            "name": "physical_token_norm_kv_prune",
            "mode": "physical",
            "press": _press_config(mode="physical", scorer="token_norm", operator="kv_prune"),
        },
        {
            "name": "physical_attention_kv_prune",
            "mode": "physical",
            "press": _press_config(
                mode="physical",
                scorer="action_attention",
                operator="kv_prune",
                scorer_options={"layer": 15, "head_mode": "mean", "action_mode": "mean"},
            ),
        },
        {
            "name": "physical_attention_vnorm_kv_prune",
            "mode": "physical",
            "press": _press_config(
                mode="physical",
                scorer="action_attention_vnorm",
                operator="kv_prune",
                scorer_options={"layer": 15, "head_mode": "mean", "action_mode": "mean"},
            ),
        },
        persistent_attention_vnorm_spec(
            15,
            domain=domain,
            name="physical_attention_vnorm_kv_prune_persistent",
        ),
        {
            "name": "physical_similarity_merge",
            "mode": "physical",
            "press": _press_config(mode="physical", name="similarity_merge"),
        },
    ]
    specs = causal + physical
    if retention_policy is not None:
        # Similarity merge constructs groups rather than a Top-K selection and
        # cannot honor the six exact per-latent quotas. Other perturbation
        # controls remain useful and retain their explicit operator metadata.
        specs = [
            spec
            for spec in specs
            if spec["press"].get("name") != "similarity_merge"
        ]
    for spec in specs:
        if spec["press"].get("name") != "noop":
            if retention_policy is None:
                spec["press"]["domain"] = domain
            else:
                spec["press"] = apply_history_retention_policy(
                    spec["press"], retention_policy
                )
    return specs


def _gradient_forward(tokens: torch.Tensor, ctx):
    """Small differentiable stand-in for the DriveVA trajectory head."""

    pooled_video = tokens[:, ctx.layout.video.start : ctx.layout.video.end].mean(dim=1, keepdim=True)
    action = tokens[:, ctx.layout.future_action.start : ctx.layout.future_action.end, :2]
    return torch.tanh(action + 0.1 * pooled_video[..., :2].expand(-1, action.shape[1], -1))


def _resolve_output_root(path: Path) -> Path:
    path = path if path.is_absolute() else FRAMEWORK_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def _build_config(spec: dict[str, Any], *, round_index: int, round_seed: int, scenes: int, run_dir: Path) -> dict[str, Any]:
    mode = spec["mode"]
    return {
        "experiment": {
            "name": "full_compression_suite",
            "seed": round_seed,
            "round": round_index,
            "method": spec["name"],
        },
        "benchmark": {"name": "synthetic", "max_scenes": scenes},
        "model": {"checkpoint": "synthetic_framework_cohort"},
        "press": deepcopy(spec["press"]),
        "evaluation": {
            "mode": mode,
            "backend": "synthetic",
            "random_baseline": False,
            "random_seeds": [round_seed + 101, round_seed + 102],
        },
        "output": {"dir": str(run_dir)},
        "synthetic": {
            "frames": 4,
            "height": 3,
            "width": 4,
            "num_cond_latents": 2,
            "trajectory_length": 6,
            "trajectory_prefix_length": 2,
            "hidden_dim": 16,
            "heads": 2,
            "scenes": scenes,
        },
    }


def run_suite(args) -> tuple[Path, dict[str, Any]]:
    if args.rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if args.max_scenes < 1:
        raise ValueError("--max-scenes must be at least 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    suite_root = _resolve_output_root(args.output_root)
    selected = None
    if args.methods:
        selected = {value.strip() for value in args.methods.split(",") if value.strip()}
    manifest: dict[str, Any] = {
        "suite_name": "full_compression_suite",
        "version": 1,
        "backend": "synthetic",
        "device": str(device),
        "rounds": int(args.rounds),
        "max_scenes": int(args.max_scenes),
        "seed_base": int(args.seed_base),
        "retention_policy": args.retention_policy,
        "methods": [],
        "runs": [],
    }
    all_method_names = [
        spec["name"]
        for spec in method_specs(args.seed_base, retention_policy=args.retention_policy)
    ]
    manifest["methods"] = all_method_names if selected is None else [name for name in all_method_names if name in selected]
    if not manifest["methods"]:
        raise ValueError("--methods did not select a known method")
    _write_json(suite_root / "suite_manifest.json", manifest)

    evaluator = Evaluator(PROJECT_ROOT)
    for round_index in range(int(args.rounds)):
        round_seed = int(args.seed_base) + round_index
        specs = [
            spec
            for spec in method_specs(round_seed, retention_policy=args.retention_policy)
            if spec["name"] in manifest["methods"]
        ]
        for spec in specs:
            run_dir = suite_root / f"round_{round_index:02d}" / spec["name"]
            config = _build_config(
                spec,
                round_index=round_index,
                round_seed=round_seed,
                scenes=args.max_scenes,
                run_dir=run_dir,
            )
            print(f"[suite] round={round_index} method={spec['name']} mode={spec['mode']}", flush=True)
            layout, samples = make_synthetic_cohort(config, device)
            if spec.get("gradient"):
                press = build_press(
                    config["press"],
                    gradient_forward=_gradient_forward,
                    gradient_objective=TrajectoryObjective(),
                )
            else:
                press = build_press(config["press"])
            result = evaluator.evaluate(
                samples,
                press,
                layout,
                mode=spec["mode"],
                output_dir=run_dir,
                config=config,
                predict_fn=synthetic_predict,
                metric_fn=synthetic_metrics,
                max_scenes=args.max_scenes,
            )
            manifest["runs"].append(
                {
                    "round": round_index,
                    "method": spec["name"],
                    "protocol": spec["mode"],
                    "backend": "synthetic",
                    "output_dir": str(run_dir.relative_to(suite_root)),
                    "summary": result["summary"],
                }
            )
            _write_json(suite_root / "suite_manifest.json", manifest)

    summary = aggregate_suite(suite_root)
    table_paths = write_suite_tables(summary, suite_root / "statistics")
    plot_paths = {} if args.skip_plots else generate_suite_visualizations(summary, suite_root / "visualizations")
    report = {
        "suite_root": str(suite_root.resolve()),
        "rounds": args.rounds,
        "methods": manifest["methods"],
        "run_count": len(summary["run_rows"]),
        "table_paths": table_paths,
        "plot_paths": plot_paths,
        "method_rows": summary["method_rows"],
    }
    _write_json(suite_root / "suite_summary.json", report)
    return suite_root, report


def main(argv=None) -> int:
    args = parse_args(argv)
    suite_root, report = run_suite(args)
    print(json.dumps(jsonable({"suite_root": str(suite_root), **report}), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

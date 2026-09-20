#!/usr/bin/env python3
"""Token-level, set-level future oracle search over the official NAVSIM runner.

Why this exists
---------------
The 2026-09-18/20 tile oracle could only choose 12 tiles per latent and its
best subset was still ``dPDM -0.0182`` at keep 0.50; the conclusion report
asked for a finer instrument.  This driver implements the token-level
set-level oracle:

* ``random-best-of-n``  -- N per-scene random token masks, evaluated in one
  runner call, giving a per-scene best-of-N upper bound;
* ``independent-topk``  -- leave-one-token-group-out importance, then compose
  the top groups (the token-group analogue of the failed tile oracle);
* ``greedy-forward``    -- add the best-scoring token group each round;
* ``greedy-backward``   -- remove the least harmful token group each round;
* ``beam``              -- beam search over token-group additions.

Every candidate subset is scored *jointly* through the official evaluator, and
each evaluation keeps PDM, trajectory displacement and planning harm so the
three signals can be compared.  The runner side gains
``--future-oracle-token-mask-jsons`` so a whole candidate batch is one
invocation instead of one invocation per candidate.

The search itself lives in :mod:`videopress.oracle.token_set`; this file is the
GPU-facing orchestration around it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
for _path in (FRAMEWORK_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from videopress.oracle.metrics import (  # noqa: E402
    OBJECTIVES,
    combined_harm,
    load_target_trajectory,
    load_trajectory,
    objective_higher_is_better,
    planning_harm,
    trajectory_displacement,
)
from videopress.oracle.token_set import (  # noqa: E402
    GROUP_MODES,
    SetScorer,
    beam_search,
    build_token_groups,
    greedy_backward_elimination,
    greedy_forward_selection,
    latent_local_mask,
    mean_over_scenes,
    per_scene_oracle,
    random_search,
    random_token_masks,
    write_token_mask_json,
)
from scripts.compare_official_test_arms import paired_bootstrap  # noqa: E402

DEFAULT_TOKENS_PER_LATENT = 390
DEFAULT_NUM_LATENTS = 2
DEFAULT_BOOTSTRAP_SEED = 20260920


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "random-best-of-n",
            "independent-topk",
            "greedy-forward",
            "greedy-backward",
            "beam",
            "evaluate-masks",
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-method-dir",
        type=Path,
        default=None,
        help=(
            "round01/physical_no_press directory of an existing official run; "
            "when omitted the driver runs its own baseline first"
        ),
    )
    parser.add_argument(
        "--scene-tokens-file",
        type=Path,
        default=None,
        help="optional jsonl/json list of scene tokens fixing the paired panel",
    )
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tokens-per-latent", type=int, default=DEFAULT_TOKENS_PER_LATENT)
    parser.add_argument("--num-future-latents", type=int, default=DEFAULT_NUM_LATENTS)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--layer", type=int, default=15)

    # grouping / search
    parser.add_argument("--group-mode", choices=GROUP_MODES, default="linear")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--block-h", type=int, default=None)
    parser.add_argument("--block-w", type=int, default=None)
    parser.add_argument("--grid-width", type=int, default=None, help="latent grid width for block groups")
    parser.add_argument("--group-seed", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--max-harm", type=float, default=math.inf)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=16, help="samples for random modes")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--mask-jsons", default=None, help="comma-separated masks for --mode evaluate-masks")
    parser.add_argument(
        "--include-random-masks",
        type=int,
        default=0,
        help=(
            "--mode evaluate-masks only: also evaluate N per-scene random masks of "
            "the same budget, as a matched random control on the identical panel"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=tuple(sorted(OBJECTIVES)),
        default="pdm_harm",
        help="search objective; pdm_harm/traj_disp/planning_harm are minimised, pdm maximised",
    )

    # runner plumbing
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--runner",
        type=Path,
        default=FRAMEWORK_ROOT / "scripts/run_official_navsim_press.py",
    )
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=64)
    parser.add_argument("--poc-test-derived", action="store_true", default=True)
    parser.add_argument("--no-poc-test-derived", dest="poc_test_derived", action="store_false")
    parser.add_argument("--force-full-scene-set", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", default=True)
    parser.add_argument("--no-skip-plots", dest="skip_plots", action="store_false")
    parser.add_argument("--no-dump-target-trajectories", dest="dump_target", action="store_false", default=True)
    parser.add_argument("--sample-seed-diffusion", type=int, default=None)
    parser.add_argument(
        "--keep-suite-baseline",
        action="store_true",
        help=(
            "keep the in-suite physical_no_press method on every candidate runner "
            "call; by default it is skipped (--persistent-skip-baseline) because the "
            "driver already holds an external baseline and re-running it once per "
            "candidate batch is pure overhead"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write candidate masks and commands without executing the runner (random/evaluate modes only)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# baseline / records
# ---------------------------------------------------------------------------


def load_records(method_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(method_dir) / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing records: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().splitlines()
        if line.strip()
    ]
    return {str(row["scene_token"]): row for row in rows}


@dataclass
class Baseline:
    method_dir: Path
    pdm: dict[str, float]
    trajectory: dict[str, np.ndarray | None] = field(default_factory=dict)
    target: dict[str, np.ndarray | None] = field(default_factory=dict)

    def scene_tokens(self) -> list[str]:
        return sorted(self.pdm)


def load_baseline(method_dir: Path) -> Baseline:
    """Load baseline PDM per scene (trajectories are attached lazily)."""

    records = load_records(method_dir)
    pdm = {
        scene: float(row["pdm"])
        for scene, row in records.items()
        if row.get("pdm") is not None
    }
    return Baseline(method_dir=Path(method_dir), pdm=pdm)


def attach_baseline_trajectories(
    baseline: Baseline,
    scenes: Sequence[str],
    *,
    need_target: bool,
) -> None:
    """Load baseline trajectories/targets for the fixed panel only."""

    baseline.trajectory = {
        scene: load_trajectory(baseline.method_dir, scene) for scene in scenes
    }
    if need_target:
        baseline.target = {
            scene: load_target_trajectory(baseline.method_dir, scene)
            for scene in scenes
        }


def load_scene_panel(path: Path | None, scenes: Sequence[str], max_scenes: int | None) -> list[str]:
    selected = list(scenes)
    if path is not None:
        text = Path(path).expanduser().read_text(encoding="utf-8").strip()
        if text.startswith("["):
            values = json.loads(text)
        else:
            values = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        panel = []
        for value in values:
            token = value.get("scene_token") if isinstance(value, Mapping) else value
            if token is None:
                raise ValueError(f"scene panel entry {value!r} has no scene_token")
            panel.append(str(token))
        missing = [token for token in panel if token not in set(selected)]
        if missing:
            raise ValueError(
                f"scene panel contains {len(missing)} scenes absent from the baseline, "
                f"first: {missing[0]!r}"
            )
        selected = panel
    if max_scenes is not None:
        selected = selected[: int(max_scenes)]
    if not selected:
        raise ValueError("empty scene panel")
    return selected


# ---------------------------------------------------------------------------
# runner-backed evaluator
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    name: str
    mask_path: Path
    method_dir: Path | None
    per_scene: dict[str, dict[str, float]]
    metrics: dict[str, float]


def discover_method_dirs(output_root: Path) -> dict[str, Path]:
    """Map absolute selector mask path -> method directory for one suite root."""

    mapping: dict[str, Path] = {}
    round_dir = Path(output_root) / "round01"
    if not round_dir.is_dir():
        raise FileNotFoundError(f"missing round01 under {output_root}")
    for config_path in sorted(round_dir.glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        press = config.get("press") or {}
        selector = press.get("selector") if isinstance(press, dict) else None
        if not isinstance(selector, dict):
            continue
        if str(selector.get("name")) != "oracle_future_token_mask":
            continue
        path = selector.get("path")
        if path is None:
            continue
        mapping[str(Path(path).resolve())] = config_path.parent
    return mapping


def _distributed_prefix(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(args.nproc_per_node)}",
        str(Path(args.runner)),
    ]


def _common_runner_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--dump-trajectories"]
    if args.dump_target:
        flags.append("--dump-target-trajectories")
    if args.max_eval_tokens is not None:
        flags += ["--max-eval-tokens", str(int(args.max_eval_tokens))]
    if args.poc_test_derived:
        flags.append("--poc-test-derived")
    if args.force_full_scene_set:
        flags.append("--force-full-scene-set")
    if args.skip_plots:
        flags.append("--skip-plots")
    return flags


def build_mask_runner_command(
    args: argparse.Namespace,
    mask_paths: Sequence[Path],
    output_root: Path,
) -> list[str]:
    """One runner invocation evaluating every candidate mask in the batch.

    By default the in-suite ``physical_no_press`` arm is skipped: the driver
    already holds a baseline (external or self-run) and re-evaluating NoPress on
    every candidate batch only burns GPU time.  ``--keep-suite-baseline`` keeps
    it for a self-contained suite.  The baseline command itself never carries
    this flag, because there it would defeat its purpose.
    """

    cmd = _distributed_prefix(args)
    cmd += [
        "--future-oracle-token-mask-jsons",
        ",".join(str(path) for path in mask_paths),
        "--future-counterfactual-layer",
        str(int(args.layer)),
    ]
    cmd += _common_runner_flags(args)
    if not bool(getattr(args, "keep_suite_baseline", False)):
        cmd.append("--persistent-skip-baseline")
    if args.sample_seed_diffusion is not None:
        cmd += ["--sample-seed", str(int(args.sample_seed_diffusion))]
    cmd += ["--output-root", str(output_root)]
    return cmd


def build_baseline_runner_command(
    args: argparse.Namespace,
    output_root: Path,
) -> list[str]:
    cmd = _distributed_prefix(args)
    cmd += ["--domain", "future_video", "--methods", "physical_no_press"]
    cmd += _common_runner_flags(args)
    cmd += ["--output-root", str(output_root)]
    return cmd


def _suite_artifacts_present(output_root: Path) -> bool:
    root = resolve_actual_suite_root(output_root)
    return (root / "suite_summary.json").is_file() and (root / "round01").is_dir()


def resolve_actual_suite_root(requested: Path) -> Path:
    """Resolve the suite root the runner actually used.

    The runner refuses to overwrite an existing root and appends ``_rerunNN``
    instead, so a search that restarts against a partially written directory
    must follow that rename or it would read stale artifacts.  Roots that
    contain a ``suite_summary.json`` win; a bare ``round01`` is only a fallback.
    """

    requested = Path(requested)
    candidates = [requested] + sorted(
        requested.parent.glob(f"{requested.name}_rerun*"),
        key=lambda path: path.stat().st_mtime,
    )
    completed = [
        path for path in candidates if (path / "suite_summary.json").is_file()
    ]
    if completed:
        return max(completed, key=lambda path: path.stat().st_mtime)
    if (requested / "round01").is_dir():
        return requested
    return candidates[-1] if len(candidates) > 1 else requested


def _run_runner(
    cmd: Sequence[str],
    output_root: Path,
    env: Mapping[str, str],
    *,
    allow_truncated_poc: bool,
) -> None:
    """Run the official runner, tolerating the documented truncated-POC exit 1.

    ``enforce_evaluation_scope`` returns 1 for every truncated run by design, so
    a POC number cannot be quoted as a full-protocol result.  Artifacts are
    complete at that point and every oracle search stage is a truncated POC, so
    only that specific case is accepted; anything else stays fatal.
    """

    completed = subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), env=dict(env))
    if completed.returncode == 0:
        return
    if allow_truncated_poc and _suite_artifacts_present(output_root):
        print(
            f"[oracle-search] runner exited {completed.returncode} "
            "(truncated POC); artifacts present, continuing",
            flush=True,
        )
        return
    raise subprocess.CalledProcessError(completed.returncode, list(cmd))


class RunnerEvaluator:
    """Evaluate candidate token masks through the official runner.

    One ``evaluate`` call writes each candidate mask JSON, launches the runner
    once with ``--future-oracle-token-mask-jsons`` so every candidate becomes a
    physical method in the same suite, then reads per-scene PDM and (optionally)
    trajectory displacement / planning harm against the baseline.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        work_dir: Path,
        baseline: Baseline,
        panel: Sequence[str],
    ):
        self.args = args
        self.work_dir = Path(work_dir)
        self.baseline = baseline
        self.panel = list(panel)
        self.objective = str(args.objective)
        # Every candidate is scored with PDM *and* the trajectory objectives the
        # 2026-09-20 conclusion asked for, independent of which one drives the
        # search.  Missing dumps simply yield NaN.
        self.need_trajectory = bool(panel) or bool(OBJECTIVES[self.objective]["needs_trajectory"])
        self.need_target = bool(getattr(args, "dump_target", True))
        self.commands_path = self.work_dir / "commands.jsonl"
        self._round = 0
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # -- commands ----------------------------------------------------------
    def build_runner_command(self, mask_paths: Sequence[Path], output_root: Path) -> list[str]:
        return build_mask_runner_command(self.args, mask_paths, output_root)

    def build_baseline_command(self, output_root: Path) -> list[str]:
        return build_baseline_runner_command(self.args, output_root)

    def _run(self, cmd: Sequence[str], output_root: Path) -> None:
        with self.commands_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(list(cmd)) + "\n")
        env = dict(os.environ)
        if self.args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(self.args.cuda_visible_devices)
        print("[oracle-search] " + " ".join(cmd), flush=True)
        _run_runner(cmd, output_root, env, allow_truncated_poc=bool(self.args.poc_test_derived))

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        candidates: Mapping[str, Mapping[str, Any]],
        *,
        tag: str,
    ) -> dict[str, CandidateResult]:
        self._round += 1
        batch_dir = self.work_dir / "masks" / f"r{self._round:03d}_{tag}"
        mask_paths: list[Path] = []
        for name, mask in candidates.items():
            path = batch_dir / f"{name}.json"
            write_token_mask_json(mask, path)
            mask_paths.append(path)
        output_root = self.work_dir / "suites" / f"r{self._round:03d}_{tag}"
        if self.args.dry_run:
            print(
                "[oracle-search][dry-run] "
                + " ".join(self.build_runner_command(mask_paths, output_root))
            )
            return {}
        self._run(self.build_runner_command(mask_paths, output_root), output_root)
        method_map = discover_method_dirs(resolve_actual_suite_root(output_root))

        results: dict[str, CandidateResult] = {}
        raw: dict[str, dict[str, float]] = {}
        for name, path in zip(candidates, mask_paths):
            method_dir = method_map.get(str(path.resolve()))
            if method_dir is None:
                raise RuntimeError(
                    f"runner did not produce a method for candidate {name!r} ({path})"
                )
            per_scene, metrics = self.candidate_metrics(method_dir)
            raw[name] = metrics
            results[name] = CandidateResult(
                name=name,
                mask_path=path,
                method_dir=method_dir,
                per_scene=per_scene,
                metrics=metrics,
            )
        if len(results) > 1:
            combined = combined_harm(
                [raw[name].get("pdm_harm", math.nan) for name in results],
                [raw[name].get("traj_disp", math.nan) for name in results],
                [raw[name].get("planning_harm", math.nan) for name in results],
            )
            for index, name in enumerate(results):
                results[name].metrics["combined_harm"] = float(combined[index])
        return results

    def candidate_metrics(
        self, method_dir: Path
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        records = load_records(method_dir)
        per_scene: dict[str, dict[str, float]] = {}
        for scene in self.panel:
            row = records.get(scene)
            if row is None or row.get("pdm") is None:
                continue
            if "valid" in row and not bool(row["valid"]):
                continue
            pdm = float(row["pdm"])
            baseline_pdm = self.baseline.pdm.get(scene)
            entry = {
                "pdm": pdm,
                "pdm_harm": (
                    math.nan if baseline_pdm is None else float(baseline_pdm) - pdm
                ),
            }
            if self.need_trajectory or self.need_target:
                candidate_traj = load_trajectory(method_dir, scene)
                baseline_traj = self.baseline.trajectory.get(scene)
                if candidate_traj is not None and baseline_traj is not None:
                    entry["traj_disp"] = trajectory_displacement(
                        baseline_traj, candidate_traj
                    )
                else:
                    entry["traj_disp"] = math.nan
            if self.need_target:
                target = self.baseline.target.get(scene)
                baseline_traj = self.baseline.trajectory.get(scene)
                candidate_traj = load_trajectory(method_dir, scene)
                if (
                    target is not None
                    and baseline_traj is not None
                    and candidate_traj is not None
                ):
                    entry["planning_harm"] = planning_harm(
                        baseline_traj, candidate_traj, target
                    )
                else:
                    entry["planning_harm"] = math.nan
            per_scene[scene] = entry
        metrics = {
            key: _nanmean([entry.get(key, math.nan) for entry in per_scene.values()])
            for key in ("pdm", "pdm_harm", "traj_disp", "planning_harm")
        }
        return per_scene, metrics


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return math.nan
    return float(finite.mean())


# ---------------------------------------------------------------------------
# search orchestration
# ---------------------------------------------------------------------------


def _check_keep_ratio(value: float) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError("--keep-ratio must be within (0, 1]")
    return value


def mask_for_panel(
    tokens: Sequence[int],
    panel: Sequence[str],
    *,
    tokens_per_latent: int,
    num_latents: int,
) -> dict[str, dict[str, list[int]]]:
    latent_mask = latent_local_mask(
        tokens, tokens_per_latent=tokens_per_latent, num_latents=num_latents
    )
    return {str(scene): latent_mask for scene in panel}


def build_evaluate_mask_candidates(
    args: argparse.Namespace,
    panel: Sequence[str],
    budget: int,
    mask_paths: Sequence[Path],
) -> dict[str, Any]:
    """Named mask candidates for ``--mode evaluate-masks``.

    Named mask JSONs first (they may use a ``"*"`` broadcast entry, which is what
    a shared searched pattern looks like), then optionally N per-scene random
    masks of the same budget as a matched random control measured in the very
    same runner call -- the point of a verification run is to compare a searched
    pattern against the random band on an identical panel.
    """

    candidates: dict[str, Any] = {
        f"mask{index:03d}": json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        for index, path in enumerate(mask_paths)
    }
    extra = int(getattr(args, "include_random_masks", 0) or 0)
    if extra > 0:
        per_sample = random_token_masks(
            panel,
            tokens_per_latent=int(args.tokens_per_latent),
            num_latents=int(args.num_future_latents),
            budget=int(budget),
            n_samples=extra,
            seed=int(args.sample_seed),
        )
        for index, mask in per_sample.items():
            candidates[f"random{index:03d}"] = mask
    return candidates


def compose_top_groups(
    groups: Sequence[Sequence[int]],
    importance: Sequence[float],
    budget: int,
) -> list[int]:
    """Keep the highest-importance groups until the token budget is reached."""

    order = sorted(
        range(len(groups)),
        key=lambda index: (
            not math.isfinite(importance[index]),
            -(importance[index] if math.isfinite(importance[index]) else 0.0),
            index,
        ),
    )
    selected: list[int] = []
    for index in order:
        group = list(groups[index])
        if len(selected) + len(group) > budget:
            continue
        selected.extend(group)
    return sorted(set(selected))


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    args.keep_ratio = _check_keep_ratio(args.keep_ratio)
    objective = str(args.objective)
    maximize = objective_higher_is_better(objective)
    need_trajectory = True
    need_target = bool(getattr(args, "dump_target", True))

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dry_run = bool(args.dry_run)

    baseline_dir = args.baseline_method_dir
    if baseline_dir is None:
        if dry_run:
            raise ValueError(
                "--dry-run requires --baseline-method-dir (it cannot run the baseline)"
            )
        baseline_suite = output_root / "baseline"
        cmd = build_baseline_runner_command(args, baseline_suite)
        env = dict(os.environ)
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        print("[oracle-search] " + " ".join(cmd), flush=True)
        _run_runner(
            cmd,
            baseline_suite,
            env,
            allow_truncated_poc=bool(args.poc_test_derived),
        )
        baseline_dir = resolve_actual_suite_root(baseline_suite) / "round01" / "physical_no_press"
    baseline_dir = Path(baseline_dir).expanduser().resolve()
    baseline = load_baseline(baseline_dir)
    panel = load_scene_panel(args.scene_tokens_file, baseline.scene_tokens(), args.max_scenes)
    if need_trajectory or need_target:
        attach_baseline_trajectories(baseline, panel, need_target=need_target)

    total_tokens = int(args.tokens_per_latent) * int(args.num_future_latents)
    budget = int(round(total_tokens * args.keep_ratio))
    groups = build_token_groups(
        args.tokens_per_latent,
        args.num_future_latents,
        mode=args.group_mode,
        group_size=args.group_size,
        block_h=args.block_h,
        block_w=args.block_w,
        width=args.grid_width,
        seed=args.group_seed,
    )
    work_dir = output_root / "search"
    evaluator = RunnerEvaluator(args, work_dir, baseline, panel)

    report: dict[str, Any] = {
        "mode": args.mode,
        "objective": objective,
        "higher_is_better": maximize,
        "keep_ratio": args.keep_ratio,
        "budget_tokens": budget,
        "total_tokens": total_tokens,
        "tokens_per_latent": int(args.tokens_per_latent),
        "num_future_latents": int(args.num_future_latents),
        "layer": int(args.layer),
        "group_mode": args.group_mode,
        "group_size": int(args.group_size) if args.group_size else None,
        "n_groups": len(groups),
        "baseline_method_dir": str(baseline_dir),
        "baseline_pdm": mean_over_scenes(baseline.pdm, panel),
        "n_scenes": len(panel),
        "scenes": panel,
        "dry_run": dry_run,
    }

    if args.mode == "evaluate-masks":
        if not args.mask_jsons:
            raise ValueError("--mode evaluate-masks requires --mask-jsons")
        paths = [Path(value.strip()) for value in str(args.mask_jsons).split(",") if value.strip()]
        candidates = build_evaluate_mask_candidates(args, panel, budget, paths)
        results = evaluator.evaluate(candidates, tag="evaluate")
        report["candidates"] = _candidate_table(results, baseline.pdm, panel)
        _write_report(output_root, report)
        return report

    if args.mode == "random-best-of-n":
        per_sample = random_token_masks(
            panel,
            tokens_per_latent=int(args.tokens_per_latent),
            num_latents=int(args.num_future_latents),
            budget=budget,
            n_samples=int(args.n_samples),
            seed=int(args.sample_seed),
        )
        candidates = {f"sample{index:03d}": mask for index, mask in per_sample.items()}
        results = evaluator.evaluate(candidates, tag="random")
        if dry_run:
            report["candidates"] = []
            _write_report(output_root, report)
            return report
        candidate_scores = {
            name: {scene: values["pdm"] for scene, values in result.per_scene.items()}
            for name, result in results.items()
        }
        oracle = per_scene_oracle(candidate_scores, higher_is_better=True, scenes=panel)
        report["candidates"] = _candidate_table(results, baseline.pdm, panel)
        report["random_per_scene_oracle"] = oracle
        _write_report(output_root, report)
        return report

    if args.mode == "independent-topk":
        full = list(range(total_tokens))
        candidates = {}
        for index, group in enumerate(groups):
            remaining = sorted(set(full) - set(group))
            candidates[f"drop_g{index:03d}"] = mask_for_panel(
                remaining,
                panel,
                tokens_per_latent=args.tokens_per_latent,
                num_latents=args.num_future_latents,
            )
        results = evaluator.evaluate(candidates, tag="drop_one_group")
        if dry_run:
            report["candidates"] = []
            _write_report(output_root, report)
            return report
        # Importance is "how much harm this group carries": for a minimised harm
        # objective the raw value is already the importance, for PDM it is the
        # baseline minus candidate PDM (pdm_harm).
        if objective == "pdm":
            importance = [results[name].metrics["pdm_harm"] for name in candidates]
        else:
            importance = [
                -results[name].metrics[objective]
                if maximize
                else results[name].metrics[objective]
                for name in candidates
            ]
        # NaN-safe: the deterministic group order is the tie-break.
        ordered = sorted(range(len(groups)), key=lambda index: -importance[index] if math.isfinite(importance[index]) else math.inf)
        group_importance = [
            {
                "group": int(index),
                "size": len(groups[index]),
                "local": list(groups[index]),
                "importance": float(importance[index]),
                **{
                    key: float(results[f"drop_g{index:03d}"].metrics.get(key, math.nan))
                    for key in ("pdm", "pdm_harm", "traj_disp", "planning_harm")
                },
            }
            for index in ordered
        ]
        composed = compose_top_groups(groups, importance, budget)
        composed_mask = mask_for_panel(
            composed,
            panel,
            tokens_per_latent=args.tokens_per_latent,
            num_latents=args.num_future_latents,
        )
        composed_results = evaluator.evaluate(
            {"topk_composed": composed_mask}, tag="topk_composed"
        )
        report["drop_one_group"] = _candidate_table(results, baseline.pdm, panel)
        report["group_importance"] = group_importance
        report["composed_selected_tokens"] = composed
        report["composed"] = _candidate_table(composed_results, baseline.pdm, panel)
        _write_report(output_root, report)
        return report

    # adaptive set-level searches
    def score_many(token_sets: Sequence[Sequence[int]]) -> list[float]:
        if dry_run:
            raise RuntimeError(
                "adaptive search modes execute the runner; use --mode random-best-of-n "
                "or --mode evaluate-masks for a dry run"
            )
        candidates = {
            f"cand{index:03d}": mask_for_panel(
                tokens,
                panel,
                tokens_per_latent=args.tokens_per_latent,
                num_latents=args.num_future_latents,
            )
            for index, tokens in enumerate(token_sets)
        }
        results = evaluator.evaluate(candidates, tag="search")
        return [float(results[name].metrics[objective]) for name in candidates]

    scorer = SetScorer(score_many=score_many)
    if args.mode == "greedy-forward":
        result = greedy_forward_selection(
            groups,
            budget,
            scorer,
            maximize=maximize,
            min_gain=float(args.min_gain),
            max_rounds=args.max_rounds,
        )
    elif args.mode == "greedy-backward":
        result = greedy_backward_elimination(
            groups,
            budget,
            scorer,
            maximize=maximize,
            max_harm=float(args.max_harm),
            max_rounds=args.max_rounds,
        )
    elif args.mode == "beam":
        result = beam_search(
            groups,
            budget,
            scorer,
            beam_width=int(args.beam_width),
            maximize=maximize,
            max_rounds=args.max_rounds,
        )
    else:  # pragma: no cover - argparse choices guard this
        raise ValueError(f"unsupported adaptive mode {args.mode!r}")

    report["search"] = result.to_dict()
    final_mask = mask_for_panel(
        result.selected,
        panel,
        tokens_per_latent=args.tokens_per_latent,
        num_latents=args.num_future_latents,
    )

    # Matched random control on the same groups/budget, so the search is never
    # reported without a budget-matched random baseline.  Both are evaluated in
    # ONE runner call so the paired (final - random) interval is free.
    random_scorer = SetScorer(score_many=score_many)
    random_result = random_search(
        groups,
        budget,
        random_scorer,
        n_samples=int(args.n_samples),
        seed=int(args.sample_seed),
        maximize=maximize,
    )
    report["matched_random"] = random_result.to_dict()
    random_mask = mask_for_panel(
        random_result.selected,
        panel,
        tokens_per_latent=args.tokens_per_latent,
        num_latents=args.num_future_latents,
    )
    same_mask = set(result.selected) == set(random_result.selected)
    batch = {"search_final": final_mask}
    if not same_mask:
        batch["matched_random"] = random_mask
    control_results = evaluator.evaluate(batch, tag="final_vs_random")
    final_result = control_results["search_final"]
    report["final"] = _candidate_table(
        {"search_final": final_result}, baseline.pdm, panel
    )
    if not same_mask:
        report["matched_random_eval"] = _candidate_table(
            {"matched_random": control_results["matched_random"]}, baseline.pdm, panel
        )
        final_pdm = {s: v["pdm"] for s, v in final_result.per_scene.items()}
        random_pdm = {
            s: v["pdm"] for s, v in control_results["matched_random"].per_scene.items()
        }
        report["final_vs_matched_random"] = paired_bootstrap(
            final_pdm, random_pdm, resamples=20000, seed=DEFAULT_BOOTSTRAP_SEED
        )
    _write_report(output_root, report)
    return report


def decision_stats(
    per_scene: Mapping[str, Mapping[str, float]],
    baseline_pdm: Mapping[str, float],
    *,
    resamples: int = 20000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap + tail audit for one candidate arm.

    A panel-mean PDM is not enough to decide anything: the 2026-09-20 token-level
    runs showed that a two-scene flip moves a 64-scene panel by 0.031 PDM.  Every
    candidate therefore carries its paired delta, 95% CI, flip counts and
    zero-score tail against the NoPress baseline.
    """

    candidate = {
        scene: float(values["pdm"])
        for scene, values in per_scene.items()
        if values.get("pdm") is not None
    }
    stats = dict(
        paired_bootstrap(candidate, dict(baseline_pdm), resamples=resamples, seed=seed)
    )
    shared = sorted(set(candidate) & set(baseline_pdm))
    diffs = [candidate[scene] - float(baseline_pdm[scene]) for scene in shared]
    stats.update(
        {
            "n": len(shared),
            "delta": float(sum(diffs) / len(diffs)) if diffs else math.nan,
            "improved": sum(1 for value in diffs if value > 1e-9),
            "tied": sum(1 for value in diffs if abs(value) <= 1e-9),
            "worse": sum(1 for value in diffs if value < -1e-9),
            "extreme_flips": sum(1 for value in diffs if abs(value) > 0.5),
            "zero_candidate": sum(1 for scene in shared if candidate[scene] <= 1e-9),
            "zero_baseline": sum(1 for scene in shared if float(baseline_pdm[scene]) <= 1e-9),
        }
    )
    return stats


def _candidate_table(
    results: Mapping[str, CandidateResult],
    baseline_pdm: Mapping[str, float] | None = None,
    panel: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    table = []
    for name, result in results.items():
        row = {"name": name, "mask_path": str(result.mask_path), "method_dir": None if result.method_dir is None else str(result.method_dir)}
        row.update({key: float(value) for key, value in result.metrics.items()})
        if baseline_pdm is not None:
            per_scene = result.per_scene
            if panel is not None:
                allowed = set(panel)
                per_scene = {
                    scene: values
                    for scene, values in per_scene.items()
                    if scene in allowed
                }
            row["vs_baseline"] = decision_stats(per_scene, baseline_pdm)
        table.append(row)
    return table


def _write_report(output_root: Path, report: Mapping[str, Any]) -> None:
    (output_root / "oracle_search_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    lines = [
        f"# Token-level set-level future oracle search ({report['mode']})",
        "",
        f"- objective: `{report['objective']}` (higher_is_better={report['higher_is_better']})",
        f"- panel: {report['n_scenes']} scenes; baseline PDM {report['baseline_pdm']:.6f}",
        f"- budget: {report['budget_tokens']}/{report['total_tokens']} tokens "
        f"(keep_ratio={report['keep_ratio']})",
        f"- groups: mode={report['group_mode']} size={report['group_size']} n={report['n_groups']}",
        f"- layer: {report['layer']}; dry_run={report['dry_run']}",
        "",
    ]
    for key in ("candidates", "drop_one_group", "composed", "final"):
        rows = report.get(key)
        if not rows:
            continue
        lines.append(f"## {key}")
        lines.append("")
        lines.append(
            "| name | pdm | pdm_harm | traj_disp | planning_harm | dPDM vs NoPress | 95% CI | "
            "improved/tied/worse | extreme flips | zero cand/base |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |"
        )
        for row in rows:
            stats = row.get("vs_baseline") or {}
            ci = stats.get("ci95") or [math.nan, math.nan]
            lines.append(
                f"| {row['name']} | {_fmt(row.get('pdm'))} | {_fmt(row.get('pdm_harm'))} | "
                f"{_fmt(row.get('traj_disp'))} | {_fmt(row.get('planning_harm'))} | "
                f"{_fmt(stats.get('delta'))} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
                f"{stats.get('improved', 'n/a')}/{stats.get('tied', 'n/a')}/"
                f"{stats.get('worse', 'n/a')} | {stats.get('extreme_flips', 'n/a')} | "
                f"{stats.get('zero_candidate', 'n/a')}/{stats.get('zero_baseline', 'n/a')} |"
            )
        lines.append("")
    if "final_vs_matched_random" in report:
        vs = report["final_vs_matched_random"]
        ci = vs.get("ci95") or [math.nan, math.nan]
        lines += [
            "## final vs matched random (paired)",
            "",
            f"- dPDM {_fmt(vs.get('delta'))}, 95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}], "
            f"n={vs.get('n')}, excludes zero: {vs.get('ci_excludes_zero')}",
            "",
        ]
    if "search" in report:
        search = report["search"]
        lines += [
            "## search",
            "",
            f"- selected {search['n_selected']} tokens; score {search['score']:.6f}; "
            f"rounds {search['metadata'].get('rounds')}; "
            f"evaluations {search['n_evaluations']}",
            "",
        ]
    if "matched_random" in report:
        random = report["matched_random"]
        lines += [
            "## matched random",
            "",
            f"- selected {random['n_selected']} tokens; score {random['score']:.6f}; "
            f"samples {random['metadata'].get('n_samples')}",
            "",
        ]
    if "random_per_scene_oracle" in report:
        oracle = report["random_per_scene_oracle"]
        lines += [
            "## per-scene best-of-N oracle",
            "",
            f"- oracle mean PDM {oracle['oracle_mean']:.6f} over {oracle['n_scenes']} scenes",
            f"- per-candidate mean PDM: {json.dumps(oracle['per_candidate_mean'])}",
            "",
        ]
    (output_root / "oracle_search_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:+.6f}" if abs(value) < 10 else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_search(args)
    print(json.dumps({key: report[key] for key in ("mode", "n_scenes", "budget_tokens")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

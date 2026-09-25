#!/usr/bin/env python3
"""Evaluate a trained Route A module on the official NAVSIM protocol.

Loads the real DriveVA pipeline, rebuilds :class:`RouteADynamicSelect` with the
same configuration the trainer used, loads the Route A weights out of a training
checkpoint (the checkpoint holds ONLY Route A plus the trajectory modules, since
the backbone was frozen), and runs the official evaluation.

``--arm route_a`` evaluates with ``physical_shortening=True`` -- the real
deployment condition, where the sequence is genuinely shortened.  ``A1`` was
trained with the dense-gated relaxation, so this run intentionally crosses the
train/inference gap; ``--physical-shortening-final-steps`` in the trainer exists
to close it later.

``--arm no_press`` runs the identical pipeline with no Route A attached, giving
the same-scene baseline for a paired delta.

Each rank writes its own CSV; merge them with ``--merge-only``.

Usage::

    CUDA_VISIBLE_DEVICES=0,1,5,6 python -m torch.distributed.run \\
        --standalone --nproc_per_node=4 scripts/eval_route_a_navsim.py \\
        --arm route_a --route-a-checkpoint <step-N.safetensors> \\
        --max-eval-tokens 256 --out-root <dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from videopress.retraining import (  # noqa: E402
    RouteAConfig,
    RouteALayoutSpec,
    SafetyClampConfig,
)

NVME = Path("/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1")
DEFAULT_FULL_CKPT = PROJECT_ROOT / "checkpoints/pdms90_9.safetensors"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models"
DEFAULT_SCENE_FILTER = (
    PROJECT_ROOT / "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml"
)
ROUTE_A_PREFIX = "dit._tokenpress_route_a."


class MatchedRandomScorer(nn.Module):
    """Keep the trained score *distribution*, destroy the ranking.

    This is the control the Route A result cannot be interpreted without.  A
    plain random top-k arm would run at a different retention than the trained
    gate, so a PDM difference could come from either the ranking or the budget.
    Permuting the trained scorer's own scores *inside each domain* leaves the
    threshold, and therefore the realised keep count and the score histogram,
    exactly as they are; the only thing that changes is which tokens survive.

    It needs no training and no extra checkpoint: it wraps whatever scorer the
    evaluated checkpoint provides.  Run several ``--random-seed`` values and
    report the band, because one permutation is one sample from the
    random-selection distribution.
    """

    def __init__(self, inner: nn.Module, domain_sizes, seed: int = 0):
        super().__init__()
        self.inner = inner
        self.domain_sizes = [int(v) for v in domain_sizes]
        self.seed = int(seed)
        self.calls = 0

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        offset = 0
        pieces = []
        for size in self.domain_sizes:
            chunk = out[:, offset : offset + size]
            generator = torch.Generator()  # CPU generator: device-agnostic
            generator.manual_seed(self.seed * 1000003 + self.calls * 7919 + offset)
            order = torch.randperm(size, generator=generator).to(chunk.device)
            pieces.append(chunk[:, order])
            offset += size
        self.calls += 1
        return torch.cat(pieces, dim=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["route_a", "route_a_random", "no_press"], required=True)
    p.add_argument("--route-a-checkpoint", type=Path, default=None)
    p.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="seed for --arm route_a_random; one seed is one sample of the "
        "matched-retention random-selection band",
    )
    p.add_argument("--max-eval-tokens", type=int, default=256,
                   help="scenes PER RANK (total = this x world_size)")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--bottleneck", type=int, default=18)
    p.add_argument("--history-threshold", type=float, default=0.5)
    # ``None`` means "follow --history-threshold / --threshold", so an explicit
    # per-domain value always wins over the shared override.
    p.add_argument("--future-threshold", type=float, default=None)
    p.add_argument("--max-kept-total", type=int, default=780)
    p.add_argument(
        "--normalize-scores",
        type=int,
        default=0,
        help="standardise each domain's logits within each scene before "
        "thresholding; must match the recipe the checkpoint was trained with",
    )
    p.add_argument("--physical-shortening", type=int, default=1)
    p.add_argument("--threshold", type=float, default=None,
                   help="override both domain thresholds; for the calibration sweep")
    p.add_argument("--num-inference-steps", type=int, default=3)
    p.add_argument("--full-ckpt", type=Path, default=DEFAULT_FULL_CKPT)
    p.add_argument("--local-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--scene-filter-yaml", type=Path, default=DEFAULT_SCENE_FILTER)
    p.add_argument("--merge-only", action="store_true")
    return p.parse_args(argv)


def merge_csvs(out_root: Path) -> int:
    files = sorted(glob.glob(str(out_root / "csv" / "*" / "*.csv")))
    if not files:
        files = sorted(glob.glob(str(out_root / "csv" / "*.csv")))
    rows = {}
    for path in files:
        for row in csv.DictReader(open(path)):
            token = row.get("token")
            if not token or token == "average":
                continue
            if str(row.get("valid")) == "True":
                rows[token] = float(row["pdm_score"])
    if not rows:
        print("[merge] no valid scenes found", flush=True)
        return 1
    vals = list(rows.values())
    payload = {
        "n_scenes": len(vals),
        "pdm_mean": statistics.mean(vals),
        "pdm_stdev": statistics.pstdev(vals),
        "files": files,
    }
    stats_files = sorted(glob.glob(str(out_root / "route_a_stats_rank*.json")))
    if stats_files:
        merged = {"thresholds": None, "calls": 0, "kept_video_total": [], "candidate": None}
        for path in stats_files:
            blob = json.loads(Path(path).read_text())
            merged["thresholds"] = blob.get("thresholds")
            merged["candidate"] = blob.get("candidate")
            merged["calls"] += int(blob.get("calls") or 0)
            summary = blob.get("kept_video_total") or {}
            if summary:
                merged["kept_video_total"].append(summary)
        if merged["kept_video_total"]:
            means = [b["mean"] for b in merged["kept_video_total"]]
            payload["kept_video_mean_across_ranks"] = statistics.mean(means)
            payload["kept_video_rank_summaries"] = merged["kept_video_total"]
        payload["thresholds"] = merged["thresholds"]
        payload["candidate"] = merged["candidate"]
    target = out_root / "aggregate.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[merge] n={len(vals)} pdm_mean={payload['pdm_mean']:.6f} -> {target}", flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        return merge_csvs(args.out_root)

    from scripts import run_official_navsim_press as official_runner

    eval_mod = official_runner._load_official_eval_module()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print(f"[eval-route-a] arm={args.arm} rank={rank}/{world} device={device}", flush=True)

    build_args = SimpleNamespace(
        local_model_path=args.local_model_path,
        full_ckpt=args.full_ckpt,
        pre_dit_register_lora_checkpoint=None,
        pre_dit_register_lora_target_modules="",
        pre_dit_register_lora_rank=32,
    )
    started = time.time()
    pipe = official_runner._build_official_pipeline(eval_mod, build_args, device)
    print(f"[eval-route-a] pipeline built in {time.time() - started:.1f}s", flush=True)

    if args.arm in ("route_a", "route_a_random"):
        if args.route_a_checkpoint is None:
            raise SystemExit("--route-a-checkpoint is required for --arm route_a")
        dit = pipe.dit
        param_dtype = next(dit.parameters()).dtype
        # Resolve the per-domain thresholds once, with an explicit per-domain
        # value always beating the shared ``--threshold`` override.
        history_tau = float(args.history_threshold)
        future_tau = (
            history_tau
            if args.future_threshold is None
            else float(args.future_threshold)
        )
        if args.threshold is not None:
            history_tau = float(args.threshold)
            if args.future_threshold is None:
                future_tau = float(args.threshold)
        module = RouteAConfig(
            token_dim=int(dit.dim),
            bottleneck_layer=int(args.bottleneck),
            num_blocks=len(dit.blocks),
            layout=RouteALayoutSpec(),
            history_threshold=history_tau,
            future_threshold=future_tau,
            # Must match the training recipe exactly.  A checkpoint trained with
            # a standardised gate evaluated without it (or vice versa) is a
            # different model: the raw gate sees a per-scene score offset that
            # swamps the ranking and only has "keep everything"/"keep nothing"
            # states, so the retention would not match the run's calibration.
            normalize_scores=bool(args.normalize_scores),
            safety_clamp=SafetyClampConfig(
                min_kept_history=8, min_kept_future=32, max_kept_total=int(args.max_kept_total)
            ),
        ).build()
        from safetensors.torch import load_file

        raw = load_file(str(args.route_a_checkpoint))
        weights = {
            key[len(ROUTE_A_PREFIX):]: value
            for key, value in raw.items()
            if key.startswith(ROUTE_A_PREFIX)
        }
        if not weights:
            raise SystemExit(f"no {ROUTE_A_PREFIX}* tensors in {args.route_a_checkpoint}")
        missing, unexpected = module.load_state_dict(weights, strict=False)
        if unexpected:
            raise SystemExit(f"unexpected Route A keys: {unexpected[:5]}")
        print(
            f"[eval-route-a] loaded {len(weights)} tensors from {args.route_a_checkpoint.name}; "
            f"missing={len(missing)} thresholds={module.gate.thresholds()} "
            f"physical_shortening={bool(args.physical_shortening)}",
            flush=True,
        )
        # The A1 trainer also left the trajectory modules trainable, so the
        # checkpoint contains them.  Loading only the Route A tensors would
        # evaluate a Route A module against the ORIGINAL trajectory head, i.e. a
        # train/eval mismatch that shows up as a large PDM drop even at full
        # retention.  Load every non-Route-A tensor that the pipeline owns.
        extra_loaded = []
        for key, value in raw.items():
            if key.startswith(ROUTE_A_PREFIX):
                continue
            owner = pipe
            parts = key.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            if owner is None or not hasattr(owner, parts[-1]):
                continue
            target = getattr(owner, parts[-1])
            if isinstance(target, torch.Tensor) and target.shape == value.shape:
                with torch.no_grad():
                    target.copy_(value.to(dtype=target.dtype, device=target.device))
                extra_loaded.append(key)
        print(f"[eval-route-a] also loaded {len(extra_loaded)} non-Route-A tensors: "
              f"{extra_loaded[:6]}", flush=True)

        module = module.to(device=device, dtype=param_dtype).eval()

        # Matched-retention random control: same checkpoint, same thresholds,
        # same score histogram -- only the ranking inside each domain is
        # replaced by a random permutation.
        if args.arm == "route_a_random":
            module.scorer = MatchedRandomScorer(
                module.scorer, module.gate.domain_sizes, seed=int(args.random_seed)
            )
            print(
                f"[eval-route-a] MATCHED RANDOM control: permuting the scored "
                f"ranking inside each domain (seed={int(args.random_seed)})",
                flush=True,
            )

        # The evaluation CSV does not carry compression statistics (the official
        # runner adds those from records.jsonl, which this path bypasses), so
        # record the gate's kept counts here.  Without this number a PDM drop
        # cannot be attributed to over-compression versus a bad ranking.
        stats = {"calls": 0, "kept_history": [], "kept_future": [], "candidate": None,
                 "thresholds": module.gate.thresholds()}
        original_forward = module.forward

        def instrumented_forward(*fargs, **fkwargs):
            out = original_forward(*fargs, **fkwargs)
            try:
                stats["calls"] += 1
                stats["kept_history"].append(int(out.gate.kept_counts[0]))
                stats["kept_future"].append(int(sum(out.gate.kept_counts[1:])))
                stats["candidate"] = [int(v) for v in out.gate.candidate_counts]
            except Exception:
                pass
            return out

        module.forward = instrumented_forward
        dit._tokenpress_route_a = module
        dit._tokenpress_route_a_physical = bool(args.physical_shortening)
        dit._tokenpress_route_a_capture_layers = ()
    else:
        print("[eval-route-a] no_press baseline: no Route A attached", flush=True)

    out_dir = args.out_root / "csv" / f"rank{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):
        stale.unlink()
    argv_eval = [
        "--repo_root", str(PROJECT_ROOT),
        "--navsim_log_path", str(NVME / "openscene-v1.1/meta_datas/test"),
        "--sensor_blobs_path", str(NVME / "openscene-v1.1/sensor_blobs/test"),
        "--metric_cache_path", str(NVME / "metric_cache_full"),
        "--output_dir", str(out_dir),
        "--full_ckpt", str(args.full_ckpt),
        "--local_model_path", str(args.local_model_path),
        "--scene_filter_yaml", str(args.scene_filter_yaml),
        "--max_eval_tokens", str(args.max_eval_tokens),
        "--num_inference_steps", str(args.num_inference_steps),
        "--height", "480", "--width", "832",
        "--num_history_frames", "5", "--num_future_frames", "10",
        "--model_future_frames", "8",
        "--infer_trajectory_only",
        "--no_print_alignment_params",
        "--no_show_eval_progress",
    ]
    eval_args = eval_mod.parse_args(argv_eval)
    started = time.time()
    try:
        eval_mod.run_eval(eval_args, external_pipe=pipe)
    finally:
        if args.arm in ("route_a", "route_a_random"):
            pipe.dit._tokenpress_route_a = None
    print(f"[eval-route-a] rank{rank} run_eval done in {time.time() - started:.1f}s", flush=True)

    if args.arm in ("route_a", "route_a_random"):
        import statistics as _st

        def _summ(values):
            if not values:
                return None
            return {"n": len(values), "mean": _st.mean(values), "min": min(values),
                    "max": max(values), "p50": _st.median(values)}
        kept_total = [h + f for h, f in zip(stats["kept_history"], stats["kept_future"])]
        payload = {
            "rank": rank,
            "calls": stats["calls"],
            "candidate": stats["candidate"],
            "thresholds": stats["thresholds"],
            "kept_history": _summ(stats["kept_history"]),
            "kept_future": _summ(stats["kept_future"]),
            "kept_video_total": _summ(kept_total),
        }
        target = args.out_root / f"route_a_stats_rank{rank}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval-route-a] compression stats(rank{rank}): {json.dumps(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

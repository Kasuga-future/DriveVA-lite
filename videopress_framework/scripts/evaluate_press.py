#!/usr/bin/env python3
"""Run one config-driven VideoTokenPress evaluation.

The default backend is a small deterministic CPU cohort.  It exercises the
same layout/domain/press/evaluator/artifact path that a DriveVA adapter uses,
without loading the 5B Wan checkpoint.  A future NavSIM backend can reuse the
Evaluator and replace only the sample/model callbacks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from evaluation.evaluator import Evaluator, SceneSample
from evaluation.artifacts import jsonable
from videopress.core.layout import build_driveva_layout
from videopress.factory import build_press
from videopress.utils.seed import stable_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a DriveVA VideoTokenPress config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--press", default=None, help="Override press name, e.g. noop, random, token_norm")
    parser.add_argument("--budget", type=int, default=None, help="Override budget with an absolute K")
    parser.add_argument("--mode", choices=["causal", "physical"], default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def make_synthetic_cohort(config: dict, device: torch.device):
    spec = config.get("synthetic", {})
    frames = int(spec.get("frames", 4))
    height = int(spec.get("height", 3))
    width = int(spec.get("width", 4))
    num_cond_latents = int(spec.get("num_cond_latents", 2))
    trajectory_length = int(spec.get("trajectory_length", 6))
    trajectory_prefix_length = int(spec.get("trajectory_prefix_length", 2))
    hidden_dim = int(spec.get("hidden_dim", 16))
    heads = int(spec.get("heads", 2))
    scenes = int(spec.get("scenes", config.get("benchmark", {}).get("max_scenes", 4)))
    if hidden_dim % heads:
        raise ValueError("synthetic.hidden_dim must be divisible by synthetic.heads")
    layout = build_driveva_layout(
        frames,
        height,
        width,
        num_cond_latents,
        trajectory_length,
        trajectory_prefix_length,
    )
    samples = []
    for scene_index in range(scenes):
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(config.get("experiment", {}).get("seed", 20260828), "scene", scene_index)
        )
        tokens = torch.randn((1, layout.total_length, hidden_dim), generator=generator, dtype=torch.float32).to(device)
        q = torch.randn((1, heads, layout.total_length, hidden_dim // heads), generator=generator, dtype=torch.float32).to(device)
        k = torch.randn((1, heads, layout.total_length, hidden_dim // heads), generator=generator, dtype=torch.float32).to(device)
        v = torch.randn((1, heads, layout.total_length, hidden_dim // heads), generator=generator, dtype=torch.float32).to(device)
        target = torch.tanh(tokens[:, layout.future_action.start : layout.future_action.end, :2]).detach()
        samples.append(
            SceneSample(
                scene_token=f"synthetic_scene_{scene_index:04d}",
                log_id=f"synthetic_log_{scene_index // 2:02d}",
                timestamp=scene_index * 500_000,
                tokens=tokens,
                q=q,
                k=k,
                v=v,
                target_trajectory=target,
                diffusion_rank=0,
                metadata={"domain": "last_history"},
            )
        )
    return layout, samples


def synthetic_predict(result, sample, ctx):
    action_length = ctx.layout.future_action.length
    if "k" in result.aux and ctx.q is not None:
        # Physical KV path: queries remain full length while K/V are shorter.
        q_action = ctx.q[:, :, ctx.layout.future_action.start : ctx.layout.future_action.end]
        k = result.aux["k"]
        v = result.aux["v"]
        weights = torch.softmax(torch.matmul(q_action.float(), k.float().transpose(-1, -2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended = torch.matmul(weights, v.float()).mean(dim=1)
        return attended[..., :2]
    video = result.output[:, : ctx.layout.video.length]
    pooled = video.mean(dim=1, keepdim=True)
    action_tokens = result.output[:, ctx.layout.future_action.start : ctx.layout.future_action.end, :2]
    return torch.tanh(action_tokens + 0.1 * pooled[..., :2].expand(-1, action_length, -1))


def synthetic_metrics(prediction, sample, ctx, result):
    target = sample.target_trajectory.to(prediction.device)
    error = prediction.float() - target.float()
    trajectory_l2 = torch.sqrt(error.pow(2).sum(dim=-1).mean()).item()
    endpoint_l2 = torch.linalg.vector_norm(error[:, -1], dim=-1).mean().item()
    return {
        "pdm": float(torch.exp(-torch.tensor(trajectory_l2)).item()),
        "trajectory_l2": trajectory_l2,
        "endpoint_l2": endpoint_l2,
        "valid": bool(torch.isfinite(prediction).all().item()),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    config = deepcopy(config)
    if args.press is not None:
        config.setdefault("press", {})["name"] = args.press
    if args.budget is not None:
        config.setdefault("press", {})["budget"] = {"type": "absolute", "value": args.budget, "reference": "eligible"}
    if args.mode is not None:
        config.setdefault("evaluation", {})["mode"] = args.mode
    if args.max_scenes is not None:
        config.setdefault("benchmark", {})["max_scenes"] = args.max_scenes
    if args.output_dir is not None:
        config.setdefault("output", {})["dir"] = str(args.output_dir)
    output_dir = Path(config.get("output", {}).get("dir", "outputs/press_run"))
    if not output_dir.is_absolute():
        output_dir = FRAMEWORK_ROOT / output_dir
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    layout, samples = make_synthetic_cohort(config, device)
    press = build_press(config.get("press", {}))
    result = Evaluator(PROJECT_ROOT).evaluate(
        samples,
        press,
        layout,
        mode=config.get("evaluation", {}).get("mode", "causal"),
        output_dir=output_dir,
        config=config,
        predict_fn=synthetic_predict,
        metric_fn=synthetic_metrics,
        max_scenes=config.get("benchmark", {}).get("max_scenes"),
    )
    print(json.dumps(jsonable({"output_dir": result["output_dir"], "summary": result["summary"]}), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

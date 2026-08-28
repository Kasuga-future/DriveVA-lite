#!/usr/bin/env python3
"""One-GPU smoke for the runtime Wan post-RoPE K/V hook."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
import json
from pathlib import Path
import sys

import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from diffsynth.models.wan_video_dit import SelfAttention, precompute_freqs_cis
from videopress.adapters import DriveVAAdapter
from videopress.core.budget import TokenBudget
from videopress.core.layout import build_driveva_layout
from videopress.core.runtime import InjectionPoint, VideoPressRuntime
from videopress.operators import KVPruneOperator
from videopress.presses import ScorerPress
from videopress.scorers import RandomScorer
from videopress.selectors import TopKSelector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPU smoke for VideoTokenPress Wan attention hook")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("gpu_smoke requires a CUDA device")
    layout = build_driveva_layout(f=4, h=3, w=4, num_cond_latents=2, traj_len=6, traj_prefix_len=2)
    dtype = torch.bfloat16
    attention = SelfAttention(dim=16, num_heads=2).to(device=device, dtype=dtype).eval()
    freqs = precompute_freqs_cis(attention.head_dim, end=layout.total_length).unsqueeze(1).to(device)
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn((1, layout.total_length, 16), generator=generator, device=device, dtype=dtype)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[SimpleNamespace(self_attn=attention)]), dit2=None)
    press = ScorerPress(
        scorer=RandomScorer(seed=20260828),
        selector=TopKSelector(),
        operator=KVPruneOperator(),
        budget=TokenBudget("absolute", 6),
        domain="last_history",
        injection_point=InjectionPoint.SELF_ATTN_KV,
    )
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=DriveVAAdapter())
    runtime.layout = layout
    runtime.current_scene = "gpu-smoke"
    runtime.current_sample = SimpleNamespace(metadata={"domain": "last_history"}, diffusion_rank=0)
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        baseline = attention(x, freqs)
        torch.cuda.synchronize(device)
        runtime.install(pipe)
        compressed = attention(x, freqs)
        torch.cuda.synchronize(device)
        result = runtime.last_result
        runtime.remove(pipe)
        restored = attention(x, freqs)
        torch.cuda.synchronize(device)
    if result is None:
        raise RuntimeError("attention hook did not emit a CompressionResult")
    q_len = int(result.aux["q"].shape[2])
    k_len = int(result.aux["k"].shape[2])
    v_len = int(result.aux["v"].shape[2])
    if q_len != layout.total_length or k_len != v_len or k_len >= layout.total_length:
        raise AssertionError(f"invalid hook lengths: q={q_len}, k={k_len}, v={v_len}")
    max_noop_diff = float((baseline - restored).abs().max().item())
    if max_noop_diff != 0.0:
        raise AssertionError(f"hook removal changed baseline output: max diff={max_noop_diff}")
    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_device_count": torch.cuda.device_count(),
        "baseline_shape": list(baseline.shape),
        "compressed_shape": list(compressed.shape),
        "q_length": q_len,
        "k_length": k_len,
        "v_length": v_len,
        "theoretical_attn_ratio": result.metadata["theoretical_attn_ratio"],
        "noop_after_remove_max_abs_diff": max_noop_diff,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

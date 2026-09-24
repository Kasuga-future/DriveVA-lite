#!/usr/bin/env python3
"""Benchmark true pre-DiT hidden-token pruning on real Wan DiT blocks.

This is a numerical/efficiency smoke, not a NAVSIM quality benchmark.  It uses
the production DriveVA adapter, selector, mapping and Wan block implementation
at the real DriveVA sequence layout while keeping the hidden width small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from diffsynth.models.wan_video_dit import DiTBlock, precompute_freqs_cis
from videopress.adapters.driveva import DriveVAAdapter
from videopress.core.layout import build_driveva_layout
from videopress.core.runtime import VideoPressRuntime
from videopress.factory import build_press


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args(argv)


def _run_blocks(blocks, x, context, t_mod, freqs):
    for block in blocks:
        x = block(x, context, t_mod, freqs)
    return x


def _timed(device, fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("pre_dit_gpu_smoke requires CUDA")
    if args.dim % args.heads:
        raise ValueError("--dim must be divisible by --heads")
    if args.dim // args.heads > 256:
        raise ValueError("FlashAttention requires head dimension <= 256")
    if not 0.0 < args.keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0, 1]")

    dtype = torch.bfloat16
    layout = build_driveva_layout(
        f=4, h=13, w=30, num_cond_latents=2, traj_len=9, traj_prefix_len=1
    )
    blocks = torch.nn.ModuleList(
        [
            DiTBlock(False, args.dim, args.heads, args.dim * 4).to(device, dtype).eval()
            for _ in range(args.layers)
        ]
    )
    generator = torch.Generator(device=device).manual_seed(20260915)
    x = torch.randn(
        1, layout.total_length, args.dim,
        device=device, dtype=dtype, generator=generator,
    )
    context = torch.randn(1, 16, args.dim, device=device, dtype=dtype, generator=generator)
    t_mod = torch.randn(
        1, layout.total_length, 6, args.dim,
        device=device, dtype=dtype, generator=generator,
    )
    freqs = precompute_freqs_cis(
        args.dim // args.heads, end=layout.total_length
    ).unsqueeze(1).to(device)

    press = build_press(
        {
            "name": "scorer_press",
            "injection_point": "block_input",
            "domain": "history",
            "scorer": {"name": "token_norm"},
            "selector": {"name": "topk"},
            "operator": {"name": "hidden_prune"},
            "budget": {
                "type": "ratio",
                "value": args.keep_ratio,
                "reference": "eligible",
            },
        }
    )
    adapter = DriveVAAdapter()
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=adapter)
    pipe = SimpleNamespace(dit=SimpleNamespace(blocks=blocks), dit2=None)
    runtime.install(pipe)

    def baseline():
        return _run_blocks(blocks, x, context, t_mod, freqs)

    observed = {}

    def compressed():
        runtime.begin_sample(
            SimpleNamespace(scene_token="gpu-smoke", metadata={}, diffusion_rank=0),
            layout,
        )
        controller = pipe.dit._tokenpress_pre_dit_controller
        short_x, short_freqs, short_t_mod = controller.begin_forward(
            x, freqs, t_mod, num_blocks=len(blocks)
        )
        short_out = _run_blocks(blocks, short_x, context, short_t_mod, short_freqs)
        restored = controller.finish_forward(short_out)
        observed["short_length"] = int(short_x.shape[1])
        observed["restored_length"] = int(restored.shape[1])
        observed["metadata"] = dict(runtime.last_result.metadata)
        return restored

    with torch.inference_mode():
        baseline_samples = _timed(device, baseline, args.warmup, args.repeats)
        compressed_samples = _timed(device, compressed, args.warmup, args.repeats)
    runtime.remove(pipe)
    baseline_ms = statistics.mean(baseline_samples)
    compressed_ms = statistics.mean(compressed_samples)
    payload = {
        "status": "SMOKE_ONLY_NOT_NAVSIM_QUALITY",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "layout_length": layout.total_length,
        "history_candidates": layout.history_video.length,
        "requested_keep_ratio": args.keep_ratio,
        "kept_history_tokens": observed["metadata"]["selected_count"],
        "short_length": observed["short_length"],
        "restored_length": observed["restored_length"],
        "layers": args.layers,
        "dim": args.dim,
        "heads": args.heads,
        "baseline_ms_mean": baseline_ms,
        "compressed_ms_mean": compressed_ms,
        "speedup_percent": 100.0 * (baseline_ms - compressed_ms) / baseline_ms,
        "baseline_ms_samples": baseline_samples,
        "compressed_ms_samples": compressed_samples,
        "selector_latency_ms": observed["metadata"]["timing"]["selector_latency_ms"],
        "hidden_sequence_ratio": observed["metadata"]["hidden_sequence_ratio"],
        "compressed_layer_count": observed["metadata"][
            "hidden_sequence_compressed_layer_count"
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

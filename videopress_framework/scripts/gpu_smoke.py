#!/usr/bin/env python3
"""One-GPU smoke for persistent runtime Wan post-RoPE K/V hooks."""

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
from videopress.scorers import ActionAttentionVNormScorer
from videopress.selectors import TopKSelector


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPU smoke for VideoTokenPress Wan attention hook")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--persistence-mode",
        choices=("kv_only", "hidden_sequence"),
        default="kv_only",
    )
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
    attentions = [
        SelfAttention(dim=16, num_heads=2).to(device=device, dtype=dtype).eval()
        for _ in range(4)
    ]
    freqs = precompute_freqs_cis(attentions[0].head_dim, end=layout.total_length).unsqueeze(1).to(device)
    generator = torch.Generator(device=device).manual_seed(7)
    x = torch.randn((1, layout.total_length, 16), generator=generator, device=device, dtype=dtype)
    pipe = SimpleNamespace(
        dit=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=attention) for attention in attentions]
        ),
        dit2=None,
    )
    press = ScorerPress(
        scorer=ActionAttentionVNormScorer(
            layer=1,
            head_mode="mean",
            action_mode="mean",
        ),
        selector=TopKSelector(),
        operator=KVPruneOperator(),
        budget=TokenBudget("absolute", 6),
        domain="last_history",
        injection_point=InjectionPoint.SELF_ATTN_KV,
        cross_layer_persistence={"enabled": True, "mode": args.persistence_mode},
    )
    runtime = VideoPressRuntime(press=press, mode="physical", adapter=DriveVAAdapter())
    runtime.layout = layout
    runtime.current_scene = "gpu-smoke"
    runtime.current_sample = SimpleNamespace(metadata={"domain": "last_history"}, diffusion_rank=0)
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        baseline = attentions[-1](x, freqs)
        torch.cuda.synchronize(device)
        runtime.install(pipe)
        runtime.begin_sample(
            SimpleNamespace(scene_token="gpu-smoke", metadata={}, diffusion_rank=0),
            layout,
        )
        compressed = x
        active_freqs = freqs
        controller = getattr(pipe.dit, "_tokenpress_hidden_sequence_controller", None)
        t_mod = torch.zeros(
            (1, layout.total_length, 6, 16), device=device, dtype=dtype
        )
        if controller is not None:
            controller.begin_forward(
                compressed, active_freqs, t_mod, num_blocks=len(attentions)
            )
        downstream_shape = None
        for layer, attention in enumerate(attentions):
            compressed = attention(compressed, active_freqs)
            if controller is not None:
                compressed, active_freqs, t_mod = controller.after_block(
                    layer, compressed, active_freqs, t_mod
                )
                if layer == 1:
                    downstream_shape = list(compressed.shape)
        if controller is not None:
            compressed = controller.finish_forward(compressed)
        torch.cuda.synchronize(device)
        results = [event.result for event in runtime.events]
        runtime.remove(pipe)
        restored = attentions[-1](x, freqs)
        torch.cuda.synchronize(device)
    expected_events = 1 if args.persistence_mode == "hidden_sequence" else 3
    if len(results) != expected_events:
        raise RuntimeError(f"expected {expected_events} events, got {len(results)}")
    result = results[-1]
    q_len = int(result.metadata["q_length"])
    k_len = int(result.metadata["k_length_after"])
    v_len = int(result.metadata["v_length_after"])
    if q_len != layout.total_length or k_len != v_len or k_len >= layout.total_length:
        raise AssertionError(f"invalid hook lengths: q={q_len}, k={k_len}, v={v_len}")
    max_noop_diff = float((baseline - restored).abs().max().item())
    if max_noop_diff != 0.0:
        raise AssertionError(f"hook removal changed baseline output: max diff={max_noop_diff}")
    mappings = [item.mapping.output_to_input for item in results]
    if not all(torch.equal(mappings[0], mapping) for mapping in mappings[1:]):
        raise AssertionError("deeper layers did not reuse the source-layer mapping")
    reused = [item.metadata["persistent_selection_reused"] for item in results]
    expected_reused = [False] if args.persistence_mode == "hidden_sequence" else [False, True, True]
    if reused != expected_reused:
        raise AssertionError(f"unexpected persistence markers: {reused}")
    if args.persistence_mode == "hidden_sequence":
        if downstream_shape is None or downstream_shape[1] != k_len:
            raise AssertionError(
                f"residual sequence was not shortened: {downstream_shape}, K={k_len}"
            )
        if compressed.shape[1] != layout.total_length:
            raise AssertionError("hidden sequence was not restored before the model head")
    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_device_count": torch.cuda.device_count(),
        "persistence_mode": args.persistence_mode,
        "baseline_shape": list(baseline.shape),
        "compressed_shape": list(compressed.shape),
        "downstream_shape": downstream_shape,
        "q_length": q_len,
        "k_length": k_len,
        "v_length": v_len,
        "theoretical_attn_ratio": result.metadata["theoretical_attn_ratio"],
        "event_layers": [event.key.layer_idx for event in runtime.events],
        "persistent_selection_reused": reused,
        "selection_source_layer": result.metadata["selection_source_layer"],
        "noop_after_remove_max_abs_diff": max_noop_diff,
    }
    if args.persistence_mode == "hidden_sequence":
        payload["hidden_sequence_ratio"] = result.metadata["hidden_sequence_ratio"]
        payload["hidden_sequence_compressed_layer_count"] = result.metadata[
            "hidden_sequence_compressed_layer_count"
        ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

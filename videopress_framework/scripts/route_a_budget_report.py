#!/usr/bin/env python3
"""Analytic cost model for Route A (Dynamic Select).

Answers the two questions a retraining plan has to justify before it is worth a
GPU week:

1. **How much compute can Route A actually save?**  Route A only shortens the
   sequence *after* the bottleneck layer, so the saving is
   ``(30 - Lb) / 30`` of the per-block sequence scaling, not a flat 90%.
2. **What does the compression machinery cost?**  The scorer runs once, but the
   dense recovery decoder is a real cross-attention module over the full grid.

The model is analytic (MAC counts from the module shapes), not a measurement.
Every assumption is a named flag so the numbers can be recomputed for the real
Wan2.2-5B geometry::

    python scripts/route_a_budget_report.py --dim 3072 --ffn-dim 14336 --layers 30
    python scripts/route_a_budget_report.py --bottleneck 15 --mean-kept 240

MAC convention: one "MAC" is one multiply-accumulate.  Attention scores and
their application are counted as ``2 * L^2 * d``, projections as
``L * d_in * d_out``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from torch import nn


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parent
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from videopress.retraining import (  # noqa: E402
    RouteAConfig,
    RouteALayoutSpec,
    RouteADynamicSelect,
    SafetyClampConfig,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--ffn-dim", type=int, default=14336)
    parser.add_argument("--layers", type=int, default=30)
    parser.add_argument("--bottleneck", type=int, default=18)
    parser.add_argument("--context-length", type=int, default=512,
                        help="text context length used by the cross-attention term")
    parser.add_argument("--history-tokens", type=int, default=780)
    parser.add_argument("--future-tokens", type=int, default=780)
    parser.add_argument("--traj-tokens", type=int, default=9)
    parser.add_argument("--mean-kept", type=float, default=240.0,
                        help="mean kept *video* tokens at deployment")
    parser.add_argument("--selector-hidden", type=int, default=256)
    parser.add_argument("--recovery-layers", type=int, default=2)
    parser.add_argument("--sweep-bottleneck", type=str, default="18,15,12",
                        help="comma-separated bottleneck layers for a saving curve")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def block_macs(seq: int, dim: int, ffn_dim: int, context: int) -> int:
    """Per-block MACs: q/k/v/o projections + attention + ffn + cross-attn."""
    projections = 4 * seq * dim * dim
    attention = 2 * seq * seq * dim
    ffn = 2 * seq * dim * ffn_dim
    cross = 4 * seq * dim * dim + 2 * seq * context * dim
    return int(projections + attention + ffn + cross)


def backbone_macs(layers: int, seq: int, dim: int, ffn_dim: int, context: int) -> int:
    return int(block_macs(seq, dim, ffn_dim, context) * layers)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        int(p.numel())
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


def module_macs(model: RouteADynamicSelect, layout: RouteALayoutSpec) -> Dict[str, int]:
    """Forward MACs of the three new modules at deployment time."""
    dim = model.config.token_dim
    hidden = model.config.selector_hidden
    candidates = layout.video_tokens
    kept = int(model.config.safety_clamp.max_kept_total or candidates)
    # Scorer: token MLP + position MLP + action/time MLP + interaction + head,
    # all evaluated once over every candidate.
    scorer = (
        candidates * (dim * hidden + hidden * hidden)          # token_proj
        + candidates * (3 * 64 + 64 * hidden)                  # pos_proj
        + (dim * hidden + hidden * hidden)                     # action_proj
        + (5 * hidden + hidden * hidden)                       # time_proj
        + candidates * (2 * hidden * hidden)                   # interaction
        + candidates * (4 * hidden * hidden + hidden)          # scoring
    )
    # Gate: elementwise, negligible but counted for completeness.
    gate = candidates * 4
    # Recovery decoder: queries = full grid, keys/values = kept tokens.
    d = dim
    per_layer = (
        4 * d * d                                            # q,k,v,o projections (approx)
        + 2 * layout.video_tokens * kept * d                 # attention
        + 2 * layout.video_tokens * d * int(d * 2)           # ffn
    )
    recovery = model.config.recovery_layers * int(per_layer)
    return {"scorer": int(scorer), "gate": int(gate), "recovery": int(recovery)}


def build_report(args: argparse.Namespace) -> Dict[str, object]:
    layout = RouteALayoutSpec(
        history_tokens=args.history_tokens,
        future_tokens=args.future_tokens,
        traj_tokens=args.traj_tokens,
    )
    context = int(args.context_length)
    full = layout.total_tokens
    kept = int(round(float(args.mean_kept)))
    sparse = kept + layout.traj_tokens
    if sparse >= full:
        raise ValueError("--mean-kept leaves no compression")

    baseline = backbone_macs(args.layers, full, args.dim, args.ffn_dim, context)

    curve = []
    for layer_text in str(args.sweep_bottleneck).split(","):
        if not layer_text.strip():
            continue
        layer = int(layer_text)
        if not 0 <= layer < args.layers:
            raise ValueError(f"bottleneck {layer} outside [0, {args.layers - 1}]")
        front = backbone_macs(layer, full, args.dim, args.ffn_dim, context)
        back = backbone_macs(args.layers - layer, sparse, args.dim, args.ffn_dim, context)
        curve.append(
            {
                "bottleneck_layer": layer,
                "compressed_layer_count": args.layers - layer,
                "backbone_macs": front + back,
                "backbone_saving": 1.0 - (front + back) / baseline,
            }
        )

    config = RouteAConfig(
        token_dim=args.dim,
        bottleneck_layer=args.bottleneck,
        num_blocks=args.layers,
        layout=layout,
        selector_hidden=args.selector_hidden,
        recovery_layers=args.recovery_layers,
        recovery_heads=max(1, args.dim // 128),
        safety_clamp=SafetyClampConfig(
            min_kept_history=8, min_kept_future=32, max_kept_total=max(kept, 64)
        ),
    )
    model = config.build()
    new_macs = module_macs(model, layout)

    scorer_params = count_parameters(model.scorer)
    recovery_params = count_parameters(model.recovery)
    total_new = count_parameters(model)

    selected = next((row for row in curve if row["bottleneck_layer"] == args.bottleneck), None)
    if selected is None:
        front = backbone_macs(args.bottleneck, full, args.dim, args.ffn_dim, context)
        back = backbone_macs(
            args.layers - args.bottleneck, sparse, args.dim, args.ffn_dim, context
        )
        selected = {
            "bottleneck_layer": args.bottleneck,
            "compressed_layer_count": args.layers - args.bottleneck,
            "backbone_macs": front + back,
            "backbone_saving": 1.0 - (front + back) / baseline,
        }

    # Deployable trajectory-only inference skips the recovery decoder entirely
    # (plan section 42).
    trajectory_only_overhead = new_macs["scorer"] + new_macs["gate"]
    with_recovery_overhead = trajectory_only_overhead + new_macs["recovery"]

    return {
        "kind": "ANALYTIC_MAC_MODEL_NOT_A_MEASUREMENT",
        "geometry": {
            "dim": args.dim,
            "ffn_dim": args.ffn_dim,
            "layers": args.layers,
            "context_length": context,
            "full_sequence": full,
            "video_candidates": layout.video_tokens,
            "traj_tokens": layout.traj_tokens,
            "mean_kept_video": kept,
            "sparse_sequence": sparse,
            "sequence_ratio": sparse / full,
        },
        "backbone_macs_baseline": baseline,
        "selected": selected,
        "bottleneck_curve": curve,
        "new_module_parameters": {
            "scorer": scorer_params,
            "gate": count_parameters(model.gate),
            "dense_recovery_decoder": recovery_params,
            "total": total_new,
        },
        "new_module_macs": new_macs,
        "overhead_fraction": {
            "trajectory_only_inference": trajectory_only_overhead / baseline,
            "training_or_video_flow": with_recovery_overhead / baseline,
        },
        "net_saving": {
            "trajectory_only_inference": selected["backbone_saving"]
            - trajectory_only_overhead / baseline,
            "training_or_video_flow": selected["backbone_saving"]
            - with_recovery_overhead / baseline,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

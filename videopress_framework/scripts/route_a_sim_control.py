#!/usr/bin/env python3
"""Untrained-scorer control for the controlled-redundancy simulation.

Why this exists: in ``train_route_a_smoke.py`` the informative subset is injected
as a *magnitude* difference (signal tokens are scaled 3x).  A top-k selection
over any monotone-but-random score is then already biased towards the
high-variance tokens, so an untrained scorer can show a kept-signal overlap far
above the base rate.  Without this control, such an artifact is easily mistaken
for learned selection.

This script measures that null distribution directly, so a trained run's overlap
can be compared against "what a random scorer would have kept at the same K".

Usage::

    python scripts/route_a_sim_control.py --keep-counts 40,100,200,384 --scenes 200
"""
import argparse
import sys, json
sys.path.insert(0, "videopress_framework")
sys.path.insert(0, ".")
import torch
from videopress.retraining import RouteAConfig, RouteALayoutSpec, SafetyClampConfig

torch.manual_seed(20260924)
PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
PARSER.add_argument("--keep-counts", default="40,100,200,384")
PARSER.add_argument("--scenes", type=int, default=200)
PARSER.add_argument("--dim", type=int, default=128)
PARSER.add_argument("--signal-tokens", type=int, default=120)
PARSER.add_argument("--signal-scale", type=float, default=3.0)
PARSER.add_argument("--seed", type=int, default=20260924)
PARSER.add_argument("--output", default=None)
ARGS = PARSER.parse_args()

LAY = RouteALayoutSpec()
SIGNAL = int(ARGS.signal_tokens)
SCALE = float(ARGS.signal_scale)
DIM = int(ARGS.dim)
SEED = int(ARGS.seed)

def sample(batch, gen):
    video = torch.randn(batch, LAY.video_tokens, DIM, generator=gen)
    mask = torch.zeros(batch, LAY.video_tokens, dtype=torch.bool)
    x = video.clone()
    for r in range(batch):
        idx = torch.randperm(LAY.video_tokens, generator=gen)[:SIGNAL]
        mask[r, idx] = True
        x[r, idx] *= SCALE
    traj = torch.zeros(batch, LAY.traj_tokens, DIM)
    return torch.cat([x, traj], 1), mask

results = {}
for k in [int(v) for v in str(ARGS.keep_counts).split(',') if v.strip()]:
    cfg = RouteAConfig(token_dim=DIM, bottleneck_layer=8, num_blocks=12, layout=LAY,
                       selector_hidden=256,
                       safety_clamp=SafetyClampConfig(min_kept_history=8, min_kept_future=32,
                                                       max_kept_total=None))
    m = cfg.build().eval()
    gen = torch.Generator().manual_seed(7)
    overlaps = []
    with torch.no_grad():
        for _ in range(int(ARGS.scenes)):
            x, mask = sample(1, gen)
            logits = m.scorer(x[:, :LAY.video_tokens],
                              action_hidden=x[:, LAY.traj_slice],
                              timestep=torch.tensor([716.0]),
                              positions=None,
                              token_type=m.token_type)
            top = logits[0].topk(k).indices
            overlaps.append(float(mask[0, top].float().mean()))
    t = torch.tensor(overlaps)
    results[k] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False)),
                  "p05": float(t.quantile(0.05)), "p95": float(t.quantile(0.95))}
    print(f"K={k:>3}  untrained overlap mean={t.mean():.4f} sd={t.std(unbiased=False):.4f} "
          f"p05={t.quantile(0.05):.3f} p95={t.quantile(0.95):.3f}   (chance={SIGNAL/LAY.video_tokens:.4f})")
if ARGS.output:
    Path = __import__("pathlib").Path
    target = Path(ARGS.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {target}")

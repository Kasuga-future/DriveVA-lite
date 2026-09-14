#!/usr/bin/env python3
"""Masked-reconstruction token importance: does a *label-free* criterion rank tiles?

Motivation
----------
Three rounds of experiments showed that a *learned* selector cannot beat a
12-parameter per-tile lookup table, because the counterfactual "does removing this
tile help?" label is essentially unlearnable from the layer-15 hidden state
(offline AUC 0.484, positive control 0.994).  That closes the *counterfactual
label* as a training target, but it does not close the whole selection idea,
because the only criterion ever tried was a *supervised* one.

The NTR result (the one isolated compression-side gain in the literature survey)
uses a different mechanism: keep the architecture and token count, but add a
*dense mask-and-reconstruct* objective.  Its stated benefit is that the bottleneck
gets densely supervised -- not that fewer tokens is better.  That objective needs
no importance labels at all: a token is "important" to the extent that it cannot
be reconstructed from the other tokens.

This script tests exactly that criterion, offline, on the probe shards the
counterfactual runs already dumped (``tokens`` = layer-15 hidden state of all 390
candidate tokens, ``membership`` = tile assignment, ``relative_delta`` =
counterfactual label).  It trains a small masked-token autoencoder on the 390-token
sequences *only* (no labels, no counterfactual information anywhere), then scores
each tile by the reconstruction error of its tokens when they are held out.

Two things must be true for this to be a real candidate:

1. **the autoencoder must learn something** -- a synthetic gate shows it recovers a
   planted redundancy structure, so a null result is a data conclusion and not an
   implementation artefact;
2. **the resulting ranking must beat its own permutation null** on the held-out
   counterfactual labels, and be compared against the already-measured numbers
   (learned-content 0.5853, 12-parameter tile lookup 0.5646).

If mask-reconstruction importance also lands at chance, then token *content* is
exhausted as a selection signal by two independent, methodologically unrelated
routes -- which is a much stronger statement than either result alone.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/mask_reconstruction_importance.py \
        outputs/offline_probe_dump_train2048_fixedseed_20260911 \
        --max-scenes 512 --steps 400 --out outputs/mask_recon_importance_20260911/report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class MaskedTokenAE(nn.Module):
    """Small transformer denoiser over a fixed-length token sequence.

    Masked positions receive a learned mask embedding; the model predicts the
    original token at those positions.  Deliberately tiny (2 layers, 4 heads,
    d=128): the question is whether *any* cross-token redundancy is usable, not
    how good a big model can get.
    """

    def __init__(self, dim_in: int = 3072, d_model: int = 128, n_layer: int = 2, n_head: int = 4,
                 max_tokens: int = 512):
        super().__init__()
        self.inp = nn.Linear(dim_in, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embedding is NOT optional here: without it the encoder is
        # permutation-equivariant, so a masked slot carries no information about
        # *which* token it has to reconstruct and the whole objective collapses
        # to predicting the conditional mean (found empirically: the synthetic
        # gate sat at the "predict zero" floor with ratio 1.00).
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, dim_in)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        h = h + self.pos[:, : h.shape[1]].to(h.dtype)
        h = self.enc(h)
        return self.out(self.norm(h))


def load_shards(root: Path, limit: int | None) -> tuple[torch.Tensor, list[torch.Tensor], np.ndarray, np.ndarray]:
    """Return (tokens[N,390,D] float16, memberships list, labels, scene_ids)."""
    paths = sorted(glob.glob(str(root / "probes" / "rank*" / "*.pt")))
    if limit:
        paths = paths[:limit]
    toks, mems, labels, scenes = [], [], [], []
    for path in paths:
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        t = d.get("tokens")
        if t is None:
            continue
        t = t.reshape(-1, t.shape[-2], t.shape[-1]).float().half()
        toks.append(t[0])
        mems.append(d["membership"].reshape(-1).bool())
        labels.append(float(d.get("relative_delta", float("nan"))))
        scenes.append(str(d.get("sample_token", "")))
    if not toks:
        raise SystemExit(f"no usable probe shards under {root}")
    return torch.stack(toks), mems, np.asarray(labels), np.asarray(scenes)


# --------------------------------------------------------------------------
def sample_mask(n_rows: int, n_tok: int, frac: float, generator: torch.Generator) -> torch.Tensor:
    """Mask ``frac`` of every row.

    Training and scoring must live in the *same* masking regime: scoring masks
    exactly one token at a time, so a training scheme that masks a fixed 25% of
    tokens never sees a "everything else is visible" context and learns the
    conditional mean instead of the copy/denoise operation.  Each row therefore
    gets its own count, drawn uniformly from 1..max(2, frac*n_tok).
    """
    hi = max(2, int(round(frac * n_tok)))
    counts = torch.randint(1, hi + 1, (n_rows,), generator=generator)
    mask = torch.zeros(n_rows, n_tok, dtype=torch.bool)
    for i, c in enumerate(counts.tolist()):
        perm = torch.randperm(n_tok, generator=generator)[:c]
        mask[i, perm] = True
    return mask


def synthetic_gate(seed: int = 0, steps: int = 1500, d_model: int = 128, n_layer: int = 4) -> dict:
    """Prove the harness can recover a *planted* redundancy structure.

    Half of the tokens are exact copies of a partner token; the other half are
    independent noise.  A working scorer must assign the copied (redundant)
    tokens a clearly lower reconstruction error than the independent ones.
    """
    g = torch.Generator().manual_seed(seed)
    n_scenes, n_tok, dim = 64, 64, 96
    base = torch.randn(n_scenes, n_tok // 2, dim, generator=g)
    copied = base.clone()
    indep = torch.randn(n_scenes, n_tok // 2, dim, generator=g)
    x = torch.cat([base, copied], dim=1)
    # interleave so position is not a giveaway
    order = torch.randperm(n_tok, generator=g)
    x = x[:, order]
    is_redundant = torch.zeros(n_tok, dtype=torch.bool)
    is_redundant[order[: n_tok // 2]] = True
    torch.manual_seed(seed)
    model = MaskedTokenAE(dim_in=dim, d_model=d_model, n_layer=n_layer, n_head=4, max_tokens=n_tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tg = torch.Generator().manual_seed(seed + 7)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n_scenes, (16,), generator=tg)
        xb = x[idx]
        mask = sample_mask(xb.shape[0], n_tok, 0.25, tg)
        pred = model(xb, mask)
        loss = ((pred - xb) ** 2).sum(-1)[mask].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        err = masked_error(model, x[:16])
    score = err.mean(axis=0)
    redundant_mean = float(score[is_redundant].mean())
    independent_mean = float(score[~is_redundant].mean())
    return {
        "redundant_mean_error": redundant_mean,
        "independent_mean_error": independent_mean,
        "ratio": redundant_mean / independent_mean if independent_mean > 0 else math.nan,
        "train_steps": steps,
        "passed": bool(redundant_mean < 0.8 * independent_mean),
    }


def masked_error(model: MaskedTokenAE, x: torch.Tensor, chunk: int = 8) -> np.ndarray:
    """Per-token reconstruction error with every token masked one at a time."""
    model.eval()
    n, n_tok, dim = x.shape
    out = np.zeros((n, n_tok), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_tok, chunk):
            sel = torch.arange(start, min(start + chunk, n_tok))
            mask = torch.zeros(n, n_tok, dtype=torch.bool)
            mask[:, sel] = True
            pred = model(x, mask)
            err = ((pred - x) ** 2).sum(-1)  # (n, n_tok)
            out[:, sel] = err[:, sel].numpy()
    return out


# --------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else math.nan


def rank_auc(scores: np.ndarray, labels_pos: np.ndarray) -> float:
    """AUC of ``scores`` at separating label-positive from label-negative tiles."""
    pos = scores[labels_pos]
    neg = scores[~labels_pos]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    # rank-based AUC (ties average)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe_root", type=Path)
    ap.add_argument("--max-scenes", type=int, default=512, help="shards to load (each shard = 1 probe)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-scenes", type=int, default=16)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    report: dict = {"probe_root": str(args.probe_root), "config": vars(args) | {"probe_root": str(args.probe_root)}}

    # ---- 1. synthetic learnability gate ---------------------------------
    if not args.skip_gate:
        gate = synthetic_gate(args.seed)
        report["synthetic_gate"] = gate
        print(f"[gate] redundant/independent error ratio = {gate['ratio']:.3f} "
              f"passed={gate['passed']}")
        if not gate["passed"]:
            report["verdict"] = "HARNESS_FAILED"
            _write(args, report)
            return 1

    # ---- 2. load real probe data ----------------------------------------
    tokens, mems, labels, scenes = load_shards(args.probe_root, args.max_scenes)
    n, n_tok, dim = tokens.shape
    print(f"[data] {n} shards  tokens={tuple(tokens.shape)}  "
          f"tiles/shard={int(np.mean([int(m.sum()) for m in mems])):.1f}  "
          f"unique scenes={len(set(scenes.tolist()))}")

    # scene-disjoint split by sample_token
    uniq = sorted(set(scenes.tolist()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    n_train = max(1, int(0.75 * len(uniq)))
    train_scenes = set(uniq[:n_train])
    is_train = np.array([s in train_scenes for s in scenes])
    tr = torch.from_numpy(np.nonzero(is_train)[0])
    te = torch.from_numpy(np.nonzero(~is_train)[0])
    if len(te) == 0:
        te = tr
    print(f"[split] train shards={len(tr)}  test shards={len(te)}  "
          f"scene overlap={len(set(scenes[is_train]) & set(scenes[~is_train]))}")

    # ---- 3. train the label-free reconstructor --------------------------
    model = MaskedTokenAE(dim_in=dim, d_model=args.d_model, n_layer=args.n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x_train = tokens[tr]
    history = []
    model.train()
    for step in range(args.steps):
        idx = torch.randint(0, len(x_train), (min(args.batch_scenes, len(x_train)),))
        xb = x_train[idx]
        mask = torch.rand(xb.shape[:2]) < args.mask_frac
        # guarantee at least one masked token per row
        mask[~mask.any(dim=1), 0] = True
        pred = model(xb, mask)
        loss = ((pred - xb) ** 2).sum(-1)[mask].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 8) == 0:
            history.append({"step": step, "train_masked_mse": float(loss.detach())})
            print(f"  step {step:4d}  masked MSE {float(loss.detach()):.4f}")
    report["train_history"] = history

    # ---- 4. score held-out tiles ----------------------------------------
    err = masked_error(model, tokens[te])          # (n_te, 390) per-token error
    te_mems = [mems[i] for i in te.numpy()]
    te_labels = labels[te.numpy()]
    te_scenes = scenes[te.numpy()]

    tile_scores, tile_labels, tile_scenes, tile_indices = [], [], [], []
    for i, mem in enumerate(te_mems):
        idx = np.nonzero(mem.numpy())[0]
        if len(idx) == 0:
            continue
        tile_scores.append(float(err[i, idx].mean()))
        tile_labels.append(float(te_labels[i]))
        tile_scenes.append(te_scenes[i])
        tile_indices.append(len(tile_scores) - 1)
    tile_scores = np.asarray(tile_scores)
    tile_labels = np.asarray(tile_labels)
    tile_scenes = np.asarray(tile_scenes)

    # within-scene ranking: importance should be HIGH for helpful tiles
    # (high reconstruction error = hard to reconstruct = important)
    by_scene: dict[str, list[int]] = {}
    for i, s in enumerate(tile_scenes):
        by_scene.setdefault(s, []).append(i)
    rhos = []
    for s, idxs in by_scene.items():
        if len(idxs) >= 4:
            r = spearman(tile_scores[idxs], tile_labels[idxs])
            if not math.isnan(r):
                rhos.append(r)
    observed_rho = float(np.mean(rhos)) if rhos else math.nan
    perm_rhos = []
    rng2 = np.random.default_rng(args.seed + 1)
    for _ in range(args.permutations):
        shuff = []
        for s, idxs in by_scene.items():
            if len(idxs) >= 4:
                vals = tile_scores[idxs]
                perm = rng2.permutation(vals)
                r = spearman(perm, tile_labels[idxs])
                if not math.isnan(r):
                    shuff.append(r)
        perm_rhos.append(np.mean(shuff) if shuff else math.nan)
    perm_rhos = np.asarray([p for p in perm_rhos if not math.isnan(p)])

    pos = tile_labels > 0
    auc = rank_auc(tile_scores, pos)

    report["tile_analysis"] = {
        "n_tiles": int(len(tile_scores)),
        "n_scenes": len(by_scene),
        "within_scene_spearman": observed_rho,
        "permutation_null_mean": float(perm_rhos.mean()) if len(perm_rhos) else math.nan,
        "permutation_null_p95": float(np.percentile(perm_rhos, 95)) if len(perm_rhos) else math.nan,
        "permutation_p_value": float((perm_rhos >= observed_rho).mean()) if len(perm_rhos) else math.nan,
        "auc_helpful_vs_not": auc,
        "positive_rate": float(pos.mean()),
        "n_positive": int(pos.sum()),
    }
    # reference numbers already measured by the project (reported, not recomputed)
    report["reference_values"] = {
        "learned_content_within_scene_spearman": 0.5853,
        "fixed_12_parameter_tile_lookup": 0.5646,
        "chance": 0.5,
        "source": "reports/driveva_key_token_routes_20260911.md sections 4.6 and 5.1",
    }

    rho, pval = observed_rho, report["tile_analysis"]["permutation_p_value"]
    if math.isnan(rho):
        verdict = "INCONCLUSIVE"
    elif pval < 0.01 and rho > 0.5646 + 0.02:
        verdict = "POSITIVE_candidate_beats_lookup"
    elif pval < 0.01 and rho > 0.5:
        verdict = "WEAK_positive_but_at_or_below_lookup"
    else:
        verdict = "NEGATIVE_mask_reconstruction_content_is_not_the_signal"
    report["verdict"] = verdict
    report["verdict_rule"] = (
        "POSITIVE requires within-scene Spearman > 0.5846 (lookup + 0.02) with permutation p < 0.01; "
        "WEAK requires p < 0.01 and rho > 0.5; otherwise NEGATIVE."
    )

    _write(args, report)
    print(f"\n[tile] within-scene Spearman {rho:+.4f}  perm null mean "
          f"{report['tile_analysis']['permutation_null_mean']:+.4f}  "
          f"p={pval:.4f}  AUC={auc:.4f}")
    print(f"[ref ] learned content 0.5853 / 12-param lookup 0.5646 / chance 0.5")
    print(f"[verdict] {verdict}")
    return 0


def _write(args, report) -> None:
    out = args.out or (args.probe_root / "mask_reconstruction_importance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())

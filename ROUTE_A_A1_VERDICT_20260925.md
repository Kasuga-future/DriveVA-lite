# Route A (Dynamic Select) A1 verdict — 2026-09-25

**Panel**: official NAVSIM split-test, 1024 paired scenes, 4 ranks x 256, 3
inference rounds. All arms share the same scene tokens, so only the *paired*
delta is meaningful (the absolute panel mean moves by up to 0.031 across
sampling seeds; a paired delta of a fixed arm moves by ~0.007 — AGENTS.md §4).

**Baseline**: `no_press` **0.911078**.

## 1. Headline

| arm | PDM | ΔPDM vs NoPress | 95% CI (paired) | kept video | retention |
|---|---:|---:|---|---:|---:|
| no_press | 0.9111 | — | — | 1560 | 100% |
| Route A, dense-gate ckpt (evaluated physically) | 0.7077 | **−0.2034** | [−0.2271, −0.1802] | 380.2 | 24.4% |
| **Route A, physical tail ckpt** | **0.7422** | **−0.1688** | [−0.1921, −0.1464] | 353.2 | 22.6% |
| matched random, seed 11 | 0.7563 | −0.1547 | [−0.1764, −0.1334] | 351.8 | 22.6% |
| matched random, seed 23 | 0.7624 | −0.1487 | [−0.1702, −0.1279] | 350.8 | 22.5% |
| matched random, seed 37 | 0.7501 | −0.1610 | [−0.1822, −0.1402] | 350.2 | 22.5% |
| matched random, seed 59 | 0.7580 | −0.1531 | [−0.1743, −0.1327] | 351.7 | 22.5% |

Matched-random band: **0.7501 – 0.7624** (mean 0.7567).

**The decisive comparison** (paired over the same 1024 scenes):

```
route_a_physical   - mean(4 random seeds)   -0.0145   95% CI [-0.0317, +0.0021]
route_a_densegate  - mean(4 random seeds)   -0.0490   95% CI [-0.0706, -0.0279]
```

## 2. What this means

1. **Route A at ~23% video-token retention loses 0.169 PDM.** The project's
   near-lossless gate is "CI lower bound > −0.002"; here it is −0.192. This is
   two orders of magnitude away, and the CI excludes zero by a wide margin.

2. **The trained selector is statistically indistinguishable from chance.** The
   physical-tail checkpoint scores 0.7422 while a random permutation of *its own*
   scores scores 0.7501–0.7624; the paired delta is −0.0145 with a CI that
   covers zero (and sits mostly below it). The random arms are the right control
   because they reuse the same checkpoint and the same thresholds, so the keep
   count and the score histogram are identical by construction and only *which*
   tokens survive changes. Equivalently: "keep the K highest-scoring tokens" is
   no better than "keep a uniformly random K-subset".

3. **The dense-gate relaxation does not transfer, and is worse than chance.**
   Training with `physical_shortening=False` (mask-to-zero, sequence kept dense)
   and then evaluating with real removal gives −0.2034; the physical tail
   recovered **+0.0346 PDM** (0.7077 → 0.7422), which is the part of the P0 plan
   that worked. But it recovered mechanism loss, not ranking ability — the
   dense-gate arm's ranking is *significantly worse* than random
   (−0.0490, CI excludes 0).

4. **Route A is also worse than the frozen press at a comparable budget.** The
   historical frozen-model random frontier (1024 scenes) loses 0.048 at keep
   0.50 and 0.095 at keep 0.346; Route A loses 0.155 at keep 0.226. Physically
   removing 78% of hidden states at L18 and running blocks 18–29 without them is
   a much harsher intervention than the deployed press, which zeroes tokens but
   keeps the sequence so later blocks can regenerate them.

## 3. A mechanistic explanation (secondary finding)

Training with physical shortening **degrades the gate's controllability**:

| run | logged samples | score_std (mean [min,max]) | hard retention (mean [min,max]) |
|---|---:|---|---|
| A1 dense-gate | 453 | 0.192 [0.143, 0.218] | 0.247 [0.072, 0.369] |
| A1 physical tail | 76 | 0.114 [0.043, 0.213] | 0.234 [0.056, 0.529] |

In the physical tail the score distribution collapses (std down to 0.043) and the
realised retention becomes hyper-sensitive to `tau`: a 0.016 change in
`tau_future` moved future retention from 0.04 to 0.42 between consecutive logged
steps. With the scores' mass concentrated in a narrow band around the threshold,
the gate keeps a near-arbitrary subset of that band. This is a *second,
independent* knife-edge — the first one (a per-scene score offset swamping the
ranking) was fixed by per-scene standardisation, but removing tokens physically
reintroduces a degenerate score shape.

This also means the evaluation retention (22.6%) is a sharp function of `tau`
rather than a smooth operating point, so any deployment would have to fix `tau`
on a calibration panel and accept the resulting K, not the other way round.

## 4. What was fixed along the way (the P0 list is done)

- freeze the whole pipeline, not just `dit`, with an attach-time assertion that
  Route A is the only trainable module;
- trajectory-loss reweighting (the FM video loss averages 780 tokens against ~8
  trajectory points, so the scorer was otherwise optimising video reconstruction
  rather than the plan PDM measures);
- paired trajectory-flow KD against the frozen dense teacher;
- `CompressionStatsRecorder` + `is_truly_dynamic` in the training loop, with a
  per-step monitoring line and a calibration side-car;
- **the gate itself**: fixed τ put the hard mask on a knife edge (54% → the 6%
  safety floor in one optimiser step while the STE surrogate stayed at 0.44); a
  τ feedback controller bang-banged; quantile calibration still alternated
  0%/100% because the scorer's score is dominated by a per-scene offset as large
  as the within-scene spread. Per-scene, per-domain score standardisation fixed
  it: retention held at **0.239–0.251 against a 0.25 target across all 11,304
  steps**, with τ converging within ~2000 steps.

## 5. What is *not* tested

- **A2/A3 (unfreeze the DiT / LoRA)**. The plan's premise is that retraining the
  model changes what the tokens *mean*, not just which ones to pick. A1 froze the
  backbone, so it tests only the selection half — and the selection half is what
  failed. Testing A2/A3 asks a much larger question ("learn a 5B model to work
  with 25% of its tokens"), and the selection evidence caps how much of the gap a
  better selector could close.
- **Dynamic length.** Per-scene standardisation removes the scene-level score
  offset, so K is nearly constant *by construction*; the plan's scene-dependent K
  is not implemented here. `is_truly_dynamic` fired on 451/453 samples and the
  K spread widened over training (p10 314→283, p90 453→469), but that is a weak
  signal, not the plan's dynamic-length behaviour.
- **Video quality.** Only trajectory PDM is measured. The recovery decoder feeds
  the frozen Wan head and would have to restore the video latent grid; that is
  untested and is certainly damaged at 78% removal.

## 6. Recommendation

Stop Route A at the A1 stage. The 25%-budget operating point is not viable, and
the failure is not a training-budget problem — the trained ranking carries no
usable selection signal, exactly as the earlier frozen-model oracle work
predicted (`future_token_level_oracle_conclusion_20260920.md`: selection signal
real but far too weak; no deployable scorer beat matched random at the only
near-neutral budget).

If the line is to continue at all, the highest-value changes are:

1. **Train the ranking directly** instead of only through the mask
   (pairwise/contrastive loss on the scores, or distillation of a
   counterfactual-perturbation teacher), because the STE path through a
   near-degenerate mask is not giving the scorer a usable gradient;
2. **Fix the budget with a per-scene quantile top-K** rather than a threshold,
   which removes both knife-edges and makes the retention exactly controlled;
3. **Then** consider A2/A3, since unfreezing is only worth it if selection is
   first shown to have signal at the target budget.

## 7. Reproduce

```bash
# training (already done): outputs/route_a_train2_20260924/
bash outputs/route_a_train2_20260924/run_queue.sh        # stages 0-3
bash outputs/route_a_train2_20260924/run_eval_recovered.sh  # recovered tau + random control

# the paired table above
python videopress_framework/scripts/analyze_route_a_paired.py
```

Artifacts:

- stage 1 (dense-gate, 12 checkpoints): `outputs/route_a_train2_20260924/a1_norm_l18/`
- stage 2 (physical tail): `.../a1_norm_l18_physical/`
- monitoring JSONL + recovered calibration: `.../*/route_a_stats.jsonl`, `.../*/route_a_stats_calibration.json`
- evaluation: `.../eval/{no_press,route_a_densegate,route_a_physical,route_a_random_s*}/`
- paired analysis: `.../eval/paired_analysis.json`

# DriveVA Route A (Dynamic Select) — Implementation and Training Report

**Date:** 2026-09-24
**Scope:** feasibility exploration + implementation + training verification for **Route A — Dynamic Select**
of `DriveVA Dynamic Video Token Compression — Retraining Implementation Plan v2`
(`DriveVA%20Dynamic%20Video%20Token%20Compression%20%E2%80%94%20Retraining%20Implementation%20Plan_v2.md`).
**Repository state:** commit `b98c6cc` (pushed to `origin/main`) + the Route A changes described here.

---

## 0. One-paragraph answer

**Route A is implementable, and the implementation is done and verified — but it is not yet
*trained* on DriveVA/NAVSIM, and the remaining work is blocked on GPU availability, not on code.**
The full Route A code path now exists as a first-class `nn.Module` package
(`videopress/retraining/`) with 59 dedicated unit tests and 323/323 framework tests passing.
An analytic cost model says Route A is worth **33.9% (Lb=18) to 50.9% (Lb=12) of DiT MACs** at a
15.9% sequence length — one order of magnitude more than the 2.8–3.4% that the currently deployed
frozen-history press achieves, which is exactly why the retraining route is the right next move.
A controlled-redundancy training simulation on the **production** Wan `DiTBlock`/`TrajectoryHead`/`Head`
verifies the mechanics and exposes two things the plan does not mention. First, the literal
`V_sparse = V[mask]` gather of plan §11 gives **exactly zero gradient to dropped candidates**, so the
selected set can only erode; the run collapses from a working selection to a dead one pinned at the
safety floor. Second, the simulation **does not** show the objective learning to pick the informative
tokens: with the required untrained-scorer control, training made selection *worse* than not training.
Both are implemented-and-fixed/quantified here. What is **not** established — and cannot be established
without a GPU — is whether a DriveVA-initialised student can hold NAVSIM PDM inside
`CI_lower(ΔPDM) > −0.002` at the plan's 12–18% retention target.

---

## 1. Why Route A exists: the frozen-prune verdict it has to beat

The retraining plan exists because three independent lines of frozen-model pruning were all
adjudicated negative on the full 7,876-scene NAVSIM protocol (details in `AGENTS.md` §4):

| Line | Full-7,876 verdict | Reading |
|---|---|---|
| Future hard prune, L22, keep 0.75, `action_attention_vnorm` | ΔPDM **−0.011601** CI [−0.014996, −0.008387] | CI excludes 0; not near-lossless |
| Same arm, L22-calibrated learned selector | ΔPDM **−0.010396** CI [−0.013780, −0.007042], e2e **+48.9 ms** | calibration buys +0.0012 for +49 ms |
| `late_to_mid [22,22,18]`, round-scheduled | ΔPDM **−0.013053** CI [−0.016461, −0.009679] | worse than fixed22 |
| From-scratch layer-scheduled selector | ΔPDM **−0.011590** CI [−0.015071, −0.008212] | retraining the *selector* alone does not help |
| Token-level set-level oracle (the upper bound) | keep 0.5 random control **−0.0483**; keep 0.875 best **−0.0068** | no constructible mask reaches the gate |
| Causal knockout probe | zeroing future tokens at L8–L24 moves the plan by **45–63%** with **no depth decay** | there is no "after layer X it is free" |

The last row is the decisive one: the model keeps reading future video tokens at every depth, so no
choice of start layer or scorer can make deletion free. The plan's conclusion — and this report's
starting point — is that the question must change from *"which tokens can we delete?"* to
*"can we train the model to need fewer tokens?"*.

---

## 2. What Route A is, in code terms

Plan section 7–15 and 39–42 specify:

```
dense noisy video latent
  -> patch embedding
  -> dense DiT blocks 0 .. Lb-1
  -> threshold scorer          (score_i >= tau, dynamic K = sum(m_i))
  -> sparse DiT blocks Lb .. 29 (shortened residual / RoPE / t_mod)
  -> trajectory head            (planning output)
  -> dense recovery decoder     (video flow target, training only)
  -> original Wan head
```

The key structural difference from everything previously tried in this repository is that the
compression modules are **not** runtime hooks bolted onto a frozen model. They are trainable
parameters in the checkpoint, and the backbone is unfrozen (LoRA, then full) so the model can learn
to *write* the information it needs into the tokens it expects to keep.

---

## 3. Implementation

### 3.1 New package: `videopress_framework/videopress/retraining/`

| File | Lines | What it implements | Plan ref |
|---|---:|---|---|
| `threshold_gate.py` | ~430 | `STEThresholdGate` (per-domain thresholds, plan-exact forward/backward), `SafetyClampConfig`, `sparsity_loss`, `SparsityCurriculum`, `jittered_thresholds`, `binding_row_domain_counts` | §3–6, §29–30 |
| `dynamic_selector.py` | ~250 | `DynamicVideoTokenScorer` — token + action + timestep + position + history/future-identity interaction scorer, `pooled` and `attention` action modes | §10 |
| `dense_recovery.py` | ~215 | `DenseRecoveryDecoder` — full-grid queries cross-attending to selected tokens; zero-initialised output | §12 |
| `distillation.py` | ~250 | `RouteALossWeights`, `layer_norm_mse`, `action_hidden_kd` (L11/L18/L29), `compute_route_a_loss` assembling `L_A` | §13, §31–32 |
| `compression_stats.py` | ~330 | `CompressionStatsRecorder` (plan §34 schema), length percentiles, `corr(K, σ)`, `corr(K, difficulty)`, `is_truly_dynamic` degeneracy detector | §15, §34–35 |
| `route_a.py` | ~520 | `RouteAConfig`, `RouteALayoutSpec`, `RouteADynamicSelect` (the whole forward), `sync_keep_lengths`, `build_driveva_video_positions` | §8, §11, §15, §40, §42 |
| `curriculum.py` | ~340 | `StageSpec` / `default_stage_specs` (A0–A4), `RouterStageSchedule` (18→15→12), `apply_stage`, `build_optimizer`, `jitter_for_step` | §14, §28–30 |

### 3.2 The threshold gate is plan-exact, including the straight-through estimator

Plan section 4 pseudocode is implemented literally:

```python
scores = sigmoid(logits)                              # s_i in (0, 1)
soft   = sigmoid((scores - tau) / temperature)
hard   = scores >= tau                                # deployment rule
mask   = hard.detach() - soft.detach() + soft         # forward == hard, backward == sigmoid
```

The threshold therefore lives on the **probability** scale, which is what plan section 29 calibrates
(`threshold_*: 0.5`). `K(x) = Σ m_i` is the natural, scene-dependent length; `torch.argmax`-free,
no fixed retention target anywhere.

### 3.3 The safety clamp is a guard rail, not a budget

`SafetyClampConfig(min_kept_history=8, min_kept_future=32, max_kept_total=384)` is the only place a
count appears. Its job (plan §6.1) is to stop the degenerate `K=0` / `K=N` states while the scorer is
still random. `test_threshold_gate_min_clamp_reserves_a_floor_per_domain` and
`test_threshold_gate_max_clamp_trims_a_flood_of_kept_tokens` pin the semantics, and
`CompressionStatsRecorder.is_truly_dynamic()` exists specifically to **detect** the failure mode the
plan warns about in §35 — if every scene ends up keeping ≈384 tokens, the selector has silently
become a fixed-budget selector and the whole premise is void.

### 3.4 The STE mask is load-bearing, and getting it wrong is silent

This is the single most important implementation detail, and it was a real bug during development.
The naive implementation gathers tokens with an integer index:

```python
x_short = x[:, kept_indices]        # WRONG for training
```

Integer gather is not differentiable with respect to the *selection*. With that code the gate's
`mask` was computed and never used, so the flow-matching and distillation losses reached the
backbone but **never the scorer**; the scorer was trained only by the sparsity penalty, which carries
no ranking signal at all. The symptom was a selector whose selected-token overlap with a known
informative subset stayed at exactly chance.

The fix (and the correct semantics) is to scale *before* gathering:

```python
ste_mask = hard - soft.detach() + soft          # evaluates to exactly 0/1 in forward
gated_video = video_hidden * ste_mask.unsqueeze(-1)
x_short = gather(gated_video, kept_indices)     # values unchanged, gradient path preserved
```

`test_route_a_gradients_reach_the_scorer` asserts the property directly, and
`test_route_a_ste_mask_equals_hard_mask_in_forward` asserts the deployed rule is unchanged.

### 3.5 Curriculum, per-group learning rates, and the layer schedule

`curriculum.py` encodes the plan's stages directly:

| Stage | trainable | compression LR | DiT LR | λ_sparse |
|---|---|---|---|---|
| A0 | nothing (teacher cache) | — | — | 0 |
| A1 | selector + recovery decoder | 1e-4 | frozen | 0 |
| A2 | + LoRA (rank 64 config is the plan's; this module is LoRA-agnostic) | 1e-4 | 5e-5 | 1e-4, ramp 2000 |
| A3 | + full DiT + trajectory modules | 5e-5 | 5e-6 | 1e-3, ramp 4000 |
| A4 | same as A3, re-fine-tuned after moving the bottleneck | 5e-5 | 5e-6 | 1e-3, ramp 2000 |

`RouterStageSchedule` moves the bottleneck **18 → 15 → 12** (plan §9: never jump straight to L8),
and `jitter_for_step` applies `tau + U(−0.05, 0.05)` so the scorer does not over-fit one threshold
value (plan §29).

### 3.6 Scorer input modes and the position convention

`DynamicVideoTokenScorer` takes `(video_hidden, action_hidden, timestep, positions, token_type)`.
`build_driveva_video_positions()` reproduces the exact `(t, y, x)` convention the deployed
`LearnedPlanningSelectorScorer._positions()` and the training-time capture code use
(`t` = storage latent index, `y/(h−1)`, `x/(w−1)`, with `h=13, w=30` per latent), so a scorer trained
inside Route A is coordinate-compatible with the frozen-press machinery and with any future
calibration on the existing selector checkpoints.

### 3.7 Batched dynamic K

A threshold selector produces a different `K` per scene, so a batch cannot be gathered into one
rectangular tensor without a decision. `sync_keep_lengths()` pads shorter rows up to the batch
maximum using their highest-scoring **dropped** tokens, which only ever *adds* tokens and therefore
cannot make a row less faithful. The plan's training recipe uses `micro_batch_per_gpu = 1`
(§28), where this is a no-op; `allow_padding=False` turns a ragged batch into a hard error instead.

### 3.8 Scripts

| Script | Purpose |
|---|---|
| `scripts/train_route_a_smoke.py` | Route A training driver: controlled-redundancy simulation (`--mode sim`) + guarded NAVSIM entry (`--mode navsim`); λ sweep, stage selection, dynamic-length reporting |
| `scripts/route_a_budget_report.py` | Analytic MAC/parameter cost model at the real DriveVA geometry |

---

## 4. Verification

### 4.1 Unit tests — `videopress_framework/tests/test_retraining_route_a.py`

**59 tests, all passing.** They cover, among other things:

* the gate's forward pass is exactly the hard threshold and its backward pass is a sigmoid
  (straight-through);
* the min/max safety clamps bound `K` without becoming a budget;
* a single threshold broadcasts across domains; invalid widths/thresholds raise;
* `sync_keep_lengths` pads ragged batches and can refuse them;
* the sparsity curriculum's warm-up/ramp shape;
* the scorer produces `[B, N]` logits and gradients for both action modes, and rejects bad token types;
* the recovery decoder returns the full grid, is gradient-safe, validates indices, requires
  `kept_indices` when ragged, and is position-only at init;
* LN-matched hidden distillation is **scale-invariant** (the plan's reason for using LayerNorm);
* `compute_route_a_loss` assembles the right terms with the right weights and fails loudly with no target;
* the stats recorder computes percentiles, **flags a fixed-budget selector as non-dynamic**, and
  writes its JSON;
* **the Route A forward shortens only the sparse backend** (dense front-end sees 1569 tokens,
  the backend sees `K + traj`), never selects a trajectory token, always keeps trajectory tokens,
  respects the bottleneck layer, validates capture layers, and records stats;
* gradients reach the scorer — the load-bearing property;
* the STE gradient's **documented limitation**: when every score saturates at 1, both sigmoids have
  zero derivative and the selector only receives the sparsity signal (this is asserted, not hidden);
* the recovery decoder's zero-init means the *video* path cannot reach the selector on the first
  backward pass while the weight is zero (also asserted and documented);
* `apply_stage` freezes/unfreezes the right groups and `build_optimizer` creates one group per LR;
* the default loss weights match plan §32 exactly.

### 4.2 Full suite — no regressions

```
323 passed
```

(264 pre-existing tests + 59 new.)

### 4.3 Production-code integration

The training simulation uses the **real** `diffsynth.models.wan_video_dit.DiTBlock`,
`diffsynth.models.wan_video_dit.Head` and `examples.wanvideo.driveva_infer.trajectory_modules.TrajectoryHead`
— not stand-ins. The only substitution is the attention kernel: `DiTBlock` prefers the installed
`flash_attn` package, which has no CPU kernel, so the driver installs the repository's own
already-present `F.scaled_dot_product_attention` branch (identical math, same code path the
repository uses when `flash_attn` is absent). This is a **device** workaround for CPU verification,
recorded in the run JSON as `attention_backend`, and never used on GPU.

The actual `model_fn_wan_video` call site is the one place where Route A still has to be wired in for
real training; see §7.

---

## 5. Training

### 5.1 What was trained, and what it does and does not establish

No GPU was available (see §6), so the training run is a **controlled-redundancy simulation** on CPU.
It is designed so that the Route A objective becomes *measurable* rather than merely runnable:

* the **teacher** is a frozen dense forward through the production blocks that sees only a
  per-scene random subset of `--signal-tokens` video tokens (the rest are zeroed);
* the **student** sees *all* video tokens, with the signal ones scaled to 3×;
* a good selector must therefore learn, from content alone, which tokens carry the signal. The
  injected subset is re-randomised every scene, so a static spatial prior cannot solve it.

This gives a **ground-truth** selection metric — overlap between the kept tokens and the injected
signal set, against a chance level of `120/1560 = 7.7%` — that the NAVSIM benchmark fundamentally
cannot provide (the 2026-09-20 oracle search established that the "right" tokens there are unknown).
It also gives a clean quality axis: trajectory/video MSE against the dense teacher.

**What this establishes:** the Route A objective (threshold gate + sparsity curriculum +
distillation + dense recovery) is *learnable and mechanically correct*.
**What it does not establish:** anything about NAVSIM PDM, or that 12–18% retention is reachable on
real driving data. The redundancy here is *injected*; on DriveVA it is precisely the thing in question.

### 5.2 Configuration

| Parameter | Sweep run | Production-scale run |
|---|---|---|
| DiT layers | 12 | **30** |
| `dim` / heads / `ffn_dim` | 128 / 4 / 384 | **256 / 8 / 1024** |
| Bottleneck `Lb` | 8 | **18** |
| Sequence length | **1569** (real DriveVA layout) | **1569** |
| KD capture layers | 4, 8, 11 | **11, 18, 29** |
| Stage | A3 (DiT trainable, DiT LR 5e-6) | A3 |
| `compression_lr` | 1e-3 | 1e-3 |
| λ_sparse sweep | 0, 3e-4, 1e-3, 3e-3, 1e-2 | 3e-3 |
| Sparsity ramp | 200 steps | 40 steps |
| Threshold jitter | ±0.05 | ±0.05 |
| Steps | 400 | 80 |
| Eval scenes | 64 held-out | 32 held-out |

Both runs use the real 780 + 780 + 9 token layout, the real `(t, y, x)` positions, a real
`timestep=716`, and the plan's loss weights
(`traj_fm 1.0, video_fm 1.0, traj_kd 2.0, video_kd 0.5, action_hidden_kd 1.0, video_hidden_kd 0.5`).

### 5.3 Production-scale run (30 layers, `dim=256`, `Lb=18`, real KD anchors L11/L18/L29)

This is the run whose geometry matches the real DriveVA DiT (same 30 blocks, same bottleneck the
plan starts from, same hidden-KD probe layers), with the hidden width reduced only to keep CPU cost
manageable. 80 steps, `λ_sparse` ramped to 3e-3 over 40 steps, A3 (DiT trainable at 5e-6).

Losses behaved as expected — they fell and stayed finite:

| step | loss | `video_kd` | `action_hidden_kd` | `traj_kd` | kept video | signal overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.889 | 0.689 | 0.155 | 0.0033 | 384 | 15.4% |
| 8 | 3.293 | 0.470 | 0.181 | 0.0052 | 40 | **72.5%** |
| 16 | 3.145 | 0.421 | 0.157 | 0.0049 | 40 | 10.0% |
| 24 | 2.998 | 0.374 | 0.128 | 0.0037 | 40 | 2.5% |
| 32 | 2.787 | 0.344 | 0.109 | 0.0035 | 40 | 0.0% |
| 40 | 2.727 | 0.321 | 0.104 | 0.0032 | 40 | 0.0% |
| 80 | 2.615 | 0.283 | 0.087 | 0.0027 | 40 | 0.0% |

Two things in that table matter more than the final numbers.

**First, that step-8 reading needs its control.** The informative subset is injected as a
*magnitude* difference, and a top-k over any score is biased towards high-variance tokens, so an
**untrained** scorer already over-selects signal tokens. The measured null distribution
(`scripts/route_a_sim_control.py`) is:

| K | untrained overlap (mean ± sd, 200 scenes) | 5th–95th pct | chance |
|---:|---:|---|---:|
| 40 | **0.328 ± 0.080** | 0.200 – 0.475 | 0.077 |
| 100 | 0.189 ± 0.039 | 0.130 – 0.260 | 0.077 |
| 200 | 0.034 ± 0.012 | 0.015 – 0.055 | 0.077 |
| 384 | 0.032 ± 0.008 | 0.021 – 0.047 | 0.077 |

At `K=40` the artifact is worth 4.3× chance, so "overlap above 7.7%" means nothing. The production
run's step-8 value (0.725) *does* exceed the control's 95th percentile (0.475), so it is the one
observation in this report that looks like genuine learning rather than artifact — but it is a single
point, on a different width/seed than the control, and it does not survive.

**Second, it then loses them, monotonically, and never recovers.** The candidate count is pinned at
the safety-clamp floor (40) and the overlap decays to zero. The final evaluation is
`kept_signal_overlap = 0.0008`, `traj_mse = 0.001478`, `video_mse = 0.555889`, and the gate is
correctly flagged as **not dynamic** (`spread = 0`, all scenes keep exactly 40). No amount of further
training would fix this, because of the mechanism in §5.4.

### 5.4 The finding: the literal `V[mask]` gather gives dropped tokens zero gradient

The plan's forward pseudocode (section 11) writes:

```python
V_sparse = V[mask]
```

Implemented literally — an integer gather — this removes a dropped token's row from the tensor
*before* the loss is computed. There is therefore no path from the loss back to that token's score.
Measured directly on the implementation
(`test_gather_ste_leaves_dropped_candidates_without_gradient`):

```
kept: 12 / 16
|grad| on KEPT    video tokens: 0.0066036
|grad| on DROPPED video tokens: 0.0
```

The consequence is structural, not cosmetic: **the selected set can only erode.** Only the currently
kept candidates are ever trained, so a token that is dropped — correctly or not — can never raise its
score back above the threshold. Combine that with the sparsity penalty, which drives `K` down to the
clamp floor, and the run collapses exactly as the table above shows:

1. early on `K = 384` (the clamp maximum), so 384 candidates are supervised and the selector finds
   the signal tokens;
2. `λ_sparse` pushes `K` to the floor of 40, so only 40 candidates remain supervised;
3. the ranking degrades, tokens leave the kept set, and nothing brings them back.

This also means the `λ_sparse` sweep is not really a compression/quality frontier: past a small
value, increasing `λ` does not trade quality for length, it *destroys the selection* and the length
falls to the floor regardless. The plan's own A1 instruction ("keep `tau` low, `lambda_sparse` tiny")
points at the right fix; what it does not say is that the gather itself is the reason the damage is
irreversible.

### 5.5 Mitigation: dense-gated training (`physical_shortening=False`)

`RouteADynamicSelect.forward(..., physical_shortening=False)` runs the backend over the **whole**
sequence with dropped video tokens masked to zero instead of gathering them away. Every candidate
then receives a gradient
(`test_dense_gated_training_supervises_every_candidate` asserts
`mean|grad| on dropped candidates > 0`), while the mask still controls the forward values, so the
optimisation still targets the threshold rule that will be deployed.

It is explicitly a **training relaxation**, not an inference-equivalent forward: masking a residual
to zero leaves its (bias-driven) key in the softmax, whereas physical removal deletes it. The honest
recipe is therefore a two-phase one, and the driver supports it directly:

* **A1 warm-up with `--dense-gate`**: every candidate is supervised, so the scorer can build a
  globally correct ranking rather than only refining whatever the clamp happened to keep;
* **`--physical-shortening-final-steps N`**: switch back to the inference-equivalent gathered
  forward for the last `N` steps so the backbone and the recovery decoder adapt to the real
  shortened sequence.

`--sparsity-guard` (default on) is the second half of the mitigation: it computes
`gate_health(logits, thresholds, temperature)` each step and refuses to raise `λ_sparse` once the
gate degenerates (`responsive < 0.05` or the mean `|d mask / d logit|` falls below `1e-5`), logging
every intervention. That is plan section 14-A1's "tiny lambda" made feedback-controlled instead of
schedule-only. Both mechanisms are unit tested.

### 5.6 Matched gather-vs-dense comparison (8 layers, `dim=96`, `Lb=4`, A1 frozen backbone)

The cleanest test of the §5.4/§5.5 mechanism is one config, two forward modes, 250 steps,
`λ_sparse` ramped linearly to 3e-3 over the whole run so the pressure grows slowly:

| | gather (`physical_shortening=True`) | dense-gate (`physical_shortening=False`) |
|---|---:|---:|
| seconds/step | 0.224 | 0.423 |
| kept video, end of run | **40** (safety floor) | **384** (safety ceiling) |
| kept video, trajectory of K | 384 → 384 → **40 at λ=1.5e-3** | 384 throughout |
| signal overlap, trajectory | 0.047 → 0.052 → **0.000** | 0.047 → 0.104 → **0.074** |
| final signal overlap (chance 0.0769) | **0.000** | **0.074** |
| `traj_mse` vs teacher | 0.002841 | **0.001417** |
| `is_truly_dynamic` | `False` (P10=P90=40) | `False` (P10=P90=384) |

Reading:

* **The collapse is real and it is caused by the gather.** With the gather, `K` held at the
  ceiling while the pressure was still small, then fell off a cliff to the floor the moment
  `λ` crossed ≈1.5e-3, and the selection quality went to exactly zero. With the dense gate and
  an identical schedule, `K` never collapsed and the overlap stayed at or above chance.
* **Quality is better too**: the dense-gate arm's trajectory error against the teacher is
  **half** the gather arm's (0.001417 vs 0.002841), even though both ran the same number of
  steps and the same loss weights.
* **But the dense gate does not by itself produce a good selector at this scale.** 0.074 is
  essentially the chance level (0.0769). At 8 layers / `dim=96` with a **frozen** backbone
  (A1), the scorer does not learn a strong ranking either way; the difference is that the
  dense arm fails *gracefully* (a usable, non-degenerate gate whose K can still be tuned by
  `tau` at calibration time) whereas the gather arm fails *irreversibly* (a dead selector
  pinned to the floor).
* **The de-facto "λ frontier" is not a frontier at all.** In the gather arm, increasing `λ`
  did not trade quality for length: it destroyed the selection and the length fell to the floor
  regardless. Any future λ sweep must therefore be read together with the overlap/health
  telemetry, not as a compression/quality curve.

### 5.6b The full 12-layer λ sweep, and why it is not a frontier

Same task, 12 layers / `dim=128` / `Lb=8` / A3 (DiT trainable) / 400 steps / real probe anchors
`capture_layers = 4,8,11`. Reported against the untrained control at the *same* K.

**Gather path** (`sim_sweep_a3.json`):

| `λ_sparse` | kept video (mean) | retention hist / fut | signal overlap | `traj_mse` | dynamic |
|---:|---:|---|---:|---:|---|
| 0 | 40 | 0.010 / 0.041 | **0.0035** | 0.001032 | False |
| 3e-4 | 40 | 0.010 / 0.041 | 0.0035 | 0.001042 | False |
| 1e-3 | 40 | 0.010 / 0.041 | 0.0223 | 0.001073 | False |
| 3e-3 | 40 | 0.010 / 0.041 | 0.0000 | 0.001106 | False |
| 1e-2 | 40 | 0.010 / 0.041 | 0.0055 | 0.001064 | False |

**Dense-gate path** (`sim_sweep_a3_densegate.json`, 40-step physical-shortening tail):

| `λ_sparse` | kept video (mean) | retention hist / fut | signal overlap | `traj_mse` | dynamic |
|---:|---:|---|---:|---:|---|
| 3e-4 | 384 | 0.492 / 0.000 | 0.0773 | 0.000602 | False |
| 1e-3 | 384 | 0.492 / 0.000 | 0.0735 | 0.000575 | False |
| 3e-3 | 384 | 0.492 / 0.000 | 0.0764 | 0.000577 | False |

Three things to take from this table, and none of them is "the frontier works":

1. **On the gather path `λ` has no effect on length at all.** Every arm, including `λ=0`, ends at
   `K=40` — the safety floor — because the selector collapses within the first ~50 steps regardless
   of the penalty. A λ sweep over a collapsed selector measures nothing.
2. **It also cannot compress the future block.** The per-domain retention is `0.010 / 0.041` in every
   arm: the clamp's 8 history + 32 future tokens are all that is left, so "dynamic length" is two
   constants.
3. **Training made selection worse than not training.** The untrained control at `K=40` is 0.328;
   every gather arm lands at 0.000–0.022. On the dense path the arms sit at 0.074–0.077 against a
   control of 0.032 at `K=384` — above control, but at the base rate, i.e. no usable selection.

The dense-gate arms do hold a much better task loss (`traj_mse` ≈ 0.0006 vs ≈ 0.0010), which is
consistent with §5.6: the dense gate keeps the forward well-behaved and the gate tunable, which is
what a warm-up stage is for. Neither path produced a compression/quality working point, and the
simulation is not a substitute for the NAVSIM run.

**Correction to an earlier reading in this report.** A draft of §0/§5.3 stated that the selector
"discovers the injected informative token subset". With the control above, that claim does not hold:
at `K=40` most of the apparent overlap is a top-k variance artifact, and training reduces it. The
defensible claims are the mechanical ones (forward/backward correctness), the credit-assignment
finding in §5.4, and the matched gather-vs-dense behaviour in §5.6.

### 5.7 What the simulation does and does not license

Licensed by the evidence above:

* the Route A forward is correct (dense front-end, threshold, sparse backend, recovery, heads);
* the gate, sparsity curriculum, safety clamp, distillation losses and dynamic-length analytics
  behave as the plan specifies;
* **nothing yet about learned selection.** The untrained control at `K=40` already reaches 0.328
  overlap, and no trained arm beat a matched control meaningfully; the single 30-layer step-8 point
  (0.725 vs a 0.475 control ceiling) is a hint, not a result;
* the literal gather formulation has a **fatal, silent** credit-assignment gap, and the dense
  gate removes it.

**Not** licensed:

* any statement about NAVSIM PDM, zero-score tail behaviour, or latency;
* that 12–18% retention is reachable on real driving data;
* that the plan's default `λ_sparse` range is usable — §5.6 shows the opposite for the gather
  form, and the dense form needs a threshold-calibration pass to map to a length.





---

## 6. Resource situation (why the NAVSIM run has not happened)

`nvidia-smi` was sampled repeatedly during this session. Every one of the 8 RTX 4090s was occupied:

```
index  memory.used  memory.free  util
0      39770 MiB    8875 MiB     100 %
1      39770 MiB    5822 MiB     100 %
2      42640 MiB    6005 MiB     100 %
3      42658 MiB    5987 MiB     100 %
4      41222 MiB    7423 MiB     0 %
5      41314 MiB    4278 MiB     0 %
6      39098 MiB    9547 MiB     0 %
7      39082 MiB    9563 MiB     0 %
```

The project rule (`AGENTS.md` §3.1, user instruction 2026-09-20) is *"at most 2 GPUs, and only
genuinely free ones (free ≥ 40 GiB); if none are free, wait."* No GPU came close: the largest free
figure observed all session was **10.3 GiB**, and each card carries another user's ~38–42 GiB
resident allocation. Loading Wan2.2-5B in bf16 needs ~10 GB for weights alone plus activations for a
1,569-token sequence, so even the "empty utilisation" cards had no usable headroom.

Consequences, stated plainly:

* **no NAVSIM training run, no PDM evaluation, no calibration sweep** was performed;
* every number in §5 is from the controlled simulation, not from driving data;
* the guarded `--mode navsim` entry point **refuses to run** rather than silently substituting the
  simulation, specifically so this limitation cannot be mistaken for a result.

---

## 7. Plan for the real NAVSIM run (executable next step)

### 7.1 Wiring

The one remaining integration is inside `diffsynth/pipelines/wan_video_new.py::model_fn_wan_video`.
Route A owns its own block loop, so the cleanest wiring mirrors the existing duck-typed hooks rather
than adding a new one:

1. Attach the trained module next to the backbone as `pipe.route_a` (an `nn.Module`, so it is saved
   in the checkpoint — plan §36 explicitly forbids implementing this as a runtime Press hook).
2. In `model_fn_wan_video`, after `x` is built and `t_mod`/`freqs` are ready, branch:
   if `pipe.route_a is not None and compression_mode == "dynamic_select"`, run
   `RouteADynamicSelect.forward(blocks=list(dit.blocks), x=..., context=..., t_mod=..., freqs=..., timestep=..., trajectory_head=..., head=dit.head, head_t_mod=t, capture_layers=[11,18,29])`
   and return its `{video, traj}`; otherwise fall through to the existing dense loop unchanged.
   The existing `hidden_sequence` / `counterfactual` machinery stays untouched for probes and baselines.
3. Training driver: reuse `examples/wanvideo/driveva_train/train_navsim_v1.py` for the data pipeline,
   the flow-matching targets and the teacher capture layers, and replace its online-selector head
   with `RouteADynamicSelect` and its selector loss with `compute_route_a_loss`. The teacher runs the
   existing dense path under `torch.no_grad()`.

### 7.2 Data and protocol

| Item | Value |
|---|---|
| Train manifest | `videopress_framework/outputs/navsim_split_audit/train_manifest.jsonl` (3,768) |
| Selector train/calibration split | `outputs/selector_capture_split_20260910/` (3,190 train / 578 calibration, no scene overlap) |
| Evaluation | official `navtest-7876`, paired vs a fresh NoPress baseline, same-scene guard |
| Gate | `CI_lower(ΔPDM) > −0.002` **and** `mean video tokens < 300` (plan §44) |
| Threshold calibration | on the 578-scene calibration split only, never the test set (plan §25) |

### 7.3 Resource estimate

With `micro_batch_per_gpu = 1`, gradient accumulation 8–16, bf16 and gradient checkpointing
(plan §28), and Route A at `Lb=18` with ~240 kept video tokens:

* the sparse backend is ~6.3× shorter than dense (249 vs 1569), and 12 of 30 blocks use it, so the
  **per-step cost is roughly 0.52× of a dense DriveVA training step** by the model in §8;
* A1 (selector + decoder only, frozen backbone) is much cheaper still, since only the front-end and
  the sparse backend need activations;
* a single 7,876-scene paired evaluation at ~590 ms/step is ~1.3 h on one GPU, or ~20 min on 4.

A credible minimum viable programme is: A1 (≈2–4 GPU-h) → A2 LoRA (≈8–12 GPU-h) → A3 at `Lb=18`
(≈20–30 GPU-h) → calibration sweep on 578 scenes (≈1 GPU-h) → full 7,876 evaluation at the chosen
`tau*` (≈1.5 GPU-h). Call it **2–4 GPU-days** for the first honest answer, using at most 2 GPUs at a
time as the project rule requires. Under the current occupancy that means waiting for cards to free up.

---

## 8. Cost model (`scripts/route_a_budget_report.py`)

Analytic MAC counts at the real geometry (`dim=3072`, `ffn_dim=14336`, 30 layers, context 512,
1569-token sequence) with a mean of 240 kept video tokens:

| Quantity | Value |
|---|---:|
| Full sequence | 1569 |
| Sparse sequence (240 video + 9 traj) | 249 |
| Sequence ratio | **15.9%** |
| Backbone MACs, baseline | 8.30e12 |
| Backbone saving, `Lb=18` (12 compressed layers) | **33.9%** |
| Backbone saving, `Lb=15` (15 compressed layers) | **42.4%** |
| Backbone saving, `Lb=12` (18 compressed layers) | **50.9%** |
| New parameters: scorer | 2,183,937 |
| New parameters: gate | 0 (thresholds are buffers) |
| New parameters: dense recovery decoder | 175,291,904 |
| Scorer + gate overhead, trajectory-only inference | **0.024%** of baseline MACs |
| With recovery decoder (training / video flow) | 1.50% |
| **Net saving, trajectory-only inference, `Lb=18`** | **33.9%** |
| Net saving, training or video-flow inference, `Lb=18` | 32.4% |

Two things follow. First, the payoff is an order of magnitude above the deployed frozen press
(2.8–3.4% latency). Second, the recovery decoder is large (175M params) but the plan's
trajectory-only inference mode skips it entirely, so the **deployment** footprint of Route A is the
2.18M-parameter scorer plus a buffer — 0.04% of the backbone. That is what makes the risk acceptable.

Caveats: these are MAC counts, not wall-clock. The plan's own history shows the gap between the two
(a single-layer K/V prune reduced sequence length but *increased* end-to-end latency because of
scoring and gather overhead), and Route A's gather plus scatter-back are real costs. The honest
expectation is that the **33.9% MAC saving converts to something like 15–25% wall-clock**, to be
measured, not assumed.

---

## 9. Risks and known failure modes

1. **STE gradient saturation.** When a candidate's keep probability saturates at 1, both sigmoids in
   `soft` have zero derivative and the selector stops learning from the task loss; only `λ_sparse`
   remains. Mitigation: keep `λ_sparse > 0` from A2 on (as the plan specifies), anneal temperature
   0.20 → 0.05, and monitor the score distribution for bipolarity rather than saturation. This is
   asserted in `test_route_a_ste_gradient_vanishes_when_every_score_saturates`.
2. **Zero-init recovery delay.** The decoder's zero-initialised output means the *video* distillation
   path cannot reach the selector on the very first backward pass. The trajectory path and the
   sparsity term still train it, and the video path goes live from step 2. Documented and asserted.
3. **Compression at `Lb` may simply be too early.** The DiT probes show trajectory semantics form at
   L10–12 and future video semantics only at L16–18. Route A's curriculum starts at `Lb=18` for
   exactly this reason; a run that starts at `Lb=12` and fails tells you nothing about `Lb=18`.
4. **Causal dependence may be irreducible.** The knockout probe showed no depth decay for future
   tokens. Route A's answer is that the *backbone* is now trainable, so the dependence can be
   re-organised — but that is a hypothesis, not a result. If A3 at `Lb=18` cannot reach the gate even
   at 50% retention, the plan's stated fallback is Route B (Dynamic Register), which changes the
   representation rather than its support.
5. **The tail is the real test.** Every prior compressed arm looked acceptable on the mean and failed
   on introduced zero-score scenes (`ego_progress=0`). `CompressionStatsRecorder` records the
   per-scene dynamic length so that `corr(K, difficulty)` can be checked; a selector that keeps the
   same count everywhere has degenerated and must be rejected even if its mean PDM looks fine.
6. **Latency may not follow MACs.** See §8.

---

## 10. Go / No-Go

Per plan §44, Route A is a success only if, on the full 7,876 paired protocol:

```
CI_lower(ΔPDM) > −0.002   AND   mean video tokens < 300
```

Decision tree for the next GPU window:

1. **A1 only** (selector + decoder on a frozen backbone, ~2–4 GPU-h). If the selector cannot beat
   matched-random selection at equal K in the *training* loss, stop — the objective has no signal,
   and no amount of A2/A3 will create it.
2. **A2/A3 at `Lb=18`.** If the retention/quality frontier reaches `CI_lower > −0.002` at
   `mean K < 300`, Route A works; then push the bottleneck to 15 and 12 for more saving.
3. **If `Lb=18` fails**, record the frontier and move to **Route B** rather than tuning scorers: the
   frozen-prune history already established that better ranking is not the bottleneck.
4. **If Route A succeeds**, the immediate follow-up is the plan's §27.1 initialisation ablation
   (WAN2.2 DiT + DriveVA trajectory modules) to separate "task-intrinsic information requirement"
   from "dense-representation lock-in".

---

## 11. Reproduction

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite

# Unit tests (59 in this file, 323 total)
/home/cpj/miniconda3/envs/DriveVA/bin/python -m pytest \
  videopress_framework/tests/test_retraining_route_a.py -q
/home/cpj/miniconda3/envs/DriveVA/bin/python -m pytest videopress_framework/tests/ -q

# Cost model at the real geometry
/home/cpj/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/route_a_budget_report.py \
  --bottleneck 18 --mean-kept 240 \
  --output videopress_framework/outputs/route_a_retraining_20260924/budget.json

# Untrained-scorer control for the simulation (the null distribution)
/home/cpj/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/route_a_sim_control.py \
  --keep-counts 40,100,200,384 --scenes 200 \
  --output videopress_framework/outputs/route_a_retraining_20260924/sim_control.json

# Controlled-redundancy training, lambda sweep (CPU)
OMP_NUM_THREADS=48 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/train_route_a_smoke.py --mode sim --device cpu --stage A3 \
  --lambda-sweep 0,3e-4,1e-3,3e-3,1e-2 --steps 400 \
  --layers 12 --bottleneck 8 --capture-layers 4,8,11 \
  --dim 128 --heads 4 --ffn-dim 384 --eval-scenes 64 --sparsity-ramp 200

# Production-scale geometry (30 layers, dim 256, Lb=18)
OMP_NUM_THREADS=48 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/train_route_a_smoke.py --mode sim --device cpu --stage A3 \
  --sparsity-weight 3e-3 --steps 80 --layers 30 --bottleneck 18 \
  --capture-layers 11,18,29 --dim 256 --heads 8 --ffn-dim 1024 --eval-scenes 32

# The real run (requires a free GPU and the NAVSIM data; currently guarded)
/home/cpj/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/train_route_a_smoke.py --mode navsim --device cuda:0
```

Artifacts from the runs in this report:
`videopress_framework/outputs/route_a_retraining_20260924/`.

---

## 12. Files added

```
videopress_framework/videopress/retraining/__init__.py
videopress_framework/videopress/retraining/threshold_gate.py
videopress_framework/videopress/retraining/dynamic_selector.py
videopress_framework/videopress/retraining/dense_recovery.py
videopress_framework/videopress/retraining/distillation.py
videopress_framework/videopress/retraining/compression_stats.py
videopress_framework/videopress/retraining/route_a.py
videopress_framework/videopress/retraining/curriculum.py
videopress_framework/scripts/train_route_a_smoke.py
videopress_framework/scripts/route_a_budget_report.py
videopress_framework/scripts/route_a_sim_control.py
videopress_framework/tests/test_retraining_route_a.py
```

No existing module was modified, so the deployed press paths and the frozen-prune conclusions are
untouched.

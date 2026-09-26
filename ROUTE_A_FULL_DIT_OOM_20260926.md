# Route A: does *full DiT* training OOM on two GPUs?

> 2026-09-26 CST · branch `main` · scope: plan v2 §14 **A3 — Full DiT adaptation**
> Status: **analytic verdict complete; empirical confirmation queued on idle cards** (see §4).

## 0. The question, disambiguated

The request was to test "prune 路径 A 中全量训练 DiT 是否会 OOM" with 2 GPUs. Two readings
are both answered below:

* **"全量训练 DiT" = train the whole DiT backbone** (plan §14 **A3**, as opposed to A1 which
  froze the backbone and only trained Route A's own 177.48 M parameters). → **Yes, it OOMs.**
  This is the substantive answer and the rest of this document is about it.
* **"全量" = the full 3,768-scene training manifest** rather than a small panel. → **No.**
  The A1 rerun already trained the complete 3,768-scene manifest for 6 epochs on **2 GPUs**
  (`outputs/route_a_train2_20260924/`, 11,304 steps, `rc=0`, 1.03–1.04 it/s). Full data on
  two cards is fine; it is the *trainable parameter count* that does not fit.

## 1. Answer

**Full-DiT Route A training (A3) cannot run on 2× RTX 4090 (49.14 GiB each) with the current
training stack.** The steady state needs **≈ 67.5 GiB per rank**, i.e. ~20 GiB more than the
card, *before* activations, VAE, text encoder and NCCL buffers. Turning off the EMA shadow —
the single largest removable term — still leaves **48.2 GiB**, which is already at/over the
card limit with nothing left for activations.

The important corollary: **this is not a "2 GPUs" problem, and adding cards does not fix it.**
`launch_training_task` constructs a plain `Accelerator` with only
`DistributedDataParallelKwargs` / `InitProcessGroupKwargs`; there is **no FSDP or DeepSpeed
anywhere in the repository** (`grep -rn "FullyShardedDataParallel|fsdp_plugin|deepspeed_plugin"`
over `diffsynth/ examples/ videopress_framework/videopress/` returns nothing). Plain DDP
**replicates** the model, so per-rank memory is identical at 2, 4 or 8 cards. The binding
constraint is per-card capacity; the fix is *sharding*, not more GPUs.

## 2. Why: exact parameter counts

`memory_model.py` reads the counts from the artifacts the loader actually uses rather than
assuming "5B":

| quantity | value | source |
|---|---:|---|
| Wan2.2-TI2V-5B DiT parameters | **4,999,787,712** (index `total_size` 19,999,150,848 B, all `F32`) | `models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json` + shard headers |
| DiT dtype on the card | **bf16** | `WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, ...)` |
| Route A parameters | **177,475,841** (`scorer` 2,183,937, `gate` 0, `recovery` 175,291,904) | `RouteAConfig(...).build()` |
| trainable in A1 (`freeze_backbone: true`) | 177.48 M | `[route-a] attached: trainable route_a=177.48M trainable dit=177.48M` |
| trainable in A3 (`freeze_backbone: false`) | **5,177,263,553** | DiT + Route A + `trajectory_encoder/head` (the YAML `TRAINABLE_MODELS` default) |

Two implementation facts drive the arithmetic:

1. `torch.optim.AdamW(model.trainable_modules(), ...)` is created on **bf16** parameters, so
   `exp_avg` / `exp_avg_sq` are bf16 — **verified** on the project interpreter:
   `torch 2.5.0+cu124`, `p = Parameter(zeros(bf16)); AdamW([p]).step()` →
   `exp_avg torch.bfloat16`, `exp_avg_sq torch.bfloat16`. Cost: 4 B/param.
2. `DiffusionTrainingModule.init_ema` allocates a **float32** shadow of every trainable
   parameter: `param.detach().to(device=ema_device, dtype=torch.float32).clone()`.
   Cost: 4 B/param on the card when `EMA_ON_CPU=0` (the default). **This is the term that
   makes the default recipe hopeless.**

## 3. Per-rank memory model

From `memory_model.py --json memory_model.json` (all figures GiB, per rank):

| recipe | term | GiB |
|---|---|---:|
| **A1** (backbone frozen) | bf16 trainable params (Route A) | 0.33 |
| | bf16 grads | 0.33 |
| | AdamW state (bf16 ×2) | 0.66 |
| | EMA fp32 shadow | 0.66 |
| | frozen backbone resident (bf16) | 9.31 |
| | DDP grad buckets | 0.33 |
| | **subtotal** | **11.62** |
| **A3** full DiT + EMA | bf16 trainable params (DiT + Route A + traj) | 9.64 |
| | bf16 grads | 9.64 |
| | AdamW state (bf16 ×2) | 19.29 |
| | EMA fp32 shadow | 19.29 |
| | DDP grad buckets | 9.64 |
| | **subtotal** | **67.50** |
| | headroom vs 48.0 GiB usable | **−19.52** |
| **A3** full DiT, EMA off | params + grads + AdamW + buckets | **48.22** |
| | headroom vs 48.0 GiB usable | **−0.23** (before activations) |

The A1 subtotal of 11.62 GiB is consistent with the A1 rerun's measured behaviour (it trained
at 1.03–1.04 it/s with `rc=0` on 2 cards), which cross-validates the model's terms.

Two **frozen** residents are deliberately left out of the table above because they are not part
of the trainable state; they are quantified in §5.1. `prepare_model` calls
`model.to(self.device)` on the whole training module (accelerate's `verify_device_map` returns
`False` for these non-HF modules, so the `elif` branch does fire), and the pipeline's VAE and
text encoder are ordinary submodules — so they ride along to the card and stay there:
VAE 704.7 M fp32 → **1.31 GiB** after the bf16 cast, umt5-xxl text encoder 5,680.9 M natively
bf16 → **10.58 GiB**. That is **11.89 GiB replicated on every rank**, which DDP never shards.
The dataset returns a prompt *string* (`_build_prompt_fixed`), so the text encoder is genuinely
needed every step and cannot simply be dropped.

Sequence of failure for A3 with EMA: model load (9.64) → DDP wrap → `init_ema` (+19.29 =
28.9) → first backward (+ grads 9.64, + buckets 9.64 = 48.2) → `optimizer.step()`
(+19.29 states = 67.5). It dies inside the first training step, not at load, **provided the
card is genuinely idle** — which is exactly why the empirical queue insists on ≥ 40 GiB free
rather than sharing a card, where the run would die during load and tell us nothing.

## 4. Empirical test

Harness: `videopress_framework/outputs/route_a_full_dit_oom_20260926/` (gitignored, like all
experiment scaffolding in this repo), driven by `run_oom_test.sh` in a persistent tmux
session `route_a_oom`, with a 1 Hz `nvidia-smi` memory sampler (`mem_sampler.py`) so the peak
survives an OOM kill. Four variants, each pinned to **2 genuinely idle cards** (project rule:
≥ 40 GiB free, never share):

| variant | `freeze_backbone` | `USE_EMA` | grad-ckpt offload | purpose |
|---|---|---|---|---|
| `v0_a1_control` | true | 1 | 0 | baseline; must fit, gives the A1 peak on the same instrument |
| `v1_full_dit_ema` | false | 1 | 0 | the literal A3 recipe |
| `v2_full_dit_no_ema` | false | 0 | 0 | isolates the 20 GiB fp32 EMA shadow |
| `v3_full_dit_offload` | false | 0 | 1 | last cheap knob: move saved activations to host RAM |

Each variant: `MAX_SCENES=16`, `NUM_EPOCHS=1`, `SAVE_RAW_CKPT=0`, `SAVE_EMA=0` (a
fully-trainable state dict would be a ~10 GiB checkpoint per save, so none is written), and the
real NAVSIM train manifest. Success is read from the `[route-a][step N]` monitor lines; OOM is
read from `torch.OutOfMemoryError`.

**Current state: the empirical half has not run yet.** At launch (2026-09-26 11:49) and at the
time of writing, **all 8 cards are busy** (another user's vLLM workers; max free 8.4 GiB,
against a 40 GiB requirement), so the queue is parked in its acquire loop and will start on its
own when two cards free up. Evidence lands in:
`status.log`, `<variant>.log`, `<variant>.mem.jsonl(.peak.json)`, `<variant>.result.json`,
`RAW_RESULTS.md`, `QUEUE_COMPLETE`.

## 5. What actually would make A3 fit

Ordered by (effort, payoff). None of these is implemented; they are options for the next step.

1. **FSDP / ZeRO-2 or ZeRO-3** (`FullyShardedDataParallelPlugin` on the `Accelerator`). Shards
   params + grads + optimizer state across ranks: 9.64 + 9.64 + 19.29 → ~19.3 GiB/rank on 2
   cards, plus activations. This is *the* change that makes A3 possible on 2 cards, and it also
   explains why "use more GPUs" alone does nothing today.
2. **`EMA_ON_CPU=1`** (already a supported flag, `--ema_on_cpu`) removes 19.29 GiB from the
   card at the cost of a host-side copy per step. Necessary but not sufficient on its own.
3. **8-bit Adam** (`bitsandbytes`) cuts the 19.29 GiB state to ~4.8 GiB; combined with
   `EMA_ON_CPU=1` the subtotal drops to ~33.7 GiB, which fits one 49 GiB card. Requires adding
   the optimizer dependency and a code path.
4. **Plan A2 (LoRA rank 64) instead of A3.** LoRA targets ~70–140 M trainable parameters
   instead of 5.18 B, landing near the A1 profile (~12 GiB) and fitting comfortably on 2 cards.
   This is the plan's own intermediate stage; the A2 wiring needs care because
   `switch_pipe_to_training_mode` calls `freeze_except` **before** LoRA injection, so a
   correct A2 recipe must not leave the DiT base weights trainable.

## 5.1 Follow-up (2026-09-26): ZeRO-2 + EMA in host memory + averaging every N steps?

**Short answer: no.** That combination still OOMs on 2×49 GiB, because ZeRO-2's default state
precision is *worse* than the patch, and because the update interval does not change any
memory figure. Model: `sharding_model.py` (2 ranks, 48 GiB usable/card, frozen VAE + text
encoder counted at 11.89 GiB/rank).

| strategy | EMA on GPU | EMA on CPU |
|---|---:|---:|
| `ddp` (current code) | 79.4 GiB ✗ | 60.1 GiB ✗ |
| **`zero2` (DeepSpeed defaults)** | 74.6 GiB ✗ | **55.3 GiB ✗** |
| `zero2` + `bf16_master_weights_and_grads` + `bf16_optimizer_states` | 60.1 GiB ✗ | **40.8 GiB ✓** |
| `zero2` + `offload_optimizer: cpu` | 45.6 GiB ✓ | **26.4 GiB ✓** |
| `zero3` (defaults) | 69.8 GiB ✗ | 50.5 GiB ✗ |
| `zero3` + `offload_optimizer: cpu` | 40.8 GiB ✓ | **21.5 GiB ✓** |
| `fsdp` FULL_SHARD (no DeepSpeed needed) | 50.5 GiB ✗ | **31.2 GiB ✓** |
| `ddp` + frozen VAE/text encoder kept off the card | 67.5 GiB ✗ | 48.2 GiB ✗ |

Three separate reasons the proposed recipe is not enough:

1. **ZeRO-2's optimizer state is fp32 by default, and that is the dominant term.** DeepSpeed
   keeps fp32 *master weights* plus fp32 `exp_avg`/`exp_avg_sq` unless told otherwise —
   `bf16.bf16_master_weights_and_grads` and `bf16.bf16_optimizer_states` both default to
   `false` ([config docs](https://www.deepspeed.ai/docs/config-json/)). That is ~12 B/param of
   state before sharding, against **4 B/param** for the repo's current `torch.optim.AdamW` on
   bf16 parameters. Halving a 12 B/param state (19.29 GiB/rank) is not the same as halving the
   cheap one (9.64 GiB/rank). This is why `zero2` at 55.3 GiB still loses to plain FSDP at
   31.2 GiB.
2. **"相隔多轮平均" saves no GPU memory at all.** The EMA shadow is one fixed-size buffer; its
   *size* is 19.29 GiB fp32 and it does not depend on how often it is refreshed.
   `update_every=N` only reduces the number of GPU→host fp32 transfers (and, incidentally,
   changes the effective averaging horizon). Moving the shadow to host is the part that
   matters, and it is already implemented: `EMA_ON_CPU=1` / `--ema_on_cpu`. Do use
   `update_every>1` for speed, but count it as zero memory saving.
3. **Sharding does not touch the 11.89 GiB of frozen VAE + text encoder.** ZeRO-2 replicates
   them; only FSDP/ZeRO-3 would shard them, and then they are all-gathered transiently during
   the forward (a 10.58 GiB spike for the text encoder unless it is a separate unit or in
   `ignored_modules`).

**What to do instead, on 2 cards**

* **FSDP `FULL_SHARD` + `EMA_ON_CPU=1` → ~31 GiB/rank.** Best value: it fits with real
  headroom, needs **no new dependency** (`deepspeed` is not installed in the `DriveVA` env;
  `accelerate` 1.14.0 + torch FSDP are), and it keeps the cheap bf16 optimizer state instead of
  introducing an fp32 master copy. Needs `use_orig_params=True` because only a subset of
  parameters requires grad.
* **ZeRO-2 + `offload_optimizer: cpu` + `EMA_ON_CPU=1` → ~26 GiB/rank.** Fits comfortably and
  keeps the familiar 2-card DDP topology, at the cost of CPU-side optimizer steps and a
  per-step GPU→CPU gradient transfer. Host RAM is not a constraint here (755 GiB total,
  ~651 GiB available; the shadow plus offloaded state is ~50–60 GiB).
* **ZeRO-2 with bf16 master weights and bf16 optimizer states → ~41 GiB/rank.** Fits, but with
  only ~7 GiB left for activations, communication buffers and fragmentation, and it puts the
  master copy in bf16 (a convergence trade-off). Verify the option actually engages with a
  client-passed `torch.optim.AdamW` before relying on it.

**Good news on wiring:** the training loop already calls `accelerator.backward(loss)`,
`accelerator.clip_grad_norm_` and uses the prepared optimizer, so both FSDP and DeepSpeed are
drop-in at the loop level — no changes to `launch_training_task`'s inner loop are required.
The launch script does need to switch from bare `torch.distributed.run` to the corresponding
accelerate env (`ACCELERATE_USE_FSDP=true` + `FSDP_SHARDING_STRATEGY=FULL_SHARD`, or
`ACCELERATE_USE_DEEPSPEED=true` + a DeepSpeed config JSON).

**Cost warning:** with `NCCL_P2P_DISABLE=1` (required on these 4090s) every ZeRO/FSDP step pays
reduce-scatter/all-gather over PCIe with no P2P path. Expect a substantial throughput loss
relative to the current DDP A1/A3 runs; memory feasibility is not the same as viability.

## 5.2 Follow-up (2026-09-26): can the 5B backbone be swapped for "Wan2.2 1.3B"?

**Short answer: there is no Wan2.2 1.3B, and the 1.3B that does exist (Wan2.1-T2V-1.3B) is not a
drop-in swap — it would discard the DriveVA checkpoint, make the sequence 4× longer, and only
then would it fit in memory.** Memory is the one thing it fixes, and FSDP/LoRA fix that without
breaking comparability.

**(a) Factual correction.** The Wan2.2 release (July 2025) is **T2V-A14B, I2V-A14B and
TI2V-5B** (later S2V-14B / Animate-14B) — see the
[Alibaba Cloud announcement](https://www.alibabacloud.com/blog/602413). The 1.3B tier exists
only in **Wan2.1** (`Wan2.1-T2V-1.3B`, `Wan2.1-VACE-1.3B`), whose reference config is
[`wan_t2v_1_3B.py`](https://huggingface.co/spaces/VIDraft/Wan2GP/blob/8949c1a9bb2ed622d80ec2de4b820f13b8bd9db4/wan/configs/wan_t2v_1_3B.py).

**(b) The two backbones are architecturally incompatible.**

| | Wan2.2-TI2V-5B (current) | Wan2.1-T2V-1.3B |
|---|---|---|
| DiT params | 4,999,787,712 | **1,418,996,800** (built from its config to count) |
| `dim` / `ffn_dim` / heads | 3072 / 14336 / 24 | 1536 / 8960 / 12 |
| latent channels | **48** | **16** |
| VAE stride | **4×16×16** (high-compression) | **4×8×8** |
| task | TI2V (text+image→video) | T2V (text→video) |

The 5B's high-compression VAE is the whole reason its token count is small: at 480×832,
16×16 compression gives a 30×52 latent → 390 tokens/latent frame; 8×8 gives 60×104 →
**1560 tokens/latent frame**.

**(c) Token count goes UP 4×, not down.** With the same 4 latent frames (2 history + 2 future):

| backbone | tokens/latent frame | 4 latent frames | DiT sequence |
|---|---:|---:|---:|
| Wan2.2-TI2V-5B | 390 | **1,560** ✓ (matches the project docs) | ~1,569 |
| Wan2.1-T2V-1.3B | 1,560 | **6,240** | ~6,249 |

Attention cost scales ~L² and MLP ~L, so the 4× shorter parameter count is largely given back.
More importantly for this project, the compression target changes from "1560 → ~240" to
"6240 → ~240" — a far more aggressive problem, so any Route A result on 1.3B would not transfer
to the 5B setting the paper is about.

**(d) The DriveVA checkpoint cannot be loaded at all.** `pdms90_9.safetensors` is shape-locked
to the 5B. Read from the safetensors header:

```
dit.patch_embedding.weight      [3072, 48, 1, 2, 2]   vs 1.3B: [1536, 16, 1, 2, 2]
dit.blocks.0.self_attn.q.weight [3072, 3072]          vs 1.3B: [1536, 1536]
dit.head.head.weight            [192, 3072]           vs 1.3B: [16, 1536]
trajectory_head.proj.2.weight   [3, 3072]             vs 1.3B: [3, 1536]
```

Every DiT and trajectory-head key differs. Swapping the backbone therefore **throws away the
DriveVA driving prior** and turns the task into "re-finetune DriveVA on NAVSIM from a generic
Wan2.1 base" — a much larger project than token compression, after which none of the historical
numbers (NoPress `0.909839`, the press results, the A1 verdict) are comparable.

**(e) It would fit, though.** Per-rank, plain DDP, no sharding (Route A rebuilt at `dim=1536`
comes to 48.9 M instead of 177.5 M):

| model | params | grads | AdamW | EMA | buckets | frozen VAE+TE | total (EMA GPU) | total (EMA CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan2.2-TI2V-5B | 9.64 | 9.64 | 19.29 | 19.29 | 9.64 | 11.89 | **79.40** ✗ | **60.11** ✗ |
| Wan2.1-T2V-1.3B | 2.73 | 2.73 | 5.47 | 5.47 | 2.73 | 10.82 | **29.96** ✓ | **24.49** ✓ |

So a 1.3B backbone needs no sharding, no optimizer offload and no host EMA — it fits on one
49 GiB card as-is.

**(f) Repo readiness.** DiffSynth already carries Wan2.1 plumbing: `WanVideoPipeline.from_pretrained`
has a `redirect_common_files` map that points `models_t5_umt5-xxl-enc-bf16.pth` and
`Wan2.1_VAE.pth` at `Wan-AI/Wan2.1-T2V-1.3B`, and `wan_video_dit.py` has 1.3B config branches.
But the weights are **not** on disk (`models/Wan-AI/` contains only `Wan2.2-TI2V-5B`), and
every DriveVA-specific piece is 5B-bound: the checkpoint, the trajectory head width, the PDM
eval protocol and metric cache, and the documented 1560-token layout. The press framework's
`TokenLayout` is parameterised (`tokens_per_latent`, `num_cond_latents`, `video_f/h/w`), so it
would adapt with configuration rather than rewrites — the *scientific* foundation is what does
not survive.

**Recommendation.**

* **If the goal is "A3 must fit on 2 GPUs" → do not change the backbone.** FSDP FULL_SHARD +
  `EMA_ON_CPU=1` (~31 GiB/rank) or plan A2 LoRA reach the same place while keeping the 5B
  checkpoint, the eval baseline and every prior result comparable. Changing the backbone to fix
  a memory problem trades a one-line launcher change for the entire experimental foundation.
* **If the goal is a cheap vehicle for iterating on Route A mechanics** → a 1.3B backbone is
  defensible, but only as a separately labelled *scaling study*, with a from-scratch DriveVA
  re-finetune and a rebuilt layout/eval path. Budget it as its own project.
* **If the goal is fewer video tokens** → the 5B is already the token-efficient choice
  (16×16 VAE); moving to Wan2.1 moves backwards. For reference, the only same-family way to
  shrink the DiT while keeping the VAE and token layout would be a self-constructed narrower
  DiT with `in_dim=48`, which forfeits the pretrained prior just the same.

## 6. Reproduction

```bash
PY=/home/cpj/miniconda3/envs/DriveVA/bin/python
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/route_a_full_dit_oom_20260926
$PY memory_model.py --json memory_model.json          # analytic table in §3

# empirical queue (waits for 2 idle cards, >=40 GiB, never shares)
tmux new-session -d -s route_a_oom "PYTHON=$PY bash run_oom_test.sh"
tail -f status.log
```

## 7. One-line verdict

**A3 "全量训练 DiT" OOMs on 2 GPUs with the current code — and on 4 or 8 GPUs too, because the
stack is plain DDP.** The requirement is ~79.4 GiB/rank once the frozen VAE and text encoder are
counted (60.1 GiB with the EMA shadow moved to host) against a 49 GiB card. Moving the EMA to
host RAM is necessary but not sufficient; **the proposed ZeRO-2 + host-EMA + every-N-averaging
recipe still needs ~55.3 GiB/rank, because DeepSpeed's default fp32 master weights and fp32
optimizer states cost 12 B/param versus the 4 B/param this repo currently pays, and the
averaging interval saves no memory at all.** What does fit on 2 cards: **FSDP FULL_SHARD +
`EMA_ON_CPU=1` (~31 GiB/rank, no new dependency)**, ZeRO-2 with `offload_optimizer: cpu`
(~26 GiB/rank), or ZeRO-2 with bf16 master weights and bf16 optimizer states (~41 GiB/rank).
Plan §14's **A2 (LoRA)** remains the only adaptation stage that fits on 2 cards without any
sharding at all.

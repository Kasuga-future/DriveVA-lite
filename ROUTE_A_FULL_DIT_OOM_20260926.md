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

**A3 "全量训练 DiT" OOMs on 2 GPUs — and on 4 or 8 GPUs too, because the stack is plain DDP.**
The requirement is ~67.5 GiB/rank (48.2 GiB even with EMA off) against a 49 GiB card; sharding
(FSDP/ZeRO) or an 8-bit optimizer plus host-side EMA is required before A3 can be attempted,
and plan §14's **A2 (LoRA)** remains the only adaptation stage that fits on 2 cards as-is.

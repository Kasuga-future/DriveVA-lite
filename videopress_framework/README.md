# DriveVA VideoTokenPress framework

This folder is the primary implementation boundary for VideoTokenPress. It
contains the core framework, DriveVA runtime adapter, evaluator, reusable
scripts, tests and package installer. A small set of explicit integration hooks
remains in the parent DriveVA pipeline and training entry points; policy,
selection and analysis logic belongs here.

The two evaluation protocols are intentionally separate:

| Protocol | Injection point | Allowed operation | Model-visible effect |
| --- | --- | --- | --- |
| `causal` | `VIDEO_INPUT` | Zero/Mean/Shuffle | token length is unchanged; token content is changed before trajectory concatenation |
| `physical` | `SELF_ATTN_KV` | `KVPrune`/`KVMerge` | Q remains full length; post-RoPE K/V length is reduced and mapping is recorded |
| `physical` | `BLOCK_INPUT` | `HiddenPrune` | selected history tokens are removed before DiT block 0; Q/K/V, residual and MLP paths all run at the shorter length, then the layout is restored before the heads |

`SELF_ATTN_OUTPUT` is currently unsupported and fails fast.

True pre-DiT pilot (use a truncated scene count before any full protocol run):

```bash
CUDA_VISIBLE_DEVICES=0 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  scripts/run_official_navsim_press.py \
  --pre-dit-token-matrix --domain history --max-eval-tokens 16 \
  --poc-test-derived --output-root outputs/pre_dit_token_pilot
```

Use `--pre-dit-token-matrix-profile conservative` for the 90%/95% random and
TokenNorm controls. `--pre-dit-token-methods METHOD[,METHOD...]` selects an
independent subset so those arms can be sharded across GPUs without sharing
output state. The runner keeps truncated results marked test-derived and fails
the final official-scope guard by design.

## Layout

```text
videopress_framework/
├── videopress/       # core, domains, scorers, selectors, operators, adapters
├── evaluation/       # scene evaluator, artifacts, timing, statistics
├── configs/          # smoke configurations
├── scripts/          # evaluation, data preflight and sweep entry points
├── tests/            # unit/integration checks
└── setup.py          # install this folder as a separate package
```

## Install in the existing DriveVA environment

From the parent repository:

```bash
python -m pip install \
  --no-deps --no-build-isolation ./videopress_framework
```

The package is also runnable directly from this folder, so source tests do not
depend on the parent project's package metadata. All new framework code stays
under `videopress_framework/`.

The adapter resolves the configured `press.domain` as the default source of
truth. A sample metadata domain is ignored unless
`evaluation.allow_sample_domain_override: true` is explicitly set.

For the official NAVSIM path, the framework also installs a runtime-only
scene-window guard around the existing lite `SceneLoader`. It preserves the
official log/token/route filter, then rejects a window unless all frames have
the same `scene_token` and `scene_name`. This is needed because the migrated
lite NAVSIM snapshot can otherwise slice a log across two scene segments.
The guard is installed from `videopress_framework/` and does not edit
`third_party/navsim/` or the official evaluator.

## Tests

```bash
cd videopress_framework
/home/cpj/miniconda3/envs/DriveVA/bin/python -m compileall -q .
/home/cpj/miniconda3/envs/DriveVA/bin/python -m pytest -q
```

The repair regression suite covers NoPress artifacts, independent mapping and
score artifacts, checked video-index decoding, strict Gradient objectives,
frozen probe rankings, deterministic Random scopes, Attention V-norm semantics,
protocol validation, real K/V shortening and the official-backend boundary.

For causal Attention, `build_execution_plan` marks the method as
`FORWARD_PROBE`: Q/K are not available yet at the pre-concatenation
`VIDEO_INPUT` hook. Run the uncompressed probe forward, save its scores and
stable ranking in `ScoreCache`, then run the intervention forward with the
same key. A missing probe cache is an error; it is never replaced with a
different objective.

Set the following fields to generate an equal-budget Random control alongside a
method run. The selector, operator, budget, domain and injection point are
copied; only the scorer is replaced.

```yaml
evaluation:
  random_baseline: true
  random_seeds: [0, 1, 2]
```

Deterministic causal smoke round:

```bash
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/evaluate_press.py \
  --config configs/press/synthetic_smoke.yaml \
  --output-dir outputs/causal_round1
```

Physical post-RoPE KV smoke round:

```bash
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/evaluate_press.py \
  --config configs/press/synthetic_physical.yaml \
  --output-dir outputs/physical_round1
```

One-GPU Wan attention hook smoke (uses `cuda:0`, not the full 5B model):

```bash
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/gpu_smoke.py \
  --device cuda:0
```

Each output directory contains a config/environment snapshot, scene records,
separate `artifacts/scores/`, `artifacts/masks/` and
`artifacts/mappings/` files, and `events.jsonl`. Timing records distinguish
`selector_latency_ms`, `model_latency_ms`, `e2e_latency_ms` and peak memory.
The output also includes `tokens.parquet` (or `tokens.jsonl`) when token rows
are available. Runtime events are appended per layer/diffusion step, rather
than only retaining the last result.

## Full multi-round compression suite, tables and plots

`scripts/run_full_compression_suite.py` runs the complete independent
framework matrix on the deterministic synthetic cohort. It is an end-to-end
test of layout, scorer/probe, selector, operator, evaluator, artifact and
reporting paths; it does not claim to be an official NAVSIM PDM run.

The default matrix has 19 methods:

| Protocol | Methods covered |
| --- | --- |
| `causal` | NoPress; Random + Zero/Mean/ShuffleAll/ShuffleDropped/ShuffleKept; TokenNorm + Zero/Mean; ActionAttention + Zero; ActionAttention-VNorm + Zero; GradientNorm + Zero; GradientInput + Zero |
| `physical` | NoPress; Random/TokenNorm/ActionAttention/ActionAttention-VNorm + KVPrune; persistent ActionAttention-VNorm + KVPrune; SimilarityMerge + KVMerge |

`physical_attention_vnorm_kv_prune_persistent` scores and selects at layer 15,
then reuses the exact same global history-token indices for every deeper
self-attention layer. Each deeper layer gathers its newly computed K/V values;
Q and the hidden-token sequence remain full length. Runtime events distinguish
the source selection from downstream reuse with `selection_source_layer` and
`persistent_selection_reused`.

Run three paired rounds on four scenes per method (use `cuda:0` when the
installed PyTorch build exposes CUDA):

```bash
cd videopress_framework
MPLCONFIGDIR=/tmp/driveva_mpl /home/cpj/miniconda3/envs/DriveVA/bin/python \
  scripts/run_full_compression_suite.py \
  --rounds 3 --max-scenes 4 --device cuda:0 \
  --output-root outputs/full_compression_suite
```

If this environment's PyTorch build is CPU-only, use `--device cpu`; the
separate `scripts/gpu_smoke.py` remains the CUDA hook check. The runner never
overwrites an existing suite: it creates a `_rerunNN` sibling.

Every suite root contains one directory per round/method, plus:

```text
suite_manifest.json                 # exact method/round/output provenance
suite_summary.json                  # report index and pooled rows
statistics/run_statistics.csv       # one row per method and round
statistics/method_statistics.csv    # scene-weighted pooled rows across rounds
statistics/*.md                     # readable versions of both tables
visualizations/pdm_by_method.png
visualizations/latency_by_method.png
visualizations/compression_ratios.png
visualizations/pdm_vs_latency.png
visualizations/round_stability.png
visualizations/summary_table.png
```

Tables and plots can be rebuilt without rerunning evaluation:

```bash
MPLCONFIGDIR=/tmp/driveva_mpl /home/cpj/miniconda3/envs/DriveVA/bin/python \
  scripts/aggregate_suite.py outputs/full_compression_suite
```

The reporting API is in `evaluation.statistics` and
`evaluation.visualization`. It reports PDM/trajectory/endpoint mean and scene
standard deviation, exact K and candidate counts, causal eligible/history
ratios, physical K/V lengths and attention ratio, selector/model/end-to-end
latency, and peak memory.

## NavSIM data preflight

This checks the migrated metadata, sensor and metric-cache paths without
loading the 5B Wan model:

```bash
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/preflight_press_data.py
```

## Official DriveVA + NAVSIM backend

`DriveVANavsimBackend` is the strict production integration boundary. It
requires an official feature builder and inference callback. For official PDM,
provide the NAVSIM metric-cache loader, simulator, scorer and future sampling
objects; there is no synthetic metric fallback.

```python
from evaluation.evaluator import Evaluator
from evaluation.navsim_evaluator import DriveVANavsimBackend
from videopress.presses import NoPress

backend = DriveVANavsimBackend(
    pipe=pipe,
    samples=samples,
    layout=layout,
    feature_builder=official_feature_builder,
    inference_fn=official_inference_fn,
    metric_cache_loader=metric_cache_loader,
    simulator=official_simulator,
    scorer=official_scorer,
    future_sampling=official_future_sampling,
    adapter=driveva_adapter,
    device="cuda:0",
)
result = Evaluator(repo_root).evaluate_backend(
    backend, NoPress(), mode="causal", output_dir="outputs/navsim_full"
)
```

Full and compressed experiments must call this same backend/evaluator. The
bundled `evaluate_press.py` synthetic backend is only a deterministic framework
smoke test and reports `backend: synthetic`; it is not an official NAVSIM PDM
benchmark.

### Official full-scene runner

`scripts/run_official_navsim_press.py` is the executable official integration.
It delegates scene loading, `VideoDriveFeatureBuilder`, Wan inference,
trajectory conversion and PDM scoring to
`examples/wanvideo/driveva_infer/eval_navsim_pdm.py`; the independent adapter
only installs runtime hooks on the live pipeline. The 18-method matrix is the
same matrix used by the synthetic reporting test, but these runs report
`backend: official_navsim` and official PDM values.

The complete official cache is stored on NVMe at
`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full`.

For the scene-safe split generated by `audit_navsim_split.py`, the complete
test cache is stored separately at
`/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_split_test`.
It contains all 45,206 route-valid windows from `test_manifest.jsonl`; the
official guarded `navtest` subset (7,876 windows) has zero missing entries.
It can be rebuilt with the independent wrapper:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite
PYTHONPATH=$PWD/third_party \
NUPLAN_MAPS_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0 \
/mnt/nvme/chenpeijian/miniconda3/envs/DriveVA/bin/python \
  videopress_framework/scripts/cache_navsim_split.py --workers 24
```

If it must be rebuilt, the official command is resumable at the file level and
does not touch the old cache:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite
PYTHONPATH=$PWD/third_party \
OPENSCENE_DATA_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1 \
NUPLAN_MAPS_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0 \
/home/cpj/miniconda3/envs/DriveVA/bin/python \
  third_party/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  navsim_log_path=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/meta_datas/test \
  cache.cache_path=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full \
  worker=single_machine_thread_pool worker.max_workers=8 worker.use_process_pool=True
```

First validate the runner with one scene and one method:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0 /home/cpj/miniconda3/envs/DriveVA/bin/python \
  scripts/run_official_navsim_press.py \
  --methods causal_no_press --max-eval-tokens 1 \
  --output-root outputs/official_runner_smoke
```

Before a model run, audit the exact official windows and their cache
intersection:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/check_scene_windows.py
```

For the current `navtest.yaml` data, the unguarded lite loader returns 12,123
windows, including 4,247 cross-segment windows. The guarded official loader
returns 7,876 valid same-segment windows; all 7,876 have metric-cache entries
in the 12,146-entry NVMe cache. The audit must report `ok: true`,
`guarded_official_loader.errors: 0`, and
`metric_cache.missing_guarded_tokens: 0`.

Run all 19 methods on the complete corrected official set with six GPUs. The
logical GPUs 0–5 are used and GPUs 6–7 remain idle:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
/home/cpj/miniconda3/envs/DriveVA/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=6 \
  scripts/run_official_navsim_press.py \
  --metric-cache-path /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full \
  --score-cache-root /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/official_press_score_cache_same_scene_6gpu_final \
  --output-root outputs/official_navsim_same_scene_6gpu_final \
  --force-full-scene-set --enable-nuscenes-metrics
```

For the released `navtest.yaml` plus the evaluator's default one-frame
interval, the runner checks the guarded scene-filter/cache intersection before
starting. `--force-full-scene-set` stops immediately if any corrected
same-segment scene is missing from the cache. Each method directory contains
the untouched official PDM CSV, compact press events, frozen probe caches
where needed, `records.jsonl`, and `summary.json`; the suite root contains
`suite_manifest.json`, `suite_summary.json`, CSV/Markdown tables and plots.
When `--score-cache-root` is supplied, frozen attention/gradient score caches
are written there rather than beside the report, which keeps large runs off
the system disk.
Use `--save-viz --viz-total-tokens N` only for a deliberately small visual
sample, because it enables full video decoding for those selected scenes.

To compare persistent-compression start layers, run a paired no-press baseline
plus layers 0 through 29. Start with a small independent scene filter before
increasing `--max-eval-tokens`:

```bash
cd videopress_framework
python \
  scripts/run_official_navsim_press.py \
  --persistent-layer-sweep all \
  --persistent-mode hidden_sequence \
  --persistent-keep-ratio 0.5 \
  --rounds 2 \
  --max-eval-tokens 1 \
  --enable-nuscenes-metrics \
  --poc-test-derived \
  --scene-filter-yaml /path/to/independent_scene_filter.yaml \
  --skip-plots \
  --output-root outputs/persistent_layer_sweep_gpu6
```

The sweep syntax also accepts subsets such as `8,10-20,24`. The generated
method names encode the source layer, so the normal aggregate tables directly
compare PDM, latency, peak memory and compression metadata layer by layer.
`--persistent-mode kv_only` preserves the original behavior: it reuses the
source mapping but gathers only fresh K/V at deeper layers. The
`hidden_sequence` mode additionally gathers the residual stream, original
RoPE positions and per-token timestep modulation after the source block, so
self-attention Q/K/V, cross-attention Q and FFN computation all shrink until
the configured end layer. The full layout is restored before the trajectory
and video heads. For a true source-layer-only control with no downstream mask reuse, add
`--persistent-oneshot`; this control is K/V-only and cannot be combined with
`--persistent-end-layer` or `--persistent-mode hidden_sequence`.
For a clean learned-selector start-layer experiment, keep the checkpoint's
feature layer fixed while moving only the compression source, for example
`--persistent-feature-layer 15 --persistent-layer-sweep 20`. This option is
restricted to `learned_planning_selector`; omitting it preserves the historical
same-feature/same-source behavior.

Complete-history experiments can select the two stored history latents
independently (order is always `oldest,newest`):

```bash
# Dynamic number selected by one threshold per latent.
--domain history --persistent-selector history_threshold \
  --per-latent-thresholds 0.10,0.40

# Exact allocation control: 100% older and 41.5% newer.
--domain history --persistent-selector history_quota \
  --per-latent-keep-ratios 1.0,0.415
```

Per-latent thresholds use the full eligible budget as a safety cap. Quotas are
converted to exact token counts for DriveVA's 390-token history latents.
TeaCache, VACE, Animate, gradient checkpointing and unified
sequence parallel are rejected in this mode until those combinations receive
separate correctness tests.

The event metadata records the resolved candidate range and selected global
positions. For the official matrix, `last_history` must equal
`[history_video.end - tokens_per_latent, history_video.end)`; any selection
outside that range, duplicate position, domain override, missing segment
identity, non-contiguous frame index, or cross-scene window fails the audit.

### Dynamic persistent token budget

The layer-16 hidden-sequence path supports a risk-gated dynamic budget. It
sorts non-negative excess attention-V-norm utility, then chooses 37.5%, 50%,
or the full candidate set. A compressed tier must pass both its cumulative
mass and local boundary-gap checks; otherwise selection falls back to the next
larger tier. The full tier is deliberately unconditional. Dynamic K is chosen
independently for every diffusion step, while a multi-sample batch uses the
largest proposed K so the tensor remains rectangular.

The dynamic policy can be configured directly from the command line:

```bash
cd /path/to/DriveVA-lite
python \
  videopress_framework/scripts/run_official_navsim_press.py \
  --persistent-layer-sweep 16 \
  --persistent-mode hidden_sequence \
  --persistent-scorer action_attention_vnorm \
  --persistent-selector adaptive_mass \
  --adaptive-ratios 0.375,0.5,1.0 \
  --adaptive-mass-thresholds 0.60,0.68 \
  --adaptive-gap-thresholds 0.04,0.025 \
  --max-eval-tokens 90 --rounds 2 --num-inference-steps 3 \
  --enable-nuscenes-metrics --poc-test-derived --skip-plots \
  --output-root videopress_framework/outputs/attention_vnorm_dynamic
```

Every scene record reports K averaged across diffusion steps, plus per-step K
snapshots, the K histogram, tier mass/gap diagnostics, and sequence lengths.
`action_contribution_stability` is also available for controlled ablations;
the current 90-scene POC did not support replacing attention-V-norm with it.

### Two-history-latent retention policies

The official and synthetic suite runners accept `--retention-policy` with six
auditable deletion policies:

| Policy | Previous history latent | Last history latent |
| --- | --- | --- |
| `drop_previous_keep_last_100` | drop all | keep 100% |
| `drop_previous_keep_last_50` | drop all | keep 50% |
| `joint_keep_50` | jointly ranked with last | keep 50% across both |
| `joint_keep_25` | jointly ranked with last | keep 25% across both |
| `per_latent_keep_50` | keep 50% independently | keep 50% independently |
| `per_latent_keep_25` | keep 25% independently | keep 25% independently |

These policies require exactly two VAE history latents and use the complete
`history` domain. `zero` discards unselected values and `kv_prune` physically
removes them; mean/shuffle methods remain clearly labeled perturbation
controls. Similarity merge is excluded because grouping cannot enforce the
six exact Top-K quotas. With the official 390-token latent, independent
25% retention rounds 97.5 to 98 tokens per latent. Every event reports both
selected and effective per-latent counts.

For example:

```bash
python scripts/run_official_navsim_press.py \
  --retention-policy per_latent_keep_50 \
  --methods causal_token_norm_zero,physical_token_norm_kv_prune \
  --output-root outputs/per_latent_keep_50
```

The six standalone CPU-checkable configurations are under `configs/press/`
with names beginning `history_`.

Audit scene identity, rank sharding, method isolation, and frozen probe-cache
keys after a run:

```bash
cd /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework
/home/cpj/miniconda3/envs/DriveVA/bin/python scripts/check_scene_integrity.py \
  --suite-root outputs/official_navsim_same_scene_6gpu_final
```

The audit requires every official CSV token to occur exactly once in the
matching event journal and `records.jsonl`, verifies CSV rank versus event
rank/order, checks that probe cache keys carry the same scene token, and
validates the sampled segment and candidate positions recorded by each runtime
event. The runner also refuses a pipeline call when no active scene token has
been bound.
In distributed runs, the runner resolves `--output-root` on rank 0 and
broadcasts that resolved path to every rank, so rank-local event journals and
official CSVs remain in one auditable suite. If the requested directory
already exists, either use `--allow-existing-output` intentionally or let the
runner choose a single shared `*_rerunNN` directory.

### Future video token compression

The physical runtime supports `future_video` and per-latent future domains
using the same layout / cross-layer persistence path as history. Future latent
order is fixed to storage order: `future_latent_0` is nearest to the history
block and `future_latent_1` is farther ahead.

```bash
# Generic no-training scorer on both future latents.
python scripts/run_official_navsim_press.py \
  --domain future_video \
  --persistent-layer-sweep 15 \
  --persistent-mode hidden_sequence \
  --persistent-scorer action_attention_vnorm \
  --persistent-selector topk \
  --persistent-keep-ratio 0.5 \
  --max-eval-tokens 1 --poc-test-derived \
  --output-root outputs/future_action_attention_smoke

# History-trained learned selector, zero-shot per-latent future thresholds.
python scripts/run_official_navsim_press.py \
  --domain future_video \
  --persistent-layer-sweep 15 \
  --persistent-mode hidden_sequence \
  --persistent-scorer learned_planning_selector \
  --persistent-learned-checkpoint <checkpoint.safetensors> \
  --persistent-feature-layer 15 \
  --persistent-selector future_threshold \
  --per-future-latent-thresholds 0.05,0.40 \
  --persistent-future-position-mode history_compatible \
  --output-root outputs/future_learned_smoke
```

`--persistent-future-position-mode` controls the learned selector temporal
feature:

- `storage` uses the true latent storage index (`t=2,3` for the default
  two-history/two-future layout);
- `history_compatible` remaps near/far future to the history range
  (`t=0,1`) to reduce OOD position features.

`future_quota` uses `--per-future-latent-keep-ratios` in the same near-to-far
order. `--retention-policy` remains history-only and fails loudly on future
domains. Matched random controls are available through
`--persistent-scorer random`.

Current 64-scene POC results are **not** a full-protocol conclusion: future
hard prune was not near-lossless and the zero-shot history selector did not
beat matched random. See
`outputs/future_poc64_report_20260917.md` before starting a future-selector
training run.

### History-guided future compression

Early future video latents are noise-dominated, so selecting them with a
history-trained scorer is close to random.  The history-guided path instead
runs the selector on history only and copies the retained spatial positions to
future latents:

```bash
python scripts/run_official_navsim_press.py \
  --history-guided-future-mapping union_history \
  --history-guided-layer 15 \
  --history-guided-thresholds 0.05,0.40 \
  --persistent-learned-checkpoint <history-trained-selector.safetensors> \
  --max-eval-tokens 1 --poc-test-derived \
  --output-root outputs/history_guided_future_smoke
```

Supported `--history-guided-future-mapping` values are `same_latent`,
`reverse_latent`, `nearest_history`, `oldest_history`, `union_history`,
`intersection_history`, and `majority_history`.  `same_latent` copies the
history mask one-to-one (`future latent 0 <- history latent 0`), which doubles
physical deletion when history and future have equal candidate counts.
`union_history` keeps a future position when either history latent keeps it.

`--history-guided-future-keep-ratio <r>` adds an optional per-future-latent cap:
positions inside the mapped history mask are ranked by transferred history score
and only the top `round(r * tokens_per_latent)` are kept.  This is useful for
frontier sweeps but is not required for plain mask copying.

Current 512/1024-scene POC conclusions:

- `same_latent` doubles compression but costs about 2.8 PDM points;
- `union_history` is safer but still costs about 1.2 PDM points at the default
  thresholds;
- keep-ratio caps below the natural union mask make quality worse quickly;
- the only nearly neutral guided point (`union_history [0.02,0.40]`) compresses
  less than history-only best and is slower.

This is not a deployable near-lossless future Press.  See
`outputs/history_guided_future_poc1024_report_20260917.md` and
`outputs/history_guided_future_frontier_512_report_20260917.md`.

For oracle headroom experiments, `--persistent-random-seed` fixes the random
future-mask seed.  A 64-scene best-of-N random-mask oracle found that at 50%
future retention the oracle upper bound is statistically indistinguishable from
NoPress, while 25-37.5% retention still loses quality.  See
`outputs/oracle_future_subset_upper_bound_20260917.md`.  The oracle is optimistic
by construction and is not a deployable selector.

### Token-level set-level future oracle

The 2026-09-18/20 tile oracle (12 normalized tiles per future latent, ~30 tokens
each) was not near-lossless even at keep 0.50 (`dPDM -0.0182`), and the
conclusion report attributed part of that to the tile granularity.  The
token-level instrument replaces tiles with explicit token sets:

- `videopress/oracle/token_set.py` -- token grouping (`token` / `linear` /
  `block` / `random`), set-level (jointly scored) searches: best-of-N random,
  greedy forward, greedy backward, beam, plus per-scene best-of-N aggregation;
- `videopress/oracle/metrics.py` -- trajectory displacement / planning harm /
  combined harm, so PDM and the trajectory objectives are always reported
  together;
- `videopress/selectors/future_oracle.py::OracleFutureTokenMaskSelector`
  (`oracle_future_token_mask`) -- applies a token mask as a real physical
  `hidden_sequence` prune;
- `scripts/search_future_token_set_oracle.py` -- the GPU-facing driver.

A token mask JSON maps scene token -> per-latent local token indices (a flat
list of future offsets is also accepted; missing latents mean "keep nothing"):

```json
{
  "scene-token": {
    "future_latent_0": [0, 5, 17, 389],
    "future_latent_1": [1, 2, 3]
  }
}
```

Evaluate one mask (or a whole candidate batch, one physical method per file, in
a single runner invocation):

```bash
python scripts/run_official_navsim_press.py \
  --future-oracle-token-mask-jsons outputs/cand_a.json,outputs/cand_b.json \
  --future-counterfactual-layer 15 \
  --dump-trajectories --dump-target-trajectories \
  --max-eval-tokens 64 --poc-test-derived --skip-plots \
  --output-root outputs/token_mask_eval
```

Search for a near-lossless token subset with the driver (it runs its own
no-press baseline unless `--baseline-method-dir` is given):

```bash
# Per-scene best-of-N random token masks (N samples = N physical methods).
python scripts/search_future_token_set_oracle.py \
  --mode random-best-of-n --n-samples 16 --keep-ratio 0.5 \
  --max-eval-tokens 64 --output-root outputs/token_oracle_random

# Token-group leave-one-out importance, then top-K composition.
python scripts/search_future_token_set_oracle.py \
  --mode independent-topk --group-mode linear --group-size 8 \
  --keep-ratio 0.5 --max-eval-tokens 64 --output-root outputs/token_oracle_topk

# Set-level greedy forward / backward / beam search.
python scripts/search_future_token_set_oracle.py \
  --mode greedy-forward --group-mode linear --group-size 30 \
  --keep-ratio 0.5 --max-eval-tokens 64 --output-root outputs/token_oracle_greedy
```

By default each candidate runner call passes `--persistent-skip-baseline`: the
driver already holds a baseline (external `--baseline-method-dir` or its own
first run), so re-evaluating `physical_no_press` in every candidate batch is
pure overhead.  Pass `--keep-suite-baseline` when you want each suite to be
self-contained instead.

Each mode writes `oracle_search_report.json` plus a markdown table with PDM,
PDM harm, trajectory displacement and planning harm per candidate, and records
every runner command in `search/commands.jsonl`.  Adaptive modes also evaluate a
budget-matched random control; `--objective` chooses which signal drives the
search (`pdm`, `pdm_harm`, `traj_disp`, `planning_harm`, `combined`).

The prepared 4-GPU queue is
`outputs/future_token_set_oracle_queue_20260920/run_queue.sh` (random best-of-N,
independent top-K, greedy forward, beam).  GPU work must stay queued: at most
four GPUs, never preempt a running job.


# Wan2.1-1.3B static migration audit

No checkpoint forward, training, overfit, or pruning was run for this audit.

## Backbone configurations

The registry is `examples/wanvideo/backbone_config.py`. Architecture values in
this table are validation metadata, not dimensions embedded in trajectory
layers. Loaded DiT attributes remain authoritative and are checked against the
selected entry before use.

| Field | `wan2.2-ti2v-5b` | `wan2.1-t2v-1.3b` |
| --- | --- | --- |
| model id | `Wan-AI/Wan2.2-TI2V-5B` | `Wan-AI/Wan2.1-T2V-1.3B` |
| default directory | `models/Wan-AI/Wan2.2-TI2V-5B` | `models/Wan2.1-T2V-1.3B` |
| model type | TI2V | T2V |
| VAE | `Wan2.2_VAE.pth` | `Wan2.1_VAE.pth` |
| hidden width | 3072 | 1536 |
| attention heads | 24 | 12 |
| DiT input/output channels | 48 | 16 |
| patch size | `(1, 2, 2)` | `(1, 2, 2)` |

`WanVideoPipeline.from_pretrained` constructs `TrajectoryEncoder.output_dim`
and `TrajectoryHead.dim` from the loaded `pipe.dit.dim`. Attention head count,
RoPE head width, patching, timestep projection, LayerNorm, and unpatchify all
come from the loaded Wan model configuration. No trajectory/action `Linear`,
`LayerNorm`, view, or rearrange operation fixes the hidden width at 3072 or
1536.

## Action and trajectory data flow

1. `NavsimDriveVADataset._build_sample` returns future XY-heading trajectory,
   ego velocity, history positions, full RGB video, and history RGB video.
2. `DriveVANavsimTrainingModule.forward_preprocess` forwards those fields to
   the pipeline units. `WanVideoUnit_Trajectory` initially encodes normalized
   trajectory/ego-state tokens with `TrajectoryEncoder`.
3. `WanVideoPipeline.training_loss` adds flow-matching noise to future
   trajectory points and rebuilds the tokens. Input semantics are optional
   history/velocity prefix followed by future noisy XY-heading points. Output
   semantics are hidden-width trajectory tokens. The width is the loaded DiT
   width and is compatible with either registry entry.
4. `model_fn_wan_video` patchifies video latents, flattens the runtime latent
   grid, appends trajectory tokens, extends timestep modulation and RoPE for
   those tokens, and sends the joint sequence through every `dit.blocks`
   transformer block. This path does not pool features.
5. After the final block, `model_fn_wan_video` slices the appended hidden
   states and sends them directly to `TrajectoryHead`. `TrajectoryHead` applies
   config-width LayerNorm and MLP projection to three trajectory channels.
6. `WanVideoPipeline.training_loss` removes history/velocity prefix outputs,
   compares future trajectory flow predictions with the scheduler target using
   MSE, combines trajectory and video losses, and applies scheduler weighting.
7. During inference, the same head predicts the trajectory scheduler update at
   every denoising step. The final normalized points are denormalized, then
   `eval_navsim_pdm.py` converts them to NAVSIM `Trajectory` objects for PDM.

## Conditioning migration

The dataset preserves DriveVA semantics: history frames include the current
frame as their final image, while the full video adds future target frames.
Both backbones receive the full video through the native VAE during training.
History frames are separately VAE-encoded by `WanVideoUnit_LongCatVideo` and
replace the leading noisy video latents. Their per-token timestep is zero;
future video and trajectory tokens are denoised normally. Text enters through
UMT5 context, ego state through the trajectory velocity prefix, and future
trajectory through noisy action tokens.

Wan2.2 TI2V additionally supports first-image latent fusion
(`fuse_vae_embedding_in_latents`) and its checkpoint has 48 DiT input/output
channels. Wan2.1 T2V has no TI2V image/CLIP condition and has 16 DiT channels.
The minimal adapter does not synthesize `y` or CLIP tensors: it preserves visual
history through clean VAE history latents instead. Training does not pass
`input_image`, so the TI2V-only image units are inactive for this data path.

Runtime validation is still required for native Wan2.1 VAE temporal length,
history replacement boundaries, per-token timestep broadcasting, joint RoPE,
checkpoint state-dict recognition, memory capacity, and gradient flow through
the trajectory path.

## Hard-coded assumption audit

- `diffsynth/models/wan_video_dit.py`: 3072/24 and 1536/12 occur only in the
  state-dict-to-Wan-config converter. They are required architecture detection,
  not action-head hard coding; retained.
- `diffsynth/pipelines/wan_video_new.py`: legacy default model id and TI2V image
  unit remain for backward compatibility. DriveVA entrypoints now pass explicit
  registry-built configs; retained.
- Train, NAVSIM, nuScenes, and Bench2Drive entrypoints previously repeated the
  5B model id and VAE filename. They now use the shared registry. nuScenes and
  Bench2Drive keep 5B as their default; NAVSIM train/eval default to 1.3B.
- `TrajectoryEncoder` had a generic default output width of 4096, but the
  pipeline always supplies `pipe.dit.dim`; no 5B dependency was found.
- No `Linear(3072, ...)`, `Linear(1536, ...)`, `LayerNorm(3072)`, fixed hidden
  reshape, fixed action query width, or fixed trajectory decoder width exists
  in the active DriveVA path.

## Dataset audit

All fixed roots exist. Sizes observed: train metadata 2.7G, test metadata 974M,
train sensors 240G, test sensors 121G, metric cache 3.1G. Metadata counts are
3768 train and 1920 test.

With deterministic seed 20260926, 20 files from each split were parsed. All 40
selected files were valid non-empty frame lists with valid token, log, and
camera dictionaries. Across 1600 frames, all 6400 camera paths resolved. Lidar
paths did not resolve for 800/800 sampled train frames and 781/800 sampled test
frames. DriveVA's current camera-only feature path does not consume lidar, but
the discrepancy remains a data limitation for any lidar-enabled configuration.

The training dataset receives `navsim_log_path` and `sensor_blobs_path` directly
from CLI arguments, so the documented smoke command wires the fixed roots
without copying data.

## Runtime gate

`smoke_wan21_1p3b.py` validates all native model files and rejects any
`.incomplete` download before constructing the model. Runtime diagnostics are
enabled only by `--debug-shapes`; there is no synthetic fallback.

# DriveVA Wan2.1-T2V-1.3B baseline

This branch is a fresh DriveVA baseline. It uses the official Wan2.1 T2V
components and never loads the released DriveVA Wan2.2-5B checkpoint or a
selector checkpoint.

## Required model layout

Place the official `Wan-AI/Wan2.1-T2V-1.3B` repository here:

`models/Wan2.1-T2V-1.3B/`

The directory must contain `diffusion_pytorch_model*.safetensors`,
`Wan2.1_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, and `google/` tokenizer
files. No Wan2.2 file may be used as a substitute.

## Smoke test

```bash
/root/miniconda3/envs/driveva/bin/python examples/wanvideo/driveva_train/smoke_wan21_1p3b.py \
  --repo_root "$PWD" \
  --navsim_log_path /mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/train \
  --sensor_blobs_path /mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/extra_trainval_32/openscene-v1.1/sensor_blobs/trainval \
  --local_model_path "$PWD/models" \
  --output_path "$PWD/outputs/wan21_1p3b_baseline/smoke" \
  --debug-shapes
```

The smoke script has no mocked tensors and exits with an error until the native
checkpoint and a readable NAVSIM sample are available.

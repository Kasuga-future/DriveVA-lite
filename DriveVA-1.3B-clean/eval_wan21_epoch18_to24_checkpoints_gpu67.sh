#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/xiangyike/DriveVA-official-base
export PATH=/root/miniconda3/envs/driveva/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHON=/root/miniconda3/envs/driveva/bin/python
export CUDA_VISIBLE_DEVICES=6,7
export GPUS_PER_NODE=2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export BACKBONE_TYPE=wan2.1-t2v-1.3b
export LOCAL_MODEL_PATH=/mnt/nvme/xiangyike/DriveVA-1.3B-clean/models
export NAVSIM_LOG_PATH=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/test
export NAVSIM_SENSOR_BLOBS_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/sensor_blobs/test
export NAVSIM_METRIC_CACHE_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full
export NUPLAN_MAPS_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0
export NUPLAN_DATA_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1
export INFER_TRAJECTORY_ONLY=1
export SAVE_VIZ=0
export SHOW_EVAL_PROGRESS=1
export NUM_INFERENCE_STEPS=3

root=/mnt/nvme/xiangyike/DriveVA-official-base/outputs/wan21_audit_full3768_epoch16_to24_8gpu
for step in 8478 9420 10362 11304; do
  export FULL_CKPT="${root}/step-${step}-ema.safetensors"
  export OUTPUT_DIR="${root}/eval_full_step${step}_ema_gpu67"
  echo "[checkpoint-eval] start step=${step} time=$(date --iso-8601=seconds)"
  bash examples/wanvideo/driveva_infer/scripts/eval_navsim_v1.sh \
    > "${root}/eval_full_step${step}_ema_gpu67.log" 2>&1
  echo "[checkpoint-eval] done step=${step} time=$(date --iso-8601=seconds)"
done

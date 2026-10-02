#!/usr/bin/env bash
set -euo pipefail

cd /mnt/nvme/xiangyike/DriveVA-official-base
export PATH=/root/miniconda3/envs/driveva/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHON=/root/miniconda3/envs/driveva/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export BACKBONE_TYPE=wan2.1-t2v-1.3b
export LOCAL_MODEL_PATH=/mnt/nvme/xiangyike/DriveVA-1.3B-clean/models
export NAVSIM_LOG_PATH=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/train
export SENSOR_BLOBS_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/extra_trainval_32/openscene-v1.1/sensor_blobs/trainval
export CACHE_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full
export NUPLAN_MAPS_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0
export NUPLAN_DATA_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1

base_output=/mnt/nvme/xiangyike/DriveVA-official-base/outputs/wan21_audit_full3768_16epoch_8gpu
export OUTPUT_PATH=/mnt/nvme/xiangyike/DriveVA-official-base/outputs/wan21_audit_full3768_epoch16_to24_8gpu
export FULL_CKPT="${base_output}/step-7536.safetensors"
export EMA_CHECKPOINT="${base_output}/step-7536-ema.safetensors"
export RESUME_TRAINING_STATE="${base_output}/latest-training-state.pt"

export MAX_SCENES=3768
export NUM_EPOCHS=8
export SAVE_STEPS=3768
export CHECKPOINT_STEPS=8478,9420,10362,11304
export LOG_EVERY_STEPS=50
export DATASET_NUM_WORKERS=4
export TRAINABLE_MODELS=dit,trajectory_encoder,trajectory_head
export LR=1e-4
export LR_SCHEDULER_TYPE=constant
export WARMUP_STEPS=100
export WARMUP_START_FACTOR=0.001
export WEIGHT_DECAY=0.01
export GRADIENT_ACCUMULATION_STEPS=1
export USE_GRADIENT_CHECKPOINTING=1
export USE_EMA=1
export EMA_DECAY=0.999
export SAVE_EMA=1
export SAVE_RAW_CKPT=1
export AUTO_EVAL=0

bash examples/wanvideo/driveva_train/scripts/train_navsim_v1.sh

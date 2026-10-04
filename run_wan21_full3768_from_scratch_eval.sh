#!/usr/bin/env bash
# Usage:
#   bash run_wan21_full3768_from_scratch_eval.sh \
#     <num_epochs> <eval_every_epochs> <output_dir> [gpu_list]
# Example:
#   bash run_wan21_full3768_from_scratch_eval.sh \
#     8 2 outputs/wan21_audit_full3768_scratch_8epoch 0,1,2,3,4,5,6,7

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <num_epochs> <eval_every_epochs> <output_dir> [gpu_list]" >&2
    exit 2
fi

num_epochs=$1
eval_every_epochs=$2
output_dir=$3
gpu_list=${4:-0,1,2,3,4,5,6,7}
steps_per_epoch=${STEPS_PER_EPOCH:-471}

if ! [[ $num_epochs =~ ^[1-9][0-9]*$ && $eval_every_epochs =~ ^[1-9][0-9]*$ && $steps_per_epoch =~ ^[1-9][0-9]*$ ]]; then
    echo "num_epochs, eval_every_epochs, and STEPS_PER_EPOCH must be positive integers" >&2
    exit 2
fi
if [[ -e $output_dir ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | rg -q .; then
    echo "Output directory is not empty: $output_dir" >&2
    exit 1
fi

checkpoint_steps=()
for ((epoch=eval_every_epochs; epoch<=num_epochs; epoch+=eval_every_epochs)); do
    checkpoint_steps+=( $((epoch * steps_per_epoch)) )
done
if (( num_epochs % eval_every_epochs != 0 )); then
    checkpoint_steps+=( $((num_epochs * steps_per_epoch)) )
fi
checkpoint_csv=$(IFS=,; echo "${checkpoint_steps[*]}")

repo=/mnt/nvme/xiangyike/DriveVA-official-base
mkdir -p "$output_dir"
cd "$repo"
export PATH=/root/miniconda3/envs/driveva/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHON=/root/miniconda3/envs/driveva/bin/python
export CUDA_VISIBLE_DEVICES="$gpu_list" GPUS_PER_NODE=8 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export BACKBONE_TYPE=wan2.1-t2v-1.3b LOCAL_MODEL_PATH=/mnt/nvme/xiangyike/DriveVA-1.3B-clean/models
export NAVSIM_LOG_PATH=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/train
export SENSOR_BLOBS_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/extra_trainval_32/openscene-v1.1/sensor_blobs/trainval
export CACHE_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full
export NUPLAN_MAPS_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0 NUPLAN_DATA_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1
export OUTPUT_PATH="$output_dir" MAX_SCENES=3768 NUM_EPOCHS="$num_epochs" SAVE_STEPS="$steps_per_epoch" CHECKPOINT_STEPS="$checkpoint_csv"
export LOG_EVERY_STEPS=50 DATASET_NUM_WORKERS=4 TRAINABLE_MODELS=dit,trajectory_encoder,trajectory_head
export LR=1e-4 LR_SCHEDULER_TYPE=constant WARMUP_STEPS=100 WARMUP_START_FACTOR=0.001 WEIGHT_DECAY=0.01 GRADIENT_ACCUMULATION_STEPS=1
export USE_GRADIENT_CHECKPOINTING=1 USE_EMA=1 EMA_DECAY=0.999 SAVE_EMA=1 SAVE_RAW_CKPT=1 AUTO_EVAL=0
unset FULL_CKPT EMA_CHECKPOINT RESUME_TRAINING_STATE INITIAL_GLOBAL_STEP

echo "[from_scratch] train start $(date --iso-8601=seconds) epochs=$num_epochs checkpoints=$checkpoint_csv"
bash examples/wanvideo/driveva_train/scripts/train_navsim_v1.sh
echo "[from_scratch] train done $(date --iso-8601=seconds)"

export NAVSIM_LOG_PATH=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/test
export NAVSIM_SENSOR_BLOBS_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/sensor_blobs/test
export NAVSIM_METRIC_CACHE_PATH=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full
unset MAX_SCENES MAX_EVAL_TOKENS EVAL_MAX_EVAL_TOKENS
export INFER_TRAJECTORY_ONLY=1 SAVE_VIZ=0 SHOW_EVAL_PROGRESS=1 NUM_INFERENCE_STEPS=3
for step in "${checkpoint_steps[@]}"; do
    export FULL_CKPT="$output_dir/step-${step}-ema.safetensors" OUTPUT_DIR="$output_dir/eval_full_step${step}_ema_8gpu"
    echo "[from_scratch] eval start step=$step $(date --iso-8601=seconds)"
    bash examples/wanvideo/driveva_infer/scripts/eval_navsim_v1.sh > "$output_dir/eval_full_step${step}_ema_8gpu.log" 2>&1
    echo "[from_scratch] eval done step=$step $(date --iso-8601=seconds)"
done

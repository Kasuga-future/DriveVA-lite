#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/nvme/xiangyike/DriveVA-1.3B-clean
PYTHON=/root/miniconda3/envs/driveva/bin/python
GPU_SET=2,3,6,7
TRAIN_META=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/train
TEST_META=/mnt/chenpeijian/autodrive/DriveVA-lite/videopress_framework/outputs/navsim_split_audit/metadata/test
TRAIN_SENSOR=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/extra_trainval_32/openscene-v1.1/sensor_blobs/trainval
TEST_SENSOR=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/openscene-v1.1/sensor_blobs/test
METRIC_CACHE=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1/metric_cache_full
MAP_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/nuplan/nuplan-maps-v1.0
DATA_ROOT=/mnt/nvme/chenpeijian/autodrive/DriveVA/data/navsim_v1.1
ROOT_OUT="$REPO/outputs/baselines"
STATUS="$ROOT_OUT/background_status.log"

mkdir -p "$ROOT_OUT"
echo $$ > "$ROOT_OUT/background.pid"
echo "$(date -Is) START pid=$$ gpus=$GPU_SET" >> "$STATUS"
cd "$REPO"

run_train() {
  local name="$1"
  local config="$2"
  echo "$(date -Is) TRAIN_START $name" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU_SET" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  MPLCONFIGDIR=/tmp/driveva-matplotlib NUPLAN_MAPS_ROOT="$MAP_ROOT" NUPLAN_DATA_ROOT="$DATA_ROOT" \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  examples/wanvideo/driveva_train/train_from_config.py --config "$config"
  echo "$(date -Is) TRAIN_DONE $name" >> "$STATUS"
}

run_eval() {
  local name="$1"
  local backbone="$2"
  local model_path="$3"
  local checkpoint="$4"
  local output_dir="$5"
  echo "$(date -Is) EVAL_START $name" >> "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU_SET" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  MPLCONFIGDIR=/tmp/driveva-matplotlib NUPLAN_MAPS_ROOT="$MAP_ROOT" NUPLAN_DATA_ROOT="$DATA_ROOT" \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  examples/wanvideo/driveva_infer/eval_navsim_pdm.py \
  --repo_root "$REPO" \
  --navsim_log_path "$TEST_META" \
  --sensor_blobs_path "$TEST_SENSOR" \
  --metric_cache_path "$METRIC_CACHE" \
  --output_dir "$output_dir" \
  --local_model_path "$model_path" \
  --backbone_type "$backbone" \
  --full_ckpt "$checkpoint" \
  --height 480 --width 832 \
  --num_history_frames 5 --num_future_frames 10 --model_future_frames 8 \
  --target_fps 2 --num_inference_steps 3 --cfg_scale 1.0 --seed 42 \
  --pdm_num_poses 40 --pdm_interval_length 0.1 \
  --traffic_agents_policy non_reactive \
  --trajectory_condition_mode velocity \
  --no_infer_replace_history_latents_before_decode \
  --no_print_errors --debug_prompt_steps 0
  echo "$(date -Is) EVAL_DONE $name" >> "$STATUS"
}

WAN21_OUT="$ROOT_OUT/wan21_1p3b_4gpu"
WAN22_OUT="$ROOT_OUT/wan22_5b_4gpu"
run_train wan21_1p3b configs/baseline_wan21_1p3b_4gpu.json
run_eval wan21_1p3b wan2.1-t2v-1.3b \
  "$REPO/models/Wan2.1-T2V-1.3B" \
  "$WAN21_OUT/step-7515.safetensors" \
  "$WAN21_OUT/eval_final"

run_train wan22_5b configs/baseline_wan22_5b_4gpu.json
run_eval wan22_5b wan2.2-ti2v-5b \
  /mnt/nvme/xiangyike/DriveVA-main/models/Wan-AI/Wan2.2-TI2V-5B \
  "$WAN22_OUT/step-7515.safetensors" \
  "$WAN22_OUT/eval_final"

echo "$(date -Is) ALL_DONE" >> "$STATUS"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVEVA_TRAIN_DIR="${DRIVEVA_TRAIN_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${DRIVEVA_TRAIN_DIR}/../../.." && pwd)"}"
DRIVEVA_INFER_DIR="${DRIVEVA_INFER_DIR:-"${REPO_ROOT}/examples/wanvideo/driveva_infer"}"
CONFIG="${CONFIG:-${NAVSIM_TRAIN_CONFIG:-"${DRIVEVA_TRAIN_DIR}/configs/navsim_v1.yaml"}}"

# shellcheck source=/dev/null
source "${DRIVEVA_INFER_DIR}/scripts/distributed_env.sh"
driveva_resolve_python

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export MAX_SCENES="${MAX_SCENES:-8}"
  export NUM_EPOCHS="${NUM_EPOCHS:-1}"
  export SAVE_STEPS="${SAVE_STEPS:-2}"
  export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
  export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
  export OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/train_navsim_v1_smoke}"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[driveva] config not found: ${CONFIG}" >&2
  exit 1
fi

export REPO_ROOT DRIVEVA_TRAIN_DIR DRIVEVA_INFER_DIR CONFIG
config_exports="$("${PYTHON:-python}" "${DRIVEVA_INFER_DIR}/scripts/load_yaml_config.py" "${CONFIG}")"
eval "${config_exports}"

driveva_setup_distributed_env

export PYTHONPATH="${REPO_ROOT}:${DRIVEVA_TRAIN_DIR}:${DRIVEVA_INFER_DIR}:${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/nuscenes-devkit/python-sdk:${PYTHONPATH:-}"
export NUPLAN_MAPS_ROOT
export NUPLAN_DATA_ROOT
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${DDP_TIMEOUT_SECONDS}}"

mkdir -p "${DRIVEVA_TMPDIR}"
export TMPDIR="${TMPDIR:-${DRIVEVA_TMPDIR}}"
export TMP="${TMP:-${DRIVEVA_TMPDIR}}"
export TEMP="${TEMP:-${DRIVEVA_TMPDIR}}"

args=(
  --repo_root "${REPO_ROOT}"
  --navsim_log_path "${NAVSIM_LOG_PATH}"
  --sensor_blobs_path "${SENSOR_BLOBS_PATH}"
  --local_model_path "${LOCAL_MODEL_PATH}"
  --output_path "${OUTPUT_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_history_frames "${NUM_HISTORY_FRAMES}"
  --num_future_frames "${NUM_FUTURE_FRAMES}"
  --target_fps "${TARGET_FPS}"
  --frame_interval "${FRAME_INTERVAL}"
  --learning_rate "${LR}"
  --seed "${SEED:-20260910}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --weight_decay "${WEIGHT_DECAY}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --ddp_timeout_seconds "${DDP_TIMEOUT_SECONDS}"
  --warmup_steps "${WARMUP_STEPS}"
  --warmup_start_factor "${WARMUP_START_FACTOR}"
  --log_every_steps "${LOG_EVERY_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --train_log_file "${TRAIN_LOG_FILE}"
  --trainable_models "${TRAINABLE_MODELS}"
  --lora_base_model "${LORA_BASE_MODEL}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --extra_inputs "${EXTRA_INPUTS}"
  --trajectory_condition_mode "${TRAJECTORY_CONDITION_MODE}"
  --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
  --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}"
  --selector-warmup-steps "${SELECTOR_WARMUP_STEPS}"
  --selector-layer "${SELECTOR_LAYER:-0}"
  --selector-loss-weight "${SELECTOR_LOSS_WEIGHT}"
  --selector-input-variant "${SELECTOR_INPUT_VARIANT}"
  --selector-feature-mode "${SELECTOR_FEATURE_MODE:-all}"
  --selector-keep-schedule "${SELECTOR_KEEP_SCHEDULE}"
  --selector-teacher-keep-ratio "${SELECTOR_TEACHER_KEEP_RATIO}"
  --selector-gradient-interval "${SELECTOR_GRADIENT_INTERVAL}"
  --selector-mask-start-step "${SELECTOR_MASK_START_STEP}"
  --selector-teacher-mode "${SELECTOR_TEACHER_MODE:-gradient_abs}"
  --selector-signed-temperature "${SELECTOR_SIGNED_TEMPERATURE:-1.0}"
  --selector-counterfactual-interval "${SELECTOR_COUNTERFACTUAL_INTERVAL:-4}"
  --selector-counterfactual-weight "${SELECTOR_COUNTERFACTUAL_WEIGHT:-1.0}"
  --selector-counterfactual-scale "${SELECTOR_COUNTERFACTUAL_SCALE:-0.05}"
  --selector-counterfactual-tile-h "${SELECTOR_COUNTERFACTUAL_TILE_H:-3}"
  --selector-counterfactual-tile-w "${SELECTOR_COUNTERFACTUAL_TILE_W:-4}"
  --selector-counterfactual-injection-point "${SELECTOR_COUNTERFACTUAL_INJECTION_POINT:-post_block}"
  --selector-teacher-timesteps "${SELECTOR_TEACHER_TIMESTEPS:-}"
  --selector-counterfactual-dump-dir "${SELECTOR_COUNTERFACTUAL_DUMP_DIR:-}"
  --selector-counterfactual-replicate "${SELECTOR_COUNTERFACTUAL_REPLICATE:-0}"
  --auto_eval_ckpt_kind "${AUTO_EVAL_CKPT_KIND}"
  --infer_all_output_root "${INFER_ALL_OUTPUT_ROOT}"
)

if [[ -n "${FULL_CKPT:-}" ]]; then args+=(--full_ckpt "${FULL_CKPT}"); fi
if [[ -n "${CACHE_PATH:-}" ]]; then args+=(--cache_path "${CACHE_PATH}"); fi
if [[ -n "${TRAIN_LOG_NAMES:-}" ]]; then args+=(--train_log_names "${TRAIN_LOG_NAMES}"); fi
if [[ -n "${TRAIN_SCENE_MANIFEST:-}" ]]; then args+=(--train_scene_manifest "${TRAIN_SCENE_MANIFEST}"); fi
if [[ -n "${FORBIDDEN_SCENE_MANIFEST:-}" ]]; then args+=(--forbidden_scene_manifest "${FORBIDDEN_SCENE_MANIFEST}"); fi
if [[ "${ALLOW_MISSING_ROUTE:-0}" == "1" ]]; then args+=(--allow_missing_route); fi
args+=(--windows_per_scene "${WINDOWS_PER_SCENE:-1}")
if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max_scenes "${MAX_SCENES}"); fi
if [[ -n "${LORA_CHECKPOINT:-}" ]]; then args+=(--lora_checkpoint "${LORA_CHECKPOINT}"); fi
if [[ -n "${SELECTOR_CHECKPOINT:-}" ]]; then args+=(--selector-checkpoint "${SELECTOR_CHECKPOINT}"); fi
if [[ -n "${SELECTOR_TEACHER_SEED:-}" ]]; then args+=(--selector-teacher-seed "${SELECTOR_TEACHER_SEED}"); fi
if [[ -n "${GRADIENT_CLIP_NORM:-}" ]]; then args+=(--gradient_clip_norm "${GRADIENT_CLIP_NORM}"); fi

if [[ "${USE_CACHE_ONLY:-0}" == "1" ]]; then args+=(--use_cache_only); fi
if [[ "${FORCE_CACHE_COMPUTATION:-0}" == "1" ]]; then args+=(--force_cache_computation); fi
if [[ "${SKIP_MISSING_FILES:-0}" == "1" ]]; then args+=(--skip_missing_files); fi
if [[ "${FIND_UNUSED_PARAMETERS:-1}" == "1" ]]; then
  args+=(--find_unused_parameters)
else
  args+=(--no_find_unused_parameters)
fi
if [[ "${PRINT_NAVSIM_TOKENS:-0}" == "1" ]]; then args+=(--print_navsim_tokens); fi
if [[ "${SURROUND_VIEW:-0}" == "1" ]]; then args+=(--surround_view); fi
if [[ "${USE_TRAJECTORY:-1}" == "1" ]]; then args+=(--use_trajectory); fi
if [[ "${ENABLE_ONLINE_SELECTOR:-0}" == "1" ]]; then args+=(--enable-online-selector); fi
if [[ "${SELECTOR_ONLY:-0}" == "1" ]]; then args+=(--selector-only); fi
if [[ "${SELECTOR_COUNTERFACTUAL_PHYSICAL:-0}" == "1" ]]; then args+=(--selector-counterfactual-physical); fi
if [[ "${SELECTOR_COUNTERFACTUAL_SWEEP_ALL:-0}" == "1" ]]; then args+=(--selector-counterfactual-sweep-all); fi
args+=(--selector-counterfactual-replays "${SELECTOR_COUNTERFACTUAL_REPLAYS:-1}")
if [[ -n "${SELECTOR_COUNTERFACTUAL_SCALES:-}" ]]; then args+=(--selector-counterfactual-scales "${SELECTOR_COUNTERFACTUAL_SCALES}"); fi
if [[ -n "${SELECTOR_COUNTERFACTUAL_ABSTAIN_EPS:-}" ]]; then args+=(--selector-counterfactual-abstain-eps "${SELECTOR_COUNTERFACTUAL_ABSTAIN_EPS}"); fi
if [[ -n "${SELECTOR_COUNTERFACTUAL_NOISE_SEED:-}" ]]; then args+=(--selector-counterfactual-noise-seed "${SELECTOR_COUNTERFACTUAL_NOISE_SEED}"); fi
if [[ -n "${DRIVEVA_RUN_ID:-}" ]]; then args+=(--run-id "${DRIVEVA_RUN_ID}"); fi
if [[ -n "${SELECTOR_TEACHER_DISP_SCALE:-}" ]]; then args+=(--selector-teacher-disp-scale "${SELECTOR_TEACHER_DISP_SCALE}"); fi
if [[ -n "${SELECTOR_TEACHER_DISP_NORMALIZE:-}" ]]; then args+=(--selector-teacher-disp-normalize "${SELECTOR_TEACHER_DISP_NORMALIZE}"); fi
if [[ -n "${SELECTOR_TEACHER_DISP_MIN_SPREAD:-}" ]]; then args+=(--selector-teacher-disp-min-spread "${SELECTOR_TEACHER_DISP_MIN_SPREAD}"); fi
if [[ -n "${SELECTOR_RANKING_LOSS_WEIGHT:-}" ]]; then args+=(--selector-ranking-loss-weight "${SELECTOR_RANKING_LOSS_WEIGHT}"); fi
if [[ -n "${SELECTOR_RANKING_MARGIN:-}" ]]; then args+=(--selector-ranking-margin "${SELECTOR_RANKING_MARGIN}"); fi
if [[ -n "${SELECTOR_RANKING_MAX_PAIRS:-}" ]]; then args+=(--selector-ranking-max-pairs "${SELECTOR_RANKING_MAX_PAIRS}"); fi
if [[ -n "${SELECTOR_CRITICAL_TOKEN_MODE:-}" ]]; then args+=(--selector-critical-token-mode "${SELECTOR_CRITICAL_TOKEN_MODE}"); fi
if [[ -n "${SELECTOR_CRITICAL_TOKEN_DILATION:-}" ]]; then args+=(--selector-critical-token-dilation "${SELECTOR_CRITICAL_TOKEN_DILATION}"); fi
if [[ -n "${SELECTOR_LONG_HORIZON_WEIGHTS:-}" ]]; then args+=(--selector-long-horizon-weights "${SELECTOR_LONG_HORIZON_WEIGHTS}"); fi
if [[ "${USE_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  args+=(--use_gradient_checkpointing 1)
else
  args+=(--no_use_gradient_checkpointing)
fi
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-0}" == "1" ]]; then args+=(--use_gradient_checkpointing_offload); fi
if [[ "${TRAIN_FUTURE_VIDEO_NOISE_ONLY:-1}" == "1" ]]; then
  args+=(--train_future_video_noise_only 1)
else
  args+=(--no_train_future_video_noise_only)
fi
if [[ "${INFER_REPLACE_HISTORY_LATENTS_BEFORE_DECODE:-1}" == "1" ]]; then
  args+=(--infer_replace_history_latents_before_decode 1)
else
  args+=(--no_infer_replace_history_latents_before_decode)
fi
if [[ "${USE_EMA:-0}" == "1" ]]; then args+=(--use_ema); fi
if [[ "${EMA_ON_CPU:-0}" == "1" ]]; then args+=(--ema_on_cpu); fi
if [[ "${SAVE_EMA:-0}" == "1" ]]; then args+=(--save_ema); fi
if [[ "${SAVE_RAW_CKPT:-1}" != "1" ]]; then args+=(--no_save_raw_ckpt); fi
if [[ "${AUTO_EVAL:-0}" == "1" ]]; then args+=(--auto_eval); else args+=(--no_auto_eval); fi
if [[ "${AUTO_EVAL_STRICT:-0}" == "1" ]]; then args+=(--auto_eval_strict); fi
args+=(--ema_decay "${EMA_DECAY}" --ema_update_after_step "${EMA_UPDATE_AFTER_STEP}" --ema_update_every "${EMA_UPDATE_EVERY}")

echo "[driveva] config=${CONFIG}"
echo "[driveva] ckpt=${FULL_CKPT:-}"
echo "[driveva] model=${LOCAL_MODEL_PATH}"
echo "[driveva] navsim train: log=${NAVSIM_LOG_PATH} sensor=${SENSOR_BLOBS_PATH}"
echo "[driveva] train: output=${OUTPUT_PATH} epochs=${NUM_EPOCHS} save_steps=${SAVE_STEPS} lr=${LR}"
echo "[driveva] trajectory: enabled=${USE_TRAJECTORY} mode=${TRAJECTORY_CONDITION_MODE} future=${NUM_FUTURE_FRAMES}"
echo "[driveva] auto_eval: enabled=${AUTO_EVAL} kind=${AUTO_EVAL_CKPT_KIND} output=${INFER_ALL_OUTPUT_ROOT} navsim=${RUN_NAVSIM} nuscenes=${RUN_NUSCENES} b2d=${RUN_B2D}"

driveva_launch_torchrun \
  "${REPO_ROOT}/examples/wanvideo/driveva_train/train_navsim_v1.py" \
  "${args[@]}"

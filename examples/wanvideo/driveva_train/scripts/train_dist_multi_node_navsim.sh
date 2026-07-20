#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVEVA_TRAIN_DIR="${DRIVEVA_TRAIN_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
export DRIVEVA_TRAIN_DIR
export CONFIG="${CONFIG:-${NAVSIM_TRAIN_CONFIG:-"${DRIVEVA_TRAIN_DIR}/configs/navsim_v1.yaml"}}"

exec bash "${SCRIPT_DIR}/train_navsim_v1.sh" "$@"

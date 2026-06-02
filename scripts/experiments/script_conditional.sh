#!/usr/bin/env bash
set -euo pipefail

# Train + sample + evaluate one conditional generation experiment.
# Default: ETTh1, prediction length 24, pacodi_ddpm backbone.

PYTHON=${PYTHON:-python}
CONFIG_FILE=${CONFIG_FILE:-config/conditional/etth1.yaml}
NAME=${NAME:-etth1_24}
GPU=${GPU:-0}
MODEL_NAME=${MODEL_NAME:-pacodi_ddpm}
MILESTONE=${MILESTONE:-5}
NUM_SAMPLES=${NUM_SAMPLES:-5}

"${PYTHON}" main.py \
  --name "${NAME}" \
  --config_file "${CONFIG_FILE}" \
  --gpu "${GPU}" \
  --mode conditional \
  --model_name "${MODEL_NAME}" \
  --train

"${PYTHON}" main.py \
  --name "${NAME}" \
  --config_file "${CONFIG_FILE}" \
  --gpu "${GPU}" \
  --mode conditional \
  --model_name "${MODEL_NAME}" \
  --milestone "${MILESTONE}" \
  --num_samples "${NUM_SAMPLES}"

"${PYTHON}" evaluate_conditional.py

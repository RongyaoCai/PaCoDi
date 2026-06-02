#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:-config/unconditional/etth1.yaml}
DATASET_NAME=${DATASET_NAME:-etth1}
MODEL_NAME=${MODEL_NAME:-pacodi_sde}
BACKBONE=${BACKBONE:-dit_solver_v1}
BACKBONE_TAG=${BACKBONE_TAG:-ditv1}
REAL_IMAG_INTERACTION=${REAL_IMAG_INTERACTION:-true}
PYTHON=${PYTHON:-python}

# seq_length gpu patch_size emb_size num_layers batch_size sample_size
EXPERIMENTS=(
  "64 0 1 128 4 128 256"
  "128 1 1 128 4 128 256"
  "256 2 1 256 4 64 128"
  "512 3 1 256 6 32 64"
)

mkdir -p logs

case "${REAL_IMAG_INTERACTION,,}" in
  true|1|yes|y|on) INTERACTION_TAG=it1 ;;
  false|0|no|n|off) INTERACTION_TAG=it0 ;;
  *)
    echo "REAL_IMAG_INTERACTION must be true/false, got ${REAL_IMAG_INTERACTION}" >&2
    exit 1
    ;;
esac

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  read -r SEQ_LENGTH GPU PATCH_SIZE EMB_SIZE NUM_LAYERS BATCH_SIZE SAMPLE_SIZE <<< "${EXPERIMENT}"
  EXP_NAME="${DATASET_NAME}_seq${SEQ_LENGTH}_${BACKBONE_TAG}_${INTERACTION_TAG}"

  if pgrep -f -- "--name ${EXP_NAME} .*--mode uncond" >/dev/null; then
    echo "SKIP ${EXP_NAME}: already running"
    continue
  fi

  LOG_FILE="logs/train_${EXP_NAME}_uncond.log"
  nohup "${PYTHON}" main.py \
    --name "${EXP_NAME}" \
    --config_file "${CONFIG_FILE}" \
    --gpu "${GPU}" \
    --mode uncond \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --tensorboard \
    --train \
    model.params.backbone "${BACKBONE}" \
    model.params.real_imag_interaction "${REAL_IMAG_INTERACTION}" \
    model.params.seq_length "${SEQ_LENGTH}" \
    model.params.patch_size "${PATCH_SIZE}" \
    model.params.emb_size "${EMB_SIZE}" \
    model.params.num_layers "${NUM_LAYERS}" \
    dataloader.train_dataset.params.window "${SEQ_LENGTH}" \
    dataloader.test_dataset.params.window "${SEQ_LENGTH}" \
    dataloader.batch_size "${BATCH_SIZE}" \
    dataloader.sample_size "${SAMPLE_SIZE}" \
    > "${LOG_FILE}" 2>&1 &

  echo "START ${EXP_NAME}: gpu=${GPU} backbone=${BACKBONE} interaction=${REAL_IMAG_INTERACTION} pid=$! log=${LOG_FILE}"
done

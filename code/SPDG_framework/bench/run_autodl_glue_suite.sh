#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$ROOT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/results/autodl_glue_suite}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_DIR/data/hf_cache}"

export HF_HOME="${HF_HOME:-$CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "[AutoDL] Project: $PROJECT_DIR"
echo "[AutoDL] Output:  $OUTPUT_DIR"
echo "[AutoDL] Cache:   $HF_HOME"
echo "[AutoDL] Device:  ${DEVICE:-cuda}"
echo "[AutoDL] Tasks:   ${TASKS:-sst2 cola mrpc rte qnli}"
echo "[AutoDL] HF API:  $HF_ENDPOINT"

cd "$PROJECT_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python code/SPDG_framework/bench/run_autodl_glue_suite.py \
  --mode "${MODE:-full}" \
  --tasks ${TASKS:-sst2 cola mrpc rte qnli} \
  --ablation-task "${ABLATION_TASK:-sst2}" \
  --device "${DEVICE:-cuda}" \
  --tokenizer-name "${TOKENIZER_NAME:-bert-base-uncased}" \
  --hf-endpoint "$HF_ENDPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-32}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --gradient-accumulation "${GRAD_ACCUM:-1}" \
  --train-limit "${TRAIN_LIMIT:-0}" \
  --eval-limit "${EVAL_LIMIT:-0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --seeds ${SEEDS:-42 43 44} \
  --d-model "${D_MODEL:-256}" \
  --n-heads "${N_HEADS:-4}" \
  --n-layers "${N_LAYERS:-2}" \
  --dim-feedforward "${DIM_FEEDFORWARD:-1024}" \
  --sparsity "${SPARSITY:-0.1}" \
  --dropout "${DROPOUT:-0.1}" \
  --run-ablation \
  --run-scaling \
  ${CHECK_ONLY:+--check-only} \
  ${FP16:+--fp16} \
  ${SAVE_CHECKPOINTS:+--save-checkpoints}

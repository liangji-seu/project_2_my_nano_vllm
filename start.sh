#!/usr/bin/env bash
set -euo pipefail

# Edit these launch settings directly in this file.
MODEL_DIR="/mnt/hdd/huggingface/Qwen2.5-0.5B-Instruct"
# 仅API别名，为兼容send_prompt.sh保留；实际模型由MODEL_DIR决定。
SERVED_MODEL_NAME="qwen2.5-0.5b"
GPU_IDS="0"
SERVER_HOST="127.0.0.1"
PORT="8010"
DTYPE="bfloat16"
TP_SIZE="1"
PP_SIZE="1"
GPU_MEMORY_UTILIZATION="0.5"
MAX_MODEL_LEN="auto"
MAX_NUM_SEQS="1"
MAX_NUM_BATCHED_TOKENS="128"
ENABLE_PREFIX_CACHING="false"
ENABLE_CUDA_GRAPH="false"
# max_tokens is a per-request output cap configured in benchmarks/send_prompt.sh.

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_DIR" >&2
  exit 1
fi

if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ && "$PP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TP_SIZE and PP_SIZE must be positive integers.\n' >&2
  exit 1
fi
IFS=',' read -r -a gpu_id_list <<< "$GPU_IDS"
if (( ${#gpu_id_list[@]} < TP_SIZE * PP_SIZE )); then
  printf 'GPU_IDS provides %d GPU(s), but TP_SIZE * PP_SIZE requires %d.\n' \
    "${#gpu_id_list[@]}" "$((TP_SIZE * PP_SIZE))" >&2
  exit 1
fi

if ! command -v my_vllm >/dev/null 2>&1; then
  printf 'my_vllm is not on PATH. Activate the project environment first: conda activate vllm_study\n' >&2
  exit 127
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
printf 'GPU IDs: %s\nModel path: %s\nAPI model alias: %s\nEndpoint: http://%s:%s\n' \
  "$GPU_IDS" "$MODEL_DIR" "$SERVED_MODEL_NAME" "$SERVER_HOST" "$PORT"

cmd=(
  my_vllm serve
  --model "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --host "$SERVER_HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --pp-size "$PP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
)
if [[ "$MAX_MODEL_LEN" != "auto" ]]; then
  cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ "$ENABLE_PREFIX_CACHING" != "true" ]]; then
  cmd+=(--disable-prefix-caching)
fi
if [[ "$ENABLE_CUDA_GRAPH" != "true" ]]; then
  cmd+=(--disable-cuda-graph)
fi

exec "${cmd[@]}"

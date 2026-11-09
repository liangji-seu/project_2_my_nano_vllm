#!/usr/bin/env bash
set -euo pipefail

# Edit these benchmark settings directly. The server in start.sh owns GPU selection.
MODEL="qwen2.5-0.5b"
TOKENIZER="/mnt/hdd/huggingface/Qwen2.5-0.5B-Instruct/tokenizer.json"
BASE_URL="http://127.0.0.1:8010"
REQUESTS=(4)
CONCURRENCY=(1)
INPUT_LENS=(64)
OUTPUT_LENS=(32)
WARMUP="1"
REPEATS="1"
DATASET="synthetic"
SEED="0"
MAX_MODEL_LEN="32768"
TIMEOUT="120"
LABEL="gpu0-single-smoke"
OUTPUT_DIR="benchmarks/results"
DRY_RUN="false"
PYTHON_BIN="python3"

usage() {
  printf 'Usage: %s [--dry-run]\n' "$0"
  printf 'Only --dry-run is accepted; edit other settings at the top of this script.\n'
}
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN="true"
  else
    usage >&2
    exit 2
  fi
done

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python was not found. Activate the project environment first: conda activate vllm_study\n' >&2
  exit 127
fi
if ! "$PYTHON_BIN" -c 'import tokenizers' >/dev/null 2>&1; then
  printf 'Python package tokenizers is unavailable. Activate the project environment: conda activate vllm_study\n' >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER" ]]; then
  printf 'Tokenizer file does not exist: %s\n' "$TOKENIZER" >&2
  exit 1
fi

cmd=("$PYTHON_BIN" benchmarks/benchmark_matrix.py
  --model "$MODEL" --tokenizer "$TOKENIZER" --base-url "$BASE_URL"
  --dataset "$DATASET" --requests "${REQUESTS[@]}"
  --concurrency "${CONCURRENCY[@]}" --input-lens "${INPUT_LENS[@]}"
  --output-lens "${OUTPUT_LENS[@]}" --seed "$SEED" --warmup "$WARMUP"
  --repeats "$REPEATS" --max-model-len "$MAX_MODEL_LEN"
  --timeout "$TIMEOUT" --label "$LABEL" --output-dir "$OUTPUT_DIR")
if [[ "$DRY_RUN" == "true" ]]; then
  cmd+=(--dry-run)
fi
printf 'Benchmark: model=%s endpoint=%s dataset=%s requests=%s concurrency=%s input=%s output=%s max_model_len=%s\n' \
  "$MODEL" "$BASE_URL" "$DATASET" "${REQUESTS[*]}" "${CONCURRENCY[*]}" \
  "${INPUT_LENS[*]}" "${OUTPUT_LENS[*]}" "$MAX_MODEL_LEN"
printf 'Mode: %s; output root: %s (each run gets its own timestamped directory)\n' \
  "$([[ "$DRY_RUN" == "true" ]] && printf 'dry-run/no HTTP' || printf 'HTTP benchmark; server must already be running')" "$OUTPUT_DIR"
printf 'Command:'
for arg in "${cmd[@]}"; do printf ' %q' "$arg"; done
printf '\n'
exec "${cmd[@]}"

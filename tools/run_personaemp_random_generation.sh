#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="${PERSONAEMP_RUN_ROOT:-$HOME/Ly/personaemp-exp1}"
REPOSITORY_ROOT="${PERSONAEMP_REPOSITORY_ROOT:-$RUN_ROOT/memory}"
DATASET="${PERSONAEMP_RANDOM_DATASET:-$RUN_ROOT/runs/splits-v1/random_test.json}"
OUTPUT_ROOT="${PERSONAEMP_RANDOM_OUTPUT:-$RUN_ROOT/runs/random-qwen3-8b-ours-v1}"
SECRETS_FILE="${PERSONAEMP_SECRETS_FILE:-$RUN_ROOT/secrets/personaemp.env}"
HF_CACHE="${PERSONAEMP_HF_HOME:-$RUN_ROOT/cache/huggingface}"
MAX_STAGE_RESTARTS="${PERSONAEMP_MAX_STAGE_RESTARTS:-12}"

mkdir -p "$OUTPUT_ROOT"
cd "$REPOSITORY_ROOT"

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

export PERSONAEMP_GENERATOR_MODEL="${PERSONAEMP_GENERATOR_MODEL_OVERRIDE:-qwen3-8b}"
export PERSONAEMP_GENERATOR_ENABLE_THINKING=false
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""

# shellcheck disable=SC1091
source .venv/bin/activate
read -r -a METHODS <<< "${PERSONAEMP_METHODS:-ours}"

for ((attempt = 1; attempt <= MAX_STAGE_RESTARTS; attempt++)); do
  printf '[%s] generation attempt %d/%d\n' \
    "$(date --iso-8601=seconds)" "$attempt" "$MAX_STAGE_RESTARTS"

  if uv run --active python -m src.experiments.personaemp.cli \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_ROOT" \
    --methods "${METHODS[@]}" \
    --dataset-provenance public_reconstruction; then
    touch "$OUTPUT_ROOT/PIPELINE_COMPLETE"
    printf '%s\n' \
      'Generation complete; official criteria and two-judge evaluation require DeepSeek-v4-flash.' \
      > "$OUTPUT_ROOT/WAITING_FOR_DEEPSEEK"
    exit 0
  fi

  if ((attempt < MAX_STAGE_RESTARTS)); then
    sleep_seconds=$((attempt * 30))
    printf '[%s] retrying from checkpoint in %d seconds\n' \
      "$(date --iso-8601=seconds)" "$sleep_seconds"
    sleep "$sleep_seconds"
  fi
done

printf '[%s] generation did not complete after %d attempts\n' \
  "$(date --iso-8601=seconds)" "$MAX_STAGE_RESTARTS" >&2
exit 1

#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="${PERSONAEMP_RUN_ROOT:-$HOME/Ly/personaemp-exp1}"
REPOSITORY_ROOT="${PERSONAEMP_REPOSITORY_ROOT:-$RUN_ROOT/memory}"
OFFICIAL_ROOT="${PERSONAEMP_OFFICIAL_ROOT:-$RUN_ROOT/PersonalizedEmpathy-official}"
GENERATION_ROOT="${PERSONAEMP_GENERATION_ROOT:-$RUN_ROOT/runs/random-qwen3-8b-four-methods-v4}"
EVALUATION_ROOT="${PERSONAEMP_EVALUATION_ROOT:-$RUN_ROOT/runs/random-qwen3-8b-official-eval-v1}"
SECRETS_FILE="${PERSONAEMP_SECRETS_FILE:-$RUN_ROOT/secrets/personaemp.env}"
MAX_STAGE_RESTARTS="${PERSONAEMP_MAX_STAGE_RESTARTS:-8}"

cd "$REPOSITORY_ROOT"
set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

export PERSONAEMP_CRITERIA_API_KEY="$PERSONAEMP_GENERATOR_API_KEY"
export PERSONAEMP_CRITERIA_BASE_URL="$PERSONAEMP_GENERATOR_BASE_URL"
export PERSONAEMP_CRITERIA_MODEL=deepseek-v4-flash
export PERSONAEMP_QWEN_JUDGE_API_KEY="$PERSONAEMP_GENERATOR_API_KEY"
export PERSONAEMP_QWEN_JUDGE_BASE_URL="$PERSONAEMP_GENERATOR_BASE_URL"
export PERSONAEMP_QWEN_JUDGE_MODEL=qwen3-30b-a3b-instruct-2507
export PERSONAEMP_DEEPSEEK_JUDGE_API_KEY="$PERSONAEMP_GENERATOR_API_KEY"
export PERSONAEMP_DEEPSEEK_JUDGE_BASE_URL="$PERSONAEMP_GENERATOR_BASE_URL"
export PERSONAEMP_DEEPSEEK_JUDGE_MODEL=deepseek-v4-flash
export CUDA_VISIBLE_DEVICES=""

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p "$EVALUATION_ROOT/input/predictions"
for method in base_model memory rag ours; do
  cp "$GENERATION_ROOT/predictions/$method.json" \
    "$EVALUATION_ROOT/input/predictions/$method.json"
done

run_with_restarts() {
  local stage="$1"
  shift
  local attempt
  for ((attempt = 1; attempt <= MAX_STAGE_RESTARTS; attempt++)); do
    printf '[%s] %s attempt %d/%d\n' \
      "$(date --iso-8601=seconds)" "$stage" "$attempt" "$MAX_STAGE_RESTARTS"
    if "$@"; then
      return 0
    fi
    if ((attempt < MAX_STAGE_RESTARTS)); then
      sleep $((attempt * 30))
    fi
  done
  return 1
}

run_with_restarts criteria \
  uv run --active python -m src.experiments.personaemp.official_eval \
  prepare-criteria \
  --official-repo "$OFFICIAL_ROOT" \
  --dataset "$GENERATION_ROOT/evaluation_dataset.json" \
  --output "$EVALUATION_ROOT/criteria.json" \
  --concurrency 8 \
  --max-retries 6 || exit 1

run_with_restarts dual-judge \
  uv run --active python -m src.experiments.personaemp.official_eval \
  suite \
  --official-repo "$OFFICIAL_ROOT" \
  --dataset "$GENERATION_ROOT/evaluation_dataset.json" \
  --predictions-dir "$EVALUATION_ROOT/input/predictions" \
  --criteria "$EVALUATION_ROOT/criteria.json" \
  --output-dir "$EVALUATION_ROOT/evaluation" \
  --split-name random \
  --judge Qwen:PERSONAEMP_QWEN_JUDGE \
  --judge DeepSeek:PERSONAEMP_DEEPSEEK_JUDGE \
  --concurrency 8 \
  --resume || exit 1

touch "$EVALUATION_ROOT/PIPELINE_COMPLETE"

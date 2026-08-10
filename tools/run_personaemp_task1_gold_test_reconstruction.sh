#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PERSONAEMP_RUN_ROOT:-$HOME/linyi/personaemp-exp1}"
REPOSITORY_ROOT="${PERSONAEMP_REPOSITORY_ROOT:-$RUN_ROOT/memory}"
OFFICIAL_ROOT="${PERSONAEMP_OFFICIAL_ROOT:-$RUN_ROOT/PersonalizedEmpathy-official}"
OUTPUT_ROOT="${PERSONAEMP_OUTPUT_ROOT:-$RUN_ROOT/runs/task1-gold-test-reconstruction-v1}"
SECRETS_FILE="${PERSONAEMP_SECRETS_FILE:-$RUN_ROOT/secrets/personaemp.env}"
EXPECTED_COMMIT="${PERSONAEMP_EXPECTED_COMMIT:?set PERSONAEMP_EXPECTED_COMMIT}"
MAX_RESTARTS="${PERSONAEMP_MAX_STAGE_RESTARTS:-12}"

mkdir -p "$RUN_ROOT/logs" "$OUTPUT_ROOT"
LOG_FILE="${PERSONAEMP_LOG_FILE:-$RUN_ROOT/logs/task1-gold-test-reconstruction.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing secrets file: $SECRETS_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

: "${PERSONAEMP_MEMORY_API_KEY:?missing intent API key}"
: "${PERSONAEMP_MEMORY_BASE_URL:?missing intent base URL}"
: "${PERSONAEMP_MEMORY_MODEL:?missing intent model}"
: "${PERSONAEMP_DATA_API_KEY:?missing data API key}"
: "${PERSONAEMP_DATA_BASE_URL:?missing data base URL}"
: "${PERSONAEMP_DATA_MODEL:?missing data model}"
: "${PERSONAEMP_BIG5_API_KEY:?missing Big Five API key}"
: "${PERSONAEMP_BIG5_BASE_URL:?missing Big Five base URL}"
: "${PERSONAEMP_BIG5_MODEL:?missing Big Five model}"

[[ "$PERSONAEMP_MEMORY_MODEL" == "deepseek-v3.2" ]]
[[ "$PERSONAEMP_DATA_MODEL" == "MiniMax-M2.5" ]]
[[ "$PERSONAEMP_BIG5_MODEL" == "deepseek-v4-flash" ]]

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CURRENT_COMMIT="$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)"
OFFICIAL_COMMIT="$(git -C "$OFFICIAL_ROOT" rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Repository commit mismatch: expected $EXPECTED_COMMIT got $CURRENT_COMMIT" >&2
  exit 2
fi
if [[ "$OFFICIAL_COMMIT" != "b555447f267b8057039aab39a4be44725718ea7f" ]]; then
  echo "Official repository commit mismatch: $OFFICIAL_COMMIT" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Task 1 gold-Memory test reconstruction"
echo "repository_commit=$CURRENT_COMMIT"
echo "official_commit=$OFFICIAL_COMMIT"
echo "output_root=$OUTPUT_ROOT"
echo "scope=test-only; train_artifacts=false"

COMMAND=(
  uv run python -m src.experiments.personaemp.task1_gold_test_reconstruction
  --output-dir "$OUTPUT_ROOT"
  --official-repo "$OFFICIAL_ROOT"
  --intent-env-prefix PERSONAEMP_MEMORY
  --data-env-prefix PERSONAEMP_DATA
  --big5-env-prefix PERSONAEMP_BIG5
  --intent-workers 4
  --target-test-users 278
)

attempt=1
while (( attempt <= MAX_RESTARTS )); do
  echo "[$(date --iso-8601=seconds)] attempt=$attempt"
  if (cd "$REPOSITORY_ROOT" && "${COMMAND[@]}"); then
    touch "$OUTPUT_ROOT/DATASET_CONSTRUCTION_COMPLETE"
    echo "[$(date --iso-8601=seconds)] dataset construction complete"
    exit 0
  fi
  if (( attempt == MAX_RESTARTS )); then
    break
  fi
  wait_seconds=$((attempt * 60))
  if (( wait_seconds > 600 )); then
    wait_seconds=600
  fi
  echo "[$(date --iso-8601=seconds)] failed; retrying in ${wait_seconds}s"
  sleep "$wait_seconds"
  attempt=$((attempt + 1))
done

touch "$OUTPUT_ROOT/DATASET_CONSTRUCTION_FAILED"
echo "[$(date --iso-8601=seconds)] reconstruction exhausted retries" >&2
exit 1

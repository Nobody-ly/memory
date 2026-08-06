#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PERSONAEMP_RUN_ROOT:-$HOME/linyi/personaemp-exp1}"
REPOSITORY_ROOT="${PERSONAEMP_REPOSITORY_ROOT:-$RUN_ROOT/memory}"
OFFICIAL_ROOT="${PERSONAEMP_OFFICIAL_ROOT:-$RUN_ROOT/PersonalizedEmpathy-official}"
OUTPUT_ROOT="${PERSONAEMP_OUTPUT_ROOT:-$RUN_ROOT/runs/full-reconstruction-v1}"
SECRETS_FILE="${PERSONAEMP_SECRETS_FILE:-$RUN_ROOT/secrets/personaemp.env}"
MAX_STAGE_RESTARTS="${PERSONAEMP_MAX_STAGE_RESTARTS:-12}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cache/huggingface" "$OUTPUT_ROOT"
LOG_FILE="$RUN_ROOT/logs/full-reconstruction.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date --iso-8601=seconds)] PersonaEmp reconstruction starting"
echo "run_root=$RUN_ROOT"
echo "repository_root=$REPOSITORY_ROOT"
echo "output_root=$OUTPUT_ROOT"

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing secrets file: $SECRETS_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

: "${PERSONAEMP_GENERATOR_API_KEY:?missing generator API key}"
: "${PERSONAEMP_GENERATOR_BASE_URL:?missing generator base URL}"
: "${PERSONAEMP_GENERATOR_MODEL:?missing generator model}"

# The reconstruction uses remote model APIs and does not need a GPU.
export CUDA_VISIBLE_DEVICES=""
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ ! -d "$REPOSITORY_ROOT/.git" ]]; then
  echo "Repository is missing: $REPOSITORY_ROOT" >&2
  exit 2
fi
if [[ ! -d "$OFFICIAL_ROOT/.git" ]]; then
  echo "Official repository is missing: $OFFICIAL_ROOT" >&2
  exit 2
fi

cd "$REPOSITORY_ROOT"
CURRENT_COMMIT="$(git rev-parse HEAD)"
CURRENT_BRANCH="$(git branch --show-current)"
OFFICIAL_COMMIT="$(git -C "$OFFICIAL_ROOT" rev-parse HEAD)"
echo "repository_branch=$CURRENT_BRANCH"
echo "repository_commit=$CURRENT_COMMIT"
echo "official_commit=$OFFICIAL_COMMIT"

if [[ "$CURRENT_BRANCH" != "paper-boost/personaemp-public-reproduction" ]]; then
  echo "Unexpected experiment branch: $CURRENT_BRANCH" >&2
  exit 2
fi
if [[ "$OFFICIAL_COMMIT" != "b555447f267b8057039aab39a4be44725718ea7f" ]]; then
  echo "Unexpected official repository commit: $OFFICIAL_COMMIT" >&2
  exit 2
fi

attempt=1
while (( attempt <= MAX_STAGE_RESTARTS )); do
  echo "[$(date --iso-8601=seconds)] reconstruction attempt $attempt"
  if uv run python -m src.experiments.personaemp.reconstruction \
    --output-dir "$OUTPUT_ROOT" \
    --official-repo "$OFFICIAL_ROOT"; then
    touch "$OUTPUT_ROOT/RECONSTRUCTION_COMPLETE"
    echo "[$(date --iso-8601=seconds)] reconstruction completed"
    exit 0
  fi

  if (( attempt == MAX_STAGE_RESTARTS )); then
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

touch "$OUTPUT_ROOT/RECONSTRUCTION_FAILED"
echo "[$(date --iso-8601=seconds)] reconstruction exhausted retries" >&2
exit 1

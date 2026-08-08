#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PERSONAEMP_RUN_ROOT:-$HOME/linyi/personaemp-exp1}"
REPOSITORY_ROOT="${PERSONAEMP_REPOSITORY_ROOT:-$RUN_ROOT/memory}"
OFFICIAL_ROOT="${PERSONAEMP_OFFICIAL_ROOT:-$RUN_ROOT/PersonalizedEmpathy-official}"
OUTPUT_ROOT="${PERSONAEMP_OUTPUT_ROOT:-$RUN_ROOT/runs/full-wildchat-reconstruction-v1}"
SECRETS_FILE="${PERSONAEMP_SECRETS_FILE:-$RUN_ROOT/secrets/personaemp.env}"
MAX_STAGE_RESTARTS="${PERSONAEMP_MAX_STAGE_RESTARTS:-12}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cache/huggingface" "$OUTPUT_ROOT"
LOG_FILE="$RUN_ROOT/logs/full-wildchat-reconstruction.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date --iso-8601=seconds)] PersonaEmp WildChat reconstruction starting"
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

: "${PERSONAEMP_MEMORY_API_KEY:?missing memory API key}"
: "${PERSONAEMP_MEMORY_BASE_URL:?missing memory base URL}"
: "${PERSONAEMP_MEMORY_MODEL:?missing memory model}"
MEMORY_ONLY="${PERSONAEMP_MEMORY_ONLY:-0}"

if [[ "$MEMORY_ONLY" != "1" ]]; then
  : "${PERSONAEMP_DATA_API_KEY:?missing data-construction API key}"
  : "${PERSONAEMP_DATA_BASE_URL:?missing data-construction base URL}"
  : "${PERSONAEMP_DATA_MODEL:?missing data-construction model}"
  : "${PERSONAEMP_BIG5_API_KEY:?missing Big Five API key}"
  : "${PERSONAEMP_BIG5_BASE_URL:?missing Big Five base URL}"
  : "${PERSONAEMP_BIG5_MODEL:?missing Big Five model}"
fi

if [[ "$PERSONAEMP_MEMORY_MODEL" != "deepseek-v3.2" ]]; then
  echo "Memory model must be deepseek-v3.2, got $PERSONAEMP_MEMORY_MODEL" >&2
  exit 2
fi
if [[ "$MEMORY_ONLY" != "1" ]]; then
  if [[ "$PERSONAEMP_DATA_MODEL" != "MiniMax-M2.5" ]]; then
    echo "Data-construction model must be MiniMax-M2.5, got $PERSONAEMP_DATA_MODEL" >&2
    exit 2
  fi
  if [[ "$PERSONAEMP_BIG5_MODEL" != "deepseek-v4-flash" ]]; then
    echo "Big Five model must be deepseek-v4-flash, got $PERSONAEMP_BIG5_MODEL" >&2
    exit 2
  fi
fi

# This pipeline uses remote APIs plus a CPU E5 encoder; it must not occupy a GPU.
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

COMMAND=(
  uv run python -m src.experiments.personaemp.wildchat_full_reconstruction
  --output-dir "$OUTPUT_ROOT"
  --official-repo "$OFFICIAL_ROOT"
)

if [[ -n "${PERSONAEMP_WILDCHAT_SNAPSHOT_DIR:-}" ]]; then
  COMMAND+=(--snapshot-dir "$PERSONAEMP_WILDCHAT_SNAPSHOT_DIR")
else
  IFS=',' read -r -a PATTERNS <<< "${PERSONAEMP_WILDCHAT_DOWNLOAD_PATTERNS:-}"
  for pattern in "${PATTERNS[@]}"; do
    if [[ -n "$pattern" ]]; then
      COMMAND+=(--download-pattern "$pattern")
    fi
  done
fi
if [[ -n "${PERSONAEMP_SOURCE_LIMIT:-}" ]]; then
  COMMAND+=(--source-limit "$PERSONAEMP_SOURCE_LIMIT")
fi
if [[ "${PERSONAEMP_ENABLE_RECONSTRUCTED_SEMANTIC_DEDUP:-0}" == "1" ]]; then
  COMMAND+=(--enable-reconstructed-semantic-dedup)
fi
if [[ "$MEMORY_ONLY" == "1" ]]; then
  COMMAND+=(--memory-only)
fi
if [[ -n "${PERSONAEMP_GOLD_INPUT:-}" || -n "${PERSONAEMP_GOLD_REFERENCE:-}" ]]; then
  : "${PERSONAEMP_GOLD_INPUT:?missing gold input}"
  : "${PERSONAEMP_GOLD_REFERENCE:?missing gold reference}"
  COMMAND+=(--gold-input "$PERSONAEMP_GOLD_INPUT" --gold-reference "$PERSONAEMP_GOLD_REFERENCE")
fi

attempt=1
while (( attempt <= MAX_STAGE_RESTARTS )); do
  echo "[$(date --iso-8601=seconds)] reconstruction attempt $attempt"
  if "${COMMAND[@]}"; then
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

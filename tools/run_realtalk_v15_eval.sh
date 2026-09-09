#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 DATASET_DIR V9_PREDICTIONS V15_PREDICTIONS V9_JUDGE_CHECKPOINT OUTPUT_ROOT JUDGE_MODEL" >&2
  exit 2
fi

dataset_dir="$1"
v9_predictions="$2"
v15_predictions="$3"
v9_judge_checkpoint="$4"
output_root="$5"
judge_model="$6"

mkdir -p "${output_root}"

python -m src.experiments.realtalk_local_metrics \
  --predictions "${v9_predictions}" \
  --output-dir "${output_root}/v9_local"
python -m src.experiments.realtalk_local_metrics \
  --predictions "${v15_predictions}" \
  --output-dir "${output_root}/v15_local"

python -m src.experiments.realtalk_gpt_judge \
  --predictions "${v9_predictions}" \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_root}/v9_judge" \
  --model "${judge_model}" \
  --reference-checkpoint "${v9_judge_checkpoint}"
python -m src.experiments.realtalk_gpt_judge \
  --predictions "${v15_predictions}" \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_root}/v15_judge" \
  --model "${judge_model}" \
  --reference-checkpoint "${v9_judge_checkpoint}"

python -m src.experiments.realtalk_v15_report \
  --v9-local "${output_root}/v9_local/results_with_local_metrics.jsonl" \
  --v15-local "${output_root}/v15_local/results_with_local_metrics.jsonl" \
  --v9-gpt "${output_root}/v9_judge/scored.jsonl" \
  --v15-gpt "${output_root}/v15_judge/scored.jsonl" \
  --output-dir "${output_root}/paired_report"

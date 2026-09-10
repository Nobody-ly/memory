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

v9_matched_predictions="${output_root}/v9_predictions_matched.jsonl"
jq -c --slurpfile selected "${v15_predictions}" '
  ($selected | map(.result_id) | INDEX(.)) as $wanted
  | select($wanted[.result_id])
' "${v9_predictions}" > "${v9_matched_predictions}"

expected_count="$(wc -l < "${v15_predictions}")"
matched_count="$(wc -l < "${v9_matched_predictions}")"
if [[ "${matched_count}" -ne "${expected_count}" ]]; then
  echo "V9/V15 sample mismatch: expected ${expected_count}, matched ${matched_count}" >&2
  exit 1
fi

python -m src.experiments.realtalk_local_metrics \
  --predictions "${v9_matched_predictions}" \
  --output-dir "${output_root}/v9_local"
python -m src.experiments.realtalk_local_metrics \
  --predictions "${v15_predictions}" \
  --output-dir "${output_root}/v15_local"

v9_scored_source="$(dirname "${v9_judge_checkpoint}")/scored.jsonl"
if [[ -f "${v9_scored_source}" ]]; then
  mkdir -p "${output_root}/v9_judge"
  jq -c --slurpfile selected "${v15_predictions}" '
    ($selected | map(.result_id) | INDEX(.)) as $wanted
    | select($wanted[.result_id])
  ' "${v9_scored_source}" > "${output_root}/v9_judge/scored.jsonl"
  v9_scored_count="$(wc -l < "${output_root}/v9_judge/scored.jsonl")"
  if [[ "${v9_scored_count}" -ne "${expected_count}" ]]; then
    echo "V9 scored subset mismatch: expected ${expected_count}, matched ${v9_scored_count}" >&2
    exit 1
  fi
else
  python -m src.experiments.realtalk_gpt_judge \
    --predictions "${v9_matched_predictions}" \
    --dataset-dir "${dataset_dir}" \
    --output-dir "${output_root}/v9_judge" \
    --model "${judge_model}" \
    --reference-checkpoint "${v9_judge_checkpoint}"
fi
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

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GATE DATASET_DIR V9_RUN_DIR OUTPUT_DIR" >&2
  exit 2
fi

gate="$1"
dataset_dir="$2"
v9_run_dir="$3"
output_dir="$4"

for name in predictions.jsonl self_domains.json; do
  if [[ ! -f "${v9_run_dir}/${name}" ]]; then
    echo "missing canonical V9 artifact: ${v9_run_dir}/${name}" >&2
    exit 2
  fi
done

mode="--fresh"
if [[ -f "${output_dir}/checkpoint.json" ]]; then
  mode="--resume"
fi

python -m src.experiments.realtalk_v15 \
  --dataset-dir "${dataset_dir}" \
  --v9-predictions "${v9_run_dir}/predictions.jsonl" \
  --v9-self-domains "${v9_run_dir}/self_domains.json" \
  --output-dir "${output_dir}" \
  --gate "${gate}" \
  --model deepseek-v4-flash \
  "${mode}"

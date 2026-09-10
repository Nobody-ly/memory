#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODE GATE DATASET_DIR OUTPUT_DIR" >&2
  exit 2
fi

mode="$1"
gate="$2"
dataset_dir="$3"
output_dir="$4"

run_mode="--fresh"
if [[ -f "${output_dir}/checkpoint.json" ]]; then
  run_mode="--resume"
fi

python -m src.experiments.realtalk_evidence_conditioned \
  --mode "${mode}" \
  --gate "${gate}" \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_dir}" \
  --model deepseek-v4-flash \
  "${run_mode}"

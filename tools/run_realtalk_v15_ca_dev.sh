#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GATE DATASET_DIR OUTPUT_DIR" >&2
  exit 2
fi

gate="$1"
dataset_dir="$2"
output_dir="$3"

mode="--fresh"
if [[ -f "${output_dir}/checkpoint.json" ]]; then
  mode="--resume"
fi

python -m src.experiments.realtalk_v15_ca_dev \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_dir}" \
  --gate "${gate}" \
  --model deepseek-v4-flash \
  "${mode}"

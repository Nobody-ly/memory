#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 START_1BASED COUNT DATASET_DIR V9_RUN_DIR OUTPUT_DIR MODEL" >&2
  exit 2
fi

start_1based="$1"
count="$2"
dataset_dir="$3"
v9_run_dir="$4"
output_dir="$5"
model="$6"

if [[ ! -f "${v9_run_dir}/predictions.jsonl" ]]; then
  echo "missing V9 predictions: ${v9_run_dir}/predictions.jsonl" >&2
  exit 2
fi
if [[ ! -f "${v9_run_dir}/self_domains.json" ]]; then
  echo "missing V9 Self Domains: ${v9_run_dir}/self_domains.json" >&2
  exit 2
fi

mode="--fresh"
if [[ -f "${output_dir}/checkpoint.json" ]]; then
  mode="--resume"
fi

python -m src.experiments.realtalk_v14 \
  --dataset-dir "${dataset_dir}" \
  --v9-predictions "${v9_run_dir}/predictions.jsonl" \
  --v9-self-domains "${v9_run_dir}/self_domains.json" \
  --output-dir "${output_dir}" \
  --contiguous-window "${start_1based}" "${count}" \
  --model "${model}" \
  "${mode}"

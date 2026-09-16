#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=/amax/xidian_ty/Ly/personaemp-exp2/worktrees/realtalk-ours-v15-1015e45/.venv/bin/python
set -a
source /amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_ours.env
source /amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_judge.env
set +a
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
exec "$python_bin" -u tools/run_realtalk_v321_all_prefixes.py --output-root "$1"

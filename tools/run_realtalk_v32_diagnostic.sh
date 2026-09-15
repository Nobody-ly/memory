#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: run_realtalk_v32_diagnostic.sh OUTPUT_ROOT COUNT' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python_bin=/amax/xidian_ty/Ly/personaemp-exp2/worktrees/realtalk-ours-v15-1015e45/.venv/bin/python
test -x "$python_bin"
set -a
source /amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_ours.env
source /amax/xidian_ty/Ly/personaemp-exp2/secrets/realtalk_judge.env
set +a
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.
exec "$python_bin" -u tools/run_realtalk_v32_diagnostic.py --output-root "$1" --count "$2"

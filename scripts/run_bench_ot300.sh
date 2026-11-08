#!/usr/bin/env bash
# Re-run PIOT-B/U/A on the twelve benchmark tasks with the relaxed OT solver
# (300 iterations / tol 1e-7, the WDC-grid setting), on four 3090s.
#
# 3 methods x 12 tasks x 5 seeds = 180 runs; four workers pinned to GPUs 0-3,
# each with a 3-task shard (45 runs; big tasks spread across shards).
# Results land in results_ot300/ — a FRESH root: run identities do not
# include solver settings, so pointing at the old results root would mark
# the existing runs as completed and silently skip these re-runs.
# Resume-safe: completed runs are skipped on re-run; per-instance manifests
# (--instance-tag) avoid concurrent writes to the shared run log.
#
# Usage:
#   conda activate daer2 && cd <project root>
#   tmux new -s bench
#   bash scripts/run_bench_ot300.sh              # GPUs 4 5 6 7 (default)
#   GPUS="0 1 2 3" bash scripts/run_bench_ot300.sh

set -u
PY=${PY:-python}
# four GPU ids to use; default to the four idle cards 4-7
read -r -a GPU_IDS <<< "${GPUS:-4 5 6 7}"
mkdir -p logs

run_shard() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu $PY run_experiments.py \
    --manifest manifests/experiment_A_ot300.json \
    --methods ot_balanced ot_unbalanced ot_auto \
    --tasks "$@" \
    --results-root results_ot300 \
    --instance-tag "gpu${gpu}" \
    > "logs/bench_ot300_gpu${gpu}.log" 2>&1
  echo "[gpu $gpu] shard done: $*"
}

run_shard "${GPU_IDS[0]}" T1 T5 T9  &
run_shard "${GPU_IDS[1]}" T2 T6 T10 &
run_shard "${GPU_IDS[2]}" T3 T7 T11 &
run_shard "${GPU_IDS[3]}" T4 T8 T12 &
wait
echo "== all 4 shards done (gpus: ${GPU_IDS[*]}) =="

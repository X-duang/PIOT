#!/usr/bin/env bash
# Faithful DeepJDOT: original joint feature-label OT with fixed-plan updates.
# 12 tasks x 5 seeds = 60 runs; 20 epochs, final checkpoint, no target-label selection.
set -u
PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p logs

run_shard() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu $PY run_experiments.py \
    --manifest manifests/experiment_A_deepjdot_faithful_ep20.json \
    --methods deepjdot_faithful \
    --tasks "$@" \
    --results-root results/results_deepjdot_faithful_lr15_ep20 \
    --instance-tag "djf20_gpu${gpu}" \
    > "logs/deepjdot_faithful_ep20_gpu${gpu}.log" 2>&1
  echo "[gpu $gpu] faithful ep20 shard done: $*"
}

run_shard "${GPU_IDS[0]}" T2     &
run_shard "${GPU_IDS[1]}" T8     &
run_shard "${GPU_IDS[2]}" T1     &
run_shard "${GPU_IDS[3]}" T7     &
run_shard "${GPU_IDS[4]}" T3 T12 &
run_shard "${GPU_IDS[5]}" T4 T10 &
run_shard "${GPU_IDS[6]}" T5 T11 &
run_shard "${GPU_IDS[7]}" T9 T6  &
wait
echo "== faithful DeepJDOT ep20 done (gpus: ${GPU_IDS[*]}) =="

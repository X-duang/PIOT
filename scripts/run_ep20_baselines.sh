#!/usr/bin/env bash
# 12 directed transfers x 5 seeds x 2 methods = 120 runs.
# MMD: 20 epochs. InvGAN+KD: 5 source pretraining + 15 adaptation epochs.
# Both use the final checkpoint, with no target-validation selection.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
read -r -a gpu_ids <<< "${GPUS:-0 1 2 3 4 5 6 7}"
if (( ${#gpu_ids[@]} != 8 )); then echo 'GPUS must contain exactly eight GPU IDs' >&2; exit 2; fi
if [[ ! -f utils.py && ! -f utils/__init__.py ]]; then
  echo 'Missing utils.py or utils package in research root; restore the server runtime before running.' >&2
  exit 2
fi
mkdir -p logs
root=${RESULTS_ROOT:-results_ep20_mmd_advda}
run_shard() {
  local gpu=$1; shift
  echo "[gpu $gpu] starting: $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u run_experiments.py \
    --manifest manifests/experiment_A_ep20_baselines.json \
    --methods mmd invgan_kd --tasks "$@" --results-root "$root" \
    --instance-tag "gpu${gpu}" --continue-on-error \
    > "logs/ep20_baselines_gpu${gpu}.log" 2>&1
}
pids=()
run_shard "${gpu_ids[0]}" T2 & pids+=("$!")
run_shard "${gpu_ids[1]}" T8 & pids+=("$!")
run_shard "${gpu_ids[2]}" T1 & pids+=("$!")
run_shard "${gpu_ids[3]}" T7 & pids+=("$!")
run_shard "${gpu_ids[4]}" T3 T12 & pids+=("$!")
run_shard "${gpu_ids[5]}" T4 T10 & pids+=("$!")
run_shard "${gpu_ids[6]}" T5 T11 & pids+=("$!")
run_shard "${gpu_ids[7]}" T9 T6 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo 'One or more shards failed; inspect logs and rerun to resume.' >&2; exit 1; fi
echo "All shards completed. Results: $root"

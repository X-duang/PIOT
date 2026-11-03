#!/usr/bin/env bash
# Label-free checkpoint-selection check: fixed 20-epoch budget, last-epoch
# evaluation (no labeled target validation anywhere in model selection).
#
# 4 methods x 12 tasks x 5 seeds = 240 runs; eight workers pinned to 8 GPUs.
# auto 的 20 轮预算 = 5 预热 + 15 联合（代码自动扣减）；崩坏重试在
# last_epoch 模式下自然关闭（best_valid_f1=-1 不满足触发条件），协议全程
# 不接触目标标签。
#
# Results land in results_ep20/ — a FRESH root: run identities do not include
# num_epochs/selection, so pointing at an old root would silently skip runs.
# Resume-safe: completed runs are skipped on re-run; --instance-tag avoids
# concurrent writes to the shared run log.
#
# Usage:
#   conda activate daer2 && cd <project root>
#   tmux new -s ep20
#   bash scripts/run_ep20.sh                 # GPUs 0-7 (default)

set -u
PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p logs

run_shard() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu $PY run_experiments.py \
    --manifest manifests/experiment_A_ep20.json \
    --methods source_only ot_balanced ot_unbalanced ot_auto \
    --tasks "$@" \
    --results-root results_ep20 \
    --instance-tag "gpu${gpu}" \
    > "logs/ep20_gpu${gpu}.log" 2>&1
  echo "[gpu $gpu] shard done: $*"
}

# 大任务（T2/T8/T1/T7）各占一卡（25 runs），小任务两两一组（50 runs）
run_shard "${GPU_IDS[0]}" T2     &
run_shard "${GPU_IDS[1]}" T8     &
run_shard "${GPU_IDS[2]}" T1     &
run_shard "${GPU_IDS[3]}" T7     &
run_shard "${GPU_IDS[4]}" T3 T12 &
run_shard "${GPU_IDS[5]}" T4 T10 &
run_shard "${GPU_IDS[6]}" T5 T11 &
run_shard "${GPU_IDS[7]}" T9 T6  &
wait
echo "== all 8 shards done (gpus: ${GPU_IDS[*]}) =="

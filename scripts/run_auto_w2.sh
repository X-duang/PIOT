#!/usr/bin/env bash
# PIOT-A 预热 2 轮整批（协议候选：probe 预热 5 -> 2，其余与 ot300 主批一致）
# 12 任务 x 5 seeds = 60 runs；8 卡贪心装箱（按 max(源训练集, 目标池) 规模加权，
# 各卡负载 13650-13737，最大差 0.6%）。
#
# 结果落 results_auto_w2/（全新根：run 身份不含预热轮数，指向旧根会误判已完成）。
# 断点重跑：summary 已完成的 run 目录已存在时 main_ot.py 会照常重训，
# 如需跳过请手动处理；中断后直接重执行本脚本（已完成的目录可保留观察）。
#
# Usage:  conda activate daer2 && cd <project root>
#         tmux new -s autow2 && bash scripts/run_auto_w2.sh

set -u
PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p logs

src_of() { case "$1" in
  T1) echo wa1;; T2) echo ds;; T3) echo dzy;; T4) echo fz;; T5) echo b2;; T6) echo ri;;
  T7) echo ab;; T8) echo da;; T9) echo ri;; T10) echo ia;; T11) echo ia;; T12) echo b2;; esac; }
tgt_of() { case "$1" in
  T1) echo ab;; T2) echo da;; T3) echo fz;; T4) echo dzy;; T5) echo fz;; T6) echo wa1;;
  T7) echo wa1;; T8) echo ds;; T9) echo ab;; T10) echo da;; T11) echo ds;; T12) echo dzy;; esac; }

run_one() {  # $1=task $2=train_seed $3=gpu
  local task=$1 tseed=$2 gpu=$3
  if find results_auto_w2 -name "${task}_train${tseed}_split42_ot_auto_experiment_A_w2_summary.json" 2>/dev/null | grep -q .; then
    echo "[gpu $gpu] skip $task seed=$tseed (summary exists)"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu $PY main/main_ot.py \
    --method ot_auto --experiment experiment_A --tier tier_1 \
    --task "$task" --src "$(src_of "$task")" --tgt "$(tgt_of "$task")" \
    --seed 42 --train_seed "$tseed" \
    --ot_normalize --lambda 0.01 --ot_reg 0.1 \
    --ot_max_iter 300 --ot_tol 1e-7 \
    --batch_size 32 --num_epochs 40 \
    --probe_pretrain_epochs 2 \
    --analysis_size 128 --analysis_seed 42 \
    --run_tag experiment_A_w2 --results_root results_auto_w2 \
    >> "logs/autow2_gpu${gpu}.log" 2>&1
}

worker() {  # $1=gpu, 其余=task:seed 列表
  local gpu=$1; shift
  for cs in "$@"; do
    run_one "${cs%%:*}" "${cs##*:}" "$gpu"
    echo "[gpu $gpu] done $cs"
  done
  echo "[gpu $gpu] shard finished"
}

worker "${GPU_IDS[0]}" T2:0 T11:10 T10:10 T7:3000 T4:0 T4:10 T4:1000 T3:42 T12:10 T5:0 T5:3000 &
worker "${GPU_IDS[1]}" T2:10 T11:42 T10:42 T6:0 T9:42 &
worker "${GPU_IDS[2]}" T2:42 T11:1000 T10:1000 T6:10 T9:1000 &
worker "${GPU_IDS[3]}" T2:1000 T11:3000 T10:3000 T6:42 T9:3000 &
worker "${GPU_IDS[4]}" T2:3000 T1:0 T1:10 T10:0 T7:1000 T9:10 &
worker "${GPU_IDS[5]}" T8:0 T8:1000 T1:42 T7:0 T6:1000 T4:3000 T3:3000 T12:1000 T5:42 &
worker "${GPU_IDS[6]}" T8:10 T8:3000 T1:1000 T7:10 T6:3000 T3:0 T3:1000 T12:42 T5:10 &
worker "${GPU_IDS[7]}" T8:42 T11:0 T1:3000 T7:42 T9:0 T4:42 T3:10 T12:0 T12:3000 T5:1000 &
wait
echo "== all 60 auto-w2 runs done =="

#!/usr/bin/env bash
# 少预热/不预热诊断批：6 个最差 PIOT-A 用例（T4/T6/T9 脆弱任务）×
# 2 个预热档位（0 轮 / 2 轮，默认 5 轮），其余协议与 ot300 主批逐参数一致。
#
# 12 runs，每 run 约 20 min（3090），8 卡并行约 40 min。
# 结果落 results_warmup_probe/（全新根：run 身份不含预热轮数，
# 指向旧根会覆盖/误判）。断点重跑安全（summary 存在即跳过由 runner 保证，
# 本脚本直接调 main_ot.py，重跑前删目标目录即可重来单个 run）。
#
# 注意：预热 0 轮时 matcher 未训练，BBSE 大概率退化触发 fallback τ=1.0——
# 读 artifacts/probe_decision.json 的 reason 字段区分
# "rule/rule_floored"（probe 正常工作）与 "bbse_singular_fallback"（probe 退化）。
#
# Usage:  conda activate daer2 && cd <project root>
#         tmux new -s warm && bash scripts/run_warmup_probe.sh

set -u
PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p logs

CASES="T4:fz:dzy:0 T4:fz:dzy:10 T4:fz:dzy:1000 T6:ri:wa1:0 T6:ri:wa1:42 T9:ri:ab:10"

i=0
for c in $CASES; do
  task="${c%%:*}"; rest="${c#*:}"
  src="${rest%%:*}"; rest="${rest#*:}"
  tgt="${rest%%:*}"; tseed="${rest##*:}"
  for warm in 0 2; do
    gpu=${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}
    echo "[launch] $task seed=$tseed warmup=$warm -> gpu $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY main/main_ot.py \
      --method ot_auto --experiment experiment_A --tier tier_1 \
      --task "$task" --src "$src" --tgt "$tgt" \
      --seed 42 --train_seed "$tseed" \
      --ot_normalize --lambda 0.01 --ot_reg 0.1 \
      --ot_max_iter 300 --ot_tol 1e-7 \
      --batch_size 32 --num_epochs 40 \
      --probe_pretrain_epochs "$warm" \
      --analysis_size 128 --analysis_seed 42 \
      --run_tag "warmup${warm}" --results_root results_warmup_probe \
      > "logs/warmup_${task}_s${tseed}_w${warm}.log" 2>&1 &
    i=$((i + 1))
    sleep 2
  done
done
wait
echo "== all 12 warmup-probe runs done =="

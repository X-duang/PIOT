#!/usr/bin/env bash
# 第二 backbone（DistilBERT）两批对照实验：bcb = best-checkpoint（主协议），
# le = label-free（20 轮末轮）。3 任务 {T2,T3,T4} x 4 方法 x 5 seeds = 60 runs/批。
# 单卡串行（每卡同一时刻只跑一个 run），分片按任务规模加权装箱。
#
# 结果根：results_xlm_bcb / results_xlm_le（与 DistilBERT 的旧根
# results_distil_bcb / results_distil_le 分开，避免覆盖或混淆）。
# 断点续跑：summary 已存在的 run 自动跳过。
#
# Usage:
#   conda activate daer2 && cd <project root>
#   tmux new -s xlm
#   ENCODER=xlm-roberta-base bash scripts/run_distil_backbone.sh            # 主协议（bcb）
#   ENCODER=xlm-roberta-base MODE=le bash scripts/run_distil_backbone.sh    # label-free（le）

set -u
PY=${PY:-python}
MODE=${MODE:-bcb}
ENCODER=${ENCODER:-distilbert-base-multilingual-cased}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
mkdir -p logs

if [ "$MODE" = "le" ]; then
  ROOT=results_xlm_le; TAG=xlm_le; EPOCHS=20; SEL_TXT="--selection last_epoch"
else
  ROOT=results_xlm_bcb; TAG=xlm_bcb; EPOCHS=40; SEL_TXT=""
fi
echo "mode=$MODE root=$ROOT epochs=$EPOCHS encoder=$ENCODER"

src_of() { case "$1" in T2) echo ds;; T3) echo dzy;; T4) echo fz;; esac; }
tgt_of() { case "$1" in T2) echo da;; T3) echo fz;; T4) echo dzy;; esac; }

run_one() {  # $1=task $2=method $3=train_seed $4=gpu
  local task=$1 method=$2 tseed=$3 gpu=$4
  if find "$ROOT" -name "${task}_train${tseed}_split42_${method}_${TAG}_summary.json" 2>/dev/null | grep -q .; then
    echo "[gpu $gpu] skip $task $method seed=$tseed"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu $PY main/main_ot.py \
    --method "$method" --experiment experiment_A --tier tier_1 \
    --task "$task" --src "$(src_of "$task")" --tgt "$(tgt_of "$task")" \
    --seed 42 --train_seed "$tseed" \
    --encoder_model "$ENCODER" \
    --ot_normalize --lambda 0.01 --ot_reg 0.1 \
    --ot_max_iter 300 --ot_tol 1e-7 \
    --batch_size 32 --num_epochs "$EPOCHS" $SEL_TXT \
    --analysis_size 128 --analysis_seed 42 \
    --run_tag "$TAG" --results_root "$ROOT" \
    >> "logs/distil_${MODE}_gpu${gpu}.log" 2>&1
}

worker() {  # $1=gpu, 其余=task:method:seed 列表
  local gpu=$1; shift
  for cs in "$@"; do
    task="${cs%%:*}"; rest="${cs#*:}"
    method="${rest%%:*}"; tseed="${rest##*:}"
    run_one "$task" "$method" "$tseed" "$gpu"
    echo "[gpu $gpu] done $cs"
  done
  echo "[gpu $gpu] shard finished"
}

worker "${GPU_IDS[0]}" T2:source_only:3000 T3:source_only:1000 T4:source_only:1000 T2:source_only:42 T3:source_only:10 T4:source_only:10 T2:source_only:0 &
worker "${GPU_IDS[1]}" T2:ot_balanced:3000 T3:ot_balanced:1000 T4:ot_balanced:1000 T2:ot_balanced:42 T3:ot_balanced:10 T4:ot_balanced:10 T2:ot_balanced:0 &
worker "${GPU_IDS[2]}" T2:ot_unbalanced:3000 T3:ot_unbalanced:1000 T4:ot_unbalanced:1000 T2:ot_unbalanced:42 T3:ot_unbalanced:10 T4:ot_unbalanced:10 T2:ot_unbalanced:0 &
worker "${GPU_IDS[3]}" T2:ot_auto:3000 T3:ot_auto:1000 T4:ot_auto:1000 T2:ot_auto:42 T3:ot_auto:10 T4:ot_auto:10 T2:ot_auto:0 &
worker "${GPU_IDS[4]}" T3:source_only:3000 T4:source_only:3000 T2:source_only:1000 T3:source_only:42 T4:source_only:42 T2:source_only:10 T3:source_only:0 T4:source_only:0 &
worker "${GPU_IDS[5]}" T3:ot_balanced:3000 T4:ot_balanced:3000 T2:ot_balanced:1000 T3:ot_balanced:42 T4:ot_balanced:42 T2:ot_balanced:10 T3:ot_balanced:0 T4:ot_balanced:0 &
worker "${GPU_IDS[6]}" T3:ot_unbalanced:3000 T4:ot_unbalanced:3000 T2:ot_unbalanced:1000 T3:ot_unbalanced:42 T4:ot_unbalanced:42 T2:ot_unbalanced:10 T3:ot_unbalanced:0 T4:ot_unbalanced:0 &
worker "${GPU_IDS[7]}" T3:ot_auto:3000 T4:ot_auto:3000 T2:ot_auto:1000 T3:ot_auto:42 T4:ot_auto:42 T2:ot_auto:10 T3:ot_auto:0 T4:ot_auto:0 &
wait
echo "== all 60 runs done (mode=$MODE) =="

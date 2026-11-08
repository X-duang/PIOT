#!/usr/bin/env bash
# PIOT-A: 12 directed WDC tasks x six priors x three seeds = 216 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
if (( ${#GPU_IDS[@]} != 8 )); then echo "GPUS must contain eight GPU IDs" >&2; exit 1; fi
if (( $(printf "%s\n" "${GPU_IDS[@]}" | sort -u | wc -l) != 8 )); then echo "GPU IDs must be distinct" >&2; exit 1; fi
ROOT=${RESULTS_ROOT:-results_wdc_auto}
OUT="$ROOT/manifests"
mkdir -p "$OUT" "$ROOT/logs"
echo "Preparing shared resampled data and feature caches serially..."
"$PY" scripts/prepare_wdc_auto.py > "$ROOT/logs/prepare.log" 2>&1 || { echo "Preparation failed: $ROOT/logs/prepare.log" >&2; exit 1; }

run_task() {
  local gpu=$1 src=$2 tgt=$3
  echo "[$(date '+%F %T')] GPU $gpu: $src -> $tgt"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u analysis/prior_shift.py \
    --src "$src" --tgt "$tgt" --methods ot_auto \
    --levels 0.05 0.10 0.17 0.27 0.40 0.55 --seeds 0 10 42 \
    --link-replica-to-seed --reuse-target-files \
    --selection last_epoch --num-epochs 40 --batch-size 32 \
    --ot-max-iter 300 --ot-tol 1e-7 --ot-reg 0.1 --tau 1.0 --lambda 0.01 --lr 1.5e-5 \
    --analysis-size 256 --analysis-seed 0 --eval-protocol inductive \
    --results-root "$ROOT" --out "$OUT/${src}_to_${tgt}.csv" \
    > "$ROOT/logs/${src}_to_${tgt}.log" 2>&1
}
worker() {
 local gpu=$1; shift
 while (( $# )); do run_task "$gpu" "$1" "$2" || return 1; shift 2; done
}
pids=()
worker "${GPU_IDS[0]}" cameras computers cameras shoes & pids+=("$!")
worker "${GPU_IDS[1]}" cameras watches computers cameras & pids+=("$!")
worker "${GPU_IDS[2]}" computers shoes computers watches & pids+=("$!")
worker "${GPU_IDS[3]}" shoes cameras shoes computers & pids+=("$!")
worker "${GPU_IDS[4]}" shoes watches & pids+=("$!")
worker "${GPU_IDS[5]}" watches cameras & pids+=("$!")
worker "${GPU_IDS[6]}" watches computers & pids+=("$!")
worker "${GPU_IDS[7]}" watches shoes & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo "A worker failed. Inspect $ROOT/logs and rerun this script to resume." >&2; exit 1; fi
"$PY" scripts/summarize_wdc_auto.py "$ROOT"

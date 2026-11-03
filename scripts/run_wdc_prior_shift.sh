#!/usr/bin/env bash
# WDC controlled prior-shift grid (paper Section V-E companion figure).
#
# 12 directed WDC tasks x 6 prior levels x 2 methods (B/U) x 3 seeds = 432 runs.
# Resampling replica is linked to the seed index: the three runs of each
# (level, method) cell see different resampling compositions AND different
# training seeds, so mean+-std per cell covers both randomness sources at
# one third of the crossed-design cost.
#
# Layout: one worker per task, at most LANES running concurrently, all on a
# single GPU (default card 1; override with GPU=<id>). Measured VRAM is
# ~8.2 GB per run, so the default 8 lanes take ~66 GB of an 81 GB H100 with
# headroom; VRAM exhaustion is a hard crash (OOM), never a queue, so do not
# raise LANES without measuring again.
# The runner skips runs whose summary is already completed, so the script is
# safe to re-run after an interruption.
#
# Usage:
#   bash scripts/run_wdc_prior_shift.sh            # single idle card 1, 8 lanes
#   GPU=0 LANES=6 bash scripts/run_wdc_prior_shift.sh
#   SKIP_WARMUP=1 bash scripts/run_wdc_prior_shift.sh
#
# After the grid finishes, manifests are merged and aggregated automatically
# (results/experiment_B/wdc/experiment_b_summary.csv / _runs.csv).

set -u
PY=${PY:-python}
GPU=${GPU:-1}
LANES=${LANES:-8}
OUT_DIR=results/experiment_B/wdc
mkdir -p "$OUT_DIR" logs

LEVELS="0.05 0.10 0.17 0.27 0.40 0.55"
SEEDS="0 10 42"
METHODS="ot_balanced ot_unbalanced"
# Solver: measured convergence is <=~100 iterations at float32 precision;
# 300/1e-7 leaves 3x margin and changes no result at the reported precision.
OT_MAX_ITER=${OT_MAX_ITER:-300}
OT_TOL=${OT_TOL:-1e-7}

run_task() {
  local src=$1 tgt=$2 extra_methods=$3 extra_levels=$4 extra_seeds=$5 out=$6 log=$7
  CUDA_VISIBLE_DEVICES=$GPU $PY analysis/prior_shift.py \
    --src "$src" --tgt "$tgt" \
    --levels $extra_levels \
    --seeds $extra_seeds \
    --link-replica-to-seed \
    --methods $extra_methods \
    --ot-max-iter $OT_MAX_ITER \
    --ot-tol $OT_TOL \
    --results-root results \
    --out "$out" \
    > "$log" 2>&1
}

# 0) preflight: two serial warm-up runs covering all four datasets, to build
#    feature caches before the concurrent launch (also validates env, model
#    and data paths end to end). Warm-ups use seed 999, deliberately OUTSIDE
#    the grid's seed set {0,10,42}: they get their own run dirs and never
#    mask a grid cell via the completed-summary skip (their pilot-free config
#    also differs in target size, so they must never be counted as grid runs).
if [ "${SKIP_WARMUP:-0}" != "1" ]; then
  echo "== preflight warm-up (2 serial runs: cameras->computers, shoes->watches, seed 999) =="
  run_task cameras computers "ot_balanced" "0.27" "999" "$OUT_DIR/warmup.csv" logs/wdc_warmup1.log \
    || { echo "warm-up FAILED, see logs/wdc_warmup1.log"; exit 1; }
  run_task shoes watches "ot_balanced" "0.27" "999" "$OUT_DIR/warmup.csv" logs/wdc_warmup2.log \
    || { echo "warm-up FAILED, see logs/wdc_warmup2.log"; exit 1; }
fi

# 1) pre-generate all resampled target CSVs serially (identical content per
#    (target, level, replica) key), so workers never write them concurrently.
echo "== pre-generating shifted target CSVs =="
$PY - <<'EOF'
import os
from analysis.prior_shift import resample_target, _fixed_sample_size, _read_dataset, SHIFT_DIR
LEVELS = [0.05, 0.10, 0.17, 0.27, 0.40, 0.55]
for tgt in ("cameras", "computers", "shoes", "watches"):
    rows, _ = _read_dataset(os.path.join("data", tgt, tgt + ".csv"))
    pos = [r for r in rows if int(float(r["labels"])) == 1]
    neg = [r for r in rows if int(float(r["labels"])) == 0]
    size = _fixed_sample_size(pos, neg, LEVELS, len(rows))
    for lv in LEVELS:
        for rep in (0, 1, 2):
            resample_target(tgt, lv, rep,
                            os.path.join(SHIFT_DIR, f"{tgt}_prior{lv:g}_rep{rep}.csv"),
                            sample_size=size)
print("shifted CSVs ready")
EOF

# 2) full grid: one worker per task, at most LANES concurrent (VRAM is a hard
#    ceiling: OOM crashes a run, it does not queue).
TASKS="cameras:computers cameras:shoes cameras:watches
       computers:cameras computers:shoes computers:watches
       shoes:cameras shoes:computers shoes:watches
       watches:cameras watches:computers watches:shoes"
running=0
for pair in $TASKS; do
  run_task "${pair%%:*}" "${pair##*:}" "$METHODS" "$LEVELS" "$SEEDS" \
    "$OUT_DIR/${pair%%:*}_to_${pair##*:}.csv" "logs/wdc_${pair%%:*}_to_${pair##*:}.log" &
  running=$((running + 1))
  sleep 2
  if [ "$running" -ge "$LANES" ]; then
    wait -n          # bash>=4.3: reap any finished worker, freeing one lane
    running=$((running - 1))
  fi
done
wait
echo "== all 12 tasks finished (gpu $GPU); merging manifests =="

# 2) merge per-task manifests (warmup.csv is excluded by the glob) and aggregate
FIRST=$(ls "$OUT_DIR"/*_to_*.csv | head -1)
head -n 1 "$FIRST" > "$OUT_DIR/all.csv"
for f in "$OUT_DIR"/*_to_*.csv; do tail -n +2 "$f"; done >> "$OUT_DIR/all.csv"
echo "merged manifest: $OUT_DIR/all.csv"

$PY analysis/experiment_b_analysis.py \
  --manifest "$OUT_DIR/all.csv" \
  --out-prefix "$OUT_DIR/experiment_b"
echo "done. aggregated: $OUT_DIR/experiment_b_summary.csv / _runs.csv"

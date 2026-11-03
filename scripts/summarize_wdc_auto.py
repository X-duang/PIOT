#!/usr/bin/env python3
"""Validate 216 WDC PIVOT-A runs and extract F1 and pilot decisions."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_wdc_auto")
files = sorted((root / "manifests").glob("*_to_*.csv"))
if len(files) != 12:
    raise SystemExit(f"Expected 12 task manifests; found {len(files)}")
records, keys, errors = [], set(), []
for path in files:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 18:
        errors.append(f"{path}: expected 18 rows, found {len(rows)}")
    for r in rows:
        key = (r["source"], r["target"], r["target_prior_requested"], r["seed"])
        if key in keys: errors.append(f"duplicate: {key}")
        keys.add(key)
        run = Path(r["run_dir"])
        summaries = sorted(run.glob("*_summary.json"))
        decision = run / "artifacts" / "probe_decision.json"
        if r["status"] not in ("completed", "already_completed") or len(summaries) != 1 or not decision.is_file():
            errors.append(f"incomplete run: {key} ({run})")
            continue
        data = json.loads(summaries[0].read_text(encoding="utf-8"))
        pilot = json.loads(decision.read_text(encoding="utf-8"))
        hist = data.get("history", [])
        if data.get("status") != "completed" or data.get("num_epochs") != 40 or not hist or any(h.get("selection") != "last_epoch_no_target_label_selection" for h in hist):
            errors.append(f"unexpected evaluation protocol: {key}")
        final = data.get("final", {})
        records.append({"source":r["source"], "target":r["target"], "target_prior":r["target_prior_actual"], "source_prior":r["source_prior"], "seed":r["seed"], "replica":r["replica"], "f1":final.get("final_f1"), "precision":final.get("final_precision"), "recall":final.get("final_recall"), "mode":pilot.get("variant"), "tau":pilot.get("tau"), "estimated_target_prior":pilot.get("pi_t_hat"), "estimated_source_prior":pilot.get("pi_s"), "bootstrap_halfwidth":pilot.get("conf_halfwidth"), "estimated_gap":pilot.get("rho_hat"), "cross_cost":pilot.get("c_cross"), "reason":pilot.get("reason"), "run_dir":str(run)})
if len(keys) != 216 or len(records) != 216: errors.append(f"expected 216 distinct completed runs; got {len(keys)} keys and {len(records)} records")
if errors:
    print("\n".join(errors[:50]), file=sys.stderr)
    raise SystemExit(1)
out = root / "wdc_auto_runs.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
print(f"Verified 216 PIVOT-A runs; wrote {out}")

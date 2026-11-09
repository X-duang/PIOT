"""Aggregate the released benchmark F1 values without training dependencies."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "results_summary" / "benchmark_per_run.csv"
runs = defaultdict(dict)
with path.open(encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream):
        key = (row["protocol"], row["task_id"], row["transfer"], row["method"])
        seed = int(row["training_seed"])
        if seed in runs[key]:
            raise ValueError(f"Duplicate seed {seed} in {key}")
        runs[key][seed] = float(row["test_f1"])

expected = {0, 10, 42, 1000, 3000}
for key, values in sorted(runs.items()):
    if set(values) != expected:
        raise ValueError(f"Unexpected seeds for {key}: {sorted(values)}")
    print(*key, f"{statistics.mean(values.values()):.4f}",
          f"{statistics.stdev(values.values()):.4f}", sep="\t")

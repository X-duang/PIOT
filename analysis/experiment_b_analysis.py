"""Aggregate existing Experiment B artifacts into paper-ready data.

This module never trains a model. It reads the prior-shift run manifest and
the artifacts emitted by the final OT runtime, then writes per-run and
per-prior-level CSV/JSON files for later plotting.
"""
import argparse
import csv
import json
import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


NUMERIC_FIELDS = (
    "f1",
    "precision",
    "recall",
    "match_recall",
    "discrepancy_before",
    "discrepancy_after",
    "cross_class_mass",
    "cross_class_cost",
    "m_hat",
    "empirical_delta_pi",
    "empirical_floor",
    "rho_star",
    "abs_delta_pi_actual",
)


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_summary(run_dir):
    if not os.path.isdir(run_dir):
        return ""
    for name in sorted(os.listdir(run_dir)):
        if name.endswith("_summary.json"):
            return os.path.join(run_dir, name)
    return ""


def _load_summary(run_dir):
    path = _find_summary(run_dir)
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_transport_stats(run_dir, tau, margin_run_dir=""):
    plan_path = os.path.join(
        run_dir, "artifacts", "transport_plan.pt"
    )
    if not os.path.exists(plan_path):
        return {}
    stats_path = os.path.join(
        run_dir, "artifacts", "transport_stats.json"
    )
    if os.path.exists(stats_path):
        with open(stats_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if not margin_run_dir:
            return cached
        if cached.get("margin_reference_run_dir") == margin_run_dir:
            return cached
    if not margin_run_dir:
        raise FileNotFoundError(
            "matched source_only run is required for Experiment B floor margin"
        )
    from analysis.transport_stats import compute_floor_stats

    stats = compute_floor_stats(
        run_dir, tau_s=float(tau), margin_run_dir=margin_run_dir
    )
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    return stats


def _load_analysis_metadata(run_dir):
    path = os.path.join(
        run_dir, "artifacts", "analysis_metadata.json"
    )
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _source_only_key(row):
    return (
        row.get("source"),
        row.get("target"),
        row.get("target_file"),
        row.get("replica"),
        row.get("seed"),
    )


def _source_only_index(rows):
    return {
        _source_only_key(row): row.get("run_dir", "")
        for row in rows
        if row.get("method") == "source_only"
    }


def _run_record(row, source_only_runs=None, discrepancy_projections=200):
    run_dir = row.get("run_dir", "")
    record = dict(row)
    record["run_dir"] = run_dir
    if not run_dir or not os.path.isdir(run_dir):
        record["status"] = "MISSING_RUN_DIR"
        return record

    summary = _load_summary(run_dir)
    analysis_metadata = _load_analysis_metadata(run_dir)
    final = summary.get("final") or {}
    record.update({
        "status": summary.get("status", "missing_summary"),
        "f1": final.get("final_f1"),
        "precision": final.get("final_precision"),
        "recall": final.get("final_recall"),
        "match_recall": final.get("match_recall"),
    })
    before_run_dir = (source_only_runs or {}).get(
        _source_only_key(row), ""
    )

    try:
        from analysis.discrepancy import analyze_pair

        if row.get("method") == "source_only":
            before_run_dir = run_dir
        if before_run_dir:
            discrepancy = analyze_pair(
                before_run_dir,
                run_dir,
                projections=discrepancy_projections,
                seed=int(row.get("analysis_seed", 42)),
            )
        else:
            raise FileNotFoundError(
                "matched source_only run is required for paper discrepancy"
            )
        record.update({
            "discrepancy_before": discrepancy["before_discrepancy"],
            "discrepancy_after": discrepancy["after_discrepancy"],
            "discrepancy_reduction": discrepancy["discrepancy_reduction"],
            "discrepancy_before_run_dir": discrepancy["before_run_dir"],
            "discrepancy_after_run_dir": discrepancy["after_run_dir"],
            "discrepancy_before_state": discrepancy["before_state"],
            "discrepancy_after_state": discrepancy["after_state"],
        })
    except (
        FileNotFoundError, ImportError, OSError, RuntimeError, ValueError
    ) as exc:
        record["discrepancy_error"] = str(exc)

    try:
        stats = _load_transport_stats(
            run_dir, float(row.get("tau", 1.0)), before_run_dir
        )
        record.update({
            "cross_class_mass": stats.get("cross_class_mass"),
            "cross_class_cost": stats.get("cross_class_cost"),
            "m_hat": stats.get("m_hat"),
            "empirical_delta_pi": stats.get("abs_delta_pi_hat"),
            "empirical_floor": stats.get("empirical_floor"),
            "rho_star": stats.get("rho_star"),
            "analysis_size_source": stats.get("analysis_size_source"),
            "analysis_size_target": stats.get("analysis_size_target"),
            "analysis_seed_artifact": stats.get("analysis_seed"),
        })
    except (
        FileNotFoundError, ImportError, OSError, RuntimeError, ValueError
    ) as exc:
        record["transport_stats_error"] = str(exc)
    if record.get("empirical_delta_pi") in (None, ""):
        record["empirical_delta_pi"] = analysis_metadata.get(
            "abs_delta_pi_hat"
        )
    if record.get("empirical_delta_pi") in (None, ""):
        record["empirical_delta_pi"] = row.get("abs_delta_pi_actual")
    record["abs_delta_pi_actual"] = row.get("abs_delta_pi_actual")
    record["pi_hat_s"] = analysis_metadata.get("pi_hat_s")
    record["pi_hat_t"] = analysis_metadata.get("pi_hat_t")
    record["analysis_size_source"] = (
        record.get("analysis_size_source")
        or analysis_metadata.get("analysis_size_actual_source")
    )
    record["analysis_size_target"] = (
        record.get("analysis_size_target")
        or analysis_metadata.get("analysis_size_actual_target")
    )
    record["analysis_seed_artifact"] = (
        record.get("analysis_seed_artifact")
        or analysis_metadata.get("analysis_seed")
    )
    return record


def _mean_std(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _group_key(row):
    return (
        row.get("target_prior_requested"),
        row.get("target_prior_actual"),
        row.get("method"),
    )


def _aggregate(rows):
    rows = [
        row for row in rows
        if row.get("status") == "completed"
    ]
    groups = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    summary = []
    for key, grouped in sorted(
        groups.items(),
        key=lambda item: (float(item[0][0]), str(item[0][2])),
    ):
        requested, actual, method = key
        output = {
            "target_prior_requested": requested,
            "target_prior_actual": actual,
            "method": method,
            "n_runs": len(grouped),
            "seeds": "|".join(
                str(row.get("seed")) for row in grouped
            ),
        }
        for field in NUMERIC_FIELDS:
            mean, std = _mean_std(
                [row.get(field) for row in grouped]
            )
            output[field + "_mean"] = mean
            output[field + "_std"] = std
        summary.append(output)

    by_level = {}
    for row in summary:
        level = (
            row["target_prior_requested"],
            row["target_prior_actual"],
        )
        by_level.setdefault(level, {})[row["method"]] = row

    for row in summary:
        level = (
            row["target_prior_requested"],
            row["target_prior_actual"],
        )
        balanced = by_level[level].get("ot_balanced")
        unbalanced = by_level[level].get("ot_unbalanced")
        if balanced and unbalanced:
            balanced_f1 = balanced.get("f1_mean")
            unbalanced_f1 = unbalanced.get("f1_mean")
            row["observed_unbalanced_minus_balanced_f1"] = (
                None
                if balanced_f1 is None or unbalanced_f1 is None
                else unbalanced_f1 - balanced_f1
            )
        else:
            row["observed_unbalanced_minus_balanced_f1"] = None
    return summary


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out-prefix",
        default="results/experiment_B/experiment_b",
    )
    parser.add_argument("--n-projections", type=int, default=200)
    args = parser.parse_args()

    manifest_rows = _load_csv(args.manifest)
    source_only_runs = _source_only_index(manifest_rows)
    run_rows = [
        _run_record(row, source_only_runs, args.n_projections)
        for row in manifest_rows
    ]
    summary_rows = _aggregate(run_rows)
    _write_csv(args.out_prefix + "_runs.csv", run_rows)
    _write_csv(args.out_prefix + "_summary.csv", summary_rows)
    with open(
        args.out_prefix + "_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)
    print(
        "wrote {} run rows and {} summary rows".format(
            len(run_rows), len(summary_rows)
        )
    )


if __name__ == "__main__":
    main()

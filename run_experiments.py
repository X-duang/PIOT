#!/usr/bin/env python3
"""Manifest-driven runner for the final Experiment A protocol.

Only methods with a verified final-protocol entrypoint are scheduled. Missing
baselines are recorded explicitly as BASELINE_MISSING; legacy runners are not
silently invoked because they do not share the final split/artifact contract.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from argparse import Namespace

from main.run_identity import (
    DEFAULT_LEARNING_RATE,
    build_run_dir,
    canonical_run_config,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(
    PROJECT_ROOT, "manifests", "experiment_A_ot300.json"
)


def load_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("experiment") != "experiment_A":
        raise ValueError("This runner currently supports Experiment A only")
    return manifest


def _method_specs(manifest, requested):
    specs = {
        item["method"]: item
        for item in manifest.get("methods", [])
    }
    if requested:
        missing = sorted(set(requested) - set(specs))
        if missing:
            raise ValueError(
                "methods absent from manifest: {}".format(", ".join(missing))
            )
        return [specs[name] for name in requested]
    return list(specs.values())


def _task_specs(manifest, requested):
    tasks = manifest.get("tasks", [])
    if not requested:
        return tasks
    selected = set(requested)
    return [task for task in tasks if task["task"] in selected]


def _methods_for_task(manifest, task, selected_methods):
    if selected_methods:
        return _method_specs(manifest, selected_methods)
    policy = manifest.get("tier_method_policy", {})
    names = policy.get(task.get("tier"), [])
    if names:
        return _method_specs(manifest, names)
    return _method_specs(manifest, None)


# Grid axes each method actually consumes; methods not listed here expand over
# every axis present in the manifest's "grid" section. Axes a method does not
# consume must not be varied for it: they would create distinct run identities
# with identical training behavior (wasted duplicate runs).
_METHOD_GRID_AXES = {
    "source_only": ("learning_rate",),
    "target_only_oracle": ("learning_rate",),
    "ot_balanced": ("alignment_weight", "learning_rate", "ot_reg"),
    "ot_unbalanced": ("alignment_weight", "tau", "learning_rate", "ot_reg"),
    # ot_auto: tau 由 probe 决策，不作为网格轴
    "ot_auto": ("alignment_weight", "learning_rate", "ot_reg"),
    "deepjdot_faithful": ("deepjdot_alpha", "deepjdot_lambda_t", "learning_rate"),
    # mmd: alignment_weight = MMD beta; invgan_kd: alignment_weight = KD beta,
    # learning_rate = 适配 lr（见 manifests/experiment_A.json method 注释）。
    "mmd": ("alignment_weight", "learning_rate"),
    "invgan_kd": ("alignment_weight", "learning_rate"),
}

_GRID_TAG_NAMES = {
    "alignment_weight": "lam",
    "learning_rate": "lr",
    "ot_reg": "reg",
    "tau": "tau",
    "deepjdot_alpha": "dja",
    "deepjdot_lambda_t": "djlt",
}


def _grid_points(manifest, method_spec):
    """Cartesian product of the grid axes relevant to this method.

    Returns [{}] (a single no-op override) when the manifest has no "grid"
    section or the method consumes none of the grid axes.
    """
    grid = manifest.get("grid") or {}
    if not grid:
        return [{}]
    axes = _METHOD_GRID_AXES.get(
        method_spec["method"], tuple(sorted(grid.keys()))
    )
    axes = [axis for axis in axes if axis in grid]
    if not axes:
        return [{}]
    return [
        dict(zip(axes, values))
        for values in itertools.product(*(grid[axis] for axis in axes))
    ]


def _grid_tag(base_tag, overrides):
    if not overrides:
        return base_tag
    fragments = [
        "{}{:g}".format(_GRID_TAG_NAMES.get(key, key), overrides[key])
        for key in sorted(overrides)
    ]
    return "{}_{}".format(base_tag, "_".join(fragments))


def _seeds_for_task(manifest, task, cli_seeds):
    if cli_seeds:
        return cli_seeds
    if manifest.get("grid"):
        return manifest.get("grid_seeds") or manifest.get("seeds") or []
    policy = manifest.get("tier_seed_policy") or {}
    return policy.get(task.get("tier")) or manifest.get("seeds") or []


def _resolved_config(manifest, task, method_spec, seed):
    defaults = dict(manifest.get("defaults", {}))
    task = dict(task)
    # Per-task, per-method hyperparameter overrides written by
    # analysis/select_grid.py into the final manifest.
    method_overrides = task.pop("method_overrides", None) or {}
    defaults.update(task)
    defaults.update(method_overrides.get(method_spec["method"]) or {})
    defaults.update({
        "experiment": manifest["experiment"],
        "seed": defaults.get("split_seed", 42),
        "train_seed": seed,
        "method": method_spec.get(
            "runtime_method", method_spec["method"]
        ),
    })
    # deepjdot/ablation_class_aware 在 main() 内强制 class_aware；在 config
    # 真源处同步，使命令行、规划身份、实际行为三者一致
    if defaults["method"] in ("ablation_class_aware",):
        defaults["cost_mode"] = "class_aware"
    if defaults["method"] == "deepjdot_faithful":
        # 原论文方法不使用 PIVOT 的外层权重或 Sinkhorn 正则；
        # 联合代价内部使用 α 与 λ_t。
        defaults["cost"] = "sq_l2"
        defaults["cost_mode"] = "class_blind"
        defaults["alignment_weight"] = 0.0
        defaults["ot_normalize"] = False
        defaults.setdefault("deepjdot_alpha", 0.001)
        defaults.setdefault("deepjdot_lambda_t", 0.0001)
        defaults.setdefault("deepjdot_solver", "emd")
    if defaults.get("learning_rate") is None:
        defaults["learning_rate"] = DEFAULT_LEARNING_RATE
    if defaults["method"] in ("source_only", "oracle"):
        defaults["alignment_weight"] = 0.0
    defaults.setdefault("task", "{}_to_{}".format(
        defaults["src"], defaults["tgt"]
    ))
    defaults.setdefault("tier", "unspecified")
    defaults.setdefault("run_tag", "experiment_A")
    defaults.setdefault("max_seq_length", 128)
    return defaults


def _namespace(config):
    args = Namespace(**config)
    args.alignment_weight = float(config["alignment_weight"])
    args.tau = float(config["tau"])
    args.ot_reg = float(config["ot_reg"])
    args.lr = float(config["learning_rate"])
    args.results_root = config.get("results_root", "results")
    args.target_file = config.get("target_file", "")
    args.cost = config.get("cost", "l2")
    args.cost_mode = config.get("cost_mode", "class_blind")
    args.class_cost_penalty = float(
        config.get("class_cost_penalty", 1.0)
    )
    args.analysis_size = int(config.get("analysis_size", 256))
    args.analysis_seed = int(
        config.get("analysis_seed", config["train_seed"])
    )
    args.sequential_align_epochs = int(
        config.get("sequential_align_epochs", 0)
    )
    args.ot_max_iter = int(config.get("ot_max_iter", 2000))
    args.ot_tol = float(config.get("ot_tol", 1e-9))
    # ot_normalize 只影响 main_ot.py 系方法；其他方法的身份里必须为 False，
    # 否则规划侧与实际运行侧（入口无此参数，默认 False）的 run_dir 哈希不一致
    args.ot_normalize = bool(config.get("ot_normalize", False)) and str(
        config.get("method", "")
    ) in ("source_only", "ot_balanced", "ot_unbalanced", "ot_auto")
    args.max_grad_norm = float(config.get("max_grad_norm", 1.0))
    args.amp = bool(config.get("amp", False))
    args.max_steps = int(config.get("max_steps", 0))
    args.deterministic = bool(config.get("deterministic", False))
    args.deepjdot_alpha = float(config.get("deepjdot_alpha", 0.001))
    args.deepjdot_lambda_t = float(config.get("deepjdot_lambda_t", 0.0001))
    args.deepjdot_solver = config.get("deepjdot_solver", "emd")
    return args


def _run_record(config, method_spec, status, **extra):
    record = dict(config)
    record.update({
        "method": method_spec["method"],
        "runtime_method": method_spec.get("runtime_method"),
        "status": status,
    })
    record.update(extra)
    return record


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(
                record, sort_keys=True, ensure_ascii=True
            ) + "\n")


def _command(config, method_spec):
    entrypoint = method_spec["entrypoint"]
    if method_spec["method"] == "target_only_oracle":
        command = [
            sys.executable, os.path.join(PROJECT_ROOT, entrypoint),
            "--src", config["src"],
            "--tgt", config["tgt"],
        ]
    else:
        command = [
            sys.executable, os.path.join(PROJECT_ROOT, entrypoint),
            "--src", config["src"],
            "--tgt", config["tgt"],
            "--method", config["method"],
            "--experiment", config["experiment"],
        ]
    scalar_args = {
        "--task": config["task"],
        "--tier": config["tier"],
        "--train_seed": config["train_seed"],
        "--seed": config["seed"],
        "--max_seq_length": config["max_seq_length"],
        "--batch_size": config["batch_size"],
        "--num_epochs": config["num_epochs"],
        "--eval_protocol": config["eval_protocol"],
        "--results_root": os.path.abspath(
            config.get("results_root", "results")
        ),
        "--run_tag": config["run_tag"],
        "--analysis_size": config.get("analysis_size", 256),
        "--analysis_seed": config.get(
            "analysis_seed", config["train_seed"]
        ),
    }
    if method_spec["method"] != "target_only_oracle":
        scalar_args.update({
            "--lambda": config["alignment_weight"],
            "--tau": config["tau"],
            "--ot_reg": config["ot_reg"],
            "--cost": config.get("cost", "l2"),
            "--cost_mode": config.get("cost_mode", "class_blind"),
            "--class_cost_penalty": config.get(
                "class_cost_penalty", 1.0
            ),
        })
    for flag, value in scalar_args.items():
        command.extend([flag, str(value)])
    if config.get("learning_rate") is not None:
        command.extend(["--lr", str(config["learning_rate"])])
    _is_ot_entry = str(method_spec.get("entrypoint") or "").endswith(
        "main_ot.py"
    )
    # --ot_normalize / --ot_max_iter / --ot_tol 仅 main_ot.py 系入口具备；
    # mmd/invgan_kd/oracle 入口没有，不能传
    if _is_ot_entry:
        command.extend([
            "--ot_max_iter", str(config.get("ot_max_iter", 2000)),
            "--ot_tol", str(config.get("ot_tol", 1e-9)),
        ])
    if config.get("ot_normalize", False) and _is_ot_entry:
        command.append("--ot_normalize")
    if config.get("selection") and (_is_ot_entry or method_spec["method"] in ("mmd", "invgan_kd")):
        command.extend(["--selection", str(config["selection"])])
    if config.get("deterministic", False):
        command.append("--deterministic")
    if config.get("keep_checkpoint", False):
        command.append("--keep_checkpoint")
    if config.get("amp", False):
        command.append("--amp")
    if method_spec["method"] == "deepjdot_faithful":
        command.extend([
            "--deepjdot_alpha", str(config.get("deepjdot_alpha", 0.001)),
            "--deepjdot_lambda_t", str(config.get("deepjdot_lambda_t", 0.0001)),
            "--deepjdot_solver", str(config.get("deepjdot_solver", "emd")),
        ])
    if config.get("max_steps", 0) > 0 and _is_ot_entry:
        command.extend(["--max_steps", str(config["max_steps"])])
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Manifest-driven final Experiment A runner"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="override manifest defaults.batch_size for every scheduled run",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--instance-tag", default="",
        help="parallel instances: write run_manifest_<tag>.jsonl instead of "
             "overwriting the shared run_manifest.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.batch_size is not None:
        manifest.setdefault("defaults", {})["batch_size"] = args.batch_size
    methods = _method_specs(manifest, args.methods)
    tasks = _task_specs(manifest, args.tasks)
    if not tasks:
        raise ValueError("manifest selection contains no tasks")

    records = []
    scheduled = []
    for task in tasks:
        task_methods = (
            methods if args.methods
            else _methods_for_task(manifest, task, None)
        )
        seeds = _seeds_for_task(manifest, task, args.seeds)
        if not seeds:
            raise ValueError(
                "no seeds resolved for task {}".format(task.get("task"))
            )
        for method_spec in task_methods:
            grid_points = _grid_points(manifest, method_spec)
            for seed in seeds:
                for overrides in grid_points:
                    config = _resolved_config(
                        manifest, task, method_spec, seed
                    )
                    config.update(overrides)
                    config["run_tag"] = _grid_tag(
                        config.get("run_tag", ""), overrides
                    )
                    config["results_root"] = args.results_root
                    if method_spec["status"] != "available":
                        records.append(_run_record(
                            config, method_spec, "BASELINE_MISSING",
                            reason=method_spec.get("reason"),
                            minimum_interface=method_spec.get(
                                "minimum_interface"
                            ),
                        ))
                        continue
                    runtime_args = _namespace(config)
                    run_dir = build_run_dir(runtime_args)
                    config["run_dir"] = run_dir
                    config["run_config"] = canonical_run_config(runtime_args)
                    config["run_config_hash"] = os.path.basename(run_dir).split(
                        "_cfg", 1
                    )[-1]
                    completed_summary = ""
                    if os.path.isdir(run_dir):
                        for name in sorted(os.listdir(run_dir)):
                            if not name.endswith("_summary.json"):
                                continue
                            summary_path = os.path.join(run_dir, name)
                            try:
                                with open(
                                    summary_path, encoding="utf-8"
                                ) as handle:
                                    summary = json.load(handle)
                            except (OSError, ValueError):
                                continue
                            if summary.get("status") == "completed":
                                completed_summary = summary_path
                                break
                    if completed_summary:
                        records.append(_run_record(
                            config, method_spec, "already_completed",
                            summary_path=completed_summary,
                        ))
                        continue
                    scheduled.append((config, method_spec))
                    records.append(_run_record(
                        config, method_spec, "scheduled"
                    ))

    manifest_name = "run_manifest.jsonl"
    if args.instance_tag:
        manifest_name = "run_manifest_{}.jsonl".format(args.instance_tag)
    manifest_output = os.path.join(
        args.results_root, "experiment_A", manifest_name
    )
    _write_jsonl(manifest_output, records)
    print("planned records: {}".format(len(records)))
    print("scheduled runs: {}".format(len(scheduled)))
    print("manifest: {}".format(manifest_output))

    if args.dry_run:
        for config, method_spec in scheduled:
            print("DRY-RUN", " ".join(_command(config, method_spec)))
        return 0

    failures = 0
    for index, (config, method_spec) in enumerate(scheduled, start=1):
        command = _command(config, method_spec)
        run_dir = config["run_dir"]
        log_dir = os.path.join(args.results_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, "{}.log".format(os.path.basename(run_dir))
        )
        print("[{}/{}] {}".format(index, len(scheduled), run_dir))
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            failures += 1
            for record in records:
                if (
                    record.get("run_dir") == run_dir
                    and record.get("status") == "scheduled"
                ):
                    record["status"] = "failed"
                    record["returncode"] = result.returncode
                    record["log_path"] = log_path
                    break
            print("FAILED returncode={} log={}".format(
                result.returncode, log_path
            ))
            if not args.continue_on_error:
                _write_jsonl(manifest_output, records)
                return result.returncode
        else:
            for record in records:
                if (
                    record.get("run_dir") == run_dir
                    and record.get("status") == "scheduled"
                ):
                    record["status"] = "completed"
                    record["returncode"] = 0
                    record["log_path"] = log_path
                    break
            print("OK {:.1f} min".format(
                (time.time() - started) / 60.0
            ))
        _write_jsonl(manifest_output, records)
    print("completed with failures={}".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment B controlled prior-shift runner.

Ground-truth target labels are used only to construct resampled target files.
The training entry receives only the generated pair strings and labels are
never exposed through the target adaptation batches.
"""
import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from main.run_identity import build_run_dir
import param

SHIFT_DIR = os.path.join(PROJECT_ROOT, "data", "_shift")

# 方法 -> 入口。source_only/ot_* 共用 main_ot.py；mmd/invgan_kd 有独立入口。
METHOD_ENTRYPOINTS = {
    "source_only": "main/main_ot.py",
    "ot_balanced": "main/main_ot.py",
    "ot_auto": "main/main_ot.py",
    "ot_unbalanced": "main/main_ot.py",
    "mmd": "main/main_mmd.py",
    "invgan_kd": "main/main_invgan_kd.py",
}


def _method_hyper(method, args):
    """(effective_lambda, effective_lr)：mmd/invgan_kd 用 imbalance 组
    legacy 调参值（与 manifests/experiment_A.json 的 T3/T4 overrides 一致），
    可被 --mmd-lambda / --invgan-lr / --invgan-kd-weight 覆盖。"""
    if method == "mmd":
        return args.mmd_lambda, param.d_learning_rate
    if method == "invgan_kd":
        return args.invgan_kd_weight, args.invgan_lr
    lam = 0.0 if method == "source_only" else args.alignment_weight
    return lam, args.lr


def _completed_summary(run_dir):
    """run_dir 下已有 status=completed 的 summary 则返回其路径（断点续跑）。"""
    if not os.path.isdir(run_dir):
        return ""
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith("_summary.json"):
            continue
        path = os.path.join(run_dir, name)
        try:
            with open(path, encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            continue
        if summary.get("status") == "completed":
            return path
    return ""


def _label_value(row):
    return int(float(row["labels"]))


def _read_dataset(path):
    with open(path, newline="", encoding="latin") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames or "labels" not in fieldnames:
        raise ValueError(f"{path} must contain a labels column")
    return rows, fieldnames


def _fixed_sample_size(positives, negatives, levels, requested_size):
    """Find one target size feasible for every requested prior level."""
    size = min(int(requested_size), len(positives) + len(negatives))
    while size > 0:
        feasible = True
        for level in levels:
            n_positive = int(round(float(level) * size))
            n_negative = size - n_positive
            if n_positive > len(positives) or n_negative > len(negatives):
                feasible = False
                break
        if feasible:
            return size
        size -= 1
    raise ValueError("no common feasible target sample size for levels")


def resample_target(
    target, target_pos_rate, seed, output_path, sample_size=None,
    write_file=True,
):
    source_path = os.path.join(PROJECT_ROOT, "data", target, target + ".csv")
    rows, fieldnames = _read_dataset(source_path)
    positives = [row for row in rows if _label_value(row) == 1]
    negatives = [row for row in rows if _label_value(row) == 0]
    if sample_size is None:
        sample_size = len(rows)
    n_positive = int(round(float(target_pos_rate) * sample_size))
    n_negative = int(sample_size - n_positive)
    if n_positive > len(positives):
        raise ValueError(
            f"target prior {target_pos_rate} needs {n_positive} positives, "
            f"but only {len(positives)} are available"
        )
    if n_negative > len(negatives):
        raise ValueError(
            f"target prior {target_pos_rate} needs {n_negative} negatives, "
            f"but only {len(negatives)} are available"
        )
    rng = random.Random(seed)
    output = (
        rng.sample(positives, n_positive)
        + rng.sample(negatives, n_negative)
    )
    rng.shuffle(output)
    if write_file:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="", encoding="latin") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)
    actual_prior = (
        sum(1 for row in output if _label_value(row) == 1)
        / max(len(output), 1)
    )
    return {
        "output_path": output_path,
        "target_prior_requested": float(target_pos_rate),
        "target_prior_actual": actual_prior,
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "n_total": int(len(output)),
        "resample_seed": int(seed),
    }


def _planned_run_dir(args, target_file, level, replica, seed, method):
    effective_lambda, effective_lr = _method_hyper(method, args)
    config = SimpleNamespace(
        experiment="experiment_B",
        tier="tier_3",
        task=f"{args.src}_to_{args.tgt}",
        src=args.src,
        tgt=args.tgt,
        target_file=target_file,
        method=method,
        seed=replica,
        train_seed=seed,
        alignment_weight=effective_lambda,
        tau=args.tau,
        ot_reg=args.ot_reg,
        cost="l2",
        cost_mode="class_blind",
        batch_size=args.batch_size,
        lr=effective_lr,
        # 与 run_experiments._namespace 同规则：非 main_ot.py 系方法身份里置 False
        ot_normalize=bool(args.ot_normalize)
        and method in ("source_only", "ot_balanced", "ot_unbalanced", "ot_auto"),
        eval_protocol=args.eval_protocol,
        run_tag=f"prior{level:g}_rep{replica}",
        max_seq_length=args.max_seq_length,
        num_epochs=args.num_epochs,
        ot_max_iter=args.ot_max_iter,
        ot_tol=args.ot_tol,
        analysis_size=args.analysis_size,
        analysis_seed=args.analysis_seed,
        deterministic=args.deterministic,
        results_root=os.path.abspath(args.results_root),
    )
    return build_run_dir(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--levels", nargs="+", type=float, required=True)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--link-replica-to-seed", action="store_true",
        help="replica index := seed index (each seed gets a different "
             "resampling composition at the same run count)")
    parser.add_argument("--methods", nargs="+",
                        default=["source_only", "ot_balanced", "ot_unbalanced"],
                        choices=sorted(METHOD_ENTRYPOINTS),
                        help="论文图默认三方法；可加 mmd invgan_kd 作对照曲线")
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ot-normalize", dest="ot_normalize",
                        action="store_true", default=True,
                        help="OT cost 前做 L2 归一化（默认开启，与 manifest 一致；"
                             "仅 main_ot.py 系方法生效）")
    parser.add_argument("--no-ot-normalize", dest="ot_normalize",
                        action="store_false")
    parser.add_argument("--lambda", dest="alignment_weight", type=float,
                        default=0.01)
    parser.add_argument("--mmd-lambda", type=float, default=0.01,
                        help="mmd 的 MMD 权重（imbalance 组 legacy 值）")
    parser.add_argument("--invgan-lr", type=float, default=1e-6,
                        help="invgan_kd 适配 lr（imbalance 组 legacy 值）")
    parser.add_argument("--invgan-kd-weight", type=float, default=1.0,
                        help="invgan_kd 的 KD 权重（imbalance 组 legacy 值）")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--ot-reg", type=float, default=0.1)
    parser.add_argument("--ot-max-iter", dest="ot_max_iter", type=int,
                        default=2000, help="透传给 main_ot.py 的求解器迭代上限")
    parser.add_argument("--ot-tol", dest="ot_tol", type=float,
                        default=1e-9, help="透传给 main_ot.py 的求解器容差")
    parser.add_argument("--lr", type=float, default=param.d_learning_rate)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--eval-protocol",
                        choices=["transductive", "inductive"],
                        default="inductive")
    parser.add_argument("--analysis-size", type=int, default=256)
    parser.add_argument("--analysis-seed", type=int, default=None)
    parser.add_argument("--target-sample-size", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--selection", choices=["validation", "last_epoch"], default="validation")
    parser.add_argument("--reuse-target-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default="results/experiment_B/prior_shift.csv")
    args = parser.parse_args()

    source_path = os.path.join(PROJECT_ROOT, "data", args.src, args.src + ".csv")
    source_rows, _ = _read_dataset(source_path)
    source_prior = (
        sum(1 for row in source_rows if _label_value(row) == 1)
        / max(len(source_rows), 1)
    )
    target_path = os.path.join(
        PROJECT_ROOT, "data", args.tgt, args.tgt + ".csv"
    )
    target_rows, _ = _read_dataset(target_path)
    positives = [row for row in target_rows if _label_value(row) == 1]
    negatives = [row for row in target_rows if _label_value(row) == 0]
    requested_size = args.target_sample_size or len(target_rows)
    target_sample_size = _fixed_sample_size(
        positives, negatives, args.levels, requested_size
    )
    analysis_seed = (
        args.analysis_seed
        if args.analysis_seed is not None
        else args.seeds[0]
    )
    args.analysis_seed = analysis_seed
    rows = []

    if args.link_replica_to_seed:
        combos = (
            (level, replica, seed, method)
            for level in args.levels
            for replica, seed in enumerate(args.seeds)
            for method in args.methods
        )
    else:
        combos = itertools.product(
            args.levels, range(args.replicas), args.seeds, args.methods
        )
    for level, replica, seed, method in combos:
        target_file = os.path.join(
            SHIFT_DIR, f"{args.tgt}_prior{level:g}_rep{replica}.csv"
        )
        if args.reuse_target_files and not os.path.isfile(target_file):
            raise FileNotFoundError(target_file)
        resampled = resample_target(
            args.tgt, level, replica, target_file, target_sample_size,
            write_file=not args.dry_run and not args.reuse_target_files,
        )
        tag = f"prior{level:g}_rep{replica}"
        effective_lambda, effective_lr = _method_hyper(method, args)
        entrypoint = METHOD_ENTRYPOINTS[method]
        command = [
            sys.executable, os.path.join(PROJECT_ROOT, entrypoint),
            "--src", args.src,
            "--tgt", args.tgt,
            "--target_file", target_file,
            "--method", method,
            "--experiment", "experiment_B",
            "--train_seed", str(seed),
            "--seed", str(replica),
            "--max_seq_length", str(args.max_seq_length),
            "--num_epochs", str(args.num_epochs),
            "--selection", args.selection,
            "--batch_size", str(args.batch_size),
            "--lambda", str(effective_lambda),
            "--tau", str(args.tau),
            "--ot_reg", str(args.ot_reg),
            "--ot_max_iter", str(args.ot_max_iter),
            "--ot_tol", str(args.ot_tol),
            "--lr", str(effective_lr),
            "--eval_protocol", args.eval_protocol,
            "--analysis_size", str(args.analysis_size),
            "--analysis_seed", str(analysis_seed),
            "--task", f"{args.src}_to_{args.tgt}",
            "--tier", "tier_3",
            "--run_tag", tag,
            "--results_root", os.path.abspath(args.results_root),
        ]
        if args.ot_normalize and entrypoint.endswith("main_ot.py"):
            # 仅 main_ot.py 系入口具备 --ot_normalize 参数
            command.append("--ot_normalize")
        if args.deterministic:
            command.append("--deterministic")
        run_dir = _planned_run_dir(
            args, target_file, level, replica, seed, method
        )
        if args.dry_run:
            result = SimpleNamespace(returncode=None)
            run_status = "DRY_RUN"
        elif _completed_summary(run_dir):
            result = SimpleNamespace(returncode=0)
            run_status = "already_completed"
        else:
            result = subprocess.run(command)
            run_status = (
                "completed" if result.returncode == 0 else "failed"
            )
        record = {
            "source": args.src,
            "target": args.tgt,
            "experiment": "experiment_B",
            "tier": "tier_3",
            "task": f"{args.src}_to_{args.tgt}",
            "method": method,
            "target_prior_requested": resampled["target_prior_requested"],
            "target_prior_actual": resampled["target_prior_actual"],
            "source_prior": source_prior,
            "abs_delta_pi_actual": abs(
                source_prior - resampled["target_prior_actual"]
            ),
            "target_n_positive": resampled["n_positive"],
            "target_n_negative": resampled["n_negative"],
            "target_n_total": resampled["n_total"],
            "target_sample_size_fixed": target_sample_size,
            "replica": replica,
            "seed": seed,
            "split_seed": replica,
            "analysis_seed": analysis_seed,
            "resample_seed": resampled["resample_seed"],
            "lambda": effective_lambda,
            "tau": args.tau,
            "ot_reg": args.ot_reg,
            "cost": "l2",
            "cost_mode": "class_blind",
            "batch_size": args.batch_size,
            "learning_rate": effective_lr,
            "num_epochs": args.num_epochs,
            "eval_protocol": args.eval_protocol,
            "status": run_status,
            "returncode": result.returncode,
            "target_file": target_file,
            "run_tag": tag,
            "run_dir": run_dir,
            "command": " ".join(command),
        }
        rows.append(record)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote Experiment B run manifest: {args.out}")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit("One or more runs failed; inspect the worker log and rerun.")


if __name__ == "__main__":
    main()

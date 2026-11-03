"""Dependency-light canonical run identity helpers."""
import hashlib
import json
import os
import re

import param


DEFAULT_LEARNING_RATE = 1.5e-5


def _target_identity(path):
    if not path:
        return "base"
    absolute = os.path.abspath(path)
    digest = hashlib.sha256()
    digest.update(absolute.encode("utf-8"))
    if os.path.isfile(absolute):
        with open(absolute, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def canonical_run_config(args):
    learning_rate = getattr(args, "lr", None)
    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE
    config = {
        "experiment": getattr(args, "experiment", "unspecified"),
        "tier": getattr(args, "tier", "unspecified"),
        "task": getattr(
            args,
            "task",
            f"{getattr(args, 'src', 'src')}_to_{getattr(args, 'tgt', 'tgt')}",
        ),
        "src": getattr(args, "src", None),
        "tgt": getattr(args, "tgt", None),
        "target_file_id": _target_identity(
            getattr(args, "target_file", "")
        ),
        "method": getattr(args, "method", None),
        "seed": getattr(args, "seed", None),
        "train_seed": getattr(args, "train_seed", None),
        "alignment_weight": float(
            getattr(args, "alignment_weight", 0.0)
        ),
        "tau": float(getattr(args, "tau", 1.0)),
        "ot_reg": float(getattr(args, "ot_reg", 0.1)),
        "ot_normalize": bool(getattr(args, "ot_normalize", False)),
        "cost": getattr(args, "cost", "l2"),
        "cost_mode": getattr(args, "cost_mode", "class_blind"),
        "class_cost_penalty": float(
            getattr(args, "class_cost_penalty", 1.0)
        ),
        "batch_size": int(getattr(args, "batch_size", 0)),
        "learning_rate": float(learning_rate),
        "eval_protocol": getattr(args, "eval_protocol", "inductive"),
        "target_valid_frac": float(
            getattr(args, "target_valid_frac", param.target_valid_frac)
        ),
        "target_test_frac": float(
            getattr(args, "target_test_frac", param.target_test_frac)
        ),
        "target_split_stratified": bool(
            getattr(
                args, "target_split_stratified",
                getattr(param, "target_split_stratified", False),
            )
        ),
        "run_tag": getattr(args, "run_tag", ""),
        "max_seq_length": int(getattr(args, "max_seq_length", 0)),
        "num_epochs": int(getattr(args, "num_epochs", 0)),
        "analysis_size": int(getattr(args, "analysis_size", 0)),
        "analysis_seed": int(
            getattr(args, "analysis_seed", getattr(args, "train_seed", 0))
        ),
        "sequential_align_epochs": int(
            getattr(args, "sequential_align_epochs", 0)
        ),
        "ot_max_iter": int(getattr(args, "ot_max_iter", 2000)),
        "ot_tol": float(getattr(args, "ot_tol", 1e-9)),
        "max_grad_norm": float(getattr(args, "max_grad_norm", 1.0)),
        "amp": bool(getattr(args, "amp", False)),
        "max_steps": int(getattr(args, "max_steps", 0)),
        "deterministic": bool(getattr(args, "deterministic", False)),
    }
    if config["method"] == "deepjdot_faithful":
        # 只给 faithful 方法加入原论文参数；旧方法的哈希保持不变。
        config.update({
            "deepjdot_alpha": float(getattr(args, "deepjdot_alpha", 0.001)),
            "deepjdot_lambda_t": float(
                getattr(args, "deepjdot_lambda_t", 0.0001)
            ),
            "deepjdot_solver": getattr(args, "deepjdot_solver", "emd"),
        })
    return config


def run_config_hash(config):
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_run_dir(args):
    config = canonical_run_config(args)
    config_hash = run_config_hash(config)
    task = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(config["task"])
    )
    run_name = (
        f"{task}_{config['method']}_train{config['train_seed']}"
        f"_split{config['seed']}_cfg{config_hash}"
    )
    return os.path.join(
        args.results_root, config["experiment"], run_name
    )

"""Shared data and tokenizer utilities for the final experiment runtime."""
import hashlib
import os

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import param

from main.run_identity import (
    build_run_dir,
    canonical_run_config,
    run_config_hash,
)
from utils import CSV2Array, GpuBatches, load_or_build_features


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tokenizer(model_name="bert-base-multilingual-cased"):
    local_paths = (
        os.path.join(PROJECT_ROOT, "Models", model_name),
        os.path.join(PROJECT_ROOT, "..", "Models", model_name),
    )
    for path in local_paths:
        if os.path.exists(path):
            return AutoTokenizer.from_pretrained(path)
    return AutoTokenizer.from_pretrained(model_name)


def _dataset_path(name, override=None):
    if override:
        return override
    return os.path.join(PROJECT_ROOT, "data", name, name + ".csv")


def _feature_tag(name, path):
    if path:
        return "external_" + hashlib.md5(path.encode()).hexdigest()[:10]
    return name


def _label_array(features):
    return np.array([
        int(f.label_id) if getattr(f, "label_id", None) is not None else 0
        for f in features
    ])


def _split_indices(indices, labels, test_size, split_seed, stratify):
    """One train_test_split with optional label stratification.

    Stratification can fail on extreme-prior resampled sets (a class with too
    few members for the requested split); fall back to the label-blind split
    with a loud warning rather than aborting the run.
    """
    strat = labels[indices] if (stratify and labels is not None) else None
    try:
        return train_test_split(
            indices, test_size=test_size, random_state=split_seed,
            stratify=strat,
        )
    except ValueError as exc:
        if strat is None:
            raise
        print("[split] stratified split failed ({}); "
              "falling back to label-blind split".format(exc))
        return train_test_split(
            indices, test_size=test_size, random_state=split_seed,
        )


def split_target_features(features, split_seed, eval_protocol, stratify=False):
    """Create the one shared target split used by all final methods.

    Fractions come from param.target_valid_frac / param.target_test_frac
    (pool/valid/test = 70/10/20 under the inductive protocol) and are part of
    the canonical run identity (main/run_identity.py).

    With ``stratify=True`` (the final-protocol default, see
    param.target_split_stratified) the split is class-stratified: ER pair sets
    are heavily imbalanced (9%~32% positive), and a label-blind draw lets the
    valid/test positive counts drift far from the full-set rate on small
    targets (fz/b2/ri), corrupting both model selection and the reported F1.
    Stratification only uses target labels to *construct* the split, keeping
    pool/valid/test class rates aligned with the full set; adaptation batches
    remain label-free (``include_labels=False`` in ``load_transfer_batches``),
    so the target-label firewall is unaffected. ``stratify=False`` remains
    available for ablation/diagnostic runs, and ``_split_indices`` falls back
    to a label-blind split with a loud warning if stratification is
    infeasible (e.g. a class too small for the requested fractions).
    """
    indices = np.arange(len(features))
    labels = _label_array(features) if stratify else None
    pool_idx, valid_idx = _split_indices(
        indices, labels, param.target_valid_frac, split_seed, stratify
    )

    test_idx = None
    if eval_protocol == "inductive":
        # test_size is relative to the remaining pool, so rescale the
        # absolute fraction: test / (1 - valid).
        rel_test = param.target_test_frac / (1.0 - param.target_valid_frac)
        pool_idx, test_idx = _split_indices(
            pool_idx, labels, rel_test, split_seed, stratify
        )

    pick = lambda idx: [features[i] for i in idx]
    target_pool = pick(pool_idx)
    target_valid = pick(valid_idx)
    target_eval = (
        target_pool
        if eval_protocol == "transductive"
        else pick(test_idx)
    )
    return target_pool, target_valid, target_eval


def split_source_features(features, analysis_seed):
    """Hold out a fixed source analysis split without using target labels."""
    indices = np.arange(len(features))
    train_idx, analysis_idx = train_test_split(
        indices, test_size=0.1, random_state=analysis_seed
    )
    return (
        [features[i] for i in train_idx],
        [features[i] for i in analysis_idx],
    )


def deterministic_analysis_indices(n_samples, analysis_size, seed, salt):
    """Select a reproducible subset without depending on GPU RNG state."""
    n = min(int(n_samples), int(analysis_size))
    if n <= 0:
        raise ValueError("analysis subset must contain at least one sample")
    rng = np.random.default_rng(int(seed) + int(salt))
    return np.sort(rng.choice(int(n_samples), size=n, replace=False))


def load_transfer_batches(args):
    """Load source/target features and construct the common batch objects."""
    tokenizer = load_tokenizer(getattr(args, "encoder_model", "bert-base-multilingual-cased"))
    use_cache = not args.no_cache

    source_path = _dataset_path(args.src)
    target_path = _dataset_path(args.tgt, args.target_file)
    source_pairs, source_labels = CSV2Array(source_path)
    target_pairs, target_labels = CSV2Array(target_path)

    source_features = load_or_build_features(
        _feature_tag(args.src, None),
        source_pairs,
        source_labels,
        args.max_seq_length,
        tokenizer,
        use_cache=use_cache,
    )
    source_train_features, source_analysis_features = split_source_features(
        source_features, args.analysis_seed
    )
    target_features = load_or_build_features(
        _feature_tag(args.tgt, args.target_file),
        target_pairs,
        target_labels,
        args.max_seq_length,
        tokenizer,
        use_cache=use_cache,
    )
    target_pool, target_valid, target_eval = split_target_features(
        target_features, args.seed, args.eval_protocol,
        stratify=bool(getattr(param, "target_split_stratified", False)),
    )

    batch_size = args.batch_size
    return {
        "source": GpuBatches(
            source_train_features, batch_size, shuffle=True, drop_last=True,
            seed=int(args.train_seed) + 101
        ),
        "source_analysis": GpuBatches(
            source_analysis_features, batch_size * 2, shuffle=False,
            seed=int(args.analysis_seed) + 103
        ),
        "target_pool": GpuBatches(
            target_pool, batch_size, shuffle=True, drop_last=True,
            include_labels=False, seed=int(args.train_seed) + 107
        ),
        "target_valid": GpuBatches(
            target_valid, batch_size * 2, shuffle=False,
            seed=int(args.seed) + 109
        ),
        "target_eval": GpuBatches(
            target_eval, batch_size * 2, shuffle=False,
            seed=int(args.seed) + 113
        ),
        "source_features": source_features,
        "target_features": target_features,
        "target_pool_features": target_pool,
        "target_valid_features": target_valid,
        "target_eval_features": target_eval,
        "source_path": source_path,
        "target_path": target_path,
    }

#!/usr/bin/env python3
"""Prepare identical WDC resamples and feature caches before parallel workers."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from analysis.prior_shift import resample_target, _fixed_sample_size, _read_dataset, SHIFT_DIR
from main.experiment_common import load_tokenizer, _feature_tag
from utils import CSV2Array, load_or_build_features

LEVELS = [0.05, 0.10, 0.17, 0.27, 0.40, 0.55]
CATEGORIES = ["cameras", "computers", "shoes", "watches"]
tokenizer = load_tokenizer("bert-base-multilingual-cased")
for target in CATEGORIES:
    source_file = ROOT / "data" / target / (target + ".csv")
    rows, _ = _read_dataset(str(source_file))
    pos = [r for r in rows if int(float(r["labels"])) == 1]
    neg = [r for r in rows if int(float(r["labels"])) == 0]
    size = _fixed_sample_size(pos, neg, LEVELS, len(rows))
    pairs, labels = CSV2Array(str(source_file))
    load_or_build_features(_feature_tag(target, None), pairs, labels, 128, tokenizer)
    for level in LEVELS:
        for rep in range(3):
            path = os.path.join(SHIFT_DIR, f"{target}_prior{level:g}_rep{rep}.csv")
            resample_target(target, level, rep, path, sample_size=size)
            pairs, labels = CSV2Array(path)
            load_or_build_features(_feature_tag(target, path), pairs, labels, 128, tokenizer)
    print(f"{target}: prepared 18 shifted datasets and caches", flush=True)

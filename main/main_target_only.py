# main/main_target_only.py
"""Target-only oracle：在目标池标签上有监督训练，作为主表上界（非竞争者）。

协议与 main_ot.py 完全一致（同划分、同 valid 选模、同终评池），
仅把 OT 替换为目标池有标签 CE 训练。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import shutil
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import param
from modules.extractor import BertEncoder
from modules.matcher import BertClassifier
from utils import CSV2Array, load_or_build_features, GpuBatches, init_model, save_model
from train.evaluate import evaluate_full
from train.run_logger import RunLogger
from main.main_ot import set_seed
from main.experiment_common import (
    build_run_dir,
    load_tokenizer,
    split_target_features,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Target-only oracle")
    parser.add_argument('--tgt', type=str, required=True)
    parser.add_argument('--src', type=str, default="oracle",
                        help="仅用于 checkpoint 目录命名")
    parser.add_argument('--task', type=str, default="")
    parser.add_argument('--tier', type=str, default="unspecified")
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--eval_protocol', type=str, default="transductive",
                        choices=["transductive", "inductive"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--model', type=str, default="bert")
    parser.add_argument('--results_root', type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument('--run_tag', type=str, default="")
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--analysis_size', type=int, default=256)
    parser.add_argument('--analysis_seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--keep_checkpoint', action='store_true',
                        help="保留 best checkpoint；默认训练结束后删除（约 700MB/run）")
    return parser.parse_args()


def main():
    args = parse_arguments()
    args.method = "oracle"
    args.experiment = "experiment_A"
    args.alignment_weight = 0.0
    args.tau = 1.0
    args.ot_reg = 0.1
    args.cost = "l2"
    args.cost_mode = "class_blind"
    args.class_cost_penalty = 1.0
    args.sequential_align_epochs = 0
    args.ot_max_iter = 500
    args.ot_tol = 1e-7
    args.max_grad_norm = 1.0
    args.amp = False
    args.max_steps = 0
    args.target_file = ""
    if args.lr is None:
        args.lr = param.c_learning_rate
    if not args.task:
        args.task = f"{args.src}_to_{args.tgt}"
    if args.analysis_seed is None:
        args.analysis_seed = args.train_seed
    args.result_dir = build_run_dir(args)
    args.checkpoint_dir = os.path.join(args.result_dir, "checkpoint")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.train_seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    tokenizer = load_tokenizer()

    tgt_path = os.path.join(PROJECT_ROOT, 'data', args.tgt, args.tgt + '.csv')
    tgt_features = load_or_build_features(args.tgt, *CSV2Array(tgt_path),
                                          args.max_seq_length, tokenizer,
                                          use_cache=not args.no_cache)

    target_pool, target_valid, target_eval = split_target_features(
        tgt_features, args.seed, args.eval_protocol,
        stratify=bool(getattr(param, "target_split_stratified", False)),
    )

    bs = args.batch_size
    pool_batches = GpuBatches(
        target_pool, bs, shuffle=True, seed=args.train_seed + 211
    )
    valid_batches = GpuBatches(
        target_valid, bs * 2, shuffle=False, seed=args.seed + 223
    )
    eval_batches = (
        pool_batches if args.eval_protocol == "transductive"
        else GpuBatches(
            target_eval, bs * 2, shuffle=False, seed=args.seed + 227
        )
    )

    encoder = init_model(args, BertEncoder())
    classifier = init_model(args, BertClassifier())
    logger = RunLogger(args, method="oracle", result_dir=args.result_dir)
    lr = args.lr
    logger.log_config(extra={
        "protocol": args.eval_protocol,
        "experiment": args.experiment,
        "tier": args.tier,
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "train_seed": args.train_seed,
        "eval_protocol": args.eval_protocol,
        "alignment_weight": 0.0,
        "tau": 1.0,
        "ot_reg": 0.1,
        "cost": "l2",
        "cost_mode": "class_blind",
        "batch_size": args.batch_size,
        "run_tag": args.run_tag,
        "learning_rate": lr,
        "target_labels_in_adaptation": True,
        "target_ground_truth_labels_in_adaptation": True,
        "protocol_note": "oracle only; not an unlabeled adaptation method",
    })

    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=lr)

    encoder.train()
    classifier.train()
    for epoch in range(args.num_epochs):
        for reviews, mask, segment, labels, _ in pool_batches:
            optimizer.zero_grad()
            loss = CE(classifier(encoder(reviews, mask, segment)), labels)
            loss.backward()
            optimizer.step()
        f1_valid, _, _, eloss = evaluate_full(encoder, classifier, valid_batches)
        if logger.log_epoch(epoch, f1_valid, eloss):
            save_model(args, encoder, param.src_encoder_path + "oraclebest")
            save_model(args, classifier, param.src_classifier_path + "oraclebest")

    encoder = init_model(args, encoder, restore=param.src_encoder_path + "oraclebest")
    classifier = init_model(args, classifier,
                            restore=param.src_classifier_path + "oraclebest")
    f1, prec, rec, _ = evaluate_full(encoder, classifier, eval_batches, verbose=True)
    logger.log_final(f1, prec, rec)
    logger.finalize()
    if not getattr(args, "keep_checkpoint", False):
        # best 权重已 restore 进内存并完成终评；磁盘副本默认删除以节省空间。
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    print(f"=== Target-only oracle: final_f1={f1:.4f} ===")


if __name__ == '__main__':
    main()

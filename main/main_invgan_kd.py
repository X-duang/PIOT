# main/main_invgan_kd.py
"""Final-protocol entry point for the InvGAN+KD baseline.

Two-stage pipeline (source pretrain, then adversarial + KD adaptation of a
target encoder) with the same data split, evaluation, RunLogger artifact
schema and run identity as main_ot.py; see train/invgan_kd_trainer.py.
The legacy v1 entry point (separate split, print-only results) is archived
at _archive/baselines_v1/main/main_invgan_kd.py.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import random
import argparse

import torch
import numpy as np

import param
from modules.extractor import BertEncoder
from modules.matcher import BertClassifier
from modules.alignment import Discriminator
from utils import init_model
from train.invgan_kd_trainer import train
from main.experiment_common import build_run_dir, load_transfer_batches


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="InvGAN+KD baseline (final protocol)"
    )

    # ---- 数据（与 main_ot.py 对齐） ----
    parser.add_argument('--src', type=str, default="b2")
    parser.add_argument('--tgt', type=str, default="fz")
    parser.add_argument('--task', type=str, default="")
    parser.add_argument('--tier', type=str, default="unspecified")
    parser.add_argument('--target_file', '--tgt_file', dest='target_file',
                        type=str, default="",
                        help="目标 CSV 覆盖路径（Experiment B 用）")
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--eval_protocol', type=str, default="inductive",
                        choices=["transductive", "inductive"])
    parser.add_argument('--seed', type=int, default=42, help="数据划分种子")
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--no_cache', action='store_true')

    # ---- 模型 ----
    parser.add_argument('--model', type=str, default="bert", choices=["bert"])
    parser.add_argument('--method', type=str, default="invgan_kd")
    parser.add_argument('--experiment', type=str, default="experiment_A",
                        choices=["experiment_A", "experiment_B",
                                 "experiment_C", "experiment_D"])

    # ---- InvGAN+KD 目标 ----
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="对抗（生成）损失权重")
    parser.add_argument('--lambda', dest='alignment_weight', type=float,
                        default=1.0, help="KD 损失权重 beta（适配损失权重槽位）")
    parser.add_argument('--temperature', type=int, default=20,
                        help="KD 温度 T")
    parser.add_argument('--clip_value', type=float, default=0.01,
                        help="判别器权重裁剪范围（legacy 口径）")
    parser.add_argument('--pre_epochs', type=int, default=5,
                        help="源域预训练 epochs（legacy 口径：5）")
    parser.add_argument('--pre_lr', type=float, default=None,
                        help="预训练 lr，默认取 param.c_learning_rate")
    # run_experiments 统一透传的占位参数；InvGAN+KD 目标不消费。
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--cost', type=str, default="l2")
    parser.add_argument('--cost_mode', type=str, default="class_blind")
    parser.add_argument('--class_cost_penalty', type=float, default=1.0)

    parser.add_argument('--selection', choices=['validation', 'last_epoch'],
                        default='validation')

    # ---- 训练 ----
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="适配阶段 epochs（legacy scripts/invgan_kd.sh 用 20）")
    parser.add_argument('--lr', type=float, default=None,
                        help="适配阶段 lr（legacy d_learning_rate），"
                             "默认取 param.d_learning_rate")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--amp', action='store_true', help="bf16 混合精度")
    parser.add_argument('--analysis_size', type=int, default=256)
    parser.add_argument('--analysis_seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--results_root', '--result_root', type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument('--run_tag', type=str, default="")
    parser.add_argument('--keep_checkpoint', action='store_true',
                        help="保留 best checkpoint；默认训练结束后删除（约 700MB/run）")

    # ---- 冒烟测试 ----
    parser.add_argument('--max_steps', type=int, default=0,
                        help=">0 时每 epoch 只跑这么多步（调试用）")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled):
    if not enabled:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def main():
    args = parse_arguments()
    if not args.task:
        args.task = f"{args.src}_to_{args.tgt}"
    if args.analysis_seed is None:
        args.analysis_seed = args.train_seed
    args.result_dir = build_run_dir(args)
    args.checkpoint_dir = os.path.join(args.result_dir, "checkpoint")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.train_seed)
    configure_determinism(args.deterministic)
    print("=== InvGAN+KD (final protocol) ===")
    print(json.dumps({k: v for k, v in vars(args).items()}, indent=2, default=str))

    data = load_transfer_batches(args)
    print(f"[data] src={data['source'].n_samples} "
          f"pool={data['target_pool'].n_samples} "
          f"valid={data['target_valid'].n_samples} "
          f"eval={data['target_eval'].n_samples}")

    src_encoder = BertEncoder()
    tgt_encoder = BertEncoder()
    classifier = BertClassifier()
    discriminator = Discriminator()
    src_encoder = init_model(args, src_encoder)
    tgt_encoder = init_model(args, tgt_encoder)
    classifier = init_model(args, classifier)
    discriminator = init_model(args, discriminator)

    tgt_encoder, classifier, best_valid_f1, final = train(
        args, src_encoder, tgt_encoder, discriminator, classifier,
        data["source"], data["target_pool"], data["source_analysis"],
        data["target_valid"], data["target_eval"],
        result_dir=args.result_dir)

    print("=== Result of InvGAN+KD ===")
    print(f"method={args.method} experiment={args.experiment} "
          f"best_valid_f1={best_valid_f1:.4f} final={final}")


if __name__ == '__main__':
    main()

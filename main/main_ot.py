# main/main_ot.py
"""Unified entry point for the final OT-DAER experiments.

The final runtime intentionally exposes only the method and experiment axes
defined by the frozen paper protocol. Legacy mass allocation and sampling
options remain in their old modules but are not imported here.
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
from utils import init_model
from train.ot_trainer import train
from main.experiment_common import build_run_dir, load_transfer_batches


def parse_arguments():
    parser = argparse.ArgumentParser(description="OT-DAER: OT for Domain-Adaptive ER")

    # ---- 数据 ----
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
    parser.add_argument('--selection', type=str, default="validation",
                        choices=["validation", "last_epoch"],
                        help="last_epoch: 不评估目标验证集，直接评估末轮（label-free 选模）")
    parser.add_argument('--seed', type=int, default=42, help="数据划分种子")
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--no_cache', action='store_true')

    # ---- 模型 ----
    parser.add_argument('--model', type=str, default="bert", choices=["bert"])
    parser.add_argument('--encoder_model', type=str,
                        default="bert-base-multilingual-cased",
                        help="编码器 HuggingFace 模型名（第二 backbone 实验用，如 distilbert-base-multilingual-cased）")

    # ---- Final method / experiment axes ----
    parser.add_argument(
        '--method',
        type=str,
        default="ot_balanced",
        choices=[
            "source_only",
            "ot_balanced",
            "ot_unbalanced",
            "ot_auto",
            "deepjdot_faithful",
            "ablation_balanced",
            "ablation_sequential",
            "ablation_class_aware",
            "ablation_squared",
            "sensitivity_tau",
            "sensitivity_lambda",
        ],
    )
    parser.add_argument('--experiment', type=str, default="experiment_A",
                        choices=["experiment_A", "experiment_B",
                                 "experiment_C", "experiment_D"])
    parser.add_argument('--cost', type=str, default="l2",
                        choices=["l2", "sq_l2"])
    parser.add_argument('--cost_mode', type=str, default="class_blind",
                        choices=["class_blind", "class_aware"])
    parser.add_argument('--class_cost_penalty', type=float, default=1.0)
    parser.add_argument('--deepjdot_alpha', type=float, default=0.001,
                        help='faithful DeepJDOT 特征距离权重 α')
    parser.add_argument('--deepjdot_lambda_t', type=float, default=0.0001,
                        help='faithful DeepJDOT 目标分类损失权重 λ_t')
    parser.add_argument('--deepjdot_solver', type=str, default='emd',
                        choices=['emd'],
                        help='faithful DeepJDOT 使用原论文的未正则化 EMD')
    parser.add_argument('--lambda', dest='alignment_weight', type=float,
                        default=0.01, help="alignment weight λ")
    parser.add_argument('--ot_reg', type=float, default=0.1, help="熵正则 ε")
    parser.add_argument('--tau', type=float, default=1.0,
                        help="unbalanced marginal penalty τ")
    parser.add_argument('--ot_max_iter', type=int, default=2000)
    parser.add_argument('--ot_tol', type=float, default=1e-9)

    # ---- 训练 ----
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--sequential_align_epochs', type=int, default=0,
                        help="Experiment C sequential ablation; 0 = half of training")
    parser.add_argument('--lr', type=float, default=None,
                        help="默认取 param.d_learning_rate")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--amp', action='store_true', help="bf16 混合精度")
    parser.add_argument('--analysis_size', type=int, default=256)
    parser.add_argument(
        '--analysis_seed', type=int, default=None,
        help="deterministic seed for the shared before/after analysis subset",
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help="enable deterministic PyTorch/cuDNN behavior where supported",
    )
    parser.add_argument('--results_root', '--result_root', type=str,
                        default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument('--run_tag', type=str, default="",
                        help="追加到 summary 文件名，区分同 src/tgt/seed 的不同实验条件")
    parser.add_argument('--keep_checkpoint', action='store_true',
                        help="保留 best checkpoint；默认训练结束后删除（约 700MB/run）")
    parser.add_argument('--ot_normalize', action='store_true',
                        help="对 encoder 表示做 L2 单位球归一化后再计算 OT 代价矩阵")

    # ---- auto 模式：pilot probe 选择器（论文 Algorithm 1）----
    parser.add_argument('--probe_frac', type=float, default=0.2,
                        help="pilot 采样比例")
    parser.add_argument('--probe_max_pairs', type=int, default=2000,
                        help="pilot 采样上限（开销常数化）")
    parser.add_argument('--probe_pretrain_epochs', type=int, default=5,
                        help="Stage 0 源预训练轮数（计入总 epoch）")
    parser.add_argument('--probe_alpha', type=float, default=0.5,
                        help="τ 规则安全因子 α∈(0,1)")
    parser.add_argument('--tau_max', type=float, default=20.0,
                        help="求解器实用上限（τ̂≥τ_max 判 balanced）")
    parser.add_argument('--probe_fallback_tau', type=float, default=1.0,
                        help="BBSE 失败时的稳健默认 τ")
    parser.add_argument('--probe_tau_floor', type=float, default=1.0,
                        help="τ̂ 下限（护栏）：规则只允许放松得比参考度少")

    parser.add_argument('--probe_estimator', type=str, default="plan",
                        choices=["plan", "margin"],
                        help="跨类成本估计器（margin 为源侧分位数消融）")
    parser.add_argument('--probe_bootstrap', type=int, default=50,
                        help="BBSE 置信区间 bootstrap 次数")
    parser.add_argument('--probe_reinit', action='store_true',
                        help='probe 后将 encoder/classifier 重置为预训练初始化'
                             '（probe-then-reinit：估计质量与初始化解耦）')

    # ---- 崩坏重试护栏（仅 ot_auto 生效）----
    parser.add_argument('--collapse_f1_threshold', type=float, default=0.05,
                        help="best_valid_f1 不高于此值判定为训练崩坏")
    parser.add_argument('--no_collapse_retry', dest='collapse_retry',
                        action='store_false',
                        help="关闭崩坏后的无预训练重试（默认开启）")
    parser.set_defaults(collapse_retry=True)

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
    if args.method != "deepjdot_faithful" and args.ot_reg <= 0:
        raise ValueError("ot_reg must be > 0; epsilon=0 exact OT is not supported")
    if args.tau <= 0:
        raise ValueError("tau must be > 0")
    if args.method == "source_only":
        args.alignment_weight = 0.0
    if args.method in ("ablation_class_aware",):
        args.cost_mode = "class_aware"
    if args.method == "ablation_squared":
        args.cost = "sq_l2"
    if args.method == "deepjdot_faithful":
        # 原论文不使用 PIOT 的外层 alignment_weight；联合代价内部
        # 已包含 α 与 λ_t。balanced 仅表示固定边际，不走 Sinkhorn 分支。
        args.alignment_weight = 0.0
        args.cost = "sq_l2"
        args.cost_mode = "class_blind"
        args.variant = "balanced"
        if args.lr is None:
            # 原论文使用 Adam(lr=2e-4)；manifest 也会显式下发该值。
            args.lr = 2e-4
    elif args.method in ("ablation_balanced",):
        args.variant = "balanced"
    elif args.method in ("ot_balanced",):
        args.variant = "balanced"
    elif args.method == "ot_auto":
        args.variant = "probe_pending"  # 由 constraint_probe 决策回填
    else:
        args.variant = "unbalanced"

    if not args.task:
        args.task = f"{args.src}_to_{args.tgt}"
    if args.analysis_seed is None:
        args.analysis_seed = args.train_seed
    args.result_dir = build_run_dir(args)
    args.checkpoint_dir = os.path.join(args.result_dir, "checkpoint")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    set_seed(args.train_seed)
    configure_determinism(args.deterministic)
    print("=== OT-DAER ===")
    print(json.dumps({k: v for k, v in vars(args).items()}, indent=2, default=str))

    data = load_transfer_batches(args)
    src_batches = data["source"]
    tgt_pool_batches = data["target_pool"]
    tgt_valid_batches = data["target_valid"]
    tgt_eval_batches = data["target_eval"]
    print(f"[data] src={src_batches.n_samples} pool={tgt_pool_batches.n_samples} "
          f"valid={tgt_valid_batches.n_samples} eval={tgt_eval_batches.n_samples}")

    # ---- 模型 ----
    encoder = BertEncoder(model_name=args.encoder_model)
    classifier = BertClassifier()
    encoder = init_model(args, encoder)
    classifier = init_model(args, classifier)

    encoder, classifier, best_valid_f1, final = train(
        args, encoder, classifier, src_batches, tgt_pool_batches,
        data["source_analysis"],
        tgt_valid_batches, tgt_eval_batches, result_dir=args.result_dir)

    # ---- 崩坏重试护栏（仅 ot_auto + 有 valid 选择的实验）----
    # 判定：best_valid_f1 <= 阈值（完全失败）。动作：保留 probe 决策
    # （variant/tau 已回填），重初始化模型，跳过 probe 与 Stage 0 预训练
    # 完整重训一次，独立目录保留两次运行证据。
    if (args.method == "ot_auto"
            and getattr(args, "collapse_retry", True)
            and 0.0 <= best_valid_f1 <= args.collapse_f1_threshold):
        print(f"[collapse-guard] best_valid_f1={best_valid_f1:.4f} "
              f"<= {args.collapse_f1_threshold}: retry without pretrain init")
        encoder = init_model(args, BertEncoder(model_name=args.encoder_model))
        classifier = init_model(args, BertClassifier())
        args.probe_skip = True          # 保留首次 probe 决策，不再探测
        args.run_tag = (args.run_tag or "") + "_retry1"
        args.result_dir = build_run_dir(args)  # run_tag 参与哈希 → 新目录
        args.checkpoint_dir = os.path.join(args.result_dir, "checkpoint")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        encoder, classifier, best_valid_f1, final = train(
            args, encoder, classifier, src_batches, tgt_pool_batches,
            data["source_analysis"],
            tgt_valid_batches, tgt_eval_batches, result_dir=args.result_dir)

    print("=== Result of OT-DAER ===")
    print(f"method={args.method} experiment={args.experiment} "
          f"best_valid_f1={best_valid_f1:.4f} final={final}")


if __name__ == '__main__':
    main()

"""Unified final-paper training loop for OT-DAER.

The only runtime design axes are the method/experiment configuration and the
balanced versus KL-relaxed marginal constraint. Source and target empirical
marginals are uniform. Legacy mass allocation, BBSE, density weighting, and
coreset sampling are intentionally absent from this module.

Batch pairing follows the legacy DADER convention shared by all baselines:
plain ``zip`` (the excess of the longer domain is discarded each epoch).
"""
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import param
from metrics.ot import (
    build_cost,
    deepjdot_faithful as deepjdot_faithful_ot,
    ot_balanced,
    ot_unbalanced,
    transported_mass_stats,
)
from train.evaluate import evaluate_full
from utils import init_model, save_model
from main.experiment_common import deterministic_analysis_indices


def _uniform_marginals(n_source, n_target, device):
    return (
        torch.full((n_source,), 1.0 / n_source, device=device),
        torch.full((n_target,), 1.0 / n_target, device=device),
    )


def _batch_from_tensors(batches, indices, include_labels):
    indices = torch.as_tensor(
        indices, dtype=torch.long, device=batches.input_ids.device
    )
    data = (
        batches.input_ids[indices],
        batches.input_mask[indices],
        batches.segment_ids[indices],
    )
    if include_labels:
        return data + (
            batches.labels[indices], batches.exm_ids[indices]
        )
    return data + (batches.exm_ids[indices],)


@torch.no_grad()
def _representations(encoder, batch):
    """分析子集的 encoder 表示。仅用于记录/分析，不建 autograd 图
    （no_grad：128 样本的完整计算图约 6-8GB，会 pin 在显存里直到 OOM）。"""
    encoder.eval()
    if len(batch) == 5:
        ids, mask, segment, labels, exm_ids = batch
    else:
        ids, mask, segment, exm_ids = batch
        labels = None
    with torch.no_grad():
        reps = encoder(ids, mask, segment).float().cpu()
    return {
        "representation": reps,
        "labels": None if labels is None else labels.detach().cpu(),
        "exm_ids": exm_ids.detach().cpu(),
    }


def _save_artifact(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _save_json_artifact(path, payload):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _method_settings(args):
    method = args.method
    if method == "source_only":
        return "source_only", 0.0, "class_blind", "l2"
    if method == "ot_auto":
        # variant/tau 已由 probe 决策回填到 args.variant / args.tau
        return args.variant, args.alignment_weight, args.cost_mode, args.cost
    if method in ("ot_balanced", "ablation_balanced"):
        return "balanced", args.alignment_weight, args.cost_mode, args.cost
    if method == "deepjdot_faithful":
        # 原论文使用平方特征距离 + soft target CE + unregularized OT；
        # 该分支不走下面的 Sinkhorn/class-aware 简化路径。
        return "faithful", 0.0, "class_blind", "sq_l2"
    if method == "ablation_class_aware":
        return "unbalanced", args.alignment_weight, "class_aware", args.cost
    if method == "ablation_squared":
        return "unbalanced", args.alignment_weight, args.cost_mode, "sq_l2"
    if method in (
        "ot_unbalanced",
        "ablation_sequential",
        "sensitivity_tau",
        "sensitivity_lambda",
    ):
        return "unbalanced", args.alignment_weight, args.cost_mode, args.cost
    raise ValueError("Unsupported final method: {}".format(method))


def _run_sequential(args):
    return args.method == "ablation_sequential"


def _repeat_batches(batches):
    """Recreate the batch iterator after exhaustion so each pass can shuffle.

    未在当前 runtime 使用（最终协议采用 zip 取短配对），保留供配对方式消融用。
    """
    while True:
        for batch in batches:
            yield batch


def _compute_ot(args, feat_src, feat_tgt, labels_src, logits_tgt):
    if args.method == "deepjdot_faithful":
        return deepjdot_faithful_ot(
            feat_src.float(), feat_tgt.float(), labels_src, logits_tgt,
            alpha=args.deepjdot_alpha,
            lambda_t=args.deepjdot_lambda_t,
            solver=args.deepjdot_solver,
        )
    variant, _, cost_mode, cost = _method_settings(args)
    a, b = _uniform_marginals(feat_src.size(0), feat_tgt.size(0), feat_src.device)
    pseudo_tgt = logits_tgt.detach().argmax(dim=1)
    kwargs = {
        "reg": args.ot_reg,
        "cost": cost,
        "cost_mode": cost_mode,
        "normalize": bool(getattr(args, "ot_normalize", False)),
        "source_labels": labels_src if cost_mode == "class_aware" else None,
        "target_pseudo_labels": pseudo_tgt if cost_mode == "class_aware" else None,
        "class_cost_penalty": args.class_cost_penalty,
        "max_iter": args.ot_max_iter,
        "tol": args.ot_tol,
        "return_plan": True,
    }
    if variant == "balanced":
        loss, plan = ot_balanced(feat_src.float(), feat_tgt.float(),
                                 a=a, b=b, **kwargs)
    else:
        loss, plan = ot_unbalanced(
            feat_src.float(), feat_tgt.float(), a=a, b=b,
            reg_m=args.tau, **kwargs
        )
    cost_matrix = build_cost(
        feat_src.float(), feat_tgt.float(),
        cost=cost, cost_mode=cost_mode,
        normalize=bool(getattr(args, "ot_normalize", False)),
        source_labels=labels_src if cost_mode == "class_aware" else None,
        target_pseudo_labels=pseudo_tgt if cost_mode == "class_aware" else None,
        class_cost_penalty=args.class_cost_penalty,
    )
    return loss, plan, cost_matrix, a, b


def _write_run_config(args, logger):
    learning_rate = args.lr if args.lr is not None else param.d_learning_rate
    logger.log_config({
        "paper_runtime": "final",
        "experiment": args.experiment,
        "tier": args.tier,
        "task": args.task,
        "method": args.method,
        "src": args.src,
        "tgt": args.tgt,
        "target_file": args.target_file,
        "seed": args.seed,
        "train_seed": args.train_seed,
        "eval_protocol": args.eval_protocol,
        "target_valid_frac": float(param.target_valid_frac),
        "target_test_frac": float(param.target_test_frac),
        "target_split_stratified": bool(
            getattr(param, "target_split_stratified", False)
        ),
        "ot_normalize": bool(getattr(args, "ot_normalize", False)),
        "alignment_weight": args.alignment_weight,
        "ot_reg": args.ot_reg,
        "tau": args.tau,
        "learning_rate": learning_rate,
        "batch_size": args.batch_size,
        "cost": args.cost,
        "cost_mode": args.cost_mode,
        "class_cost_penalty": args.class_cost_penalty,
        "uniform_marginals": True,
        "target_labels_in_adaptation": False,
        "target_ground_truth_labels_in_adaptation": False,
        "target_unlabeled_empirical_distribution_used": True,
        "target_prior_explicitly_estimated_for_adaptation": False,
        "target_prior_label_conditioned_for_adaptation": False,
        "target_prior_recorded_for_posthoc_analysis": True,
        "target_prior_semantics": (
            "not explicitly estimated or label-conditioned for adaptation; "
            "the unlabeled empirical target distribution is used"
        ),
        "analysis_size": args.analysis_size,
        "analysis_seed": args.analysis_seed,
        "pairing": "zip_discard_excess",
        "deterministic": args.deterministic,
        "deepjdot_alpha": getattr(args, "deepjdot_alpha", None),
        "deepjdot_lambda_t": getattr(args, "deepjdot_lambda_t", None),
        "deepjdot_solver": getattr(args, "deepjdot_solver", None),
        "deepjdot_fixed_plan": args.method == "deepjdot_faithful",
        "deepjdot_unregularized_ot": args.method == "deepjdot_faithful",
    })


def train(args, encoder, classifier, src_batches, tgt_pool_batches,
          src_analysis_batches, tgt_valid_batches, tgt_eval_batches,
          result_dir=None):
    from train.run_logger import RunLogger

    logger = RunLogger(args, method=args.method, result_dir=result_dir)
    _write_run_config(args, logger)
    if tgt_pool_batches.labels is not None:
        raise RuntimeError(
            "Target adaptation batches must not expose target labels"
        )
    if len(tgt_pool_batches) == 0:
        raise RuntimeError(
            "Target adaptation pool produced zero batches "
            "(n_samples={} < batch_size={} with drop_last=True); "
            "reduce --batch_size or enlarge the target pool".format(
                tgt_pool_batches.n_samples, tgt_pool_batches.batch_size
            )
        )

    # ---- auto 模式：pilot probe 选择边际约束（论文 Algorithm 1）----
    joint_epochs = args.num_epochs
    if args.method == "ot_auto" and not getattr(args, "probe_skip", False):
        from train.constraint_probe import run_probe
        device = next(encoder.parameters()).device
        decision = run_probe(
            args, encoder, classifier, src_batches, tgt_pool_batches,
            device, result_dir=result_dir,
        )
        args.variant = decision["variant"]
        if decision.get("tau") is not None:
            args.tau = decision["tau"]
        joint_epochs = args.num_epochs - int(args.probe_pretrain_epochs)
        if joint_epochs <= 0:
            raise ValueError(
                "probe_pretrain_epochs must be smaller than num_epochs"
            )
        if getattr(logger, "config", None):
            logger.config["probe_decision"] = decision
        else:
            logger.log_config(extra={"probe_decision": decision})
        # probe-then-reinit（可选）：probe 借助预热表示完成估计后，将
        # encoder/classifier 重置回预训练初始化，联合训练从干净的初始化
        # 开始；probe 决策（variant/tau）与轮次预算口径（预热计入预算）
        # 均不受影响。用于分离预热的两个角色：让 probe 可用（保留）、
        # 作为联合训练初始化（丢弃）。
        if getattr(args, "probe_reinit", False):
            from modules.extractor import BertEncoder
            from modules.matcher import BertClassifier
            encoder = init_model(args, BertEncoder(model_name=args.encoder_model))
            classifier = init_model(args, BertClassifier())

    source_indices = deterministic_analysis_indices(
        src_analysis_batches.n_samples, args.analysis_size,
        args.analysis_seed, 0
    )
    target_indices = deterministic_analysis_indices(
        tgt_valid_batches.n_samples, args.analysis_size, args.analysis_seed, 1
    )
    source_analysis = _batch_from_tensors(
        src_analysis_batches, source_indices, True
    )
    target_analysis = _batch_from_tensors(
        tgt_valid_batches, target_indices, True
    )
    analysis_metadata = {
        "analysis_size_requested": int(args.analysis_size),
        "analysis_size_actual_source": int(len(source_indices)),
        "analysis_size_actual_target": int(len(target_indices)),
        "analysis_seed": int(args.analysis_seed),
        "selection_rule": (
            "numpy.default_rng(seed + domain_salt), choice without replacement, "
            "sorted indices"
        ),
        "source_split": "source_analysis_holdout",
        "target_split": "target_valid",
        "before_state": (
            "probe_pretrained_encoder"
            if args.method == "ot_auto"
            and not getattr(args, "probe_reinit", False)
            else "initial_encoder_before_training"
        ),
        "after_state": "adapted_encoder",
        "source_indices": [int(x) for x in source_indices],
        "target_indices": [int(x) for x in target_indices],
        "source_example_ids": [
            int(x) for x in source_analysis[4].detach().cpu().tolist()
        ],
        "target_example_ids": [
            int(x) for x in target_analysis[4].detach().cpu().tolist()
        ],
        "pi_hat_s": float(source_analysis[3].float().mean().item()),
        "pi_hat_t": float(target_analysis[3].float().mean().item()),
    }
    analysis_metadata["abs_delta_pi_hat"] = abs(
        analysis_metadata["pi_hat_s"] - analysis_metadata["pi_hat_t"]
    )
    _save_artifact(
        os.path.join(result_dir, "artifacts", "representations_before.pt"),
        {
            "source": _representations(encoder, source_analysis),
            "target": _representations(encoder, target_analysis),
        },
    )
    _save_artifact(
        os.path.join(result_dir, "artifacts", "analysis_metadata.pt"),
        analysis_metadata,
    )
    _save_json_artifact(
        os.path.join(result_dir, "artifacts", "analysis_metadata.json"),
        analysis_metadata,
    )

    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if args.lr is not None else param.d_learning_rate
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=lr
    )
    use_amp = bool(args.amp) and torch.cuda.is_available()
    best_valid_f1 = -1.0
    best_epoch = -1
    # --selection last_epoch：不评估目标验证集、直接评估末轮（label-free 选模）。
    # 此模式下 best_valid_f1 恒为 -1，main_ot 的崩坏重试（依赖验证标签）
    # 因守卫条件 0.0 <= best_valid_f1 不成立而自然关闭。
    use_target_validation_selection = (
        args.experiment != "experiment_B"
        and getattr(args, "selection", "validation") != "last_epoch"
    )
    sequential = _run_sequential(args)
    sequential_epochs = args.sequential_align_epochs or max(1, args.num_epochs // 2)
    for epoch in range(joint_epochs):
        # legacy DADER 口径（与所有基线一致）：zip 配对取两者较短，
        # 较长域多余的 batch 不参与本 epoch
        n_steps = min(len(src_batches), len(tgt_pool_batches))
        if args.max_steps > 0:
            n_steps = min(n_steps, args.max_steps)

        epoch_match = 0.0
        epoch_align = 0.0
        epoch_total = 0.0
        epoch_mass = []

        if sequential and epoch >= sequential_epochs:
            if epoch == sequential_epochs:
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
            encoder.eval()
            classifier.train()
        else:
            encoder.train()
            classifier.train()

        for step, (src, tgt) in enumerate(zip(src_batches, tgt_pool_batches)):
            if step >= n_steps:
                break
            ids_s, mask_s, seg_s, labels_s, _ = src
            ids_t, mask_t, seg_t, _ = tgt
            optimizer.zero_grad()

            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else nullcontext()
            )
            with amp_context:
                feat_src = encoder(ids_s, mask_s, seg_s)
                logits_src = classifier(feat_src)
                match_loss = criterion(logits_src, labels_s)
                feat_tgt = encoder(ids_t, mask_t, seg_t)
                logits_tgt = classifier(feat_tgt)

            alignment_loss = torch.zeros((), device=ids_s.device)
            plan = None
            cost_matrix = None
            a = b = None
            alignment_stage = sequential and epoch < sequential_epochs
            use_alignment = args.method != "source_only" and (
                alignment_stage or not sequential
            )
            if use_alignment:
                alignment_loss, plan, cost_matrix, a, b = _compute_ot(
                    args, feat_src.float(), feat_tgt.float(), labels_s, logits_tgt
                )
                if alignment_stage:
                    loss = alignment_loss
                elif args.method == "deepjdot_faithful":
                    # 原论文固定 plan 后的联合损失；不再额外乘 PIOT 的 λ。
                    loss = match_loss + alignment_loss
                else:
                    loss = match_loss + args.alignment_weight * alignment_loss
            else:
                loss = match_loss

            # 响亮失败而非静默污染：NaN/Inf loss 会把 encoder 冲成 NaN，
            # 之后的训练不可逆（历史上表现为全 0 F1 + eval loss=nan 的
            # "completed" run，会污染聚合与网格选择）。
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(
                    "Non-finite loss at epoch {} step {} "
                    "(match_loss={}, alignment_loss={}). Likely OT solver "
                    "numerical underflow (cost scale vs ot_reg); check the "
                    "solver stability settings in metrics/ot.py.".format(
                        epoch + 1, step + 1,
                        float(match_loss.detach()),
                        float(alignment_loss.detach()),
                    )
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()

            epoch_match += float(match_loss.detach().item())
            epoch_align += float(alignment_loss.detach().item())
            epoch_total += float(loss.detach().item())
            if plan is not None:
                epoch_mass.append(
                    transported_mass_stats(plan, a=a, b=b)
                )
            if (step + 1) % args.log_step == 0:
                logger.log_step(
                    epoch, step, n_steps,
                    float(match_loss.detach().item()),
                    float(alignment_loss.detach().item()),
                    epoch_mass[-1]["total_mass"] if epoch_mass else 0.0,
                )

        if use_target_validation_selection:
            valid = evaluate_full(
                encoder, classifier, tgt_valid_batches, return_stats=True
            )
            f1_valid, _, _, valid_loss, valid_stats = valid
            valid_match_recall = valid_stats["match_recall"]
        else:
            f1_valid = -1.0
            valid_loss = None
            valid_match_recall = None
        mean_mass = (
            float(np.mean([x["total_mass"] for x in epoch_mass]))
            if epoch_mass else 0.0
        )
        is_best = logger.log_epoch(
            epoch,
            f1_valid,
            valid_loss,
            ot_diag={
                "match_loss": epoch_match / max(n_steps, 1),
                "alignment_loss": epoch_align / max(n_steps, 1),
                "total_loss": epoch_total / max(n_steps, 1),
                "total_mass": mean_mass,
                "valid_match_recall": valid_match_recall,
                "selection": (
                    "target_validation"
                    if use_target_validation_selection
                    else "last_epoch_no_target_label_selection"
                ),
            },
        )
        if is_best:
            best_valid_f1 = f1_valid
            best_epoch = epoch
            save_model(args, encoder, "encoder_best.pt")
            save_model(args, classifier, "classifier_best.pt")

    if best_epoch < 0:
        best_epoch = args.num_epochs - 1
        logger.best_epoch = best_epoch + 1
        save_model(args, encoder, "encoder_best.pt")
        save_model(args, classifier, "classifier_best.pt")

    encoder = init_model(args, encoder, restore="encoder_best.pt")
    classifier = init_model(args, classifier, restore="classifier_best.pt")
    final = evaluate_full(
        encoder, classifier, tgt_eval_batches, verbose=True, return_stats=True
    )
    f1, precision, recall, eval_loss, final_stats = final

    after_source = _representations(encoder, source_analysis)
    after_target = _representations(encoder, target_analysis)
    _save_artifact(
        os.path.join(result_dir, "artifacts", "representations_after.pt"),
        {"source": after_source, "target": after_target},
    )

    source_ids, source_mask, source_seg, source_labels, _ = source_analysis
    target_ids, target_mask, target_seg, target_labels, _ = target_analysis
    with torch.no_grad():
        # 仅用于最终分析工件，不需要梯度；no_grad 避免多张完整计算图
        # 同时 pin 在显存中（大任务上曾因此 OOM）
        source_rep = encoder(source_ids, source_mask, source_seg).float()
        target_rep = encoder(target_ids, target_mask, target_seg).float()
        target_logits = classifier(target_rep)
        _, plan, cost_matrix, a, b = _compute_ot(
            args, source_rep, target_rep, source_labels, target_logits
        ) if args.method != "source_only" else (None, None, None, None, None)
    transport_stats = {}
    if plan is not None:
        analysis_metadata["gamma_source_indices"] = analysis_metadata[
            "source_indices"
        ]
        analysis_metadata["gamma_target_indices"] = analysis_metadata[
            "target_indices"
        ]
        analysis_metadata["gamma_pi_hat_s"] = analysis_metadata["pi_hat_s"]
        analysis_metadata["gamma_pi_hat_t"] = analysis_metadata["pi_hat_t"]
        analysis_metadata["gamma_abs_delta_pi_hat"] = analysis_metadata[
            "abs_delta_pi_hat"
        ]
        _save_artifact(
            os.path.join(result_dir, "artifacts", "analysis_metadata.pt"),
            analysis_metadata,
        )
        _save_json_artifact(
            os.path.join(result_dir, "artifacts", "analysis_metadata.json"),
            analysis_metadata,
        )
        transport_stats = transported_mass_stats(
            plan, a=a, b=b, cost=cost_matrix,
            source_labels=source_labels,
            target_labels=target_labels,
        )
        _save_artifact(
            os.path.join(result_dir, "artifacts", "transport_plan.pt"),
            {
                "gamma": plan.detach().cpu(),
                "cost": cost_matrix.detach().cpu(),
                "source_labels": source_labels.detach().cpu(),
                "target_labels": target_labels.detach().cpu(),
                "source_marginal": a.detach().cpu(),
                "target_marginal": b.detach().cpu(),
                "analysis": analysis_metadata,
                "stats": transport_stats,
            },
        )

    logger.log_final(
        f1, precision, recall,
        extra={
            "loss": eval_loss,
            **final_stats,
            "best_epoch": best_epoch + 1,
            "transport": transport_stats,
        },
    )
    logger.finalize()
    if not getattr(args, "keep_checkpoint", False):
        # best checkpoint 已 restore 进内存、终评与 artifact 均已落盘；
        # 磁盘副本（mBERT 约 700MB/run）无任何下游消费，默认删除。
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    return encoder, classifier, best_valid_f1, {
        "final_f1": f1,
        "final_precision": precision,
        "final_recall": recall,
        "match_recall": final_stats["match_recall"],
    }

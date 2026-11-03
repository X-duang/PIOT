"""Final-protocol InvGAN+KD baseline trainer.

Two-stage objective identical to the legacy implementation
(train/pretrain.pretrain_best + train/adapt_invgan_kd.adapt_best):

1. Source pretrain: encoder+classifier with cross-entropy on labeled source
   for ``pre_epochs`` epochs, keeping the target-validation-best state
   (legacy saved it to disk; the final runtime keeps it in memory).
2. Adversarial + KD adaptation: a target encoder initialized from the source
   encoder is trained with
   ``alpha * BCE(D(F'(x_tgt)), 1) + beta * T^2 * KL(teacher || student)``
   where the discriminator D is trained with BCE on
   [F'(x_src) -> 1, F'(x_tgt) -> 0] and weight-clamped to +-clip_value,
   exactly as in the legacy code.

Training dynamics follow the legacy code as well: plain ``zip`` pairing
(discard the excess of the longer domain), no gradient clipping in the
pretrain stage, and gradient-norm clipping of the target encoder only in the
adaptation stage.

Everything around the objective — shared label-blind target split,
target-label firewall, validation-based model selection, evaluate_full
metrics, RunLogger artifacts and checkpoint identity — follows the final
paper protocol (train/ot_trainer.py).
"""
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import param
from train.evaluate import evaluate_full
from utils import init_model, save_model
from main.experiment_common import deterministic_analysis_indices
from train.ot_trainer import (
    _batch_from_tensors,
    _representations,
    _save_artifact,
    _save_json_artifact,
)


def _write_run_config(args, logger):
    adapt_lr = args.lr if args.lr is not None else param.d_learning_rate
    pre_lr = args.pre_lr if args.pre_lr is not None else param.c_learning_rate
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
        "selection": args.selection,
        "num_epochs": args.num_epochs,
        "target_valid_frac": float(param.target_valid_frac),
        "target_test_frac": float(param.target_test_frac),
        "target_split_stratified": bool(
            getattr(param, "target_split_stratified", False)
        ),
        # InvGAN+KD 的 KD 权重 beta 占用 alignment_weight 字段（对齐/适应
        # 损失权重，与 OT 的 λ、MMD 的 beta 同槽位、同 run identity）。
        "alignment_weight": args.alignment_weight,
        "kd_weight": args.alignment_weight,
        "adv_weight": args.alpha,
        "temperature": args.temperature,
        "clip_value": args.clip_value,
        "pre_epochs": args.pre_epochs,
        "pre_learning_rate": pre_lr,
        "learning_rate": adapt_lr,
        "batch_size": args.batch_size,
        "target_labels_in_adaptation": False,
        "target_ground_truth_labels_in_adaptation": False,
        "analysis_size": args.analysis_size,
        "analysis_seed": args.analysis_seed,
        "pairing": "zip_discard_excess",
        "deterministic": args.deterministic,
    })


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _pretrain_source(args, src_encoder, classifier, src_batches,
                     tgt_valid_batches, logger, epoch_offset=0):
    """Legacy pretrain_best: CE on source, target-validation-best state.

    The legacy version persisted the best-of-first-5-epochs weights to disk;
    the final runtime keeps the same selection rule but holds the best state
    in memory. Pretrain epochs are logged with stage="source_pretrain"; the
    summary best_* fields are reset afterwards so model selection reported
    for the run reflects the adaptation stage only.
    """
    pre_lr = args.pre_lr if args.pre_lr is not None else param.c_learning_rate
    optimizer = optim.Adam(
        list(src_encoder.parameters()) + list(classifier.parameters()),
        lr=pre_lr,
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(args.amp) and torch.cuda.is_available()
    use_selection = (args.experiment != "experiment_B" and
                     getattr(args, "selection", "validation") != "last_epoch")
    best_f1 = -1.0
    best_encoder_state = _cpu_state(src_encoder)
    best_classifier_state = _cpu_state(classifier)
    for epoch in range(args.pre_epochs):
        src_encoder.train()
        classifier.train()
        epoch_cls = 0.0
        n_steps = len(src_batches)
        if args.max_steps > 0:
            n_steps = min(n_steps, args.max_steps)
        for step, src in enumerate(src_batches):
            if step >= n_steps:
                break
            ids_s, mask_s, seg_s, labels_s, _ = src
            optimizer.zero_grad()
            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else nullcontext()
            )
            with amp_context:
                feat = src_encoder(ids_s, mask_s, seg_s)
                logits = classifier(feat)
                cls_loss = criterion(logits, labels_s)
            if not torch.isfinite(cls_loss.detach()):
                raise RuntimeError(
                    "Non-finite pretrain loss at epoch {} step {}.".format(
                        epoch + 1, step + 1
                    )
                )
            cls_loss.backward()
            # legacy pretrain_best 无梯度裁剪，保持原样
            optimizer.step()
            epoch_cls += float(cls_loss.detach().item())
            if (step + 1) % args.log_step == 0:
                logger.log_step(
                    epoch_offset + epoch, step, n_steps,
                    float(cls_loss.detach().item()), 0.0,
                )
        if use_selection:
            f1_valid, _, _, valid_loss, valid_stats = evaluate_full(
                src_encoder, classifier, tgt_valid_batches, return_stats=True
            )
            valid_match_recall = valid_stats["match_recall"]
        else:
            f1_valid, valid_loss, valid_match_recall = -1.0, None, None
        logger.log_epoch(
            epoch_offset + epoch, f1_valid, valid_loss,
            ot_diag={
                "stage": "source_pretrain",
                "match_loss": epoch_cls / max(n_steps, 1),
                "valid_match_recall": valid_match_recall,
                "selection": ("target_validation" if use_selection
                              else "last_epoch_no_target_label_selection"),
            },
        )
        if use_selection and f1_valid > best_f1:
            best_f1 = f1_valid
            best_encoder_state = _cpu_state(src_encoder)
            best_classifier_state = _cpu_state(classifier)

    if use_selection:
        # experiment_B 与其他方法一致：不做目标 valid 选状态，用末轮。
        src_encoder.load_state_dict(best_encoder_state)
        classifier.load_state_dict(best_classifier_state)
    # summary 的 best_* 只反映 adaptation 阶段（见 docstring）。
    logger.best_valid_f1 = -1.0
    logger.best_epoch = -1
    return src_encoder, classifier


def train(args, src_encoder, tgt_encoder, discriminator, classifier,
          src_batches, tgt_pool_batches, src_analysis_batches,
          tgt_valid_batches, tgt_eval_batches, result_dir=None):
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

    # ---- 阶段 1：源域预训练（legacy pretrain_best 口径） ----
    src_encoder, classifier = _pretrain_source(
        args, src_encoder, classifier, src_batches, tgt_valid_batches, logger
    )
    for params in src_encoder.parameters():
        params.requires_grad = False
    for params in classifier.parameters():
        params.requires_grad = False
    src_encoder.eval()
    classifier.eval()
    tgt_encoder.load_state_dict(src_encoder.state_dict())

    # ---- 分析快照（before = 适配前的目标编码器） ----
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
        "before_state": "source_pretrained_target_encoder_before_adaptation",
        "after_state": "adapted_target_encoder",
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
            "source": _representations(tgt_encoder, source_analysis),
            "target": _representations(tgt_encoder, target_analysis),
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

    # ---- 阶段 2：对抗 + KD 适配（legacy adapt_best 口径） ----
    adapt_lr = args.lr if args.lr is not None else param.d_learning_rate
    BCELoss = nn.BCELoss()
    KLDivLoss = nn.KLDivLoss(reduction="batchmean")
    optimizer_G = optim.Adam(tgt_encoder.parameters(), lr=adapt_lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=adapt_lr)
    use_amp = bool(args.amp) and torch.cuda.is_available()
    best_valid_f1 = -1.0
    best_epoch = -1
    use_target_validation_selection = (args.experiment != "experiment_B" and
                                       getattr(args, "selection", "validation") != "last_epoch")
    T = args.temperature
    device = next(tgt_encoder.parameters()).device

    for epoch in range(args.num_epochs):
        # legacy 口径：zip 配对取两者较短，较长域多余的 batch 不参与
        n_steps = min(len(src_batches), len(tgt_pool_batches))
        if args.max_steps > 0:
            n_steps = min(n_steps, args.max_steps)

        epoch_gen = 0.0
        epoch_dis = 0.0
        epoch_kd = 0.0
        epoch_dacc = 0.0

        tgt_encoder.train()
        discriminator.train()

        for step, (src, tgt) in enumerate(zip(src_batches, tgt_pool_batches)):
            if step >= n_steps:
                break
            ids_s, mask_s, seg_s, _, _ = src
            ids_t, mask_t, seg_t, _ = tgt

            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else nullcontext()
            )
            with amp_context:
                with torch.no_grad():
                    feat_src = src_encoder(ids_s, mask_s, seg_s)
                feat_src_tgt = tgt_encoder(ids_s, mask_s, seg_s)
                feat_tgt = tgt_encoder(ids_t, mask_t, seg_t)
                feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

                # ---- D step ----
                pred_concat = discriminator(feat_concat.detach())
                label_src = torch.ones(
                    feat_src_tgt.size(0), 1, device=device,
                    dtype=pred_concat.dtype,
                )
                label_tgt = torch.zeros(
                    feat_tgt.size(0), 1, device=device,
                    dtype=pred_concat.dtype,
                )
                label_concat = torch.cat((label_src, label_tgt), 0)
                dis_loss = BCELoss(pred_concat, label_concat)

            optimizer_D.zero_grad()
            dis_loss.backward()
            for p in discriminator.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)
            optimizer_D.step()

            # 诊断量：D 的分类准确率（legacy 的 max(1) 写法对 (N,1) 输出
            # 无意义，这里按 0.5 阈值算真实准确率；仅记录，不影响训练）。
            d_acc = (
                (pred_concat.detach() > 0.5).float() == label_concat
            ).float().mean()

            # ---- G step（legacy 不回传 D 的梯度更新，但图穿过 D） ----
            with amp_context:
                pred_tgt = discriminator(feat_tgt)
                with torch.no_grad():
                    src_prob = F.softmax(classifier(feat_src) / T, dim=-1)
                tgt_prob = F.log_softmax(
                    classifier(feat_src_tgt) / T, dim=-1
                )
                kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T
                gen_loss = BCELoss(
                    pred_tgt,
                    torch.ones(
                        feat_tgt.size(0), 1, device=device,
                        dtype=pred_tgt.dtype,
                    ),
                )
                loss_tgt = args.alpha * gen_loss + args.alignment_weight * kd_loss

            if not (
                torch.isfinite(loss_tgt.detach())
                and torch.isfinite(dis_loss.detach())
            ):
                raise RuntimeError(
                    "Non-finite loss at adapt epoch {} step {} "
                    "(gen_loss={}, kd_loss={}, dis_loss={}).".format(
                        epoch + 1, step + 1,
                        float(gen_loss.detach()),
                        float(kd_loss.detach()),
                        float(dis_loss.detach()),
                    )
                )

            optimizer_G.zero_grad()
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(
                tgt_encoder.parameters(), args.max_grad_norm
            )
            optimizer_G.step()

            epoch_gen += float(gen_loss.detach().item())
            epoch_dis += float(dis_loss.detach().item())
            epoch_kd += float(kd_loss.detach().item())
            epoch_dacc += float(d_acc.detach().item())
            if (step + 1) % args.log_step == 0:
                logger.log_step(
                    args.pre_epochs + epoch, step, n_steps,
                    float(gen_loss.detach().item()),
                    float(kd_loss.detach().item()),
                )

        if use_target_validation_selection:
            valid = evaluate_full(
                tgt_encoder, classifier, tgt_valid_batches, return_stats=True
            )
            f1_valid, _, _, valid_loss, valid_stats = valid
            valid_match_recall = valid_stats["match_recall"]
        else:
            f1_valid = -1.0
            valid_loss = None
            valid_match_recall = None
        is_best = logger.log_epoch(
            args.pre_epochs + epoch,
            f1_valid,
            valid_loss,
            ot_diag={
                "stage": "adversarial_kd_adaptation",
                "gen_loss": epoch_gen / max(n_steps, 1),
                "d_loss": epoch_dis / max(n_steps, 1),
                "kd_loss": epoch_kd / max(n_steps, 1),
                "d_acc": epoch_dacc / max(n_steps, 1),
                "total_loss": (epoch_gen + epoch_kd) / max(n_steps, 1),
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
            save_model(args, tgt_encoder, "encoder_best.pt")
            save_model(args, classifier, "classifier_best.pt")

    if best_epoch < 0:
        best_epoch = args.num_epochs - 1
        logger.best_epoch = args.pre_epochs + best_epoch + 1
        save_model(args, tgt_encoder, "encoder_best.pt")
        save_model(args, classifier, "classifier_best.pt")

    tgt_encoder = init_model(args, tgt_encoder, restore="encoder_best.pt")
    classifier = init_model(args, classifier, restore="classifier_best.pt")
    final = evaluate_full(
        tgt_encoder, classifier, tgt_eval_batches,
        verbose=True, return_stats=True,
    )
    f1, precision, recall, eval_loss, final_stats = final

    _save_artifact(
        os.path.join(result_dir, "artifacts", "representations_after.pt"),
        {
            "source": _representations(tgt_encoder, source_analysis),
            "target": _representations(tgt_encoder, target_analysis),
        },
    )

    logger.log_final(
        f1, precision, recall,
        extra={
            "loss": eval_loss,
            **final_stats,
            "best_epoch": args.pre_epochs + best_epoch + 1,
        },
    )
    logger.finalize()
    if not getattr(args, "keep_checkpoint", False):
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    return tgt_encoder, classifier, best_valid_f1, {
        "final_f1": f1,
        "final_precision": precision,
        "final_recall": recall,
        "match_recall": final_stats["match_recall"],
    }

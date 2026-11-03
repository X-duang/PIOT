"""Final-protocol MMD baseline trainer.

Objective identical to the legacy implementation (train/adapt_mmd.py):
cross-entropy on labeled source + ``alignment_weight`` (beta) times the
multi-kernel RBF MMD between source and target encoder features. Training
dynamics also follow the legacy code: plain ``zip`` pairing (the excess of
the longer domain is discarded) and no gradient clipping.

Everything around the objective — shared label-blind target split, target-label
firewall, validation-based model selection, evaluate_full metrics,
RunLogger artifacts and checkpoint identity — follows the final paper
protocol (train/ot_trainer.py), so results aggregate and compare directly
with the OT-DAER methods.
"""
import os
import shutil
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import param
from metrics.mmd import mmd_rbf_noaccelerate
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
        "selection": args.selection,
        "num_epochs": args.num_epochs,
        "target_valid_frac": float(param.target_valid_frac),
        "target_test_frac": float(param.target_test_frac),
        "target_split_stratified": bool(
            getattr(param, "target_split_stratified", False)
        ),
        # MMD 的 beta 就是总损失里的对齐权重，占用 alignment_weight 字段，
        # 与 OT 方法的 λ 同语义、同 run identity。
        "alignment_weight": args.alignment_weight,
        "mmd_beta": args.alignment_weight,
        "mmd_kernel": "rbf_multikernel(kernel_mul=2.0, kernel_num=5)",
        "learning_rate": learning_rate,
        "batch_size": args.batch_size,
        "target_labels_in_adaptation": False,
        "target_ground_truth_labels_in_adaptation": False,
        "analysis_size": args.analysis_size,
        "analysis_seed": args.analysis_seed,
        "pairing": "zip_discard_excess",
        "deterministic": args.deterministic,
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
        "before_state": "initial_encoder_before_training",
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
    use_target_validation_selection = (args.experiment != "experiment_B" and
                                       getattr(args, "selection", "validation") != "last_epoch")
    for epoch in range(args.num_epochs):
        # legacy 口径：zip 配对取两者较短，较长域多余的 batch 不参与
        n_steps = min(len(src_batches), len(tgt_pool_batches))
        if args.max_steps > 0:
            n_steps = min(n_steps, args.max_steps)

        epoch_match = 0.0
        epoch_align = 0.0
        epoch_total = 0.0

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

            alignment_loss = mmd_rbf_noaccelerate(
                feat_src.float(), feat_tgt.float()
            )
            loss = match_loss + args.alignment_weight * alignment_loss

            # 与 ot_trainer 一致：非有限损失直接失败，避免静默污染聚合。
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(
                    "Non-finite loss at epoch {} step {} "
                    "(match_loss={}, mmd_loss={}).".format(
                        epoch + 1, step + 1,
                        float(match_loss.detach()),
                        float(alignment_loss.detach()),
                    )
                )

            loss.backward()
            # legacy adapt_mmd 无梯度裁剪，保持原样
            optimizer.step()

            epoch_match += float(match_loss.detach().item())
            epoch_align += float(alignment_loss.detach().item())
            epoch_total += float(loss.detach().item())
            if (step + 1) % args.log_step == 0:
                logger.log_step(
                    epoch, step, n_steps,
                    float(match_loss.detach().item()),
                    float(alignment_loss.detach().item()),
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
        is_best = logger.log_epoch(
            epoch,
            f1_valid,
            valid_loss,
            ot_diag={
                "match_loss": epoch_match / max(n_steps, 1),
                "alignment_loss": epoch_align / max(n_steps, 1),
                "total_loss": epoch_total / max(n_steps, 1),
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

    _save_artifact(
        os.path.join(result_dir, "artifacts", "representations_after.pt"),
        {
            "source": _representations(encoder, source_analysis),
            "target": _representations(encoder, target_analysis),
        },
    )

    logger.log_final(
        f1, precision, recall,
        extra={
            "loss": eval_loss,
            **final_stats,
            "best_epoch": best_epoch + 1,
        },
    )
    logger.finalize()
    if not getattr(args, "keep_checkpoint", False):
        shutil.rmtree(args.checkpoint_dir, ignore_errors=True)
    return encoder, classifier, best_valid_f1, {
        "final_f1": f1,
        "final_precision": precision,
        "final_recall": recall,
        "match_recall": final_stats["match_recall"],
    }

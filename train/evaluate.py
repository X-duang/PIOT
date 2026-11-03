"""Adaptation to train target encoder."""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("..")
import param
from utils import make_cuda,save_model,init_model
import csv
import os
import datetime

def evaluate(encoder, classifier, data_loader,args=None,flag=None,discriminator=None,exp_idx=None):
    """Evaluation for encoder and classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0
    need_preds = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    count = 0
    # evaluate network
    for (reviews, mask,segment, labels,exm_id) in data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        truelen = torch.sum(mask, dim=1)

        with torch.no_grad():
            feat = encoder(reviews, mask,segment)
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        if args != None and (args.need_pred_res or args.al1 or args.al2):
            for i in range(len(labels)):
                confidence = preds.data[i].cpu().numpy().tolist()
                entro = abs(confidence[1]-confidence[0])
                need_preds.append([exm_id[i].item(),labels[i].item(),pred_cls[i].item(),confidence,entro])

        acc += pred_cls.eq(labels.data).cpu().sum().item()
        for i in range(len(labels)):
            if labels[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1

    div_safe = 0.000001
    print("===== RES ====")
    print("p",p)
    print("tp",tp)
    print("fp",fp)
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    print("recall",recall)
    print("precision",precision)
    print("f1",f1)

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    return f1


@torch.no_grad()
def evaluate_full(encoder, classifier, data_loader, verbose=False,
                  return_stats=False):
    """统一评估：返回论文要求的正类 P/R/F1。

    The default four-value return is kept for old callers. Final experiment
    code can request confusion statistics with ``return_stats=True``.

    与 evaluate 的口径一致（正类 F1），但旧 evaluate 只返回 f1 且四处打印；
    新代码一律用本函数。旧 evaluate 保留以兼容 DADER 原有管线。
    """
    encoder.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    tp = fp = fn = tn = 0
    n_batches = 0
    n_samples = 0
    for batch in data_loader:
        reviews, mask, segment, labels = batch[0], batch[1], batch[2], batch[3]
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        feat = encoder(reviews, mask, segment)
        preds = classifier(feat)
        loss += criterion(preds, labels).item()
        n_batches += 1
        n_samples += labels.numel()
        pred_cls = preds.data.max(1)[1]
        tp += ((pred_cls == 1) & (labels == 1)).sum().item()
        fp += ((pred_cls == 1) & (labels == 0)).sum().item()
        fn += ((pred_cls == 0) & (labels == 1)).sum().item()
        tn += ((pred_cls == 0) & (labels == 0)).sum().item()

    eps = 1e-6
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    loss /= max(n_batches, 1)
    if verbose:
        print(f"[evaluate_full] n={n_samples} P={precision:.4f} "
              f"R={recall:.4f} F1={f1:.4f} loss={loss:.4f}")
    if return_stats:
        return f1, precision, recall, loss, {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "match_recall": float(recall),
        }
    return f1, precision, recall, loss

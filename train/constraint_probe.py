# train/constraint_probe.py
"""边际约束选择器（论文 Algorithm 1 / Section V-B）。

在适配训练开始前执行一次 pilot probe：
  Stage 0  源域 match-loss 预训练（就地修改模型，作为主循环的初始化）；
  1        pilot 子采样（源侧按标签比例分层、目标侧纯随机，cap 保证
           开销与数据规模无关）；
  2        BBSE 聚合级估计目标先验 pi_t（带 bootstrap 置信半宽 w），
           从不产生逐样本伪标签；
  3        pilot 表示间跑一次 balanced Sinkhorn，跨类成本 C_cross 用
           源类质心的最近邻区域指派打分（一次性、无梯度、只打分不训练，
           高估方向 = 更强放松 = 安全方向）；
  4        闭式规则 tau_hat = alpha * C_cross / surcharge(rho_hat)，
           tau_hat >= tau_max 时退化 balanced；估计不可靠时回退稳健默认。

叙事纪律（论文 Remark 9）：训练目标全程 class-blind；区域指派仅用于
给 pilot plan 打分。决策与全部诊断量写入 artifacts/probe_decision.json。
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import param
from metrics.ot import build_cost, ot_balanced

PRIOR_EPS = 1e-4


def _psi(u):
    """generalized-KL 的 psi(u) = (1-u)log(1-u) + u（定义域 [0,1)）。"""
    u = min(max(float(u), 0.0), 1.0 - 1e-12)
    return (1.0 - u) * math.log(1.0 - u) + u


# ---------------- 1. 子采样 ----------------

def subsample_indices(n_total, frac, cap, seed, labels=None):
    """pilot 子采样索引。源侧给 labels 时按比例分层（保持先验无偏且
    保证稀有类至少 1 个）；目标侧 labels=None 纯随机。"""
    rng = np.random.default_rng(seed)
    n_p = max(1, min(int(round(n_total * frac)), int(cap), n_total))
    if labels is None:
        return np.sort(rng.choice(n_total, n_p, replace=False))
    labels = np.asarray(labels)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n_pos = int(round(n_p * len(idx_pos) / max(n_total, 1)))
    n_pos = min(max(n_pos, 1 if len(idx_pos) else 0), len(idx_pos))
    n_neg = min(n_p - n_pos, len(idx_neg))
    parts = []
    if n_pos:
        parts.append(rng.choice(idx_pos, n_pos, replace=False))
    if n_neg:
        parts.append(rng.choice(idx_neg, n_neg, replace=False))
    return np.sort(np.concatenate(parts))


# ---------------- 2. Stage 0：源预训练 ----------------

def source_pretrain(args, encoder, classifier, src_batches, device):
    """match-loss 源预训练（就地修改）。返回 wall-clock 秒数。"""
    epochs = int(getattr(args, "probe_pretrain_epochs", 0))
    if epochs <= 0:
        return 0.0
    t0 = time.time()
    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    lr = args.lr if args.lr is not None else param.d_learning_rate
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=lr
    )
    params = list(encoder.parameters()) + list(classifier.parameters())
    for _ in range(epochs):
        for batch in src_batches:
            ids, mask, seg, labels = (
                batch[0].to(device), batch[1].to(device),
                batch[2].to(device), batch[3].to(device),
            )
            optimizer.zero_grad()
            loss = criterion(classifier(encoder(ids, mask, seg)), labels)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError("Non-finite loss during probe pretrain")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
    return time.time() - t0


# ---------------- 表示提取 ----------------

@torch.no_grad()
def encode_subset(encoder, batches, indices, device, chunk=256):
    """对 tensor batches 的指定索引前向取表示（no_grad，分块防 OOM）。"""
    encoder.eval()
    feats = []
    for i in range(0, len(indices), chunk):
        idx = torch.as_tensor(indices[i:i + chunk], dtype=torch.long)
        ids = batches.input_ids[idx].to(device)
        mask = batches.input_mask[idx].to(device)
        seg = batches.segment_ids[idx].to(device)
        feats.append(encoder(ids, mask, seg).float().cpu())
    return torch.cat(feats)


# ---------------- 3. BBSE 目标先验 ----------------

@torch.no_grad()
def bbse_target_prior(encoder, classifier, z_src, y_src, z_tgt,
                      n_bootstrap=50, seed=0):
    """BBSE 聚合级先验估计（Lipton et al. 2018）。

    返回 (pi_t_hat, pi_s, w)：w 为 bootstrap 95% 置信半宽；
    混淆矩阵病态时 w=inf（触发上层回退）。用 softmax 软概率的
    均值（标准 BBSE），只用均值统计，不产生逐样本伪标签。
    """
    classifier.eval()
    # 表示在 CPU（encode_subset 收 CPU 防显存占用），前向需挪到模型设备
    param0 = next(classifier.parameters(), None)
    device = param0.device if param0 is not None else z_src.device
    prob_s = torch.softmax(
        classifier(z_src.to(device)), dim=1
    ).cpu()  # (n,2)
    prob_t = torch.softmax(
        classifier(z_tgt.to(device)), dim=1
    ).cpu()
    y_src = y_src.cpu()
    conf = torch.zeros(2, 2)  # conf[y, j] = P(pred=j | y) 的软估计
    for y in (0, 1):
        m = y_src == y
        if m.sum() > 0:
            conf[y] = prob_s[m].mean(dim=0)
    pi_s = float(y_src.float().mean().clamp(PRIOR_EPS, 1 - PRIOR_EPS))

    def solve(mu_vec):
        # conf[y,j]=P(pred=j|y) 时 mu = conf.T @ x（mu_j = sum_y conf[y,j] x_y），
        # 故须按 conf.T 求解；解 conf @ x = mu 会解错线性系统（非对称 conf 下
        # 系统性偏差，见 tests/test_bbse_solve.py）。
        try:
            return float(torch.linalg.solve(conf.t(), mu_vec)[1])
        except Exception:
            return float("nan")

    pi_t = solve(prob_t.mean(dim=0))
    rng = np.random.default_rng(seed)
    pt = prob_t.numpy()
    boots = []
    for _ in range(int(n_bootstrap)):
        b = rng.choice(len(pt), len(pt), replace=True)
        mu_b = torch.as_tensor(pt[b].mean(axis=0), dtype=torch.float32)
        boots.append(solve(mu_b))
    boots = np.asarray(boots, dtype=np.float64)
    if not np.isfinite(pi_t) or not np.isfinite(boots).all():
        return pi_t, pi_s, float("inf")
    w = float(1.96 * boots.std(ddof=1)) if len(boots) > 1 else float("inf")
    pi_t = float(min(max(pi_t, PRIOR_EPS), 1 - PRIOR_EPS))
    return pi_t, pi_s, w


# ---------------- 4. pilot plan 强迫成本 ----------------

@torch.no_grad()
def pilot_forced_cost(args, z_src, y_src, z_tgt):
    """pilot 表示间 balanced Sinkhorn，区域指派打分跨类成本。

    返回 (c_cross, diagnostics)。区域指派 = 目标点到源类质心最近邻，
    一次性、无梯度、只用于给 plan 打分；高估方向 = 更强放松（安全）。
    """
    normalize = bool(getattr(args, "ot_normalize", False))

    def _n(z):
        return torch.nn.functional.normalize(z, p=2, dim=1) if normalize else z

    zs, zt = _n(z_src), _n(z_tgt)
    mu_pos = zs[y_src == 1].mean(dim=0)
    mu_neg = zs[y_src == 0].mean(dim=0)
    dists = torch.cdist(zt, torch.stack([mu_neg, mu_pos]))
    assign_t = dists.argmin(dim=1)
    # 指派置信度：平均相对边距 |d- - d+| / (d- + d+)，越小越不可信
    conf = float(((dists[:, 0] - dists[:, 1]).abs()
                  / dists.sum(dim=1).clamp_min(1e-8)).mean())

    _, plan = ot_balanced(
        z_src, z_tgt, reg=args.ot_reg, cost=args.cost,
        normalize=normalize, return_plan=True,
        max_iter=args.ot_max_iter, tol=args.ot_tol,
    )
    C = build_cost(z_src, z_tgt, normalize=normalize, cost=args.cost)
    cross = (y_src.view(-1, 1) != assign_t.view(1, -1)).float()
    c_cross = float((plan * C * cross).sum())
    # 源侧 margin 分位数：仅作诊断量落盘（论文 Remark 9），不进决策
    cross_d = torch.cdist(zs[y_src == 1], zs[y_src == 0])
    diagnostics = {
        "assign_confidence": conf,
        "assigned_target_match_frac": float(assign_t.float().mean()),
        "plan_total_cost": float((plan * C).sum()),
        "plan_cross_mass": float((plan * cross).sum()),
        "margin_source_q25": float(cross_d.quantile(0.25)),
    }
    return c_cross, diagnostics


# ---------------- 5. 决策规则 ----------------

def select_constraint(args, pi_s, pi_t, w, c_cross):
    """闭式规则（论文 Eq. tau-rule）：tau_hat = alpha*C_cross/surcharge。"""
    rho = abs(pi_s - pi_t)
    base = {
        "pi_s": float(pi_s), "pi_t_hat": float(pi_t),
        "conf_halfwidth": float(w), "rho_hat": float(rho),
        "c_cross": float(c_cross),
    }
    if not np.isfinite(pi_t) or not np.isfinite(w):
        return dict(base, variant="unbalanced",
                    tau=float(args.probe_fallback_tau),
                    reason="bbse_singular_fallback")
    if rho <= w:
        return dict(base, variant="balanced", tau=None,
                    reason="within_noise")
    # 方向感知分母（论文 V-B 步骤 4 与 Algorithm 1）。
    # surcharge 记的是 dropping plan 两侧 KL 罚：亏缺一侧的边际按"该类
    # 自身质量"加权，因此分母依赖先验差的方向：
    #   源盈余 (pi_s >= pi_t)：源 A+ 亏 rho、目标 A- 欠填 rho
    #     D_s = pi_s*psi(rho/pi_s) + (1-pi_t)*psi(rho/(1-pi_t))
    #   目标盈余 (pi_t > pi_s)：源 A- 亏 rho、目标 A+ 欠填 rho
    #     D_t = (1-pi_s)*psi(rho/(1-pi_s)) + pi_t*psi(rho/pi_t)
    # 二者不可合并为无方向公式：亏缺落在哪一侧由方向决定。
    if pi_t > pi_s:
        denom = ((1.0 - pi_s) * _psi(rho / (1.0 - pi_s))
                 + pi_t * _psi(rho / pi_t))
    else:
        denom = (pi_s * _psi(rho / pi_s)
                 + (1.0 - pi_t) * _psi(rho / (1.0 - pi_t)))
    tau_hat = float(args.probe_alpha) * float(c_cross) / max(denom, 1e-12)
    if tau_hat >= float(args.tau_max):
        return dict(base, variant="balanced", tau=None,
                    reason="small_gap_limit", tau_hat=tau_hat)
    # 护栏：τ̂ 只允许"放松得更少"，不允许超过参考放松度（τ=1）。
    # 实证依据：v1 全套 60 runs 中全部崩坏 run 的 τ̂<1（ρ̂ 在 matcher
    # 失明任务上被高估 → 罚项虚增 → 过度放松）；τ̂≥1 的 run 零崩坏。
    tau_floor = float(getattr(args, "probe_tau_floor", 1.0))
    tau_final = max(tau_hat, tau_floor)
    return dict(base, variant="unbalanced", tau=tau_final,
                reason="rule" if tau_final == tau_hat else "rule_floored",
                tau_hat=tau_hat)


# ---------------- 编排 ----------------

def run_probe(args, encoder, classifier, src_batches, tgt_pool_batches,
              device, result_dir=None):
    """执行完整 probe，返回 decision dict（并落盘 probe_decision.json）。"""
    t0 = time.time()
    pretrain_sec = source_pretrain(args, encoder, classifier,
                                   src_batches, device)

    seed = int(getattr(args, "train_seed", 0)) + 7919
    src_labels = src_batches.labels.cpu().numpy()
    idx_s = subsample_indices(src_batches.n_samples, args.probe_frac,
                              args.probe_max_pairs, seed, src_labels)
    idx_t = subsample_indices(tgt_pool_batches.n_samples, args.probe_frac,
                              args.probe_max_pairs, seed + 1, None)
    z_src = encode_subset(encoder, src_batches, idx_s, device)
    z_tgt = encode_subset(encoder, tgt_pool_batches, idx_t, device)
    y_src = torch.as_tensor(src_labels[idx_s])

    pi_t, pi_s, w = bbse_target_prior(
        encoder, classifier, z_src, y_src, z_tgt,
        n_bootstrap=args.probe_bootstrap, seed=seed,
    )
    if getattr(args, "probe_estimator", "plan") == "margin":
        # 消融估计器：源侧跨类距离 25% 分位数 × rho（T13 证明源侧稳定）
        zs = torch.nn.functional.normalize(z_src, p=2, dim=1)
        cross_d = torch.cdist(zs[y_src == 1], zs[y_src == 0])
        m_src = float(cross_d.quantile(0.25))
        c_cross = m_src * abs(pi_s - pi_t) if np.isfinite(pi_t) else 0.0
        cost_diag = {"margin_source_q25": m_src}
    else:
        c_cross, cost_diag = pilot_forced_cost(args, z_src, y_src, z_tgt)

    decision = select_constraint(args, pi_s, pi_t, w, c_cross)
    decision.update({
        "pilot_source": int(len(idx_s)),
        "pilot_target": int(len(idx_t)),
        "probe_pretrain_epochs": int(args.probe_pretrain_epochs),
        "probe_alpha": float(args.probe_alpha),
        "tau_max": float(args.tau_max),
        "wallclock_pretrain_sec": pretrain_sec,
        "wallclock_total_sec": time.time() - t0,
    })
    decision.update(cost_diag)

    if result_dir:
        path = os.path.join(result_dir, "artifacts", "probe_decision.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)
    return decision

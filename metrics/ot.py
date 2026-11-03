# metrics/ot.py
"""OT-DAER 的最优传输对齐损失（对应论文 Eq.28 / Eq.29）。

设计原则（见《代码优化方案》§3.3）：
  两个变体共享同一条 特征 -> 代价 -> 求解 路径，唯一分叉是约束集合：
    - ot_balanced   : 硬约束 Pi(a,b)，论文 Eq.28，Sinkhorn
    - ot_unbalanced : KL 松弛边际，论文 Eq.29，generalized Sinkhorn
  代价函数默认未平方欧氏距离（W1 一致，Kantorovich-Rubinstein 对偶）；
  cost="sq_l2" 为论文 Experiment C 的 squared-cost 消融。

所有函数基于 POT 的 torch 后端，梯度可回传到 encoder。

数值稳定性（重要）：原始 encoder 表示的配对距离为双峰分布、量级 3~50，
C/reg 远超 float32 的 exp 范围（exp(-88) 即下溢）；任何在线性域显式构造
exp(-C/reg) 的 Sinkhorn 变体（POT 的 knopp / stabilized / translation-
invariant 在 torch+CUDA 下均属此类）都会在第 0 次迭代产生 0/0 -> NaN plan
-> NaN 梯度 -> 整个 encoder 被污染。因此：
  - balanced 走 POT 的 log 域 Sinkhorn（method="sinkhorn_log"）；
  - unbalanced 用本模块自实现的 log 域 generalized Sinkhorn
    （Chizat et al. 2018 的同一算法，logsumexp 形式，见
    _generalized_sinkhorn_log），不依赖 POT 的 unbalanced 变体选择。
两者求解的优化问题与论文 Eq.28/Eq.29 完全一致，只是数值实现不同。
"""
import numpy as np
import torch
import torch.nn.functional as F
import ot

# Sinkhorn 求解默认参数（可用 CLI --ot_max_iter / --ot_tol 覆盖）
# 与 legacy 版（dader-ot-master metrics/ot.py）一致：2000 迭代 / 1e-9
DEFAULT_MAX_ITER = 2000
DEFAULT_TOL = 1e-9


def build_cost(feat_src, feat_tgt, normalize=False, cost="l2",
               source_labels=None, target_pseudo_labels=None,
               cost_mode="class_blind", class_cost_penalty=1.0):
    """构造 mini-batch 代价矩阵 C（两变体共用，论文 §IV-C）。

    normalize=True: optional unit-sphere normalization. The final-paper
                    default is False so the runtime matches c(z_s,z_t)
                    = ||z_s-z_t||_2 in framework.tex.
    cost: "l2" / "sq_l2" -> unsquared / squared Euclidean distance.
    ``class_aware`` is only used by the Experiment C ablation and consumes
    target pseudo-labels, never target ground-truth labels.
    """
    s = F.normalize(feat_src, p=2, dim=1) if normalize else feat_src
    t = F.normalize(feat_tgt, p=2, dim=1) if normalize else feat_tgt
    C = torch.cdist(s, t, p=2)
    if cost in ("sq_l2", "sql2"):
        C = C ** 2
    elif cost != "l2":
        raise ValueError(f"unknown cost: {cost}")
    if cost_mode == "class_aware":
        if source_labels is None or target_pseudo_labels is None:
            raise ValueError("class-aware cost requires source and target pseudo-labels")
        src_y = source_labels.to(C.device).long().view(-1, 1)
        tgt_y = target_pseudo_labels.to(C.device).long().view(1, -1)
        cross_class = (src_y != tgt_y).to(C.dtype)
        C = C + float(class_cost_penalty) * cross_class
    elif cost_mode != "class_blind":
        raise ValueError(f"unknown cost_mode: {cost_mode}")
    return C


def _sinkhorn_method(stable):
    return "sinkhorn_log" if stable else "sinkhorn"


def _generalized_sinkhorn_log(C, a, b, reg, reg_m,
                              max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL):
    """log 域 generalized Sinkhorn（Chizat et al. 2018），求解论文 Eq.29：

        min_{P>=0} <C,P> + reg_m KL(P1||a) + reg_m KL(P^T1||b) - reg H(P)

    _scaling 变量 u=exp(f/reg), v=exp(g/reg)，更新规则
        u <- (a / Kv)^{reg_m/(reg_m+reg)},  v <- (b / K^T u)^{reg_m/(reg_m+reg)}
    全部改写成 logsumexp 形式，迭代过程中从不显式计算 exp(-C/reg)，
    因此对任意代价尺度数值稳定（reg_m -> inf 时退化为 balanced 的 log
    Sinkhorn）。reg_m 较小时收敛快；reg_m 很大（>=10）时收敛变慢，必要
    时增大 max_iter。
    """
    log_a = torch.log(a.clamp_min(1e-45))
    log_b = torch.log(b.clamp_min(1e-45))
    power = reg_m / (reg_m + reg)  # τ/(τ+ε) ∈ (0,1)
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(int(max_iter)):
        # log(Kv)_i = logsumexp_j (g_j - C_ij)/reg
        lKv = torch.logsumexp((g.unsqueeze(0) - C) / reg, dim=1)
        f_new = (power * reg) * (log_a - lKv)
        # log(K^T u)_j = logsumexp_i (f_i - C_ij)/reg
        lKu = torch.logsumexp((f_new.unsqueeze(1) - C) / reg, dim=0)
        g_new = (power * reg) * (log_b - lKu)
        inc = torch.maximum(
            (f_new - f).abs().max(), (g_new - g).abs().max()
        )
        f, g = f_new, g_new
        if bool(torch.isfinite(inc)) and float(inc) < tol:
            break
    # P_ij = u_i K_ij v_j = exp((f_i + g_j - C_ij)/reg)
    return torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / reg)


def ot_balanced(feat_src, feat_tgt, a=None, b=None, reg=0.1, cost="l2",
                normalize=False, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL,
                stable=True, return_plan=False, cost_mode="class_blind",
                source_labels=None, target_pseudo_labels=None,
                class_cost_penalty=1.0):
    """平衡 OT 对齐损失（论文 Eq.28）：硬约束 Pi(a,b)。

    a/b 默认均匀；最终论文 runtime 只使用 uniform marginals。
    stable=True（默认）走 log 域 Sinkhorn——原始 encoder 距离下 C/reg 远超
    float32 exp 范围，非 log 迭代会数值下溢产生 NaN plan；优化问题不变。
    返回完整熵正则目标值及可选 transport plan。
    """
    C = build_cost(
        feat_src, feat_tgt, normalize=normalize, cost=cost,
        source_labels=source_labels, target_pseudo_labels=target_pseudo_labels,
        cost_mode=cost_mode, class_cost_penalty=class_cost_penalty,
    )
    n_s, n_t = C.size(0), C.size(1)
    dev = C.device
    if a is None:
        a = torch.full((n_s,), 1.0 / n_s, device=dev)
    if b is None:
        b = torch.full((n_t,), 1.0 / n_t, device=dev)
    a = a.to(dev).float()
    b = b.to(dev).float()
    # 数值兜底：平衡 OT 要求总质量相等
    a = a / a.sum().clamp_min(1e-12)
    b = b / b.sum().clamp_min(1e-12)

    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=max_iter, stopThr=tol,
                    method=_sinkhorn_method(stable))
    entropy = -(P * (torch.log(P.clamp_min(1e-12)) - 1.0)).sum()
    loss = torch.sum(P * C) - reg * entropy
    if return_plan:
        return loss, P
    return loss


def ot_unbalanced(feat_src, feat_tgt, a=None, b=None, reg=0.1, reg_m=1.0,
                  cost="l2", normalize=False, max_iter=DEFAULT_MAX_ITER,
                  tol=DEFAULT_TOL, return_plan=False, cost_mode="class_blind",
                  source_labels=None, target_pseudo_labels=None,
                  class_cost_penalty=1.0):
    """非平衡 OT 对齐损失（论文 Eq.29）：KL(reg_m) 松弛边际。

    a/b: uniform reference marginals in the final runtime. Their transported
    marginals are relaxed through generalized KL penalties.
    求解器为本模块的 log 域 generalized Sinkhorn（见模块 docstring 的数值
    稳定性说明），与 POT 的非 log 变体求解同一个 UOT 问题。
    """
    C = build_cost(
        feat_src, feat_tgt, normalize=normalize, cost=cost,
        source_labels=source_labels, target_pseudo_labels=target_pseudo_labels,
        cost_mode=cost_mode, class_cost_penalty=class_cost_penalty,
    )
    n_s, n_t = C.size(0), C.size(1)
    dev = C.device
    if a is None:
        a = torch.full((n_s,), 1.0 / n_s, device=dev)
    if b is None:
        b = torch.full((n_t,), 1.0 / n_t, device=dev)
    a = a.to(dev).float()
    b = b.to(dev).float()

    P = _generalized_sinkhorn_log(
        C, a, b, reg=reg, reg_m=reg_m, max_iter=max_iter, tol=tol
    )
    row_mass = P.sum(dim=1)
    col_mass = P.sum(dim=0)

    def generalized_kl(p, q):
        return (p * torch.log((p + 1e-12) / (q + 1e-12)) - p + q).sum()

    entropy = -(P * (torch.log(P.clamp_min(1e-12)) - 1.0)).sum()
    loss = (
        torch.sum(P * C)
        + reg_m * generalized_kl(row_mass, a)
        + reg_m * generalized_kl(col_mass, b)
        - reg * entropy
    )
    if return_plan:
        return loss, P
    return loss


def transported_mass_stats(P, a=None, b=None, cost=None,
                           source_labels=None, target_labels=None):
    """OT 诊断统计（论文 RQ4 的 unmatched-ratio / over-transport 分析素材）。

    返回 dict:
      total_mass : sum(P)，非平衡下 < 1 即"丢弃质量"的直接证据
      tgt_fill_min/mean : 目标实际收到边际 gamma^T1 的最小/均值
      src_send_min/mean : 源实际送出边际 gamma1 的最小/均值
    """
    with torch.no_grad():
        total = float(P.sum().item())
        tgt_fill = P.sum(dim=0)
        src_send = P.sum(dim=1)
        stats = {
            "total_mass": total,
            "tgt_fill_min": float(tgt_fill.min().item()),
            "tgt_fill_mean": float(tgt_fill.mean().item()),
            "src_send_min": float(src_send.min().item()),
            "src_send_mean": float(src_send.mean().item()),
        }
        if a is not None:
            stats["source_marginal_max_error"] = float(
                (src_send - a.to(P.device)).abs().max().item()
            )
        if b is not None:
            stats["target_marginal_max_error"] = float(
                (tgt_fill - b.to(P.device)).abs().max().item()
            )
        if cost is not None:
            stats["transport_cost"] = float((P * cost).sum().item())
        if source_labels is not None and target_labels is not None:
            src_y = source_labels.to(P.device).long().view(-1, 1)
            tgt_y = target_labels.to(P.device).long().view(1, -1)
            cross = (src_y != tgt_y).to(P.dtype)
            stats["cross_class_mass"] = float((P * cross).sum().item())
            if cost is not None:
                stats["cross_class_cost"] = float((P * cost * cross).sum().item())
        return stats


def pairwise_target_cross_entropy(source_labels, target_logits):
    """DeepJDOT 的成对目标分类损失 T[i,j]。

    T[i,j] = CE(y_i^s, f(g(x_j^t)))，使用目标分类器的 soft prediction，
    不使用硬 argmax 伪标签。返回形状为 [n_source, n_target]。
    """
    if source_labels.ndim != 1 or target_logits.ndim != 2:
        raise ValueError("source_labels must be [n] and target_logits [m,K]")
    if target_logits.size(1) < 2:
        raise ValueError("target_logits must contain at least two classes")
    log_probs = F.log_softmax(target_logits.float(), dim=1)
    # log_probs[:, y_i] 对每一列取源样本标签对应的 log 概率；
    # 转置后得到 [n_source, n_target]。
    return -log_probs.index_select(
        1, source_labels.to(target_logits.device).long()
    ).transpose(0, 1)


def _solve_emd_plan(cost, a, b):
    """用未正则化 balanced OT（POT network simplex）求固定 plan。"""
    if not torch.isfinite(cost).all():
        raise RuntimeError("DeepJDOT cost contains non-finite values")
    cost_np = cost.detach().to(device="cpu", dtype=torch.float64).numpy()
    a_np = a.detach().to(device="cpu", dtype=torch.float64).numpy()
    b_np = b.detach().to(device="cpu", dtype=torch.float64).numpy()
    try:
        plan_np = ot.emd(a_np, b_np, cost_np)
    except Exception as exc:
        raise RuntimeError("DeepJDOT EMD solver failed") from exc
    plan = torch.as_tensor(
        plan_np, dtype=cost.dtype, device=cost.device
    ).detach()
    if not torch.isfinite(plan).all():
        raise RuntimeError("DeepJDOT EMD returned non-finite plan")
    if float(plan.min()) < -1e-10:
        raise RuntimeError("DeepJDOT EMD returned negative transport mass")
    if abs(float(plan.sum()) - 1.0) > 1e-6:
        raise RuntimeError("DeepJDOT EMD plan does not have unit total mass")
    row_error = (plan.sum(dim=1) - a).abs().max()
    col_error = (plan.sum(dim=0) - b).abs().max()
    if max(float(row_error), float(col_error)) > 1e-6:
        raise RuntimeError("DeepJDOT EMD plan violates prescribed marginals")
    return plan


def deepjdot_faithful(feat_src, feat_tgt, labels_src, logits_tgt,
                      alpha=0.001, lambda_t=0.0001, solver="emd"):
    """Damodaran et al. 2018 的联合 feature-label OT 目标。

    C = alpha * ||z_s-z_t||^2 + lambda_t * CE(y_s, f(z_t))。
    OT 在未正则化 balanced 约束上求解；网络损失使用固定 plan 的
    <plan, C>，不包含 entropy，也不让求解过程进入 autograd 图。
    """
    if solver != "emd":
        raise ValueError("faithful DeepJDOT uses the original unregularized EMD solver")
    feature_cost = build_cost(
        feat_src, feat_tgt, normalize=False, cost="sq_l2",
        cost_mode="class_blind",
    )
    label_cost = pairwise_target_cross_entropy(labels_src, logits_tgt)
    cost = float(alpha) * feature_cost + float(lambda_t) * label_cost
    n_s, n_t = cost.size(0), cost.size(1)
    a = torch.full((n_s,), 1.0 / n_s, dtype=cost.dtype, device=cost.device)
    b = torch.full((n_t,), 1.0 / n_t, dtype=cost.dtype, device=cost.device)
    plan = _solve_emd_plan(cost, a, b)
    # 固定 plan 后更新网络；plan 不参与梯度回传。
    loss = torch.sum(plan * cost)
    return loss, plan, cost, a, b

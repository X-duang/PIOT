# train/mass_design.py
"""质量分配（mass allocation）框架 —— OT-DAER 的第一设计轴（方案 §4）。

统一视角：采样与 marginal 加权都是"设计经验测度"：
  - 硬采样（coreset）   = 0/1 质量；
  - 类先验校正          = 软质量（match 类按 pi_t/pi_s 加权）；
  - 密度软加权          = 软化的 coreset（密集近重复簇降权而非删点）；
  - 非平衡 KL 松弛      = 对构造误差的残余吸收。

理论钩子（论文 Prop 2）：重加权源测度使 pi_s^w = pi_t 时，
跨类输运下界 m*|Δπ| 消失 —— 质量分配即测度层面的先验偏移校正。

pi_t 无标签不可直接观测，用 BBSE 黑盒估计（Lipton et al. 2018，
论文参考文献 [28]；聚合层面估计，比逐样本伪标签稳健）。
"""
import numpy as np
import torch

# 质量构造的安全范围（guardrail，对应 Remark: ESS 权衡）
BOOST_MIN, BOOST_MAX = 0.25, 4.0
PRIOR_EPS = 1e-4


def prior_matching_boost(pi_s, pi_t):
    """Corollary 3 的精确先验匹配权重（几率比形式）。

    两类权重（负类 1、正类 r）下，重加权先验
        pi^w = r*pi_s / (r*pi_s + 1 - pi_s)
    令 pi^w = pi_t 解出精确解：
        r* = pi_t*(1-pi_s) / (pi_s*(1-pi_t))
    线性比 pi_t/pi_s 只是 pi 小时的一阶近似 —— 论文用精确形式。
    """
    ps = min(max(float(pi_s), PRIOR_EPS), 1 - PRIOR_EPS)
    pt = min(max(float(pi_t), PRIOR_EPS), 1 - PRIOR_EPS)
    return pt * (1.0 - ps) / (ps * (1.0 - pt))


# ============ 1. 目标先验 pi_t 的 BBSE 估计 ============
@torch.no_grad()
def estimate_target_prior_bbse(encoder, classifier, src_loader, tgt_loader,
                               device=None):
    """BBSE (black-box shift estimation) 估计目标域 match 先验 pi_t。

    用源域混淆矩阵 C[i,j] = P(pred=i | y=j)（在源域上估计）与目标域
    预测均值 mu_hat = E_t[pred]，解 mu_t = C^{-1} mu_hat。
    返回 (pi_t_hat, pi_s)。失败/病态时回退 pi_t_hat = pi_s（即不校正）。
    """
    dev = device or (torch.device("cuda") if torch.cuda.is_available()
                     else torch.device("cpu"))
    encoder.eval()
    classifier.eval()

    # 源域混淆矩阵（行=真实类，列=预测类）
    conf = torch.zeros(2, 2, device=dev)
    n_src = 0
    n_src_pos = 0
    for batch in src_loader:
        reviews, mask, segment, labels = batch[0], batch[1], batch[2], batch[3]
        reviews, mask, segment, labels = reviews.to(dev), mask.to(dev), segment.to(dev), labels.to(dev)
        feat = encoder(reviews, mask, segment)
        preds = classifier(feat).argmax(dim=1)
        for y in (0, 1):
            m = labels == y
            cnt = m.sum()
            if cnt > 0:
                conf[y, 0] += (preds[m] == 0).sum()
                conf[y, 1] += (preds[m] == 1).sum()
        n_src += labels.numel()
        n_src_pos += (labels == 1).sum().item()
    row_sum = conf.sum(dim=1, keepdim=True).clamp_min(1.0)
    conf = conf / row_sum  # 行归一化 -> P(pred | y)

    # 目标域预测均值
    pred_sum = torch.zeros(2, device=dev)
    n_tgt = 0
    for batch in tgt_loader:
        reviews, mask, segment = batch[0], batch[1], batch[2]
        reviews, mask, segment = reviews.to(dev), mask.to(dev), segment.to(dev)
        feat = encoder(reviews, mask, segment)
        preds = classifier(feat).argmax(dim=1)
        pred_sum[0] += (preds == 0).sum()
        pred_sum[1] += (preds == 1).sum()
        n_tgt += reviews.size(0)
    mu_hat = pred_sum / max(n_tgt, 1)

    pi_s = n_src_pos / max(n_src, 1)
    try:
        mu_t = torch.linalg.solve(conf, mu_hat)  # mu_t[j] = P_t(y=j) 估计
        pi_t = float(mu_t[1].clamp(PRIOR_EPS, 1 - PRIOR_EPS).item())
    except Exception:
        pi_t = pi_s  # 病态矩阵回退：不校正
    return pi_t, pi_s


# ============ 2. 逐样本质量构造 ============
def class_weights(labels, boost):
    """类先验校正权重：match 类乘 boost，非 match 类为 1。

    boost 的理论取值 = pi_t / pi_s（BBSE 估计），由调用方给出。
    labels: (n,) 0/1；返回 (n,) float np.ndarray（未归一化）。
    """
    labels = np.asarray(labels)
    w = np.ones(len(labels), dtype=np.float64)
    w[labels == 1] = float(boost)
    return w


def density_weights(features, gamma=1.0, k=10, probe=4096, seed=0):
    """密度软加权：w_i ∝ (到 k 近邻的平均余弦距离)^gamma。

    密集近重复簇中的样本降权（软化的 coreset 效果），
    稀疏/边界样本升权。gamma=0 退化为均匀。
    features: (n, d) np.ndarray
    """
    feats = np.asarray(features, dtype=np.float64)
    n = feats.shape[0]
    if n > probe:  # 大集合用自身子集做参照，控 O(n*probe)
        rng = np.random.default_rng(seed)
        ref = feats[rng.choice(n, probe, replace=False)]
    else:
        ref = feats
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    D = 1.0 - fn @ rn.T  # (n, probe) 余弦距离
    k_eff = min(k + 1, D.shape[1])
    knn = np.partition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_d = knn.mean(axis=1) + 1e-8
    w = mean_d ** float(gamma)
    return w


def normalize_mass(w, total_mass=1.0):
    """归一化到指定总质量（总质量可与另一侧不等 —— 非平衡松弛的意义）。"""
    w = np.asarray(w, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        w = np.ones_like(w)
        s = w.sum()
    return (w / s * float(total_mass)).astype(np.float32)


def build_masses(mass_mode, src_labels=None, src_features=None,
                 n_tgt=0, prior_boost=1.0, density_gamma=1.0,
                 src_total_mass=1.0, tgt_total_mass=1.0):
    """按 mass_mode 构造全量源/目标边际质量（训练循环内按 batch 索引取用）。

    mass_mode:
      uniform        : 均匀（论文 Eq.29 的默认形态）
      class          : 源域 match 类 ×prior_boost（先验校正）
      density        : 源域密度软加权
      class_density  : 先验校正 × 密度软加权
    目标域当前恒为均匀（b 的 confidence 加权留作后续扩展）。
    返回 (a_full, b_full)：np.ndarray，长度 = 源/目标（采样后）样本数。
    """
    if src_labels is None:
        raise ValueError("build_masses 需要 src_labels")
    n_src = len(src_labels)
    if mass_mode == "uniform":
        w = np.ones(n_src, dtype=np.float64)
    elif mass_mode == "class":
        w = class_weights(src_labels, prior_boost)
    elif mass_mode == "density":
        w = density_weights(src_features, gamma=density_gamma)
    elif mass_mode == "class_density":
        w = class_weights(src_labels, prior_boost) * \
            density_weights(src_features, gamma=density_gamma)
    else:
        raise ValueError(f"unknown mass_mode: {mass_mode}")
    a_full = normalize_mass(w, src_total_mass)
    b_full = normalize_mass(np.ones(n_tgt, dtype=np.float64), tgt_total_mass)
    return a_full, b_full

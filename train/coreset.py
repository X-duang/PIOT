# train/coreset.py
"""固定比例 k-center coreset（硬质量分配的 0/1 特例，方案 §2.3）。

定位：标准预处理组件 / 效率开关。比例由对照实验选定，不写死。
动态决策层（启发式/LLM）已删除 —— 实证显示其在全部 120 个 run 上
坍缩为常数 ≈41%，无自适应价值（见《代码优化方案》§2.1）。
"""
import numpy as np
import torch


def kcenter_coreset(features, n_select, random_seed=42):
    """贪心 k-center：保留特征空间中最分散的 n_select 个样本索引。

    features: (n, d) np.ndarray 或 torch.Tensor（已驻留 GPU 亦可）
    返回: np.ndarray，长度为 min(n_select, n) 的索引
    """
    if isinstance(features, np.ndarray):
        feats = torch.from_numpy(features).float()
    else:
        feats = features.float()
    if torch.cuda.is_available() and not feats.is_cuda:
        feats = feats.cuda()

    n_samples = feats.size(0)
    n_select = min(n_select, n_samples)
    if n_select >= n_samples:
        return np.arange(n_samples)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    first_idx = torch.randint(0, n_samples, (1,)).item()
    centers = [first_idx]
    dist_to_centers = torch.cdist(feats[first_idx].unsqueeze(0), feats).squeeze(0)

    for _ in range(n_select - 1):
        farthest_idx = torch.max(dist_to_centers, dim=0)[1].item()
        centers.append(farthest_idx)
        new_dists = torch.cdist(feats[farthest_idx].unsqueeze(0), feats).squeeze(0)
        dist_to_centers = torch.minimum(dist_to_centers, new_dists)

    return np.array(centers)


def select_coreset(features, n_select, labels=None, random_seed=42):
    """coreset 选择入口。

    labels=None      : 普通 k-center；
    labels 给定      : 类比例 k-center（保护稀有 match 类，
                       --coreset_stratified 开关对应；
                       对应 Remark: 分层采样保持先验 pi(μ̂|_S)=π_s）。
    """
    n = features.shape[0]
    n_select = min(n_select, n)
    if labels is None:
        return kcenter_coreset(features, n_select, random_seed)
    labels = np.asarray(labels)
    picks = []
    classes, counts = np.unique(labels, return_counts=True)
    for c, cnt in zip(classes, counts):
        idx_c = np.where(labels == c)[0]
        quota = max(1, int(round(n_select * cnt / n)))
        sub = features[idx_c]
        local = kcenter_coreset(sub, min(quota, len(idx_c)), random_seed)
        picks.append(idx_c[local])
    return np.concatenate(picks)


def voronoi_cell_counts(features, selected_idx):
    """质量聚合（Lemma 1 的严格形式）：每个 retained 原子的 Voronoi 计数。

    把每个全量样本指派到最近的 retained 中心，计数即该原子的聚合质量
    （归一化后 Σ = 1）。对应论文 Lemma 1：质量聚合测度满足
    W1(μ̂_full, μ̂|_S) ≤ ε_S，而 uniform-over-S 只是单元计数均衡时的特例。

    features: (n, d) np.ndarray / torch.Tensor
    selected_idx: (k,) retained 原子在全量集中的索引
    返回: (k,) float np.ndarray，与 selected_idx 对齐的归一化质量
    """
    if isinstance(features, np.ndarray):
        feats = torch.from_numpy(features).float()
    else:
        feats = features.float()
    if torch.cuda.is_available() and not feats.is_cuda:
        feats = feats.cuda()
    sel = torch.as_tensor(np.asarray(selected_idx), dtype=torch.long,
                          device=feats.device)
    centers = feats[sel]                                    # (k, d)
    assign = torch.cdist(feats, centers).argmin(dim=1)      # (n,) 最近中心
    counts = torch.bincount(assign, minlength=sel.numel()).float()
    counts = counts / counts.sum().clamp_min(1.0)
    return counts.detach().cpu().numpy()

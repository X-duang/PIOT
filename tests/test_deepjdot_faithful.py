# -*- coding: utf-8 -*-
"""Tests for the faithful DeepJDOT joint feature-label OT path."""
import math

import pytest

pytest.importorskip("ot")
import torch

from metrics.ot import deepjdot_faithful, pairwise_target_cross_entropy


def test_pairwise_target_cross_entropy_matches_manual_values():
    labels = torch.tensor([0, 1])
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    cost = pairwise_target_cross_entropy(labels, logits)
    expected_same = -math.log(math.exp(2.0) / (math.exp(2.0) + 1.0))
    expected_cross = -math.log(1.0 / (math.exp(2.0) + 1.0))
    assert cost.shape == (2, 2)
    assert torch.allclose(cost[0, 0], torch.tensor(expected_same), atol=1e-6)
    assert torch.allclose(cost[1, 1], torch.tensor(expected_same), atol=1e-6)
    assert torch.allclose(cost[0, 1], torch.tensor(expected_cross), atol=1e-6)
    assert torch.allclose(cost[1, 0], torch.tensor(expected_cross), atol=1e-6)
    cost.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_faithful_deepjdot_uses_fixed_valid_emd_plan():
    feat_src = torch.tensor([[0.0, 0.0], [1.0, 0.0]], requires_grad=True)
    feat_tgt = torch.tensor([[0.0, 0.1], [0.9, 0.0]], requires_grad=True)
    labels_src = torch.tensor([0, 1])
    logits_tgt = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    loss, plan, cost, a, b = deepjdot_faithful(
        feat_src, feat_tgt, labels_src, logits_tgt,
        alpha=0.001, lambda_t=0.0001,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert plan.shape == (2, 2)
    assert not plan.requires_grad
    assert plan.min() >= -1e-10
    assert torch.allclose(plan.sum(dim=1), a, atol=1e-6)
    assert torch.allclose(plan.sum(dim=0), b, atol=1e-6)
    assert cost.shape == (2, 2)
    loss.backward()
    assert feat_src.grad is not None and torch.isfinite(feat_src.grad).all()
    assert feat_tgt.grad is not None and torch.isfinite(feat_tgt.grad).all()
    assert logits_tgt.grad is not None and torch.isfinite(logits_tgt.grad).all()


def test_faithful_deepjdot_rejects_non_original_solver():
    with pytest.raises(ValueError):
        deepjdot_faithful(
            torch.zeros(1, 2), torch.zeros(1, 2),
            torch.tensor([0]), torch.zeros(1, 2), solver="sinkhorn_log",
        )

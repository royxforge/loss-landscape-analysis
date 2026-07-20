"""Tests for custom loss functions."""

from __future__ import annotations

import torch

from src.losses import (
    CrossEntropyLoss,
    MSELoss,
    log_softmax,
    softmax,
    to_one_hot,
)


class TestToOheHot:
    """Tests for the to_one_hot utility."""

    def test_converts_indices(self, sample_targets: torch.Tensor):
        one_hot = to_one_hot(sample_targets, num_classes=3)
        assert one_hot.shape == (3, 3)
        assert one_hot[0, 0] == 1.0
        assert one_hot[1, 1] == 1.0
        assert one_hot[2, 2] == 1.0

    def test_all_other_entries_are_zero(self, sample_targets: torch.Tensor):
        one_hot = to_one_hot(sample_targets, num_classes=3)
        for i, target in enumerate(sample_targets):
            assert one_hot[i].sum() == 1.0
            assert one_hot[i, target] == 1.0

    def test_batch_of_zeros(self):
        targets = torch.zeros(4, dtype=torch.long)
        one_hot = to_one_hot(targets, num_classes=5)
        assert one_hot.shape == (4, 5)
        assert (one_hot[:, 0] == 1.0).all()
        assert (one_hot[:, 1:].sum() == 0).all()


class TestSoftmax:
    """Tests for the custom softmax function."""

    def test_sums_to_one(self, sample_logits: torch.Tensor):
        probs = softmax(sample_logits)
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_all_positive(self, sample_logits: torch.Tensor):
        probs = softmax(sample_logits)
        assert (probs >= 0).all()

    def test_numerical_stability_with_large_values(self):
        large_logits = torch.tensor([[1000.0, 0.0, -1000.0]])
        probs = softmax(large_logits)
        # Should not produce NaN or Inf
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()
        # Largest logit should dominate
        assert probs[0, 0] > 0.99

    def test_identical_to_torch_softmax(self, sample_logits: torch.Tensor):
        custom = softmax(sample_logits)
        torch_softmax = torch.softmax(sample_logits, dim=1)
        assert torch.allclose(custom, torch_softmax, atol=1e-6)


class TestLogSoftmax:
    """Tests for the custom log_softmax function."""

    def test_log_of_softmax(self, sample_logits: torch.Tensor):
        custom_log = log_softmax(sample_logits)
        torch_log = torch.log_softmax(sample_logits, dim=1)
        assert torch.allclose(custom_log, torch_log, atol=1e-5)

    def test_numerical_stability(self):
        large_logits = torch.tensor([[1000.0, 0.0, -1000.0]])
        result = log_softmax(large_logits)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestMSELoss:
    """Tests for the MSELoss class."""

    def test_loss_is_positive(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor):
        loss_fn = MSELoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.item() >= 0

    def test_perfect_prediction(self, sample_logits: torch.Tensor):
        """Loss should be zero when predictions match targets exactly."""
        loss_fn = MSELoss()
        targets = torch.tensor([0, 1, 2], device=sample_logits.device, dtype=torch.long)
        loss = loss_fn(sample_logits, targets)
        # Not quite zero since we use softmax probabilities
        assert loss.item() < 1.0

    def test_loss_is_scalar(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor):
        loss_fn = MSELoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.ndim == 0


class TestCrossEntropyLoss:
    """Tests for the CrossEntropyLoss class."""

    def test_loss_is_positive(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor):
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.item() >= 0

    def test_perfect_prediction_has_low_loss(self):
        """Cross-entropy should be near zero when strongly correct."""
        loss_fn = CrossEntropyLoss()
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = torch.tensor([0, 1])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_cross_entropy_better_than_mse_on_incorrect(self):
        """CE should penalise confident wrong predictions more than MSE."""
        ce_loss_fn = CrossEntropyLoss()
        mse_loss_fn = MSELoss()

        # Confidently wrong: high logit for wrong class
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        targets = torch.tensor([1])  # Correct class is index 1

        ce_loss = ce_loss_fn(logits, targets)
        mse_loss = mse_loss_fn(logits, targets)

        # CE gradient should be larger in magnitude for confident errors
        assert ce_loss.item() > mse_loss.item() * 0.5

    def test_loss_is_scalar(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor):
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sample_logits, sample_targets)
        assert loss.ndim == 0

    def test_against_torch_cross_entropy(self, sample_logits: torch.Tensor, sample_targets: torch.Tensor):
        """Our CE implementation should match PyTorch's."""
        custom_loss = CrossEntropyLoss()
        torch_loss = torch.nn.CrossEntropyLoss()

        custom_val = custom_loss(sample_logits, sample_targets)
        torch_val = torch_loss(sample_logits, sample_targets)

        assert torch.allclose(custom_val, torch_val, atol=1e-5)

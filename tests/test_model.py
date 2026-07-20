"""Tests for the SmallNet model."""

from __future__ import annotations

import torch

from src.model import SmallNet, get_fresh_model


class TestSmallNet:
    """Test suite for SmallNet architecture."""

    def test_initialization(self):
        """Model should initialise with Xavier uniform weights."""
        model = SmallNet()
        # Check fc1 weights are not zero (Xavier init produced values)
        assert torch.abs(model.fc1.weight).sum().item() > 0
        assert torch.abs(model.fc2.weight).sum().item() > 0
        assert torch.abs(model.fc3.weight).sum().item() > 0

    def test_forward_shape(self, sample_images: torch.Tensor):
        """Forward pass should produce logits of shape (B, 10)."""
        model = SmallNet()
        logits = model(sample_images)
        assert logits.shape == (2, 10)

    def test_forward_sums_to_one_after_softmax(self, sample_images: torch.Tensor):
        """Softmax of logits should sum to 1 across classes."""
        model = SmallNet()
        logits = model(sample_images)
        probs = torch.softmax(logits, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_get_weights_flat(self):
        """Flattened weights should be a single 1-D tensor."""
        model = SmallNet()
        flat = model.get_weights_flat()
        assert flat.ndim == 1
        # Count total parameters
        total = sum(p.numel() for p in model.parameters())
        assert flat.shape[0] == total

    def test_reset_weights_changes_weights(self):
        """Resetting weights should produce different weight values."""
        model = SmallNet()
        original = model.get_weights_flat().clone()
        model.reset_weights(seed=99)
        new_weights = model.get_weights_flat()
        # Different seed -> different init — extremely unlikely to match
        assert not torch.allclose(original, new_weights)

    def test_fresh_model_identical_to_new_smallnet(self):
        """get_fresh_model() should produce identical weights to SmallNet()."""
        model1 = SmallNet()
        model2 = get_fresh_model()
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_gradients_flow_through_all_layers(self, sample_images: torch.Tensor):
        """Backward pass should produce gradients for all parameters."""
        model = SmallNet()
        logits = model(sample_images)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"


class TestSmallNetReproducibility:
    """Test suite for model reproducibility."""

    def test_deterministic_initialization(self):
        """Same seed should produce identical weights."""
        model_a = SmallNet()
        model_b = SmallNet()
        for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(p1, p2), "Weights differ with same seed"

    def test_forward_pass_is_deterministic(self, sample_images: torch.Tensor):
        """Same input should produce identical logits."""
        model = SmallNet()
        output1 = model(sample_images)
        output2 = model(sample_images)
        assert torch.allclose(output1, output2)

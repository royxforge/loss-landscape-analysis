"""Tests for training utilities."""

from __future__ import annotations

import pytest
import torch

from src.losses import CrossEntropyLoss
from src.model import SmallNet
from src.trainer import Trainer, compute_gradient_norm


class TestComputeGradientNorm:
    """Tests for the gradient norm computation."""

    def test_zero_grad_returns_zero(self):
        model = SmallNet()
        assert compute_gradient_norm(model) == 0.0

    def test_after_backward_norm_is_positive(self, sample_images: torch.Tensor):
        model = SmallNet()
        logits = model(sample_images)
        loss = logits.sum()
        loss.backward()
        norm = compute_gradient_norm(model)
        assert norm > 0
        assert isinstance(norm, float)


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def trainer(self, device: str):
        model = SmallNet().to(device)
        loss_fn = CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    def test_initialization(self, trainer: Trainer):
        assert trainer.device in ("cpu", "cuda")

    def test_train_epoch_returns_metrics_dict(self, trainer: Trainer, device: str):
        from src.data import load_mnist

        train_loader, *_ = load_mnist(train_batch_size=64, seed=42)
        metrics = trainer.train_epoch(train_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "grad_norm" in metrics
        assert 0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] >= 0

    def test_evaluate_returns_metrics_dict(self, trainer: Trainer, device: str):
        from src.data import load_mnist

        _, test_loader, *_ = load_mnist(test_batch_size=1000)
        metrics = trainer.evaluate(test_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1.0

    def test_train_runs_multiple_epochs(self, trainer: Trainer, device: str):
        from src.data import load_mnist

        train_loader, test_loader, *_ = load_mnist(
            train_batch_size=64, test_batch_size=1000, seed=42
        )
        history = trainer.train(train_loader, test_loader, epochs=3)
        assert len(history["epochs"]) == 3
        assert len(history["train_loss"]) == 3
        assert len(history["test_loss"]) == 3
        assert len(history["train_accuracy"]) == 3
        assert len(history["test_accuracy"]) == 3
        assert len(history["grad_norms"]) == 3

    def test_accuracy_improves_with_training(self, trainer: Trainer, device: str):
        from src.data import load_mnist

        train_loader, test_loader, *_ = load_mnist(
            train_batch_size=64, test_batch_size=1000, seed=42
        )
        history = trainer.train(train_loader, test_loader, epochs=2)
        # Accuracy should improve (or stay same) after 2 epochs of training
        assert history["test_accuracy"][-1] >= history["test_accuracy"][0]

    def test_loss_decreases_with_training(self, trainer: Trainer, device: str):
        from src.data import load_mnist

        train_loader, test_loader, *_ = load_mnist(
            train_batch_size=64, test_batch_size=1000, seed=42
        )
        history = trainer.train(train_loader, test_loader, epochs=2)
        assert history["train_loss"][-1] <= history["train_loss"][0] + 0.01

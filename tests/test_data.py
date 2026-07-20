"""Tests for data loading utilities."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import get_small_batch, load_mnist


class TestLoadMNIST:
    """Test suite for load_mnist function."""

    def test_returns_dataloaders(self):
        """Should return train and test DataLoaders."""
        train_loader, test_loader, train_dataset, test_dataset = load_mnist()
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_train_loader_batch_shape(self):
        """Train batches should have correct shape."""
        train_loader, *_ = load_mnist(train_batch_size=64)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 64
        assert batch_x.ndim == 4  # (B, C, H, W)
        assert batch_x.shape[1] == 1  # Grayscale

    def test_test_loader_batch_shape(self):
        """Test batches should have correct shape."""
        _, test_loader, *_ = load_mnist(test_batch_size=1000)
        batch_x, batch_y = next(iter(test_loader))
        assert batch_x.shape[0] == 1000

    def test_labels_are_integers(self):
        """Labels should be integer tensors."""
        train_loader, *_ = load_mnist(train_batch_size=64)
        _, batch_y = next(iter(train_loader))
        assert batch_y.dtype == torch.int64

    def test_deterministic_shuffle(self):
        """Same seed should produce same batch order."""
        loader_a, *_ = load_mnist(train_batch_size=64, seed=42)
        loader_b, *_ = load_mnist(train_batch_size=64, seed=42)
        batch_a_x, _ = next(iter(loader_a))
        batch_b_x, _ = next(iter(loader_b))
        assert torch.allclose(batch_a_x, batch_b_x)

    def test_different_seed_different_order(self):
        """Different seed should produce different batch order."""
        loader_a, *_ = load_mnist(train_batch_size=64, seed=42)
        loader_b, *_ = load_mnist(train_batch_size=64, seed=99)
        batch_a_x, _ = next(iter(loader_a))
        batch_b_x, _ = next(iter(loader_b))
        # Extremely unlikely to match
        assert not torch.allclose(batch_a_x, batch_b_x)


class TestGetSmallBatch:
    """Test suite for get_small_batch function."""

    def test_returns_correct_number_of_samples(self):
        loader, *_ = load_mnist(train_batch_size=64)
        x, y = get_small_batch(loader, n=100)
        assert x.shape[0] == 100
        assert y.shape[0] == 100

    def test_raises_on_zero(self):
        loader, *_ = load_mnist(train_batch_size=64)
        with pytest.raises(ValueError, match="positive"):
            get_small_batch(loader, n=0)

    def test_raises_on_negative(self):
        loader, *_ = load_mnist(train_batch_size=64)
        with pytest.raises(ValueError, match="positive"):
            get_small_batch(loader, n=-1)

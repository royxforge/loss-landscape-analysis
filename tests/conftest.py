"""Shared fixtures for loss-landscape-analysis tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add src to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(scope="session")
def device() -> str:
    """Return the device to use for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_logits(device: str) -> torch.Tensor:
    """Return a small batch of logits for loss function tests."""
    return torch.tensor(
        [[2.0, 1.0, 0.1], [0.1, 3.0, 0.5], [1.0, 0.5, 0.8]],
        device=device,
        dtype=torch.float32,
    )


@pytest.fixture
def sample_targets(device: str) -> torch.Tensor:
    """Return class indices matching sample_logits."""
    return torch.tensor([0, 1, 2], device=device, dtype=torch.long)


@pytest.fixture
def sample_images(device: str) -> torch.Tensor:
    """Return a small batch of dummy MNIST-like images (batch=2)."""
    return torch.randn(2, 1, 28, 28, device=device, dtype=torch.float32)

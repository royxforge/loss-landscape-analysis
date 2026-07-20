"""Tests for utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import (
    EarlyStopping,
    _to_serializable,
    format_metrics,
    load_results,
    save_results,
    set_seed,
)


class TestSetSeed:
    """Tests for the set_seed utility."""

    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)

        set_seed(42)
        b = torch.randn(5)

        assert torch.allclose(a, b)

    def test_different_seeds_different_values(self):
        set_seed(1)
        a = torch.randn(5)

        set_seed(2)
        b = torch.randn(5)

        assert not torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.randn(5)

        set_seed(42)
        b = np.random.randn(5)

        assert np.allclose(a, b)


class TestSerializable:
    """Tests for the _to_serializable helper."""

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_serializable(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_float(self):
        result = _to_serializable(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_torch_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _to_serializable(t)
        assert result == [1.0, 2.0, 3.0]

    def test_dict_with_torch_keys(self):
        d = {"a": torch.tensor(1.0), "b": [1, 2, 3]}
        result = _to_serializable(d)
        assert result == {"a": 1.0, "b": [1, 2, 3]}


class TestSaveLoadResults:
    """Tests for save_results and load_results."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        data = {"loss": 0.5, "accuracy": 0.95, "epochs": [1, 2, 3]}
        filepath = save_results(data, "test_results.json")
        loaded = load_results("test_results.json")
        assert loaded["loss"] == 0.5
        assert loaded["accuracy"] == 0.95
        assert loaded["epochs"] == [1, 2, 3]

    def test_save_with_tensor(self, tmp_path: Path):
        data = {"loss": torch.tensor(0.5)}
        filepath = save_results(data, "tensor_test.json")
        loaded = load_results("tensor_test.json")
        assert loaded["loss"] == 0.5


class TestFormatMetrics:
    """Tests for format_metrics utility."""

    def test_formats_floats(self):
        metrics = {"loss": 0.1234, "accuracy": 0.9876}
        result = format_metrics(metrics)
        assert "loss: 0.1234" in result
        assert "accuracy: 0.9876" in result

    def test_formats_integers(self):
        metrics = {"epoch": 5}
        result = format_metrics(metrics)
        assert "epoch: 5" in result


class TestEarlyStopping:
    """Tests for EarlyStopping utility."""

    def test_does_not_stop_early_when_loss_decreases(self):
        stopper = EarlyStopping(patience=3)
        for loss in [1.0, 0.8, 0.6, 0.4]:
            should_stop = stopper.step(loss)
            assert not should_stop

    def test_stops_when_loss_stagnates(self):
        stopper = EarlyStopping(patience=2)
        stopper.step(1.0)  # best = 1.0, counter = 0
        assert not stopper.step(1.1), "counter=1 < patience=2, should not stop"
        assert stopper.step(1.2), "counter=2 >= patience=2, should stop"
        assert stopper.step(1.3), "already stopped, should stay stopped"

    def test_resets_counter_on_improvement(self):
        stopper = EarlyStopping(patience=3)
        stopper.step(1.0)  # best = 1.0, counter = 0
        stopper.step(1.1)  # counter = 1
        stopper.step(0.9)  # best = 0.9, counter resets to 0
        stopper.step(0.95)  # counter = 1
        stopper.step(0.98)  # counter = 2
        # Next non-improving step: counter becomes 3, which equals patience
        assert stopper.step(0.99), "counter=3 >= patience=3, should stop"
        assert stopper.step(1.0), "already stopped, should stay stopped"

    def test_reset(self):
        stopper = EarlyStopping(patience=2)
        stopper.step(1.0)
        stopper.step(1.1)
        stopper.step(1.2)
        assert stopper.should_stop
        stopper.reset()
        assert stopper.best_loss == float("inf")
        assert stopper.counter == 0
        assert not stopper.should_stop

    def test_callable(self):
        stopper = EarlyStopping(patience=1)
        assert stopper(1.0) is False
        assert stopper(1.1) is True

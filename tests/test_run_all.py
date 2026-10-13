"""Tests for ``run_all.py`` experiment selection and CLI flags."""

from __future__ import annotations

import pytest

import run_all


def test_default_selection_returns_every_experiment():
    assert [e[0] for e in run_all.select_experiments()] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("raw", ["exp1,exp3", "1,3", " exp1 , 3 "])
def test_only_accepts_prefixed_and_bare_ids(raw):
    assert [e[0] for e in run_all.select_experiments(only=raw)] == [1, 3]


def test_only_is_case_insensitive():
    assert [e[0] for e in run_all.select_experiments(only="EXP2")] == [2]


def test_skip_removes_experiments():
    assert [e[0] for e in run_all.select_experiments(skip="exp5")] == [1, 2, 3, 4]


def test_only_and_skip_can_eliminate_everything():
    assert run_all.select_experiments(only="exp1", skip="exp1") == []


@pytest.mark.parametrize("raw", ["bogus", "expbogus", "0"])
def test_unknown_ids_fail_fast(raw):
    with pytest.raises(SystemExit) as exc:
        run_all.select_experiments(only=raw)
    assert "unknown experiment" in str(exc.value)


def test_empty_selection_is_rejected():
    with pytest.raises(SystemExit):
        run_all.select_experiments(only=",, ")


@pytest.mark.parametrize("extra", ["--only", "--skip"])
def test_plots_only_rejects_a_simultaneous_selection(extra):
    """--plots-only runs nothing, so pairing it with a selection is an error."""
    argv = ["--plots-only", extra, "exp1"]

    with pytest.raises(SystemExit) as exc:
        run_all.main(argv)

    assert "--plots-only cannot be combined" in str(exc.value)


def test_help_lists_available_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        run_all.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--only" in out
    assert "--skip" in out
    assert "--plots-only" in out
    assert "exp1" in out  # available experiments are discoverable


def test_unknown_selection_exits_before_running_anything():
    # The failure happens during validation, before any experiment or plot
    # side effect (a typo must not silently run the whole suite).
    with pytest.raises(SystemExit) as exc:
        run_all.main(["--only", "nope"])
    assert "unknown experiment" in str(exc.value)


def test_selection_excluding_everything_is_rejected():
    with pytest.raises(SystemExit) as exc:
        run_all.main(["--only", "exp1", "--skip", "exp1"])
    assert "excluded every experiment" in str(exc.value)

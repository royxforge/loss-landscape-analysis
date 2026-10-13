from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

from src.utils import load_results
from visualizations.plots import (
    plot_convergence,
    plot_failure_cases,
    plot_generalization,
    plot_gradient_norms,
    plot_lr_sensitivity,
    plot_summary_dashboard,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

EXPERIMENTS = [
    (1, "Convergence", "experiments.exp1_convergence"),
    (2, "Gradient Magnitude", "experiments.exp2_gradient_magnitude"),
    (3, "Learning Rate Sensitivity", "experiments.exp3_lr_sensitivity"),
    (4, "Generalization", "experiments.exp4_generalization"),
    (5, "Failure Cases", "experiments.exp5_failure_cases"),
]


def _ascii_header() -> str:
    return r"""
 _     ___  ____  ____    _____ _   _ _   _  ____ _____ ___ ___  _   _
| |   / _ \/ ___|/ ___|  |  ___| | | | \ | |/ ___|_   _|_ _/ _ \| \ | |
| |  | | | \___ \___ \  | |_  | | | |  \| | |     | |  | | | | |  \| |
| |__| |_| |___) |__) | |  _| | |_| | |\  | |___  | |  | | |_| | |\  |
|_____\___/|____/____/  |_|    \___/|_| \_|\____| |_| |___\___/|_| \_|

                     LOSS FUNCTION ANALYSIS
"""


def _first_epoch_at_or_above(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value >= threshold:
            return idx
    return None


def _run_experiments(experiments: list | None = None) -> dict:
    status = {}
    experiments = EXPERIMENTS if experiments is None else experiments

    for exp_num, exp_name, module_path in experiments:
        print(f"Running Experiment {exp_num}: {exp_name}...")
        start = time.time()

        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, "main"):
                raise AttributeError(f"{module_path} has no main()")
            result = module.main()
            elapsed = time.time() - start
            print(f"[OK] Experiment {exp_num} complete. Time: {elapsed:.2f}s")
            status[f"exp{exp_num}"] = {"ok": True, "time_s": elapsed, "result": result}
        except Exception as exc:
            elapsed = time.time() - start
            print(f"[FAIL] Experiment {exp_num} failed. Time: {elapsed:.2f}s")
            print(f"  Error: {exc}")
            traceback.print_exc()
            status[f"exp{exp_num}"] = {"ok": False, "time_s": elapsed, "error": str(exc)}

    return status


def _regenerate_all_plots() -> None:
    print("Regenerating plots from saved JSON results...")

    try:
        mse_hist = load_results("exp1_mse_history.json")
        ce_hist = load_results("exp1_ce_history.json")
        plot_convergence(mse_hist, ce_hist)
        print("  Generated: exp1_convergence.png")
    except Exception as exc:
        print(f"  Skipped exp1 plot: {exc}")

    try:
        mse_grad = load_results("exp2_mse_gradients.json")
        ce_grad = load_results("exp2_ce_gradients.json")
        plot_gradient_norms(mse_grad, ce_grad)
        print("  Generated: exp2_gradient_norms.png")
    except Exception as exc:
        print(f"  Skipped exp2 plot: {exc}")

    try:
        lr_results = load_results("exp3_lr_sensitivity.json")
        plot_lr_sensitivity(lr_results)
        print("  Generated: exp3_lr_sensitivity.png")
    except Exception as exc:
        print(f"  Skipped exp3 plot: {exc}")

    try:
        gen_results = load_results("exp4_generalization.json")
        plot_generalization(gen_results["mse_history"], gen_results["ce_history"])
        print("  Generated: exp4_generalization.png")
    except Exception as exc:
        print(f"  Skipped exp4 plot: {exc}")

    try:
        failure_results = load_results("exp5_failure_cases.json")
        plot_failure_cases(failure_results)
        print("  Generated: exp5_failure_cases.png")
    except Exception as exc:
        print(f"  Skipped exp5 plot: {exc}")

    try:
        plot_summary_dashboard()
        print("  Generated: summary_dashboard.png")
    except Exception as exc:
        print(f"  Skipped summary dashboard: {exc}")


def _print_final_summary_table() -> None:
    print("\nFinal Summary of Key Findings")
    print("-" * 72)

    try:
        mse_hist = load_results("exp1_mse_history.json")
        ce_hist = load_results("exp1_ce_history.json")
        threshold = mse_hist.get("accuracy_threshold", 0.97)
        mse_epoch = _first_epoch_at_or_above(mse_hist["test_accuracy"], threshold)
        ce_epoch = _first_epoch_at_or_above(ce_hist["test_accuracy"], threshold)
        print(f"Exp1 | First epoch test accuracy >= {threshold:g} | MSE: {mse_epoch} | CE: {ce_epoch}")
    except Exception:
        print("Exp1 | Not available")

    try:
        mse_grad = load_results("exp2_mse_gradients.json")
        ce_grad = load_results("exp2_ce_gradients.json")
        ratio5 = ce_grad["mean_grad_norm"][4] / max(mse_grad["mean_grad_norm"][4], 1e-12)
        ratio20 = ce_grad["mean_grad_norm"][19] / max(mse_grad["mean_grad_norm"][19], 1e-12)
        print(f"Exp2 | CE/MSE mean grad norm ratio | epoch5: {ratio5:.2f} | epoch20: {ratio20:.2f}")
    except Exception:
        print("Exp2 | Not available")

    try:
        exp3 = load_results("exp3_lr_sensitivity.json")
        lrs = sorted([float(k) for k in exp3.keys()])
        mse_div = sum(1 for lr in lrs if exp3[str(lr)]["mse"]["diverged"])
        ce_div = sum(1 for lr in lrs if exp3[str(lr)]["ce"]["diverged"])
        best_mse_lr = max(lrs, key=lambda x: exp3[str(x)]["mse"]["acc"])
        best_ce_lr = max(lrs, key=lambda x: exp3[str(x)]["ce"]["acc"])
        print(
            "Exp3 | Best accuracy and divergence "
            f"| MSE best LR={best_mse_lr:g}, acc={exp3[str(best_mse_lr)]['mse']['acc']:.4f}, div={mse_div}/{len(lrs)} "
            f"| CE best LR={best_ce_lr:g}, acc={exp3[str(best_ce_lr)]['ce']['acc']:.4f}, div={ce_div}/{len(lrs)}"
        )
    except Exception:
        print("Exp3 | Not available")

    try:
        exp4 = load_results("exp4_generalization.json")
        print(
            "Exp4 | Final generalization gap "
            f"| MSE: {exp4['final_mse_gap']:.4f} "
            f"| CE: {exp4['final_ce_gap']:.4f} "
            f"| Better: {exp4['better_generalization']}"
        )
    except Exception:
        print("Exp4 | Not available")

    try:
        exp5 = load_results("exp5_failure_cases.json")
        summary = exp5["summary"]
        print(
            "Exp5 | Failure-case gradient strength "
            f"| MSE grad: {summary['mse_mean_grad_norm']:.4f} "
            f"| CE grad: {summary['ce_mean_grad_norm']:.4f} "
            f"| Ratio CE/MSE: {summary['grad_ratio_ce_over_mse']:.2f}"
        )
    except Exception:
        print("Exp5 | Not available")

    print("-" * 72)


def _experiment_key(exp_num: int) -> str:
    return f"exp{exp_num}"


def _parse_selection(raw: str, flag: str) -> set[str]:
    """Parse a comma-separated experiment list such as ``exp1,3``."""
    keys = set()
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        keys.add(token if token.startswith("exp") else f"exp{token}")
    if not keys:
        raise SystemExit(
            f"{flag} needs at least one experiment id, e.g. {flag} exp1,exp3"
        )
    return keys


def select_experiments(only: str | None = None, skip: str | None = None) -> list:
    """Filter ``EXPERIMENTS`` by ``--only``/``--skip`` selections.

    Raises ``SystemExit`` on unknown ids so typos fail fast instead of
    silently running nothing.
    """
    known = {_experiment_key(n) for n, _, _ in EXPERIMENTS}
    selected = list(EXPERIMENTS)

    for raw, flag in ((only, "--only"), (skip, "--skip")):
        if not raw:
            continue
        keys = _parse_selection(raw, flag)
        unknown = keys - known
        if unknown:
            raise SystemExit(
                f"unknown experiment(s): {', '.join(sorted(unknown))} "
                f"(valid: {', '.join(sorted(known))})"
            )
        if flag == "--only":
            selected = [e for e in selected if _experiment_key(e[0]) in keys]
        else:
            selected = [e for e in selected if _experiment_key(e[0]) not in keys]

    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the loss-landscape experiments, rebuild plots, and print a summary.",
        epilog="Available experiments: " + ", ".join(_experiment_key(n) for n, _, _ in EXPERIMENTS),
    )
    parser.add_argument(
        "--only",
        metavar="IDS",
        help="Run only these experiments (comma-separated, e.g. --only exp1,exp3).",
    )
    parser.add_argument(
        "--skip",
        metavar="IDS",
        help="Skip these experiments (comma-separated, e.g. --skip exp5).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Do not run any experiment; rebuild plots and the summary from saved results.",
    )
    args = parser.parse_args(argv)

    # Refuse contradictory selections instead of silently ignoring one of them.
    if args.plots_only and (args.only or args.skip):
        raise SystemExit(
            "--plots-only cannot be combined with --only/--skip "
            "(no experiment runs in that mode; drop the selection flags)"
        )

    # Validate the selection before touching the filesystem or running anything.
    selected = None
    if not args.plots_only:
        selected = select_experiments(args.only, args.skip)
        if not selected:
            raise SystemExit("the given selection excluded every experiment")

    start_total = time.time()

    print(_ascii_header())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        print("Plots-only mode: reusing saved results (experiments not re-run).")
        status: dict = {}
    else:
        status = _run_experiments(selected)
    _regenerate_all_plots()
    _print_final_summary_table()

    total_runtime = time.time() - start_total
    print(f"Total runtime: {total_runtime:.2f}s")

    # Surface failures to the shell so CI and wrapper scripts can detect them.
    failed = sorted(name for name, info in status.items() if not info["ok"])
    if failed:
        print(f"\n{len(failed)} experiment(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

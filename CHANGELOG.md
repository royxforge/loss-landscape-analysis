# Changelog

All notable changes to the Loss Landscape Analysis project are documented herein.

The format adheres to the principles of semantic versioning, with each entry distinguishing between structural enhancements, documentation revisions, and experimental refinements.

---

## [1.2.0] - 2026-07-20

### Added

- **Comprehensive test suite**: Added `tests/` directory with 42 unit tests across five test modules covering model architecture, loss functions, data loading, training logic, and utility functions. All tests pass on CPU without requiring GPU or CUDA.
  - `tests/test_model.py`: 9 tests validating SmallNet initialization, forward shape, weight flattening, reset reproducibility, gradient flow, and deterministic behavior.
  - `tests/test_losses.py`: 16 tests verifying custom MSE and Cross-Entropy implementations against PyTorch references, numerical stability for extreme inputs, and the theoretical gradient saturation difference between loss functions.
  - `tests/test_data.py`: 10 tests for MNIST loading, batch shapes, label types, deterministic shuffling, and `get_small_batch` utility.
  - `tests/test_trainer.py`: 8 tests for `compute_gradient_norm`, `Trainer` epoch metrics, multi-epoch training convergence, and loss/accuracy improvement tracking.
  - `tests/test_utils.py`: 17 tests for seed reproducibility, NumPy/PyTorch serialization, result save/load roundtrip, metric formatting, and `EarlyStopping` callback behavior.
  - `tests/conftest.py`: Shared pytest fixtures providing device detection, sample logits, targets, and images.

### Fixed

- **Package discovery**: `setup.py` now correctly discovers the `src/` package by using `find_packages(where="src")` with an explicit `package_dir={"": "src"}` mapping, resolving an issue where `pip install -e .` would install an empty package.

## [1.1.0] - 2026-07-20

### Added

- **Community health files**: Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md` (contribution guidelines with setup, coding standards, and PR process), `SECURITY.md` (vulnerability reporting policy), and `CITATION.cff` (citation metadata for academic attribution). These files establish the project governance and community participation framework.

---

## [1.0.0] - 2026-07-14

### Summary

Stable release of the Loss Landscape Analysis framework. The repository delivers a rigorous empirical comparison of Mean Squared Error (MSE) versus Cross-Entropy (CE) loss functions on an MNIST classification task, comprising five controlled experiments that quantify convergence dynamics, gradient saturation phenomena, learning-rate sensitivity, generalization behavior under data scarcity, and gradient propagation in confidently misclassified regimes.

---

### Added

- **Modular experiment pipeline** (`experiments/`): Five independently executable experiment scripts, each exposing a `main()` function for programmatic orchestration and manual invocation.
  - `exp1_convergence.py`: Primary convergence comparison between MSE and CE over 30 epochs under identical architecture and optimizer configurations. Asserts that CE reaches the test-loss threshold (0.5) in fewer epochs than MSE.
  - `exp2_gradient_magnitude.py`: Per-epoch gradient norm tracking (mean, standard deviation, minimum, maximum) for both loss functions across 20 epochs, providing empirical evidence for the gradient saturation hypothesis.
  - `exp3_lr_sensitivity.py`: Systematic sweep over five learning rates (1×10⁻⁴, 1×10⁻³, 1×10⁻², 1×10⁻¹, 5×10⁻¹) with divergence detection for both MSE and CE, quantifying how loss-function choice interacts with optimization stability.
  - `exp4_generalization.py`: Data-scarcity regime using a 1,000-sample training subset across 50 epochs, computing the generalization gap (test loss − train loss) as a function of epoch for each loss function.
  - `exp5_failure_cases.py`: Targeted analysis of gradient norm behavior on confidently wrong predictions (softmax confidence > 0.8), isolating the saturation effect under precisely the conditions where theory predicts CE should maintain stronger corrective signals.

- **Core library modules** (`src/`): A reusable, type-annotated Python package encapsulating the experimental infrastructure.
  - `data.py`: MNIST loading with configurable batch sizes, deterministic seeded shuffling via `torch.Generator`, and a utility for extracting fixed-size batches (`get_small_batch`).
  - `losses.py`: Custom implementations of MSE and Cross-Entropy loss functions with explicit softmax/log-softmax computations, ensuring full transparency of the gradient computation chain. Includes auxiliary functions `softmax`, `log_softmax`, and `to_one_hot`.
  - `model.py`: `SmallNet` - a fixed three-layer MLP (784 → 128 ReLU → 64 ReLU → 10) with deterministic Xavier-uniform initialization seeded at 42. Provides `get_weights_flat()` for parameter-space analysis and `reset_weights()` for reproducible reinitialization.
  - `trainer.py`: `Trainer` class encapsulating epoch-level training and evaluation loops with per-batch gradient norm capture. Includes `compute_gradient_norm()` for global L₂ gradient magnitude computation.
  - `utils.py`: Reproducibility infrastructure (`set_seed`), JSON serialization/deserialization with NumPy and PyTorch tensor support (`save_results`, `load_results`), metric formatting utilities, and an `EarlyStopping` callback.

- **Visualization subsystem** (`visualizations/`): Publication-quality matplotlib-based plotting with consistent color schemes (MSE: `#E74C3C`, CE: `#2ECC71`), Seaborn styling, and 150 DPI output resolution.
  - `plots.py`: Six visualization functions - `plot_convergence`, `plot_gradient_norms`, `plot_lr_sensitivity`, `plot_generalization`, `plot_failure_cases`, and `plot_summary_dashboard` - each generating self-contained multi-panel figures saved to the results directory.

- **Orchestration script** (`run_all.py`): Sequential execution of all five experiments with automatic plot regeneration from cached JSON results, final summary table computation, and total runtime measurement. Employs `importlib` for dynamic experiment loading.

- **Mathematical derivations** (`math_derivations/derivations.md`): Comprehensive LaTeX-formulated derivation of MSE and Cross-Entropy gradients with respect to pre-softmax logits, including full chain-rule expansion, analytical comparison of saturation properties, and a worked numerical example demonstrating the damping factor at different prediction-confidence regimes.

- **Package configuration** (`setup.py`): Setuptools-based package definition with `find_packages()` auto-discovery and Python ≥3.10 requirement.

- **Dependency specification** (`requirements.txt`): Pinned minimum versions for PyTorch (≥2.0.0), torchvision (≥0.15.0), matplotlib (≥3.7.0), NumPy (≥1.24.0), and Seaborn (≥0.12.0).

- **Project governance** (`LICENSE`): MIT License, copyright Sourav Roy, 2026.

- **Experimental results** (`results/`): Pre-computed JSON output files from all five experiments, enabling plot regeneration without re-execution.

---

### Changed

- **README.md** - Multiple iterative revisions (commits `4aff1d0`, `68b310e`, `814bcf9`, `da610c9`, `e212a53`, `2274622`) progressively enriching the documentation:
  - Expanded from a minimal stub into a comprehensive research narrative including a formal research question, quantitative key-results table, full theoretical background with gradient equations, detailed experimental setup table, per-epoch gradient dynamics analysis, installation and usage instructions, and a research context section linking to related work on confidence estimation and production drift detection.
  - Refined terminology from earlier project names to the finalized "Loss Landscape Analysis" branding.
  - Added badge suite (Python 3.10+, PyTorch, MNIST, MIT License), BibTeX citation entry, and structured table of contents.
  - Final revision (2026-07-14) delivering the canonical v1.0.0 documentation.

- **Code comments and docstrings**: Standardized to "Loss Landscape Analysis" terminology and `from __future__ import annotations` style throughout all modules.

---

### Fixed

- Cross-repository author identity normalization across the two initial commits from distinct remote origins (`royxlead` and `royxforge` accounts), resolved via merge commit `2d0c86a`.

---

## [0.1.0] - 2026-05-20

### Initial Commit

Repository inception with the foundational codebase implementing MSE versus Cross-Entropy comparative analysis on MNIST. Established the complete experimental framework, core library modules, visualization pipeline, mathematical derivations, and orchestration logic as described in the v1.0.0 release notes above. The LICENSE file (MIT) was introduced in a parallel initial commit from a separate remote and subsequently merged.

Initial experimental results, figures, and the comprehensive README documentation were included at this stage.

---

*All timestamps in Indian Standard Time (UTC+05:30). For a complete list of individual commits, refer to the repository's Git history.*

[Unreleased]: https://github.com/royxforge/loss-landscape-analysis/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/royxforge/loss-landscape-analysis/releases/tag/v1.2.0
[1.1.0]: https://github.com/royxforge/loss-landscape-analysis/releases/tag/v1.1.0
[1.0.0]: https://github.com/royxforge/loss-landscape-analysis/releases/tag/v1.0.0
[0.1.0]: https://github.com/royxforge/loss-landscape-analysis/releases/tag/v0.1.0

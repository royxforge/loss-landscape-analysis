# Contributing to Loss Landscape Analysis

Thank you for your interest in contributing to Loss Landscape Analysis! This
document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior to royxforge@proton.me.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/loss-landscape-analysis.git
   cd loss-landscape-analysis
   ```
3. Add the upstream repository:
   ```
   git remote add upstream https://github.com/royxforge/loss-landscape-analysis.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- pip

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Verify Installation

```bash
python -c "from src import utils; print('Setup OK')"
```

## Coding Standards

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use type annotations for all function signatures.
- Use `from __future__ import annotations` for deferred evaluation.
- Use descriptive variable names; avoid single-letter names except for
  loop indices and mathematical variables.
- Maximum line length: 88 characters (Black-compatible).

### Imports

Organize imports in the following order, separated by a blank line:

1. Standard library imports
2. Third-party imports
3. Local application imports

### Docstrings

Use Google-style docstrings for all public modules, classes, and functions:

```python
def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the global L2 gradient norm of all model parameters.

    Args:
        model: The PyTorch model whose gradient norm to compute.

    Returns:
        The global L2 norm of all parameter gradients.
    """
```

## Testing

- All experiments must be reproducible (use `set_seed(42)`).
- Run experiments end-to-end before submitting changes:
  ```bash
  python run_all.py
  ```
- Verify that the experiment assertions pass (e.g., CE reaches threshold
  faster than MSE).
- Add tests for new utility functions in `tests/` if applicable.

## Pull Request Process

1. Create a new branch from `main` for your changes:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commit messages.

3. Ensure your code is properly formatted:
   ```bash
   pip install black
   black .
   ```

4. Run the full experiment suite to verify nothing is broken:
   ```bash
   python run_all.py
   ```

5. Push your branch and open a Pull Request on GitHub.

6. In your PR description, include:
   - What the change does
   - Any relevant issue numbers
   - How you tested the change
   - Screenshots or results if applicable

7. Request review from a maintainer.

## Commit Message Conventions

We follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(experiment): add learning rate sensitivity sweep
fix(trainer): correct gradient accumulation for batch sizes
docs(readme): update benchmark results table
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details (OS, Python version, PyTorch version)
- Relevant code snippets or error logs
- Any recent changes that may have triggered the bug

### Feature Requests

We welcome feature suggestions! Please include:

- A clear description of the proposed feature
- The motivation or use case
- Any relevant research or references
- Whether you are willing to implement it

## Research Contributions

If you would like to extend the experimental framework (e.g., add a new loss
function comparison, new architecture, additional dataset), please open an issue
first to discuss the design. We are particularly interested in contributions
that:

- Extend the comparison to additional loss functions
- Add support for different architectures or datasets
- Improve visualization quality or add new plot types
- Expand the theoretical analysis in `math_derivations/`

Thank you for helping make Loss Landscape Analysis better!

# Loss Function Analysis
### MSE vs Cross-Entropy on Classification Tasks

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-MNIST-14b8a6?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-6366f1?style=flat-square" />
</p>

> A rigorous empirical comparison of Mean Squared Error (MSE) and Cross-Entropy (CE) loss functions on classification using MNIST. The theoretical argument for preferring CE on classification is well known. This repository makes it concrete, measurable, and reproducible.

---

## Table of Contents

- [The Research Question](#the-research-question)
- [Key Results](#key-results)
- [Theoretical Background](#theoretical-background)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Gradient Dynamics](#gradient-dynamics)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Research Context](#research-context)
- [Citation](#citation)

---

## The Research Question

MSE and Cross-Entropy are both valid loss functions. Both measure the distance between predictions and targets. Both converge on MNIST. Yet CE consistently outperforms MSE on classification tasks. Why?

> *Loss function choice is not a hyperparameter. It is a modeling decision with theoretical consequences that show up in gradient dynamics from the first epoch.*

The standard explanation is gradient saturation - but at what magnitude, at what epoch, and by how much? This project quantifies it.

---

## Key Results

| Metric | Cross-Entropy | MSE |
|---|---|---|
| Final test accuracy (epoch 30) | **98.06%** | 97.79% |
| Final train accuracy | **100%** | 99.8% |
| Epoch to reach 97% test accuracy | **Epoch 4** | Epoch 13 |
| Mean gradient norm (epoch 1) | **1.528** | 0.900 |
| Mean gradient norm (epoch 30) | 0.021 | 0.051 |

**CE reaches 97% accuracy in 4 epochs. MSE takes 13.** Both start above 90% from epoch 1, but MSE falls progressively behind at higher accuracy thresholds - exactly where gradient saturation would predict the gap to emerge.

---

## Theoretical Background

### Why CE outperforms MSE on classification

For a softmax classifier with output probabilities **p** and one-hot targets **y**:

**MSE gradient** with respect to the pre-softmax logits contains the Jacobian of the softmax:

```
dL_MSE/dz = (p - y) * p * (1 - p)
```

The factor `p(1-p)` kills the gradient when predictions are confidently wrong (p near 0 or 1). A model that is confidently incorrect receives almost no corrective signal.

**CE gradient** with respect to the pre-softmax logits simplifies cleanly:

```
dL_CE/dz = (p - y)
```

The gradient is linear in the prediction error. No saturation. The corrective signal scales directly with how wrong the model is.

This is not a subtle difference. At epoch 1, CE gradients are 1.70x larger than MSE gradients on the same architecture and data. That early-training advantage compounds across epochs.

---

## Experimental Setup

**Dataset:** MNIST

| Property | Value |
|---|---|
| Training samples | 60,000 (full MNIST train set) |
| Test samples | 10,000 (full MNIST test set) |
| Generalization experiment | 1,000 training samples (Exp 4) |
| Task | 10-class digit classification |

**Architecture:** MLP, identical for both conditions

| Component | Value |
|---|---|
| Input | 784 (28x28 flattened) |
| Hidden layer 1 | 128 units, ReLU |
| Hidden layer 2 | 64 units, ReLU |
| Output | 10 units (logits) |
| Weight initialization | Xavier uniform, seed=42 |
| Optimizer | SGD, momentum=0.9 |
| Learning rate | 0.01 (main experiments) |
| Batch size | 64 (train), 1000 (test) |
| Epochs | 30 |

**Experiments:**

| Experiment | Description |
|---|---|
| Exp 1 | Main comparison: CE vs MSE, 30 epochs, full dataset |
| Exp 2 | Gradient magnitude logging per epoch |
| Exp 3 | Learning rate sensitivity sweep: 0.0001, 0.001, 0.01, 0.1, 0.5 |
| Exp 4 | Generalization under data scarcity: 1,000 training samples |
| Exp 5 | Extended training analysis |

---

## Results

### Accuracy and Loss

| Metric | Cross-Entropy | MSE |
|---|---|---|
| Final test accuracy | **98.06%** | 97.79% |
| Final test loss | 0.0934 | 0.0348 |
| Final train accuracy | **100%** | 99.8% |

Note: MSE test loss is lower in absolute terms (0.0348 vs 0.0934) because MSE and CE operate on different scales and are not directly comparable as numbers. The accuracy gap is what matters - CE is higher despite the lower raw loss value.

### Convergence Speed

| Threshold | CE reaches at | MSE reaches at |
|---|---|---|
| 90% test accuracy | **Epoch 1 (96.34%)** | Epoch 1 (95.29%) |
| 97% test accuracy | **Epoch 4 (97.33%)** | Epoch 13 (97.78%) |

Both loss functions clear 90% in the first epoch. The gap appears and widens as the accuracy threshold increases - consistent with the saturation hypothesis. CE is 3.25x faster to reach 97% than MSE on the same architecture.

---

## Gradient Dynamics

The gradient norm comparison across training directly confirms the theoretical prediction:

| Epoch | CE Mean Grad | MSE Mean Grad | CE / MSE Ratio |
|---|---|---|---|
| 1 (early) | **1.528** | 0.900 | **1.70x** |
| 5 | **0.859** | 0.561 | **1.53x** |
| 10 | 0.410 | 0.378 | 1.09x |
| 20 | 0.039 | **0.109** | 0.36x |
| 30 | 0.021 | **0.051** | 0.42x |

**Reading the table:**

Epochs 1-10: CE gradients are larger, driving faster learning. The 1.70x advantage in epoch 1 is the early-training signal that causes CE to reach 97% accuracy 9 epochs ahead of MSE.

Epochs 20-30: MSE gradients are now larger than CE gradients. This is not MSE recovering - it is MSE still trying to converge on errors that CE has already resolved. The larger late-training MSE gradients reflect residual error, not learning signal strength.

CE gradient norms decrease sharply (1.528 to 0.021) because the model has converged. MSE gradient norms remain elevated (0.109 at epoch 20) because the model has not converged as completely. The inversion in the ratio confirms that CE's advantage is front-loaded, which is precisely where gradient saturation theory predicts it.

---

## Repository Structure

```
loss-function-analysis-python/
|
+-- loss_analysis.ipynb    # Main experiment notebook (all 5 experiments)
+-- requirements.txt
+-- LICENSE
+-- README.md
```

---

## Installation

```bash
git clone https://github.com/royxlead/loss-function-analysis-python.git
cd loss-function-analysis-python

pip install -r requirements.txt
```

**Core dependencies:** PyTorch · torchvision · Matplotlib · NumPy · Jupyter

---

## Usage

```bash
jupyter notebook loss_analysis.ipynb
```

Run cells sequentially. The notebook is structured into 5 experiments with separate sections for each. All experiments use seed=42 for reproducibility. CUDA is used automatically if available; falls back to CPU.

To test a different architecture, modify the MLP definition in the model cell. The training loop, gradient logging, and evaluation code are architecture-agnostic.

---

## Research Context

The MSE vs CE debate is often treated as settled - "use CE for classification" - without quantifying the magnitude of the difference or the mechanism that produces it. This experiment makes the mechanism measurable: gradient saturation is not theoretical, it shows up as a 1.70x gradient norm difference in the first epoch and a 9-epoch convergence gap at the 97% accuracy threshold.

The result connects to broader themes in neural network design:

**Activation function design** - ReLU replaced sigmoid/tanh partly because of gradient saturation in deep networks. The same saturation mechanism operates in the output layer when MSE is used with softmax.

**Calibration** - CE loss is derived from maximum likelihood estimation under a categorical distribution, which produces better-calibrated probability outputs than MSE. This connects directly to the calibration analysis in [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python), where ECE (Expected Calibration Error) is a primary evaluation metric.

**Production reliability** - A model trained with MSE may produce confidence scores that are less meaningful than one trained with CE. This is not just a training efficiency issue - it affects every downstream system that consumes model probabilities, including the drift and confidence monitoring in [DriftWatch](https://github.com/royxlead/driftwatch-python).

---

## Related Work

- [Self-Diagnosing Neural Models](https://github.com/royxlead/self-diagnosing-neural-models-python) - How loss function choice affects calibration and uncertainty estimation
- [DriftWatch](https://github.com/royxlead/driftwatch-python) - Production monitoring of model confidence signals
- [Multi-Objective Feature Selection](https://github.com/royxlead/multi-objective-evolutionary-feature-selection-python) - Optimization in ML systems

---

## Citation

```bibtex
@software{roy2025lossfunctionanalysis,
  author = {Roy, Sourav},
  title  = {Loss Function Analysis: MSE vs Cross-Entropy on Classification Tasks},
  year   = {2026},
  url    = {https://github.com/royxlead/loss-function-analysis-python}
}
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/royxlead">Sourav Roy</a> · Founding AI/ML Engineer · Yuga AI</sub>
</p>

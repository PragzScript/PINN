# 🔥 Physics-Informed Neural Network — 1D Heat Equation

> A neural network that learns physics. No traditional numerical solver needed.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org/)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=flat&logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to approximate the solution of the **1D Heat Diffusion PDE** — without using any traditional numerical methods like finite difference or finite element analysis.

The neural network learns by minimizing a loss function that encodes:
- The **PDE residual** (physics constraint)
- **Initial conditions**
- **Boundary conditions**

### The PDE

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0,1], \quad t \in [0,1]$$

| Term | Description |
|------|-------------|
| `u(x, t)` | Temperature at position x and time t |
| `α` (alpha) | Thermal diffusivity constant |
| IC: `u(x, 0) = sin(πx)` | Initial temperature profile |
| BC: `u(0, t) = u(1, t) = 0` | Dirichlet boundary conditions |

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| **MSE** | **1.33 × 10⁻⁶** |
| **Max Absolute Error** | 0.003209 |
| Epochs | 5,000 |
| Collocation Points | 1,000 |

> ✅ PINN successfully approximates the heat equation with research-level accuracy.
> - **4-layer fully connected network**

- **Tanh activations** (smooth, differentiable — required for autograd-based PDE residuals)

- **Adam optimizer** with StepLR scheduler

- **Automatic differentiation** via `torch.autograd.grad` for computing PDE derivatives

---

## 📂 Project Structure                                                                                                                                                           ---

## 📓 Notebook Walkthrough

| Cell | Description |

|------|-------------|

| 1 | Install libraries, setup GPU |

| 2 | Define PINN model architecture |

| 3 | PDE residual loss (heat equation) |

| 4 | Initial condition + boundary condition losses |

| 5 | Training loop with logging |

| 6 | Training loss curve |

| 7 | Analytical (exact) solution |

| 8 | 2D heatmap — Exact vs PINN vs Error |

| 9 | MSE evaluation |

| 10 | Snapshot plots at t = 0.0, 0.1, 0.5, 1.0 |

| 11 | Animation — heat diffusion over time |

| 12 | Experiment — new IC: u(x,0) = x(1−x) |

| 13 | Experiment — compare α ∈ {0.001, 0.01, 0.05, 0.1} |

---

## 🚀 Getting Started

### Run in Google Colab (Recommended)

1. Click the Open in Colab badge above

2. Go to Runtime → Change runtime type → T4 GPU

3. Run all cells sequentially with Shift + Enter

### Run Locally                                                                                                                                                         ---

## 🔬 Experiments

### Experiment 1 — Different Initial Condition

Replaced u(x,0) = sin(πx) with a parabolic profile u(x,0) = x(1−x) to test PINN generalization to unseen initial conditions.

### Experiment 2 — Thermal Diffusivity Comparison

| α | Physical Meaning | Diffusion Speed |

|---|-----------------|-----------------|

| 0.001 | Low conductivity (insulator) | Very slow |

| 0.01 | Default (baseline) | Moderate |

| 0.05 | Medium conductivity | Fast |

| 0.1 | High conductivity (metal) | Very fast |

> Higher α → heat spreads faster → flatter temperature profile at same time t.

---

## 🏭 Industrial Relevance

This approach directly applies to engineering simulations in:

- 🔧 **Turbine blade thermal analysis** — modeling heat transfer across blade cross-sections

- 🌡️ **Steam path components** — temperature distribution in high-pressure steam environments

- ⚙️ **Heat exchanger design** — optimizing thermal profiles without expensive FEM solvers

PINNs are increasingly used as a mesh-free, data-efficient alternative to traditional numerical PDE solvers like FEM and FDM.

---

## 🛠️ Tech Stack

| Tool | Purpose |

|------|---------|

| PyTorch | Neural network + autograd for PDE derivatives |

| NumPy | Analytical solution computation |

| Matplotlib | Heatmaps, snapshots, animation |

| Google Colab | GPU-accelerated training environment |

---

## 📚 References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. Journal of Computational Physics.

- Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks.

---

## 👩‍💻 Author

**Pragya Jha**

Final-year B.Tech CS | University Topper (8.4 GPA)

[LinkedIn](https://www.linkedin.com/in/pragya-jha-1399192a9) • [GitHub](https://github.com/PragzScript)

> "Neural networks can replace traditional numerical PDE solvers — this project demonstrates exactly that."

---

## 🏗️ Architecture

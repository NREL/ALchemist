# ALchemist
**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

**ALchemist** is a modular Python toolkit that brings **active learning** and **Bayesian optimization** to experimental design in chemical and materials research.  
It is built for scientists who want to efficiently explore or optimize **high-dimensional variable spaces**—without needing to write any code—through a lightweight **CustomTkinter** graphical interface.

---

## Purpose

ALchemist helps researchers accelerate discovery and optimization by combining:

- Mixed-variable search spaces (real, integer, categorical)
- Probabilistic surrogate models (Gaussian processes)
- Advanced acquisition strategies for efficient sampling
- Intuitive GUI workflows for experiment planning and model refinement

---

## Key Features (v0.1.0 beta)

| Category | Highlights |
|----------|------------|
| **Search space definition** | Supports continuous, integer, and categorical parameters with bounds and priors |
| **Model back-end** | Gaussian process regression via **BoTorch** or **scikit-optimize** backend |
| **Acquisition functions** | *qEI*, *qPI*, *qUCB*, and *qNegIntegratedPosteriorVariance* for exploratory learning |
| **GUI workflow** | Configure variable space, set up initial experiments or load existing experimental data, train model, and execute acquisition functions |
| **Experiment logging** | CSV checkpoints, reproducible random seeds, and basic error tracking |
| **Extensibility** | Abstract model and acquisition interfaces allow for future back-end additions (e.g. Ax, deep learning models)

---

## Installation

```bash
git clone https://github.com/calebcoatney/ALchemist.git
cd ALchemist
python -m pip install -e .
```

Requires Python ≥ 3.9. All dependencies are specified in `pyproject.toml` and will be installed automatically.

---

## Quick Start

To launch the graphical user interface:

```bash
python main.py
```

From the GUI, you can:

- Define the optimization variables and constraints
- Initialize experiments using built-in sampling methods
- Run active learning loops with model-based suggestions
- Export logs and visualizations of model predictions

---

## Development Status

ALchemist is in active development at NREL as part of research supported by the DataHub project within the ChemCatBio consortium. It is designed to be approachable for non-ML researchers and extensible for power users who want to incorporate custom models, acquisition strategies, or lab automation workflows. Future versions will include improved DoE initial sampling, additional model types, enhanced visualization tools, and a reimplementation of the GUI in PySide6 for better cross-platform compatibility.

---

## License

This project is licensed under the MIT License. See the LICENSE file for full terms.

---

## Repository

<https://github.com/calebcoatney/ALchemist>


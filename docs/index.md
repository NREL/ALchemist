#![ALchemist](assets/logo.png){ width="400"}

**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is designed for scientists and engineers who want to efficiently explore or optimize high-dimensional variable spaces—without writing code—using an intuitive graphical interface.

---

## Key Features

- **Flexible variable space definition:** Real, integer, and categorical variables with bounds or discrete values.
- **Probabilistic surrogate modeling:** Gaussian process regression via BoTorch or scikit-optimize backends.
- **Advanced acquisition strategies:** Efficient sampling using qEI, qPI, qUCB, and qNegIntegratedPosteriorVariance.
- **Intuitive GUI workflow:** No coding required—define variables, generate initial experiments, load data, train models, and suggest new experiments.
- **Experiment tracking:** CSV logging, reproducible random seeds, and error tracking.
- **Extensibility:** Abstract interfaces for models and acquisition functions enable future backend and workflow expansion.

---

## Installation

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) to manage your Python environments for ALchemist.

**1. Create a new environment:**
```bash
conda create -n alchemist-env python=3.12
conda activate alchemist-env
```

**2. Clone the ALchemist repository:**
```bash
git clone https://github.com/NREL/ALchemist.git
cd ALchemist
```

**3. Install ALchemist:**
```bash
python -m pip install -e .
```

All dependencies are specified in `pyproject.toml` and will be installed automatically.

After installation, launch the graphical user interface by running:

```bash
alchemist
```

---

Use the sidebar to navigate through the documentation. See [Getting Started](setup/variable_space.md) to define your variable space and generate initial experiments.
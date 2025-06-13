# ALchemist

**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is designed for scientists and engineers who want to efficiently explore or optimize high-dimensional variable spaces—without writing code—using an intuitive graphical interface.

---

## 📖 Documentation

Full user guide and documentation:  
[https://calebcoatney.github.io/ALchemist/](https://calebcoatney.github.io/ALchemist/)

---

## 🚀 Overview

ALchemist accelerates discovery and optimization by combining:

- **Flexible variable space definition:** Real, integer, and categorical variables with bounds or discrete values.
- **Probabilistic surrogate modeling:** Gaussian process regression via BoTorch or scikit-optimize backends.
- **Advanced acquisition strategies:** Efficient sampling using qEI, qPI, qUCB, and qNegIntegratedPosteriorVariance.
- **Intuitive GUI workflow:** No coding required—define variables, generate initial experiments, load data, train models, and suggest new experiments.
- **Experiment tracking:** CSV logging, reproducible random seeds, and error tracking.
- **Extensibility:** Abstract interfaces for models and acquisition functions enable future backend and workflow expansion.

---

## 🗂️ Project Structure

- **Setup:** Define variable spaces, generate initial experiments, load and visualize experimental data.
- **Surrogate Modeling:** Train and evaluate models using scikit-optimize or BoTorch.
- **Visualizations:** Analyze model performance, error metrics, and create contour plots.
- **Acquisition & Optimization:** Run acquisition functions and log/track experiment suggestions.
- **Educational Resources:** Learn about active learning and Bayesian optimization concepts.

See the [documentation](https://calebcoatney.github.io/ALchemist/) for detailed guides on each workflow step.

---

## 💻 Installation

Requirements: Python 3.9 or higher

```bash
git clone https://github.com/calebcoatney/ALchemist.git
cd ALchemist
python -m pip install -e .
```

All dependencies are specified in `pyproject.toml` and will be installed automatically.

---

## 🏁 Quick Start

To launch the graphical user interface:

```bash
alchemist
```

From the GUI, you can:

- Define optimization variables and constraints
- Generate initial experiments or load existing data
- Train surrogate models and evaluate acquisition functions
- Visualize model predictions and export logs

For step-by-step instructions, see the [Getting Started](https://calebcoatney.github.io/ALchemist/) section of the documentation.

---

## 🛠️ Development Status

ALchemist is under active development at NREL as part of the DataHub project within the ChemCatBio consortium. It is designed to be approachable for non-ML researchers and extensible for advanced users. Planned features include:

- Enhanced initial sampling and DoE methods
- Additional model types and acquisition strategies
- Improved visualization tools
- GUI reimplementation in PySide6 for broader compatibility

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Repository

[https://github.com/calebcoatney/ALchemist](https://github.com/calebcoatney/ALchemist)


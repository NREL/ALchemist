<img src="docs/assets/NEW_LOGO_LIGHT.png" alt="ALchemist" width="50%" />

**ALchemist: Active Learning Toolkit for Chemical and Materials Research**

ALchemist is a modular Python toolkit that brings active learning and Bayesian optimization to experimental design in chemical and materials research. It is designed for scientists and engineers who want to efficiently explore or optimize high-dimensional variable spaces—without writing code—using an intuitive graphical interface.

**NREL Software Record:** SWR-25-102

---

## 📖 Documentation

Full user guide and documentation:  
[https://nrel.github.io/ALchemist/](https://nrel.github.io/ALchemist/)

---

## 🚀 Overview

ALchemist accelerates discovery and optimization by combining:

- **Flexible variable space definition:** Real, integer, and categorical variables with bounds or discrete values.
- **Probabilistic surrogate modeling:** Gaussian process regression via BoTorch or scikit-optimize backends.
- **Advanced acquisition strategies:** Efficient sampling using qEI, qPI, qUCB, and qNegIntegratedPosteriorVariance.
- **Modern web interface:** React-based UI with FastAPI backend for seamless active learning workflows.
- **Autonomous optimization:** Human-out-of-the-loop optimization for real-time process control.
- **Experiment tracking:** CSV logging, reproducible random seeds, and error tracking.
- **Extensibility:** Abstract interfaces for models and acquisition functions enable future backend and workflow expansion.

### Use Cases

- **Interactive Optimization**: Desktop GUI or web UI for manual experiment design
- **Programmatic Workflows**: Python Session API for scripts and notebooks
- **Autonomous Operation**: REST API for real-time process control (reactors, synthesis, etc.)
- **Remote Collaboration**: Web-based interface accessible from any device

---

## 🧭 Quick Start

### Installation

**Requirements:** Python 3.11 or higher

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) to manage your Python environments.

**1. Create a new environment:**
```bash
conda create -n alchemist-env python=3.11
conda activate alchemist-env
```

**2. Install ALchemist:**

*Option A: From PyPI (recommended):*
```bash
pip install alchemist-nrel
```

*Option B: From GitHub:*
```bash
pip install git+https://github.com/NREL/ALchemist.git
```

*Option C: Development install (for contributors):*
```bash
git clone https://github.com/NREL/ALchemist.git
cd ALchemist
pip install -e .
```

All dependencies are specified in `pyproject.toml` and will be installed automatically.

**Note:** The web UI is pre-built and included in the package. You do **not** need Node.js/npm to use ALchemist unless you're developing the frontend.

### Running ALchemist

**Web Application (Recommended):**
```bash
alchemist-web
# Opens at http://localhost:8000
```

**Desktop Application:**
```bash
alchemist
# Launches CustomTkinter GUI
```

**Development Mode (Frontend Developers):**
```bash
# Terminal 1: Backend with hot-reload
python run_api.py

# Terminal 2: Frontend with hot-reload
cd alchemist-web
npm install  # First time only
npm run dev
# Opens at http://localhost:5173
```

**Docker Deployment:**
```bash
docker pull ghcr.io/nrel/alchemist:latest
docker run -p 8000:8000 ghcr.io/nrel/alchemist:latest

# Or build from source:
cd docker
docker-compose up --build
```

For step-by-step instructions, see the [Getting Started](https://nrel.github.io/ALchemist/) section of the documentation.

---

## 📁 Project Structure

```
ALchemist/
├── alchemist_core/       # Core Python library
├── alchemist-web/        # React frontend application
├── api/                  # FastAPI backend
├── docker/               # Docker configuration files
├── scripts/              # Build and development scripts
├── tests/                # Test suite
├── docs/                 # Documentation (MkDocs)
├── memory/               # Development notes and references
└── run_api.py           # API server entry point
```

---

## 🛠️ Development Status

ALchemist is under active development at NREL as part of the DataHub project within the ChemCatBio consortium. It is designed to be approachable for non-ML researchers and extensible for advanced users. Planned features include:

- Enhanced initial sampling and DoE methods
- Additional model types and acquisition strategies
- Improved visualization tools
- GUI reimplementation in PySide6 for broader compatibility
- Support for multi-output models and multi-objective optimization

---

## 🐞 Issues & Troubleshooting

If you encounter any issues or have questions, please [open an issue on GitHub](https://github.com/NREL/ALchemist/issues) or contact ccoatney@nrel.gov.

For the latest known issues and troubleshooting tips, see the [Issues & Troubleshooting Log](docs/ISSUES_LOG.md).

We appreciate your feedback and bug reports to help improve ALchemist!

---

## 📄 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Repository

[https://github.com/NREL/ALchemist](https://github.com/NREL/ALchemist)


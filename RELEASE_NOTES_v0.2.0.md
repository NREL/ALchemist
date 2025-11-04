# ALchemist v0.2.0: Core Library Architecture & REST API

**Release Date**: November 4, 2025  
**NREL Software Record**: SWR-25-102

## Overview

This release represents a major architectural milestone for ALchemist, transforming it from a standalone GUI application into a modular toolkit with three access modes: GUI, programmatic Python API, and REST API for web applications.

## Major Features

### Core/UI Separation

- **New `alchemist_core` package**: Headless Bayesian optimization engine suitable for scripts, notebooks, and automation
- **Session API**: Clean, programmatic interface for building optimization workflows
- **Event system**: Decoupled progress reporting enables integration with diverse UIs
- **Zero GUI dependencies**: Core library can run in server environments without display

**Example Usage**:
```python
from alchemist_core.session import OptimizationSession

session = OptimizationSession()
session.add_variable("temperature", "real", min=20, max=100)
session.add_experiment({"temperature": 50}, output=85.2)
session.train_model(backend="botorch", kernel="rbf")
suggestions = session.suggest_next(strategy="qEI", n_suggestions=1)
```

### FastAPI REST API

Complete REST API wrapper enabling web-based access to ALchemist's optimization capabilities:

- **19 RESTful endpoints** covering the complete optimization workflow
- **Session management** with configurable time-to-live
- **CSV upload support** for batch experiment data
- **OpenAPI documentation** (Swagger UI) at `/api/docs`
- **CORS-enabled** for React and other web frontends
- **Production-ready** with comprehensive test coverage

**Base URL**: `http://localhost:8000/api/v1`

Key endpoint categories:
- Sessions: Create, retrieve, update, delete optimization sessions
- Variables: Define and manage search space parameters
- Experiments: Add data points individually, in batches, or via CSV upload
- Models: Train surrogate models and make predictions
- Acquisition: Generate next experiment suggestions

See [API Documentation](api/README.md) for complete usage guide.

## Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/NREL/ALchemist.git

# For development
git clone https://github.com/NREL/ALchemist.git
cd ALchemist
pip install -e .
```

## Usage Modes

### 1. GUI Application (No-code)
```bash
alchemist
```

### 2. Python Library (Scripts/Notebooks)
```python
from alchemist_core.session import OptimizationSession
# ... your optimization workflow
```

### 3. REST API (Web Applications)
```bash
python api/main.py
# Visit http://localhost:8000/api/docs
```

## Documentation

- **User Guide**: https://nrel.github.io/ALchemist/
- **API Documentation**: [api/README.md](api/README.md)
- **Architecture Overview**: [memory/architecture_overview.md](memory/architecture_overview.md)

## Future Roadmap

- Web frontend (React) leveraging the REST API
- Enhanced DoE methods (Box-Behnken, Sobol sequences)
- Multi-objective optimization
- Improved interactive visualizations
- Feature importance and explainability tools

See [Development Roadmap](memory/ALchemist_Development_Roadmap.md) for complete vision.

## Acknowledgments

Developed at the National Renewable Energy Laboratory (NREL) as part of the DataHub project within the ChemCatBio consortium.

## License

BSD 3-Clause License

---

**Full Changelog**: https://github.com/NREL/ALchemist/compare/v0.1.2...v0.2.0

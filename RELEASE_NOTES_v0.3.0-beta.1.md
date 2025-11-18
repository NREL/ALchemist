# Release Notes - v0.3.0-beta.1

**Release Date:** November 18, 2025

**Status:** Beta Release

---

## 🎉 Major New Features

### React Web Interface

A complete modern web interface built with React, TypeScript, and Tailwind CSS provides an alternative to the desktop application:

- **Modern UI/UX**: Clean, responsive interface with shadcn/ui components
- **Real-time Visualizations**: Interactive contour plots, parity plots, Q-Q plots, and metrics tracking using Plotly.js
- **Session Management**: Create, save, load, export, and import optimization sessions
- **Full Feature Parity**: All core optimization features available in the web interface

### Autonomous Optimization Support

New capabilities for running optimization campaigns autonomously:

- **Initial Design Generation**: Built-in Design of Experiments (DoE) methods:
  - Latin Hypercube Sampling (LHS) with multiple criteria (maximin, correlation, ratio)
  - Sobol sequences
  - Halton sequences
  - Hammersly sequences
  - Random sampling
- **Monitoring Dashboard**: Dedicated read-only dashboard with auto-refresh polling for monitoring autonomous campaigns
- **Lightweight State API**: Efficient endpoint for checking optimization progress without heavy data transfers
- **Auto-training**: Optional automatic model retraining after adding experimental data

### API Enhancements

- **Comprehensive Documentation**: Complete API endpoint reference (`api/API_ENDPOINTS.md`) covering all 25+ endpoints
- **Auto-generated OpenAPI Docs**: Interactive Swagger UI at `/api/docs` and ReDoc at `/api/redoc`
- **Enhanced Workflow Guide**: Detailed autonomous optimization examples in `api/README.md`
- **Static File Serving**: Production builds can be served directly from FastAPI without separate web server

---

## 🔧 Improvements

### Bug Fixes

- **Acquisition Function Parameters**: Fixed `xi` and `kappa` parameters not being passed to sklearn acquisition functions (#fixed)
  - UI slider changes now properly affect suggested next points
  - Parameters correctly propagated through session → acquisition → optimizer chain

### Developer Experience

- **TypeScript Support**: Added type definitions for plotly.js and react-plotly.js
- **CORS Configuration**: Support for multiple development ports (3000, 5173, 5174)
- **Build Configuration**: Proper static file path resolution for production deployments
- **Package Structure**: Clear separation between core (`alchemist-core`) and full package (`alchemist-nrel`)

---

## 📦 Installation

### Core Package (Headless/Minimal)

For programmatic use without UI dependencies:

```bash
pip install alchemist-core
```

### Full Package (Desktop + Web UI)

For complete functionality including desktop and web interfaces:

```bash
pip install alchemist-nrel
```

### Web Interface Development

To run the web interface in development mode:

```bash
cd alchemist-web
npm install
npm run dev
```

### Production Deployment

Build and serve the web interface:

```bash
cd alchemist-web
npm run build
cd ..
python run_api.py
```

Access at `http://localhost:8000`

---

## 🚀 Quick Start

### Web Interface (Recommended)

1. Start the API server:
   ```bash
   python run_api.py
   ```

2. Open browser to `http://localhost:8000`

3. Create a session and define your search space

4. Add experimental data or generate initial design

5. Train a model and get acquisition function suggestions

### Autonomous Optimization

1. Generate initial experimental design:
   ```python
   import requests
   
   # Create session
   session = requests.post("http://localhost:8000/api/v1/sessions").json()
   sid = session["session_id"]
   
   # Define variables
   requests.post(f"http://localhost:8000/api/v1/sessions/{sid}/variables", json={
       "name": "temperature", "type": "real", "min": 300, "max": 500
   })
   
   # Generate initial design
   design = requests.post(f"http://localhost:8000/api/v1/sessions/{sid}/initial-design", json={
       "method": "lhs", "n_points": 10, "lhs_criterion": "maximin"
   })
   
   print(design.json()["points"])
   ```

2. Monitor progress at `http://localhost:8000?mode=monitor`

See `api/README.md` for complete autonomous workflow examples.

---

## 🔄 Breaking Changes

**None** - This release is backward compatible with v0.2.x

---

## 📝 API Changes

### New Endpoints

- `POST /api/v1/sessions/{session_id}/initial-design` - Generate DoE points
- `GET /api/v1/sessions/{session_id}/state` - Lightweight state for monitoring

### Enhanced Endpoints

- `POST /api/v1/sessions/{session_id}/experiments` - Added `auto_train` parameter
- `POST /api/v1/sessions/{session_id}/experiments/batch` - Added `auto_train` parameter

---

## 📚 Documentation

### New Documentation

- `api/API_ENDPOINTS.md` - Complete reference for all API endpoints with examples
- `api/README.md` - Enhanced with autonomous optimization workflow guide

### Updated Documentation

- `README.md` - Added autonomous optimization features and use cases
- `memory/development_tasks.md` - Marked all autonomous optimization phases complete

---

## 🐛 Known Issues

- Installation may fail if Node.js is not available (being investigated)
  - **Workaround**: Install `alchemist-core` instead for headless use
  - Web interface is optional and not required for core functionality

---

## 🙏 Acknowledgments

Developed at the National Renewable Energy Laboratory (NREL)

---

## 📋 File Changes Summary

- **New Files**: 3
  - `alchemist-web/src/features/experiments/InitialDesignPanel.tsx`
  - `alchemist-web/src/features/monitoring/MonitoringDashboard.tsx`
  - `api/API_ENDPOINTS.md`

- **Modified Files**: 14
  - Version updates across `pyproject.toml`, `pyproject_core.toml`, `package.json`, `api/main.py`
  - API integration: types, endpoints, hooks
  - Documentation: README, API guides
  - Bug fixes: session.py, main.py

- **Total Changes**: 4,173 insertions, 152 deletions

---

## 🔮 What's Next

- Address Node.js installation dependency issue
- Add more autonomous optimization examples
- Performance optimizations for large datasets
- Additional visualization types
- Enhanced batch acquisition strategies

---

For full commit history, see: https://github.com/NREL/ALchemist/compare/v0.2.0...v0.3.0-beta.1

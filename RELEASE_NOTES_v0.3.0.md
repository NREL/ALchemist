# Release Notes - ALchemist v0.3.0

**Release Date:** November 24, 2025  
**Type:** Production Release  
**Status:** Stable

## 🎉 Highlights

ALchemist v0.3.0 is now **production-ready** and fully deployable! This release marks a significant milestone with complete packaging, Docker support, and simplified installation for end users.

## ✨ New Features

### Production-Ready Packaging
- **Pre-built Web UI**: Web interface is now bundled in the Python wheel - no Node.js required for end users!
- **Single-command Installation**: `pip install alchemist-nrel` now includes everything needed to run both desktop and web apps
- **Entry Points**: Two command-line tools after installation:
  - `alchemist` - Launch desktop GUI (CustomTkinter)
  - `alchemist-web` - Launch web application (React + FastAPI)

### Docker Support
- **Production-ready Dockerfile**: Optimized multi-stage build
- **Docker Compose configuration**: One-command deployment
- **Health checks**: Automated container health monitoring
- **Volume mounting**: Persistent data for logs and cache

### Enhanced Web UI
- **Production mode**: Optimized build with minification and code splitting
- **Static file serving**: FastAPI now serves React build artifacts efficiently
- **Development mode preserved**: Hot-reload still available for frontend developers

## 🔧 Technical Improvements

### Build System
- **Custom build hooks**: Automatic React UI compilation during `python -m build`
- **Smart static file handling**: Supports both development and production workflows
- **Clean git history**: Build artifacts excluded from repository

### API Improvements
- **Flexible CORS**: Environment variable configuration for production domains
- **Static file priority**: Checks `api/static/` first (production), falls back to `alchemist-web/dist/` (development)
- **Better logging**: Informative messages about static file serving

### Configuration
- **Version synchronization**: v0.3.0 across all components (pyproject.toml, package.json, API)
- **Simplified MANIFEST.in**: Excludes node_modules from source distributions
- **Production flags**: `--production` mode for optimized serving

## 📦 Installation & Deployment

### For End Users (No Node.js Required!)

```bash
# Install from PyPI (coming soon)
pip install alchemist-nrel

# Or install from GitHub
pip install git+https://github.com/NREL/ALchemist.git

# Run the web application
alchemist-web

# Or run the desktop application
alchemist
```

### For NREL Server Deployment

**Option 1: Docker (Recommended)**
```bash
docker pull ghcr.io/nrel/alchemist:v0.3.0
docker run -p 8000:8000 ghcr.io/nrel/alchemist:v0.3.0
```

**Option 2: Docker Compose**
```bash
cd docker
docker-compose up -d
```

**Option 3: Direct Python**
```bash
pip install alchemist-nrel
alchemist-web --production
```

### For Developers

```bash
# Clone and install in editable mode
git clone https://github.com/NREL/ALchemist.git
cd ALchemist
pip install -e .

# Frontend development with hot-reload
cd alchemist-web
npm install
npm run dev  # Terminal 1

# Backend with auto-reload  
python run_api.py  # Terminal 2
```

## 🐛 Bug Fixes

- Fixed static file serving to use correct path for pip-installed packages
- Corrected CORS configuration to include localhost:8000 for production mode
- Fixed MANIFEST.in to exclude node_modules from source distributions
- Resolved Dockerfile CMD to use entry point instead of direct uvicorn call

## 🔄 Breaking Changes

None! This release is fully backward compatible with v0.2.x.

## 📝 Migration Guide

If upgrading from v0.2.x:
1. Uninstall old version: `pip uninstall alchemist-nrel`
2. Install new version: `pip install alchemist-nrel`
3. (Optional) Update your Docker images to use v0.3.0 tag

## 🚀 What's Next

### Planned for v0.3.1
- PyPI publishing automation
- Additional acquisition strategies
- Enhanced documentation

### Future Releases
- Multi-objective optimization
- Advanced DoE methods
- PySide6 desktop UI migration

## 🙏 Acknowledgments

This work is supported by the U.S. Department of Energy's Bioenergy Technologies Office (BETO) through the ChemCatBio Consortium.

## 📄 License

BSD 3-Clause License

## 🔗 Links

- **GitHub**: https://github.com/NREL/ALchemist
- **Documentation**: https://nrel.github.io/ALchemist/
- **Issues**: https://github.com/NREL/ALchemist/issues
- **NREL Software Record**: SWR-25-102

---

For detailed installation and deployment instructions, see:
- `README.md` - General overview and quick start
- `memory/RELEASE_BUILD_GUIDE.md` - Build system details
- `memory/DOCKER.md` - Docker deployment guide
- `memory/PRODUCTION_DEPLOYMENT.md` - Production configuration

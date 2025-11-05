# Scripts Directory

This directory contains utility scripts for building and running ALchemist.

## Available Scripts

### Development Scripts

**`dev_start.bat`** / **`dev_start.sh`**
- Quick launcher for development mode
- Starts both backend (port 8000) and frontend (port 5173)
- Run from project root: `scripts\dev_start.bat` (Windows) or `./scripts/dev_start.sh` (Linux/Mac)

### Production Build Scripts

**`build_production.bat`** / **`build_production.sh`**
- Builds the React frontend for production
- Copies static files to `api/static/`
- Run from project root: `scripts\build_production.bat` (Windows) or `./scripts/build_production.sh` (Linux/Mac)

### Testing Scripts

**`run_tests_coverage.bat`**
- Runs all tests with coverage report
- Run from project root: `scripts\run_tests_coverage.bat`

## Usage Examples

```bash
# Windows - Development
scripts\dev_start.bat

# Windows - Production build
scripts\build_production.bat

# Linux/Mac - Development
./scripts/dev_start.sh

# Linux/Mac - Production build
./scripts/build_production.sh
```

All scripts should be run from the project root directory. They automatically handle path resolution.

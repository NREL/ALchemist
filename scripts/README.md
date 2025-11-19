# Scripts Directory

Utility scripts for building, testing, and running ALchemist.

## Development Scripts

### `dev_start.bat` / `dev_start.sh`
**Purpose:** Quick launcher for development mode with hot-reload

**What it does:**
- Starts backend API on port 8000
- Starts frontend dev server on port 5173
- Both run simultaneously (separate terminals on Windows, background on Linux/Mac)

**Usage:**
```bash
# Windows
scripts\dev_start.bat

# Linux/Mac
./scripts/dev_start.sh
```

**When to use:** Daily development work on React frontend or API

---

## Production Build Scripts

### `build_production.bat` / `build_production.sh`
**Purpose:** Build React frontend for production deployment

**What it does:**
1. Runs `npm install` in `alchemist-web/`
2. Runs `npm run build` to create optimized production bundle
3. Copies `alchemist-web/dist/*` to `api/static/`

**Usage:**
```bash
# Windows
scripts\build_production.bat

# Linux/Mac
./scripts/build_production.sh
```

**When to use:** 
- Testing production mode locally: `python run_api.py --production`
- Preparing Docker builds
- Manual production deployments

**Note:** For PyPI releases, this happens automatically via `build_hooks.py`

---

## Testing Scripts

### `run_tests_coverage.bat` / `run_tests_coverage.sh`
**Purpose:** Run full test suite with coverage report

**What it does:**
- Runs pytest on `tests/` directory
- Generates coverage for `alchemist_core`, `api`, and `ui`
- Creates HTML report in `htmlcov/`
- Opens report in browser (Windows/Mac)

**Usage:**
```bash
# Windows
scripts\run_tests_coverage.bat

# Linux/Mac
./scripts/run_tests_coverage.sh
```

**When to use:** Before committing major changes, checking test coverage

---

## Build & Release Scripts

### `test_build.py`
**Purpose:** Verify package build includes React UI

**What it does:**
1. Cleans old build artifacts
2. Runs `python -m build`
3. Verifies `api/static/index.html` exists in wheel
4. Reports success/failure

**Usage:**
```bash
python scripts/test_build.py
```

**When to use:** Before releasing to verify React UI is bundled correctly

---

### `bump_version.py`
**Purpose:** Update version across all files

**What it does:**
- Updates `pyproject.toml`
- Updates `alchemist-web/package.json`
- Optionally creates git tag
- Optionally pushes tag

**Usage:**
```bash
# Just update version files
python scripts/bump_version.py 0.3.2

# Update version and create tag
python scripts/bump_version.py 0.3.2 --tag

# Update, tag, and push
python scripts/bump_version.py 0.3.2 --tag --push
```

**When to use:** Starting a new release

---

## Quick Reference

| Task | Command |
|------|---------|
| **Daily development** | `./scripts/dev_start.sh` |
| **Test production locally** | `./scripts/build_production.sh` → `python run_api.py --production` |
| **Run tests** | `pytest` or `./scripts/run_tests_coverage.sh` |
| **Verify build** | `python scripts/test_build.py` |
| **Release** | `python scripts/bump_version.py X.Y.Z --tag --push` |

---

## Notes

- All scripts should be run from the **project root**
- `.bat` files are for Windows, `.sh` files are for Linux/Mac
- Python scripts (`.py`) work on all platforms
- Shell scripts are made executable with `chmod +x scripts/*.sh`


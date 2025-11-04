# Testing and Coverage

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/api/test_endpoints.py -v
```

### Run with verbose output
```bash
pytest tests/api/test_catalyst_real_data.py -v -s
```

## Coverage Reports

### Generate coverage report (terminal)
```bash
pytest tests/ --cov=alchemist_core --cov=api --cov=ui --cov-report=term-missing
```

### Generate HTML coverage report
```bash
pytest tests/ --cov=alchemist_core --cov=api --cov=ui --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see detailed coverage.

### Quick coverage (Windows)
```bash
run_tests_coverage.bat
```

This will run tests, generate an HTML report, and open it in your browser.

## Continuous Integration

The repository uses GitHub Actions to automatically run tests on:
- Every push to `main` or `develop` branches
- Every pull request to `main` or `develop` branches

Tests run on:
- Multiple OS: Ubuntu, Windows, macOS
- Multiple Python versions: 3.9, 3.10, 3.11, 3.12

Coverage reports are uploaded to Codecov for tracking over time.

## Coverage Configuration

Coverage settings are in `pyproject.toml` under `[tool.coverage.*]`:
- **Source**: `alchemist_core`, `api`, `ui`
- **Omit**: test files, cache, node_modules
- **Report**: Shows missing lines and generates HTML reports in `htmlcov/`

## Adding Coverage Badge to README

Once you set up Codecov:

1. Sign up at https://codecov.io with your GitHub account
2. Add your repository
3. Add this badge to your README.md:

```markdown
[![codecov](https://codecov.io/gh/NREL/ALchemist/branch/main/graph/badge.svg)](https://codecov.io/gh/NREL/ALchemist)
```

## Pre-commit Hooks (Optional)

To automatically run tests before commits, create `.git/hooks/pre-commit`:

```bash
#!/bin/sh
pytest tests/ --cov=alchemist_core --cov=api --cov=ui --cov-report=term-missing --cov-fail-under=80
```

This will prevent commits if coverage drops below 80%.

@echo off
REM Run tests with coverage report

echo Running tests with coverage...
pytest tests/ --cov=alchemist_core --cov=api --cov=ui --cov-report=html --cov-report=term-missing

echo.
echo Coverage report generated in htmlcov/index.html
echo Opening coverage report...
start htmlcov\index.html

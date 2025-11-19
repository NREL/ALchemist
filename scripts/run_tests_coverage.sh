#!/bin/bash
# Run tests with coverage report (Linux/Mac)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "Running tests with coverage..."
cd "$PROJECT_ROOT"
pytest tests/ --cov=alchemist_core --cov=api --cov=ui --cov-report=html --cov-report=term-missing

echo ""
echo "Coverage report generated in htmlcov/index.html"

# Try to open coverage report (works on macOS and some Linux)
if command -v open &> /dev/null; then
    echo "Opening coverage report..."
    open htmlcov/index.html
elif command -v xdg-open &> /dev/null; then
    echo "Opening coverage report..."
    xdg-open htmlcov/index.html
else
    echo "Open htmlcov/index.html in your browser to view the report"
fi

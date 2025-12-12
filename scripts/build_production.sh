#!/bin/bash
# Build script for ALchemist production deployment (Linux/Mac)
# This builds the React frontend and prepares for production
# Run from project root: ./scripts/build_production.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================"
echo "Building ALchemist for Production"
echo "========================================"

echo ""
echo "Step 1: Building React frontend..."
cd "$PROJECT_ROOT/alchemist-web" || exit 1
npm install
npm run build

if [ $? -ne 0 ]; then
    echo "ERROR: Frontend build failed!"
    exit 1
fi

echo ""
echo "Step 2: Copying static files to API directory..."
cd "$PROJECT_ROOT"
rm -rf api/static
cp -r alchemist-web/dist api/static

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy static files!"
    exit 1
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  - For local production: python -m api.run_api --production"
echo "  - For Docker: cd docker && docker-compose up --build"
echo "========================================"

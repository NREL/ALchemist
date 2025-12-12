#!/bin/bash
# Quick Development Launcher for ALchemist
# Starts both backend and frontend in development mode
# Run from project root: ./scripts/dev_start.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================"
echo "Starting ALchemist in Development Mode"
echo "========================================"
echo ""
echo "This will start:"
echo "  1. Backend API (port 8000)"
echo "  2. Frontend Dev Server (port 5173)"
echo ""
echo "Press Ctrl+C to stop both servers."
echo "========================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
echo "Starting backend..."
cd "$PROJECT_ROOT"
python -m api.run_api &
BACKEND_PID=$!

# Give backend a moment to start
sleep 3

# Start frontend in background
echo "Starting frontend..."
cd "$PROJECT_ROOT/alchemist-web"
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "Development servers running!"
echo ""
echo "Backend API:  http://localhost:8000/api/docs"
echo "Frontend:     http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "========================================"

# Wait for both processes
wait

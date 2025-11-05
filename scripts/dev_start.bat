@echo off
REM Quick Development Launcher for ALchemist
REM Starts both backend and frontend in development mode
REM Run from project root: scripts\dev_start.bat

echo ========================================
echo Starting ALchemist in Development Mode
echo ========================================
echo.
echo This will open TWO terminal windows:
echo   1. Backend API (port 8000)
echo   2. Frontend Dev Server (port 5173)
echo.
echo Press Ctrl+C in each window to stop.
echo ========================================
echo.

REM Get project root directory
set "PROJECT_ROOT=%~dp0.."

REM Start backend in a new window
start "ALchemist Backend" cmd /k "cd /d "%PROJECT_ROOT%" && python run_api.py"

REM Give backend a moment to start
timeout /t 3 /nobreak > nul

REM Start frontend in a new window
start "ALchemist Frontend" cmd /k "cd /d "%PROJECT_ROOT%\alchemist-web" && npm run dev"

echo.
echo ========================================
echo Development servers starting...
echo.
echo Backend API:  http://localhost:8000/api/docs
echo Frontend:     http://localhost:5173
echo ========================================

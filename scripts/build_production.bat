@echo off
REM Build script for ALchemist production deployment
REM This builds the React frontend and prepares for production
REM Run from project root: scripts\build_production.bat

echo ========================================
echo Building ALchemist for Production
echo ========================================

echo.
echo Step 1: Building React frontend...
cd "%~dp0..\alchemist-web"
call npm install
call npm run build

if errorlevel 1 (
    echo ERROR: Frontend build failed!
    exit /b 1
)

echo.
echo Step 2: Copying static files to API directory...
cd "%~dp0.."
if exist api\static rmdir /s /q api\static
xcopy alchemist-web\dist api\static\ /E /I /Y

if errorlevel 1 (
    echo ERROR: Failed to copy static files!
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Next steps:
echo   - For local production: python -m api.run_api --production
echo   - For Docker: cd docker ^&^& docker-compose up --build
echo ========================================

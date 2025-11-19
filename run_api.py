"""
Startup script for ALchemist FastAPI server.

Usage:
    python run_api.py              # Development mode with auto-reload
    python run_api.py --production # Production mode (no reload)
    python run_api.py --dev        # Explicitly start in development mode
    alchemist-web                  # Entry point (production mode by default)
"""

def main():
    """Entry point for alchemist-web command."""
    import uvicorn
    import sys
    
    # For the alchemist-web entry point, default to production mode
    # Only use dev mode if explicitly requested
    is_script_call = any(arg.endswith('run_api.py') for arg in sys.argv)
    
    if is_script_call:
        # Called as: python run_api.py
        # Default to dev mode unless --production flag is present
        production = "--production" in sys.argv or "--prod" in sys.argv
    else:
        # Called as: alchemist-web
        # Default to production mode unless --dev flag is present
        production = "--dev" not in sys.argv and "--development" not in sys.argv
    
    if production:
        print("Starting ALchemist API in PRODUCTION mode...")
        print("Access the web UI at: http://localhost:8000")
        # Run the API server in production mode
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="warning",
            workers=1  # Increase to 4 for multi-core production
        )
    else:
        print("Starting ALchemist API in DEVELOPMENT mode...")
        print("API docs: http://localhost:8000/api/docs")
        # Run the API server in development mode
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )


if __name__ == "__main__":
    main()

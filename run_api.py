"""
Startup script for ALchemist FastAPI server.

Usage:
    python run_api.py              # Development mode with auto-reload
    python run_api.py --production # Production mode (no reload)
"""

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Check if production mode
    production = "--production" in sys.argv or "--prod" in sys.argv
    
    if production:
        print("Starting ALchemist API in PRODUCTION mode...")
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
        # Run the API server in development mode
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )

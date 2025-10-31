"""
Startup script for ALchemist FastAPI server.

Usage:
    python run_api.py
"""

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

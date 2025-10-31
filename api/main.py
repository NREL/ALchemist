"""
ALchemist FastAPI Application

RESTful API wrapper for alchemist_core Session API.
Designed for React frontend but framework-agnostic.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import sessions, variables, experiments, models, acquisition
from .middleware.error_handlers import add_exception_handlers
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ALchemist API",
    description="REST API for Bayesian optimization and active learning",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom exception handlers
add_exception_handlers(app)

# Include routers
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(variables.router, prefix="/api/v1/sessions", tags=["Variables"])
app.include_router(experiments.router, prefix="/api/v1/sessions", tags=["Experiments"])
app.include_router(models.router, prefix="/api/v1/sessions", tags=["Models"])
app.include_router(acquisition.router, prefix="/api/v1/sessions", tags=["Acquisition"])


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ALchemist API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "alchemist-api"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# ALchemist Docker Image for FastAPI Server
# This builds a container that runs the ALchemist REST API

# Start from official Python 3.11 image (smaller than 3.12, still well-supported)
FROM python:3.11-slim

# Set metadata labels
LABEL maintainer="caleb.coatney@nrel.gov"
LABEL description="ALchemist Active Learning Toolkit - REST API Server"
LABEL version="0.2.0"

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# - gcc, g++, make: Required for compiling some Python packages (numpy, scipy, torch)
# - git: Needed if any dependencies come from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for installation first
# This helps Docker cache the dependency layer
COPY pyproject.toml README.md LICENSE ./
COPY alchemist_core/ ./alchemist_core/
COPY api/ ./api/
COPY main.py ./

# Install ALchemist and all dependencies
# Using --no-cache-dir to keep image size smaller
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy built frontend static files (if they exist)
# Run 'npm run build' in alchemist-web/ before building Docker image
COPY alchemist-web/dist ./api/static

# Create directories for logs and cache
RUN mkdir -p /app/logs /app/cache

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check - Docker will periodically check if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server when container starts
# --host 0.0.0.0 allows external connections
# --port 8000 is the standard port
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

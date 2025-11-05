# Docker Directory

This directory contains Docker-related files for containerized deployment of ALchemist.

## Files

**`Dockerfile`**
- Multi-stage Docker image definition
- Based on Python 3.11-slim
- Includes both backend and frontend static files

**`docker-compose.yml`**
- Docker Compose configuration for easy deployment
- Includes volume mounts for logs and cache
- Configurable CORS origins

**`.dockerignore`**
- Excludes unnecessary files from Docker build context
- Optimizes build speed and image size

## Usage

### Build and Run

From the project root:

```bash
# Build the frontend first (required before Docker build)
cd alchemist-web
npm run build
cd ..

# Build and start with Docker Compose
cd docker
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

### View Logs

```bash
cd docker
docker-compose logs -f
```

### Stop

```bash
cd docker
docker-compose down
```

### Configuration

Edit `docker-compose.yml` to customize:
- Port mappings
- Volume mounts
- Environment variables (especially `ALLOWED_ORIGINS` for production)
- Resource limits

## Production Deployment

For production deployment:

1. Build the frontend: `npm run build` in `alchemist-web/`
2. Update `ALLOWED_ORIGINS` in `docker-compose.yml` with your domain
3. Consider using a reverse proxy (nginx/Caddy) for SSL/HTTPS
4. Set up proper backup strategies for volume-mounted directories

## Health Check

The container includes a health check at `/health`. Docker will automatically monitor this endpoint.

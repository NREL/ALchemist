# Docker Deployment Guide for ALchemist

This guide walks you through building and running ALchemist in a Docker container.

## Prerequisites

1. **Install Docker Desktop** (if you haven't already):
   - Windows: https://docs.docker.com/desktop/install/windows-install/
   - After installation, make sure Docker Desktop is running

2. **Verify Docker is working**:
   ```bash
   docker --version
   docker-compose --version
   ```

## Quick Start (Using Docker Compose - Recommended)

Docker Compose makes it easier to manage the container with one command.

### 1. Build and Start the Container

From the ALchemist directory, run:

```bash
docker-compose up -d
```

**What this does:**
- `-d` runs in "detached" mode (background)
- Builds the image if it doesn't exist
- Starts the container
- Maps port 8000 on your computer to the container
- Mounts `logs/` and `cache/` folders for data persistence

### 2. Check if it's Running

```bash
docker-compose ps
```

You should see `alchemist-api` with status "Up".

### 3. Access the API

Open your browser and go to:
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

### 4. View Logs

```bash
docker-compose logs -f
```

Press `Ctrl+C` to stop viewing logs (container keeps running).

### 5. Stop the Container

```bash
docker-compose down
```

### 6. Rebuild After Code Changes

If you modify the code and want to rebuild:

```bash
docker-compose up -d --build
```

## Manual Docker Commands (Alternative)

If you prefer not to use Docker Compose:

### 1. Build the Image

```bash
docker build -t alchemist:0.2.0 .
```

**What this does:**
- `-t alchemist:0.2.0` tags the image with a name and version
- `.` means use the current directory (where Dockerfile is)
- Takes 5-10 minutes first time (downloads Python, installs dependencies)

### 2. Run the Container

```bash
docker run -d -p 8000:8000 --name alchemist-api alchemist:0.2.0
```

**What this does:**
- `-d` runs in background (detached mode)
- `-p 8000:8000` maps port 8000 from your computer to container
- `--name alchemist-api` gives the container a friendly name
- `alchemist:0.2.0` is the image we just built

### 3. View Running Containers

```bash
docker ps
```

### 4. View Logs

```bash
docker logs alchemist-api
```

Or follow logs in real-time:

```bash
docker logs -f alchemist-api
```

### 5. Stop the Container

```bash
docker stop alchemist-api
```

### 6. Start Again

```bash
docker start alchemist-api
```

### 7. Remove the Container

```bash
docker rm alchemist-api
```

## Troubleshooting

### Container won't start

Check logs for errors:
```bash
docker-compose logs
```

### Port 8000 already in use

Either:
- Stop whatever is using port 8000, or
- Change the port mapping in `docker-compose.yml`:
  ```yaml
  ports:
    - "8080:8000"  # Use 8080 on your computer instead
  ```

### Need to access the container

Open a shell inside the running container:
```bash
docker exec -it alchemist-api bash
```

Type `exit` to leave.

### Container is unhealthy

The healthcheck verifies the `/health` endpoint responds. If unhealthy:
```bash
docker-compose logs
```

### Rebuild from scratch

Remove everything and start fresh:
```bash
docker-compose down
docker system prune -a
docker-compose up -d --build
```

## Data Persistence

The `docker-compose.yml` file mounts local directories:
- `./logs` → Container's `/app/logs`
- `./cache` → Container's `/app/cache`

This means logs and cache survive container restarts.

## Production Deployment

For production environments:

1. **Use specific version tags**:
   ```bash
   docker build -t alchemist:0.2.0 .
   ```

2. **Set resource limits** in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

3. **Configure environment variables** in `docker-compose.yml`:
   ```yaml
   environment:
     - LOG_LEVEL=INFO
     - MAX_SESSION_TTL=48
   ```

4. **Use a reverse proxy** (nginx, Caddy) for HTTPS

5. **Consider orchestration** (Docker Swarm, Kubernetes) for scaling

## Testing the API

Once running, test with curl:

```bash
# Health check
curl http://localhost:8000/health

# Create a session
curl -X POST http://localhost:8000/api/v1/sessions

# View API docs
# Open browser: http://localhost:8000/api/docs
```

## File Size

The built image will be approximately 2-3 GB due to:
- Python base image (~150 MB)
- PyTorch (~1.5 GB)
- BoTorch, GPyTorch, and dependencies (~500 MB)
- Other Python packages (~200 MB)

## Questions?

For issues or questions about Docker deployment, see:
- ALchemist Issues: https://github.com/NREL/ALchemist/issues
- Docker Documentation: https://docs.docker.com/

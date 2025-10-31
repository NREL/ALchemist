# ALchemist FastAPI Backend

REST API wrapper for the `alchemist_core` Session API. Provides language-agnostic access to Bayesian optimization functionality.

## Quick Start

### Installation

Install FastAPI dependencies:

```bash
pip install -e .
```

Or install just the API dependencies:

```bash
pip install "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.5.0" "python-multipart>=0.0.6"
```

### Running the Server

Development mode (with auto-reload):

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Production mode:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, visit:

- **Interactive Docs (Swagger UI)**: http://localhost:8000/api/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/api/redoc
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json
- **Health Check**: http://localhost:8000/health

## API Overview

### Base URL

```
http://localhost:8000/api/v1
```

### Workflow Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Create a session
response = requests.post(f"{BASE_URL}/sessions", json={"ttl_hours": 24})
session_id = response.json()["session_id"]

# 2. Define search space
requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "temperature",
        "type": "continuous",
        "bounds": [100, 500],
        "unit": "°C"
    }
)

requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "pressure",
        "type": "continuous",
        "bounds": [1, 10],
        "unit": "bar"
    }
)

# 3. Add experimental data
requests.post(
    f"{BASE_URL}/sessions/{session_id}/experiments",
    json={
        "inputs": {"temperature": 250, "pressure": 5},
        "output": 0.85
    }
)

# Or upload CSV
with open("experiments.csv", "rb") as f:
    requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/upload",
        files={"file": f}
    )

# 4. Train surrogate model
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/model/train",
    json={
        "backend": "botorch",
        "kernel": "rbf",
        "output_transform": "standardize"
    }
)

# 5. Get next experiment suggestions
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={
        "strategy": "qEI",
        "goal": "maximize",
        "n_suggestions": 3
    }
)
suggestions = response.json()["suggestions"]

# 6. Make predictions
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/model/predict",
    json={
        "inputs": [
            {"temperature": 300, "pressure": 7},
            {"temperature": 400, "pressure": 3}
        ]
    }
)
predictions = response.json()["predictions"]
```

## Endpoints

### Sessions

- `POST /sessions` - Create new session
- `GET /sessions/{session_id}` - Get session info
- `DELETE /sessions/{session_id}` - Delete session
- `PATCH /sessions/{session_id}/ttl` - Extend session TTL

### Variables (Search Space)

- `POST /sessions/{session_id}/variables` - Add single variable
- `GET /sessions/{session_id}/variables` - List all variables
- `POST /sessions/{session_id}/variables/load` - Load from JSON file

### Experiments (Data)

- `POST /sessions/{session_id}/experiments` - Add single experiment
- `POST /sessions/{session_id}/experiments/batch` - Add multiple experiments
- `GET /sessions/{session_id}/experiments` - List all experiments
- `POST /sessions/{session_id}/experiments/upload` - Upload CSV file
- `GET /sessions/{session_id}/experiments/summary` - Get data summary

### Models

- `POST /sessions/{session_id}/model/train` - Train surrogate model
- `GET /sessions/{session_id}/model` - Get model info
- `POST /sessions/{session_id}/model/predict` - Make predictions

### Acquisition

- `POST /sessions/{session_id}/acquisition/suggest` - Suggest next experiments

## Data Formats

### Variable Definition

```json
{
  "name": "temperature",
  "type": "continuous",  // or "discrete", "categorical"
  "bounds": [100, 500],  // for continuous/discrete
  "categories": null,    // for categorical: ["low", "medium", "high"]
  "unit": "°C",
  "description": "Reaction temperature"
}
```

### Experiment Data

```json
{
  "inputs": {
    "temperature": 250,
    "pressure": 5
  },
  "output": 0.85
}
```

### CSV Format

Upload CSV files with columns for each variable + "output":

```csv
temperature,pressure,output
250,5,0.85
300,7,0.92
200,3,0.71
```

### Model Training Request

```json
{
  "backend": "botorch",           // or "sklearn"
  "kernel": "rbf",                // "matern", "periodic", etc.
  "kernel_params": null,          // optional: {"nu": 2.5}
  "input_transform": "normalize", // or "standardize", null
  "output_transform": "standardize", // or "normalize", null
  "calibration_enabled": false
}
```

### Acquisition Request

```json
{
  "strategy": "qEI",      // "EI", "PI", "UCB", "qUCB", "qNIPV"
  "goal": "maximize",     // or "minimize"
  "n_suggestions": 3,
  "xi": null,            // optional: exploration parameter for EI/PI
  "kappa": null          // optional: exploration parameter for UCB
}
```

### Prediction Request

```json
{
  "inputs": [
    {"temperature": 300, "pressure": 7},
    {"temperature": 400, "pressure": 3}
  ]
}
```

### Prediction Response

```json
{
  "predictions": [
    {
      "inputs": {"temperature": 300, "pressure": 7},
      "prediction": 0.89,
      "uncertainty": 0.05
    },
    {
      "inputs": {"temperature": 400, "pressure": 3},
      "prediction": 0.78,
      "uncertainty": 0.12
    }
  ],
  "n_predictions": 2
}
```

## Architecture

### Session Management

- **Storage**: In-memory session store with UUID-based keys
- **TTL**: Configurable time-to-live (default 24 hours)
- **Cleanup**: Automatic background cleanup of expired sessions
- **Persistence**: Sessions are not persisted - restart clears all data

### Model Training

- **Synchronous**: Training is fast (seconds) and runs synchronously
- **Storage**: Models stored in session memory
- **Backends**: sklearn (Gaussian Process) or BoTorch (state-of-the-art)
- **Transforms**: Input/output normalization for better performance

### Error Handling

All errors return consistent JSON format:

```json
{
  "detail": "Human-readable error message"
}
```

HTTP status codes:
- `400` - Invalid request data
- `404` - Session or resource not found
- `422` - Validation error
- `500` - Server error

## CORS Configuration

Configured for React development servers:
- http://localhost:3000 (Create React App)
- http://localhost:5173 (Vite)

Update `api/main.py` to add additional origins.

## Testing

Run integration tests:

```bash
pytest tests/api/
```

Test individual endpoints:

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Health check
curl http://localhost:8000/health
```

## Deployment

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t alchemist-api .
docker run -p 8000:8000 alchemist-api
```

### Production Considerations

- Use process manager (e.g., systemd, supervisord)
- Configure multiple workers for concurrent requests
- Add authentication/authorization if needed
- Consider Redis for session persistence
- Add rate limiting for public APIs
- Use HTTPS with reverse proxy (nginx, caddy)

## Development

### Project Structure

```
api/
├── main.py                    # FastAPI app entry point
├── dependencies.py            # Shared dependency injection
├── models/
│   ├── requests.py           # Pydantic request schemas
│   └── responses.py          # Pydantic response schemas
├── routers/
│   ├── sessions.py           # Session lifecycle
│   ├── variables.py          # Search space management
│   ├── experiments.py        # Data management
│   ├── models.py             # Model training & prediction
│   └── acquisition.py        # Next experiment suggestions
├── services/
│   └── session_store.py      # Session storage & management
└── middleware/
    └── error_handlers.py     # Custom exceptions & handlers
```

### Adding New Endpoints

1. Define request/response models in `api/models/`
2. Add route handler in appropriate router
3. Update this README with endpoint documentation
4. Add tests in `tests/api/`

### Code Style

- Use type hints throughout
- Add docstrings to all public functions
- Follow REST conventions (resource-based URLs)
- Return appropriate HTTP status codes
- Include examples in Pydantic schemas

## License

BSD 3-Clause (same as alchemist_core)

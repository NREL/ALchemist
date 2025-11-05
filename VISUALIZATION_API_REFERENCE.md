# Visualization API Quick Reference

## Base URL
```
/api/v1/sessions/{session_id}/visualizations
```

---

## 1. Contour Plot Data

**POST** `/contour`

### Request Body
```json
{
  "x_var": "temperature",
  "y_var": "pressure",
  "fixed_values": {
    "time": 60,
    "catalyst": "A"
  },
  "grid_resolution": 50,
  "include_experiments": true,
  "include_suggestions": false
}
```

### Response
```json
{
  "x_var": "temperature",
  "y_var": "pressure",
  "x_grid": [[...], [...], ...],  // 50x50 grid
  "y_grid": [[...], [...], ...],  // 50x50 grid
  "predictions": [[...], [...], ...],  // Mean predictions
  "uncertainties": [[...], [...], ...],  // Std deviations
  "experiments": {
    "x": [25, 50, 75],
    "y": [1.0, 1.5, 2.0],
    "output": [0.85, 0.92, 0.88]
  },
  "suggestions": null,
  "x_bounds": [0, 100],
  "y_bounds": [0.5, 2.5],
  "colorbar_bounds": [0.8, 0.95]
}
```

---

## 2. Parity Plot Data

**GET** `/parity?use_calibrated=false`

### Response
```json
{
  "y_true": [0.85, 0.92, 0.88, ...],
  "y_pred": [0.84, 0.93, 0.87, ...],
  "y_std": [0.03, 0.02, 0.04, ...],
  "metrics": {
    "rmse": 0.025,
    "mae": 0.018,
    "r2": 0.95,
    "mape": 2.3
  },
  "bounds": [0.8, 0.95],
  "calibrated": false
}
```

---

## 3. Metrics Data

**GET** `/metrics?cv_splits=5`

### Response
```json
{
  "training_sizes": [5, 6, 7, 8, ..., 20],
  "rmse": [0.15, 0.12, 0.10, ..., 0.025],
  "mae": [0.12, 0.09, 0.08, ..., 0.018],
  "r2": [0.65, 0.75, 0.82, ..., 0.95],
  "mape": [12.5, 9.8, 8.2, ..., 2.3]
}
```

---

## 4. Q-Q Plot Data

**GET** `/qq-plot?use_calibrated=false`

### Response
```json
{
  "theoretical_quantiles": [-2.5, -2.0, ..., 2.0, 2.5],
  "sample_quantiles": [-2.3, -1.9, ..., 2.1, 2.4],
  "z_mean": -0.05,
  "z_std": 1.12,
  "n_samples": 20,
  "bounds": [-3.0, 3.0],
  "calibrated": false
}
```

---

## 5. Calibration Curve Data

**GET** `/calibration-curve?use_calibrated=false`

### Response
```json
{
  "nominal_coverage": [0.10, 0.15, 0.20, ..., 0.90, 0.95],
  "empirical_coverage": [0.12, 0.18, 0.22, ..., 0.88, 0.93],
  "n_samples": 20,
  "calibrated": false
}
```

---

## 6. Hyperparameters

**GET** `/hyperparameters`

### Response
```json
{
  "hyperparameters": {
    "k1": 1.0,
    "k2__k1": 2.5,
    "k2__k2": 0.1,
    "...": "..."
  },
  "backend": "sklearn",
  "kernel": "RBF",
  "input_transform": "standard",
  "output_transform": "standard",
  "calibration_enabled": true,
  "calibration_factor": 1.15
}
```

---

## Error Responses

### 400 - Model Not Trained
```json
{
  "detail": "Model must be trained before generating visualizations"
}
```

### 400 - Variable Not Found
```json
{
  "detail": "Variable 'xyz' not found in search space"
}
```

### 400 - No CV Results
```json
{
  "detail": "Model does not have cached cross-validation results"
}
```

---

## Query Parameters

### Parity, Q-Q, Calibration Curve
- `use_calibrated` (boolean, default: false)

### Metrics
- `cv_splits` (integer, 2-10, default: 5)

---

## Notes

- **All endpoints require a trained model**
- **Contour plot** only works with continuous (real/integer) variables
- **Metrics endpoint** is computationally expensive (~5-10 seconds)
- **Other endpoints** use cached CV results (fast, < 1 second)
- **Grid resolution** for contour: higher = more detail but slower
  - 30x30 = 900 predictions
  - 50x50 = 2,500 predictions (default)
  - 100x100 = 10,000 predictions (may be slow)

---

## Testing with cURL

```bash
# Get parity data
curl http://localhost:8000/api/v1/sessions/{session_id}/visualizations/parity

# Get parity data (calibrated)
curl http://localhost:8000/api/v1/sessions/{session_id}/visualizations/parity?use_calibrated=true

# Get metrics
curl http://localhost:8000/api/v1/sessions/{session_id}/visualizations/metrics?cv_splits=5

# Get contour data
curl -X POST http://localhost:8000/api/v1/sessions/{session_id}/visualizations/contour \
  -H "Content-Type: application/json" \
  -d '{
    "x_var": "temperature",
    "y_var": "pressure",
    "grid_resolution": 30
  }'
```

# Visualization Module Documentation

**Module**: `alchemist_core.visualization`  
**Created**: January 8, 2026  
**Status**: Phase 1 Complete

---

## Overview

The visualization module provides pure plotting functions for creating matplotlib figures without any session or UI dependencies. All functions are framework-agnostic and return `(Figure, Axes)` tuples for maximum flexibility.

## Architecture

```
alchemist_core/visualization/
├── __init__.py          # Module exports
├── plots.py             # Pure plotting functions (600+ lines)
├── helpers.py           # Utility functions
└── README.md            # This file
```

### Design Principles

1. **Pure Functions**: No session or UI dependencies
2. **Framework Agnostic**: Works with notebooks, scripts, desktop UI, web
3. **Flexible Embedding**: Optional `ax` parameter for existing figures
4. **Consistent API**: Similar signatures across all plot types
5. **Type Hints**: Full typing for better IDE support

---

## API Reference

### Plotting Functions

All plotting functions follow this pattern:
- Accept numpy arrays and parameters as input
- Return `(Figure, Axes)` tuple
- Accept optional `ax` parameter for embedding
- Apply `tight_layout()` only when creating new figure

#### `create_parity_plot()`

Create parity plot of actual vs predicted values.

```python
from alchemist_core.visualization import create_parity_plot

fig, ax = create_parity_plot(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...]),        # Optional
    sigma_multiplier=1.96,         # 95% CI
    show_error_bars=True,
    show_metrics=True,
    figsize=(8, 6),
    dpi=100,
    title=None,                    # Auto-generated
    ax=None                        # Create new if None
)
```

**Features**:
- Error bars with configurable sigma multiplier
- Automatic RMSE/MAE/R² calculation
- Parity line (y=x) reference
- Confidence interval labels (68%, 95%, 99%)

#### `create_contour_plot()`

Create 2D contour plot of model predictions.

```python
from alchemist_core.visualization import create_contour_plot

X, Y = np.meshgrid(x_range, y_range)
Z = predictions.reshape(X.shape)

fig, ax = create_contour_plot(
    x_grid=X,
    y_grid=Y,
    predictions_grid=Z,
    x_var='temperature',
    y_var='pressure',
    exp_x=exp_data_x,              # Optional overlay
    exp_y=exp_data_y,
    suggest_x=next_points_x,       # Optional suggestions
    suggest_y=next_points_y,
    cmap='viridis',
    figsize=(8, 6),
    dpi=100,
    title='Contour Plot',
    ax=None
)
```

**Features**:
- 20-level contour filling
- Colorbar with label
- Experimental points overlay (white circles)
- Suggestion points overlay (red stars)
- Automatic legend when overlays present

#### `create_slice_plot()`

Create 1D slice plot with uncertainty bands.

```python
from alchemist_core.visualization import create_slice_plot

fig, ax = create_slice_plot(
    x_values=np.linspace(0, 100, 100),
    predictions=y_mean,
    x_var='temperature',
    std=y_std,                     # Optional
    sigma_bands=[1.0, 2.0],        # Multiple bands
    exp_x=exp_data_x,              # Optional experiments
    exp_y=exp_data_y,
    figsize=(8, 6),
    dpi=100,
    title=None,
    ax=None
)
```

**Features**:
- Multiple uncertainty bands with proper z-stacking
- Sequential colormap (Blues) for band opacity
- Sigmoid-based alpha scaling
- Smart legend ordering (Prediction → bands → Experiments)
- Professional styling (colorblind-friendly colors)

#### `create_metrics_plot()`

Create learning curve showing metric vs training size.

```python
from alchemist_core.visualization import create_metrics_plot

fig, ax = create_metrics_plot(
    training_sizes=np.array([5, 6, 7, ...]),
    metric_values=np.array([0.15, 0.12, 0.10, ...]),
    metric_name='rmse',            # 'rmse', 'mae', 'r2', 'mape'
    figsize=(8, 6),
    dpi=100,
    ax=None
)
```

**Features**:
- Line plot with markers
- Automatic axis labels for each metric type
- Grid for readability

#### `create_qq_plot()`

Create Q-Q plot of standardized residuals.

```python
from alchemist_core.visualization import create_qq_plot

fig, ax = create_qq_plot(
    z_scores=np.array([...]),      # Standardized residuals
    figsize=(8, 6),
    dpi=100,
    show_confidence_bands=True,    # For samples < 100
    title='Q-Q Plot',
    ax=None
)
```

**Features**:
- Comparison to standard normal distribution
- Perfect calibration reference line
- Approximate 95% CI bands for small samples
- Automatic quantile computation

#### `create_calibration_plot()`

Create calibration curve (reliability diagram).

```python
from alchemist_core.visualization import create_calibration_plot

fig, ax = create_calibration_plot(
    nominal_probs=np.array([0.68, 0.95, 0.99]),
    empirical_coverage=np.array([0.70, 0.94, 0.98]),
    figsize=(8, 6),
    dpi=100,
    title='Calibration Curve',
    ax=None
)
```

**Features**:
- Nominal vs empirical coverage
- Perfect calibration reference line
- Fixed [0, 1] axis limits

---

### Helper Functions

#### `check_matplotlib()`

Check if matplotlib is available, raise ImportError if not.

```python
from alchemist_core.visualization import check_matplotlib

check_matplotlib()  # Raises ImportError if not installed
```

#### `compute_z_scores()`

Compute standardized residuals (z-scores).

```python
from alchemist_core.visualization import compute_z_scores

z = compute_z_scores(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...])
)
# z = (y_true - y_pred) / y_std
```

**Note**: Adds small epsilon (1e-10) to avoid division by zero.

#### `compute_calibration_metrics()`

Compute nominal vs empirical coverage for calibration curves.

```python
from alchemist_core.visualization import compute_calibration_metrics

nominal, empirical = compute_calibration_metrics(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...]),
    prob_levels=None               # Default: np.arange(0.10, 1.00, 0.05)
)
```

**Returns**: `(nominal_probs, empirical_coverage)` tuple

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from alchemist_core.visualization import create_parity_plot

# Synthetic data
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

# Create plot
fig, ax = create_parity_plot(y_true, y_pred)

# Save or display
fig.savefig('parity.png', dpi=300, bbox_inches='tight')
# OR in notebook: fig  (auto-displays)
```

### Embedding in Existing Figure

```python
import matplotlib.pyplot as plt
from alchemist_core.visualization import create_parity_plot, create_slice_plot

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot on existing axes
create_parity_plot(y_true, y_pred, ax=ax1)
create_slice_plot(x_values, predictions, 'temperature', ax=ax2)

fig.tight_layout()
plt.show()
```

### Complete Workflow

```python
from alchemist_core.visualization import (
    create_parity_plot,
    create_contour_plot,
    compute_z_scores,
    create_qq_plot
)

# 1. Get CV results (from model)
y_true = cv_results['y_true']
y_pred = cv_results['y_pred']
y_std = cv_results['y_std']

# 2. Create parity plot
fig1, _ = create_parity_plot(y_true, y_pred, y_std)
fig1.savefig('parity.png')

# 3. Check calibration with Q-Q plot
z = compute_z_scores(y_true, y_pred, y_std)
fig2, _ = create_qq_plot(z)
fig2.savefig('qq.png')

# 4. Create contour plot
X, Y = np.meshgrid(x_range, y_range)
Z = model.predict(grid_points).reshape(X.shape)
fig3, _ = create_contour_plot(X, Y, Z, 'x', 'y')
fig3.savefig('contour.png')
```

---

## Integration Points

### Session API Integration

The Session API delegates to these functions:

```python
# alchemist_core/session.py

def plot_parity(self, ...):
    from alchemist_core.visualization.plots import create_parity_plot
    
    # Get data from model
    cv_results = self._check_cv_results(use_calibrated)
    
    # Delegate to pure function
    fig, ax = create_parity_plot(
        y_true=cv_results['y_true'],
        y_pred=cv_results['y_pred'],
        y_std=cv_results['y_std'],
        ...
    )
    return fig
```

### UI Integration

The Desktop UI can call Session API or use functions directly:

```python
# ui/visualizations.py

def plot_parity(self):
    # Option 1: Call Session API (recommended)
    fig = self.session.plot_parity(...)
    
    # Option 2: Call pure function directly (if needed)
    from alchemist_core.visualization.plots import create_parity_plot
    fig, ax = create_parity_plot(...)
    
    # Embed in canvas
    self._embed_figure(fig)
```

### REST API Integration

The REST API can return plot data or rendered images:

```python
# api/routers/visualizations.py

@router.get("/parity")
async def get_parity_plot(session_id: str):
    session = get_session(session_id)
    
    # Get figure from session
    fig = session.plot_parity()
    
    # Option 1: Return as PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return StreamingResponse(buf, media_type='image/png')
    
    # Option 2: Return data for client-side rendering
    # Extract data from figure and return JSON
```

---

## Testing

Unit tests cover all plotting functions:

```bash
pytest tests/unit/visualization/test_plots.py -v
```

**Test Coverage**:
- ✅ All 6 plotting functions
- ✅ Helper functions
- ✅ Error handling
- ✅ Edge cases (empty arrays, single points)
- ✅ Embedding in existing axes
- ✅ Custom styling options

**Results**: 21/21 tests passing

---

## Future Enhancements

### Potential Additions

1. **Feature Importance Plot**: Bar chart of feature importance
2. **Pareto Front Plot**: For multi-objective optimization
3. **Acquisition Function Plot**: Show acquisition landscape
4. **Residual Plot**: Residuals vs predicted values
5. **Correlation Matrix**: Heatmap of variable correlations

### Styling Improvements

1. **StyleConfig Class**: Centralized styling configuration
2. **Theme Support**: Light/dark themes
3. **Publication Mode**: IEEE/Nature/Science style presets
4. **Interactive Plots**: Plotly backend support

### Performance

1. **Lazy Imports**: Defer matplotlib import until needed
2. **Caching**: Cache rendered figures for repeated requests
3. **Vectorization**: Optimize meshgrid operations

---

## Migration Guide

For developers migrating from old code:

### Before (UI code)
```python
# ui/visualizations.py - 100 lines of plotting code
def plot_parity(self):
    # Manual CV computation
    # Direct matplotlib calls
    # Embedded in self.ax
```

### After (with module)
```python
# ui/visualizations.py - 10 lines
def plot_parity(self):
    fig = self.session.plot_parity(...)
    self._embed_figure(fig)
```

**Benefits**:
- 90% code reduction
- Consistent behavior
- Easier to maintain
- Testable in isolation

---

## Dependencies

**Required**:
- `numpy`
- `matplotlib`
- `scipy` (for stats functions)
- `scikit-learn` (for metrics in parity plot)

**Optional**:
- None (all dependencies already in ALchemist)

---

## Performance Characteristics

| Function | Typical Time | Notes |
|----------|-------------|-------|
| `create_parity_plot` | ~50ms | Fast, O(n) scatter |
| `create_contour_plot` | ~200ms | Depends on grid resolution |
| `create_slice_plot` | ~100ms | Multiple fill_between calls |
| `create_metrics_plot` | ~30ms | Simple line plot |
| `create_qq_plot` | ~40ms | Sorting dominates |
| `create_calibration_plot` | ~30ms | Simple line plot |

*Benchmarked on M1 Mac with typical data sizes*

---

## Version History

- **v0.1.0** (Jan 8, 2026): Initial implementation
  - 6 plotting functions
  - 3 helper functions
  - 21 unit tests
  - Complete documentation

---

## Contact

Questions or issues? See main ALchemist documentation or contact:
- Email: ccoatney@nrel.gov
- GitHub: https://github.com/NREL/ALchemist

import numpy as np

def get_model_data(model, backend, metrics=None):
    """
    Get model data in a consistent format for display.
    
    Args:
        model: The trained model
        backend: The backend name (scikit-learn, botorch, etc.)
        metrics: Optional dictionary of metrics
        
    Returns:
        Dictionary with model information
    """
    model_data = {
        'backend': backend,
        'kernel': str(model.kernel if hasattr(model, 'kernel') else "Unknown"),
        'hyperparameters': model.get_hyperparameters() if hasattr(model, 'get_hyperparameters') else {},
        'metrics': metrics or {}
    }
    
    return model_data

def format_value(value):
    """Format a value for display, handling different data types."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    elif isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, separator=', ', suppress_small=True)
    elif isinstance(value, list):
        return str([format_value(v) for v in value])
    else:
        return str(value)
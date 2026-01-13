"""
Helper functions for visualization module.

Utilities for data preparation, validation, and computation.
"""

import numpy as np
from typing import Optional, Tuple


def check_matplotlib() -> None:
    """
    Check if matplotlib is available for plotting.
    
    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def compute_z_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray
) -> np.ndarray:
    """
    Compute standardized residuals (z-scores).
    
    z = (y_true - y_pred) / y_std
    
    Args:
        y_true: Actual experimental values
        y_pred: Model predicted values
        y_std: Prediction standard deviations
    
    Returns:
        Array of z-scores (standardized residuals)
    
    Note:
        Small epsilon (1e-10) added to denominator to avoid division by zero.
    """
    return (y_true - y_pred) / (y_std + 1e-10)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    prob_levels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nominal vs empirical coverage for calibration curves.
    
    For each nominal probability level, computes the empirical fraction of
    observations that fall within the predicted confidence interval.
    
    Args:
        y_true: Actual experimental values
        y_pred: Model predicted values
        y_std: Prediction standard deviations
        prob_levels: Nominal coverage probabilities to evaluate.
                    Default: np.arange(0.10, 1.00, 0.05)
    
    Returns:
        Tuple of (nominal_probs, empirical_coverage)
        - nominal_probs: The requested probability levels
        - empirical_coverage: Observed coverage fractions
    
    Example:
        >>> nominal, empirical = compute_calibration_metrics(y_true, y_pred, y_std)
        >>> # nominal[i] is the expected coverage (e.g., 0.68 for ±1σ)
        >>> # empirical[i] is the observed coverage fraction
    """
    from scipy import stats
    
    if prob_levels is None:
        prob_levels = np.arange(0.10, 1.00, 0.05)
    
    empirical_coverage = []
    
    for prob in prob_levels:
        # Convert probability to sigma multiplier
        # For symmetric interval: P(|Z| < z) = prob → z = Φ^(-1)((1+prob)/2)
        sigma = stats.norm.ppf((1 + prob) / 2)
        
        # Compute empirical coverage at this sigma level
        lower_bound = y_pred - sigma * y_std
        upper_bound = y_pred + sigma * y_std
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        empirical_coverage.append(np.mean(within_interval))
    
    return prob_levels, np.array(empirical_coverage)


def sort_legend_items(labels: list) -> list:
    """
    Sort legend labels for consistent ordering.
    
    Preferred order: Prediction, uncertainty bands (small to large), Experiments
    
    Args:
        labels: List of legend label strings
    
    Returns:
        List of indices for sorted order
    """
    def sort_key(lbl):
        if 'Prediction' in lbl:
            return (0, 0)
        elif 'σ' in lbl:
            # Extract sigma value for sorting bands
            import re
            match = re.search(r'±([\d.]+)σ', lbl)
            if match:
                return (1, float(match.group(1)))
            return (1, 999)
        elif 'Experiment' in lbl:
            return (2, 0)
        else:
            return (3, 0)
    
    indices = list(range(len(labels)))
    indices.sort(key=lambda i: sort_key(labels[i]))
    return indices

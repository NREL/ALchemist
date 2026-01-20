"""
Utility functions for acquisition function evaluation.

These are internal helper functions used by visualization methods.
Users should use the high-level plotting APIs in OptimizationSession instead.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any


def evaluate_acquisition(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    acq_func: str = 'ucb',
    acq_func_kwargs: Optional[Dict[str, Any]] = None,
    goal: str = 'maximize'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate acquisition function at given points using the model's backend.
    
    This is an internal utility function used by visualization methods.
    Users should call plot_acquisition_slice() or plot_acquisition_contour() instead.
    
    Args:
        model: Trained model instance (SklearnModel or BoTorchModel)
        X: Points to evaluate (DataFrame or array with shape (n, d))
        acq_func: Acquisition function name:
                 - 'ei': Expected Improvement
                 - 'pi': Probability of Improvement
                 - 'ucb/lcb': Upper/Lower Confidence Bound
                 - 'logei', 'logpi': Log variants (BoTorch only)
        acq_func_kwargs: Additional parameters:
                       - 'xi' (float): Exploration parameter for EI/PI (default: 0.01)
                       - 'kappa' (float): Exploration parameter for UCB (default: 1.96)
                       - 'beta' (float): Exploration parameter for UCB (BoTorch, default: 0.5)
        goal: 'maximize' or 'minimize' - optimization direction
        
    Returns:
        Tuple of (acq_values, None) - None because acquisition functions are deterministic
        
    Example:
        >>> from alchemist_core.utils.acquisition_utils import evaluate_acquisition
        >>> acq_vals, _ = evaluate_acquisition(
        ...     session.model, points, acq_func='ei', goal='maximize'
        ... )
    
    Note:
        - Requires trained model
        - Acquisition values are relative - only their ordering matters
        - Higher values indicate better candidates for next experiment
    """
    if model is None:
        raise ValueError("Model must be trained before evaluating acquisition functions")
    
    maximize = (goal.lower() == 'maximize')
    
    # Delegate to model's evaluate_acquisition method
    return model.evaluate_acquisition(X, acq_func, acq_func_kwargs, maximize)

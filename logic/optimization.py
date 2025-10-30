from logic.acquisition.skopt_acquisition import SkoptAcquisition
from alchemist_core.config import get_logger
import pandas as pd
import numpy as np
import warnings

logger = get_logger(__name__)

def select_optimize(search_space, experiments=None, base_estimator='GP', acq_func='gp_hedge', verbose=True, random_state=42):
    """
    Legacy wrapper for backwards compatibility. 
    Uses the new SkoptAcquisition class internally.
    """
    # Create acquisition function
    acquisition = SkoptAcquisition(
        search_space=search_space,
        model=base_estimator if base_estimator != 'GP' else None,
        acq_func=acq_func,
        random_state=random_state
    )
    
    # Update with experiments if provided
    if experiments is not None:
        if isinstance(experiments, pd.DataFrame):
            X = experiments.drop(columns='Output')
            Y = experiments['Output']
            acquisition.update(X, Y)
        elif isinstance(experiments, tuple) and len(experiments) == 2:
            X, Y = experiments
            acquisition.update(X, Y)
        else:
            warnings.warn("Invalid input: experiments must be a pandas DataFrame or a tuple (X, y).")
            return None
    
    # Ask for next point
    next_x = acquisition.select_next()
    
    if verbose:
        logger.info(f'Suggested Next Experiment:\n{next_x}')
    
    return next_x
"""
Design of Experiments (DoE) - Initial sampling strategies for Bayesian optimization.

This module provides methods for generating initial experimental designs before
starting the optimization loop. Supported methods:
- Random sampling
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Halton sequences
- Hammersly sequences
"""

from typing import List, Dict, Optional, Literal, Any
import numpy as np
from skopt.sampler import Lhs, Sobol, Hammersly
from skopt.space import Real, Integer, Categorical
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.config import get_logger

logger = get_logger(__name__)


def generate_initial_design(
    search_space: SearchSpace,
    method: Literal["random", "lhs", "sobol", "halton", "hammersly"] = "lhs",
    n_points: int = 10,
    random_seed: Optional[int] = None,
    lhs_criterion: str = "maximin"
) -> List[Dict[str, Any]]:
    """
    Generate initial experimental design using specified sampling strategy.
    
    This function creates a set of experimental conditions to evaluate before
    starting Bayesian optimization. Different methods provide different
    space-filling properties:
    
    - **random**: Uniform random sampling
    - **lhs**: Latin Hypercube Sampling (recommended for most cases)
    - **sobol**: Sobol quasi-random sequences (low discrepancy)
    - **halton**: Halton sequences (via Hammersly sampler)
    - **hammersly**: Hammersly sequences (low discrepancy)
    
    Args:
        search_space: SearchSpace object with defined variables
        method: Sampling method to use
        n_points: Number of points to generate
        random_seed: Random seed for reproducibility (applies to random and lhs)
        lhs_criterion: Criterion for LHS optimization ("maximin", "correlation", "ratio")
    
    Returns:
        List of dictionaries, each containing variable names and values.
        Does NOT include 'Output' column - experiments need to be evaluated.
    
    Raises:
        ValueError: If search_space has no variables or method is unknown
    
    Example:
        >>> from alchemist_core import SearchSpace
        >>> from alchemist_core.utils.doe import generate_initial_design
        >>> 
        >>> # Define search space
        >>> space = SearchSpace()
        >>> space.add_variable('temperature', 'real', min=300, max=500)
        >>> space.add_variable('pressure', 'real', min=1, max=10)
        >>> 
        >>> # Generate 10 LHS points
        >>> points = generate_initial_design(space, method='lhs', n_points=10)
        >>> 
        >>> # Points ready for experimentation
        >>> for point in points:
        >>>     print(point)  # {'temperature': 350.2, 'pressure': 4.7}
    """
    # Validate inputs
    if len(search_space.variables) == 0:
        raise ValueError("SearchSpace has no variables. Define variables before generating initial design.")
    
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"Set random seed to {random_seed} for reproducibility")
    
    # Get skopt dimensions from SearchSpace
    skopt_space = search_space.skopt_dimensions
    
    # Generate samples based on method
    if method == "random":
        samples = _random_sampling(skopt_space, n_points)
    
    elif method == "lhs":
        samples = _lhs_sampling(skopt_space, n_points, lhs_criterion)
    
    elif method == "sobol":
        samples = _sobol_sampling(skopt_space, n_points)
    
    elif method in ["halton", "hammersly"]:
        samples = _hammersly_sampling(skopt_space, n_points)
    
    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Choose from: random, lhs, sobol, halton, hammersly"
        )
    
    # Convert samples to list of dicts
    variable_names = [v['name'] for v in search_space.variables]
    points = []
    
    for sample in samples:
        point = {name: value for name, value in zip(variable_names, sample)}
        points.append(point)
    
    logger.info(
        f"Generated {len(points)} initial points using {method} method "
        f"for {len(variable_names)} variables"
    )
    
    return points


def _random_sampling(skopt_space, n_points: int) -> list:
    """
    Generate random samples respecting variable types.
    
    Handles Real, Integer, and Categorical dimensions appropriately.
    Returns list of lists to preserve mixed types.
    """
    samples_list = []
    
    for dim in skopt_space:
        if isinstance(dim, Categorical):
            # Random choice from categories
            samples = np.random.choice(dim.categories, size=n_points)
        
        elif isinstance(dim, Integer):
            # Random integers in [low, high] (inclusive)
            # np.random.randint is [low, high), so add 1 to include upper bound
            samples = np.random.randint(dim.low, dim.high + 1, size=n_points)
        
        elif isinstance(dim, Real):
            # Random floats in [low, high]
            samples = np.random.uniform(dim.low, dim.high, size=n_points)
        
        else:
            raise ValueError(f"Unknown dimension type: {type(dim)}")
        
        samples_list.append(samples)
    
    # Transpose to get list of samples (each sample is a list of values)
    # Don't use column_stack as it converts everything to same dtype
    samples = [[samples_list[j][i] for j in range(len(samples_list))] 
               for i in range(n_points)]
    return samples


def _lhs_sampling(skopt_space, n_points: int, criterion: str = "maximin") -> list:
    """
    Generate Latin Hypercube Sampling points.
    
    LHS provides good space-filling properties and is generally recommended
    for initial designs in Bayesian optimization.
    
    Args:
        criterion: Optimization criterion
            - "maximin": maximize minimum distance between points (default)
            - "correlation": minimize correlations between dimensions
            - "ratio": minimize ratio of max to min distance
    """
    sampler = Lhs(lhs_type="classic", criterion=criterion)
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples


def _sobol_sampling(skopt_space, n_points: int) -> list:
    """
    Generate Sobol quasi-random sequence points.
    
    Sobol sequences have low discrepancy properties, meaning they cover
    the space more uniformly than random sampling.
    """
    sampler = Sobol()
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples


def _hammersly_sampling(skopt_space, n_points: int) -> list:
    """
    Generate Hammersly sequence points.
    
    Hammersly and Halton sequences are low-discrepancy sequences similar
    to Sobol, providing good space coverage.
    """
    sampler = Hammersly()
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples

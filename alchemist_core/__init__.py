"""
ALchemist Core - Headless Bayesian Optimization Library

This package provides the core functionality for active learning and Bayesian
optimization workflows without UI dependencies.

Main Components:
    - OptimizationSession: High-level API for optimization workflows
    - SearchSpace: Define variable search spaces
    - ExperimentManager: Manage experimental data
    - Models: Surrogate modeling backends (sklearn, BoTorch)
    - Acquisition: Acquisition function strategies

Example:
    >>> from alchemist_core import OptimizationSession, SearchSpace, ExperimentManager
    >>> 
    >>> # Create session
    >>> session = OptimizationSession(
    ...     search_space=SearchSpace.from_json("variables.json"),
    ...     experiment_manager=ExperimentManager.from_csv("experiments.csv")
    ... )
    >>> 
    >>> # Train model
    >>> session.fit_model(backend="sklearn")
    >>> 
    >>> # Get next experiment suggestion
    >>> next_point = session.suggest_next(acq_func="ei")

Version: 0.2.0-dev (Core-UI Split)
"""

__version__ = "0.2.0-dev"
__author__ = "Caleb Coatney"
__email__ = "caleb.coatney@nrel.gov"

# Core data structures
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager

# Event system
from alchemist_core.events import EventEmitter

# Configuration and logging
from alchemist_core.config import configure_logging, get_logger, set_verbosity

# High-level session API
from alchemist_core.session import OptimizationSession

# Public API
__all__ = [
    # High-level API (recommended entry point)
    "OptimizationSession",
    
    # Data structures
    "SearchSpace",
    "ExperimentManager",
    
    # Events and logging
    "EventEmitter",
    "configure_logging",
    "get_logger",
    "set_verbosity",
]

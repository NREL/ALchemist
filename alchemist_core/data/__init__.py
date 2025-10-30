"""
Data management module for ALchemist Core.

Provides classes for managing search spaces and experimental data.
"""

from .search_space import SearchSpace
from .experiment_manager import ExperimentManager

__all__ = ["SearchSpace", "ExperimentManager"]

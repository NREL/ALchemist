from abc import ABC, abstractmethod
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union

class BaseModel(ABC):
    def __init__(self, random_state: int = 42, input_transform_type: str = "none", 
                 output_transform_type: str = "none"):
        """Initialize the model with a random state for reproducibility.
        
        Args:
            random_state: Random seed for reproducibility
            input_transform_type: Type of input scaling ("none", "standard", "minmax", "robust")
            output_transform_type: Type of output scaling ("none", "standard")
        """
        self.random_state = random_state
        self.input_transform_type = input_transform_type
        self.output_transform_type = output_transform_type
        self._is_trained = False
        
    @abstractmethod
    def train(self, experiment_manager: ExperimentManager, **kwargs):
        """
        Train the model on the experiment data.
        
        Args:
            experiment_manager: ExperimentManager containing the experiment data
            **kwargs: Additional backend-specific training parameters
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_std: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return predictions for input X.
        
        Args:
            X: Input features as DataFrame or numpy array
            return_std: Whether to return standard deviations
            **kwargs: Additional backend-specific prediction parameters
            
        Returns:
            If return_std is False: numpy array of predictions
            If return_std is True: tuple of (predictions, standard deviations)
        """
        pass

    @abstractmethod
    def evaluate(self, experiment_manager: ExperimentManager, cv_splits: int = 5, debug: bool = False, 
                progress_callback: Optional[callable] = None, **kwargs) -> Dict[str, List[float]]:
        """
        Evaluate the model (e.g., using cross-validation) and return performance metrics.
        
        Args:
            experiment_manager: ExperimentManager containing the experiment data
            cv_splits: Number of cross-validation splits
            debug: Whether to run in debug mode (detailed output)
            progress_callback: Optional callback function for progress updates
            **kwargs: Additional backend-specific evaluation parameters
            
        Returns:
            Dictionary of metric names to lists of values
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Return the learned hyperparameters of the model.
        
        Returns:
            Dictionary of hyperparameter names to values
        """
        pass
        
    @property
    def is_trained(self) -> bool:
        """Whether the model is trained."""
        return self._is_trained

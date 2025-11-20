from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import os
import json

class ExperimentManager:
    """
    Class for storing and managing experimental data in a consistent way across backends.
    Provides methods for data access, saving/loading, and conversion to formats needed by different backends.
    """
    def __init__(self, search_space=None):
        self.df = pd.DataFrame()  # Raw experimental data
        self.search_space = search_space  # Reference to the search space
        self.filepath = None  # Path to saved experiment file
        self._current_iteration = 0  # Track current iteration for audit log
        
    def set_search_space(self, search_space):
        """Set or update the search space reference."""
        self.search_space = search_space
        
    def add_experiment(self, point_dict: Dict[str, Union[float, str, int]], output_value: Optional[float] = None, 
                       noise_value: Optional[float] = None, iteration: Optional[int] = None, 
                       reason: Optional[str] = None):
        """
        Add a single experiment point.
        
        Args:
            point_dict: Dictionary with variable names as keys and values
            output_value: The experiment output/target value (if known)
            noise_value: Optional observation noise/uncertainty value for regularization
            iteration: Iteration number (auto-assigned if None)
            reason: Reason for this experiment (e.g., 'Initial Design (LHS)', 'Expected Improvement')
        """
        # Create a copy of the point_dict to avoid modifying the original
        new_point = point_dict.copy()
        
        # Add output value if provided
        if output_value is not None:
            new_point['Output'] = output_value
            
        # Add noise value if provided
        if noise_value is not None:
            new_point['Noise'] = noise_value
        
        # Add iteration tracking
        if iteration is not None:
            # Use provided iteration and ensure _current_iteration reflects it
            new_point['Iteration'] = int(iteration)
            # Keep _current_iteration in sync with the latest explicit iteration
            try:
                self._current_iteration = int(iteration)
            except Exception:
                pass
        else:
            # Use current iteration (doesn't increment until lock_acquisition)
            new_point['Iteration'] = int(self._current_iteration)
        
        # Add reason
        new_point['Reason'] = reason if reason is not None else 'Manual'
            
        # Convert to DataFrame and append
        new_df = pd.DataFrame([new_point])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
    def add_experiments_batch(self, data_df: pd.DataFrame):
        """Add multiple experiment points at once from a DataFrame."""
        # Ensure all required columns are present
        if self.search_space:
            required_cols = self.search_space.get_variable_names()
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        
        # Ensure each row has an Iteration value; default to current iteration
        if 'Iteration' not in data_df.columns:
            data_df = data_df.copy()
            data_df['Iteration'] = int(self._current_iteration)
        else:
            # Fill missing iterations with current iteration
            data_df = data_df.copy()
            data_df['Iteration'] = pd.to_numeric(data_df['Iteration'], errors='coerce').fillna(self._current_iteration).astype(int)
            # Update _current_iteration to the max iteration present
            if len(data_df) > 0:
                max_iter = int(data_df['Iteration'].max())
                if max_iter > self._current_iteration:
                    self._current_iteration = max_iter

        # Append the data
        self.df = pd.concat([self.df, data_df], ignore_index=True)
    
    def get_data(self) -> pd.DataFrame:
        """Get the raw experiment data."""
        return self.df.copy()
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and target (y) separated.
        
        Returns:
            X: Features DataFrame
            y: Target Series
        """
        if 'Output' not in self.df.columns:
            raise ValueError("DataFrame doesn't contain 'Output' column")
        
        # Drop metadata columns (Output, Noise, Iteration, Reason)
        metadata_cols = ['Output']
        if 'Noise' in self.df.columns:
            metadata_cols.append('Noise')
        if 'Iteration' in self.df.columns:
            metadata_cols.append('Iteration')
        if 'Reason' in self.df.columns:
            metadata_cols.append('Reason')
        
        X = self.df.drop(columns=metadata_cols)
        y = self.df['Output']
        return X, y
    
    def get_features_target_and_noise(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Get features (X), target (y), and noise values if available.
        
        Returns:
            X: Features DataFrame
            y: Target Series
            noise: Noise Series if available, otherwise None
        """
        if 'Output' not in self.df.columns:
            raise ValueError("DataFrame doesn't contain 'Output' column")
        
        # Drop metadata columns
        metadata_cols = ['Output']
        if 'Noise' in self.df.columns:
            metadata_cols.append('Noise')
        if 'Iteration' in self.df.columns:
            metadata_cols.append('Iteration')
        if 'Reason' in self.df.columns:
            metadata_cols.append('Reason')
        
        X = self.df.drop(columns=metadata_cols)
        y = self.df['Output']
        noise = self.df['Noise'] if 'Noise' in self.df.columns else None
        return X, y, noise
    
    def has_noise_data(self) -> bool:
        """Check if the experiment data includes noise values."""
        return 'Noise' in self.df.columns
    
    def save_to_csv(self, filepath: Optional[str] = None):
        """
        Save experiments to a CSV file.
        
        Args:
            filepath: Path to save the file. If None, uses the previously used path.
        """
        if filepath:
            self.filepath = filepath
        
        if not self.filepath:
            raise ValueError("No filepath specified and no previous filepath available")
            
        self.df.to_csv(self.filepath, index=False)
        
    def load_from_csv(self, filepath: str):
        """
        Load experiments from a CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        self.df = pd.read_csv(filepath)
        self.filepath = filepath
        
        # Ensure noise values are numeric if present
        if 'Noise' in self.df.columns:
            try:
                self.df['Noise'] = pd.to_numeric(self.df['Noise'])
                print(f"Loaded experiment data with noise column. Noise values will be used for model regularization.")
            except ValueError:
                print("Warning: Noise column contains non-numeric values. Converting to default noise level.")
                self.df['Noise'] = 1e-10  # Default small noise
        
        # Initialize iteration tracking from data
        if 'Iteration' in self.df.columns:
            self._current_iteration = int(self.df['Iteration'].max())
        else:
            # Add iteration column if missing (legacy data)
            self.df['Iteration'] = 0
            self._current_iteration = 0
        
        # Add reason column if missing (legacy data)
        if 'Reason' not in self.df.columns:
            self.df['Reason'] = 'Initial Design'
        
        return self
    
    @classmethod
    def from_csv(cls, filepath: str, search_space=None):
        """Class method to create an ExperimentManager from a CSV file."""
        instance = cls(search_space=search_space)
        return instance.load_from_csv(filepath)
    
    def clear(self):
        """Clear all experimental data."""
        self.df = pd.DataFrame()
    
    def get_full_history(self) -> pd.DataFrame:
        """Get the full experiment history."""
        return self.df.copy()
    
    def get_latest_experiment(self) -> pd.Series:
        """Get the most recently added experiment."""
        if len(self.df) == 0:
            return None
        return self.df.iloc[-1].copy()
    
    def __len__(self):
        return len(self.df)

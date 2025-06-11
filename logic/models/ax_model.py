from ax.service.ax_client import AxClient
from .base_model import BaseModel
import pandas as pd
import numpy as np
from skopt.space import Real, Integer, Categorical

class AxModel(BaseModel):
    def __init__(self, search_space, experiment_name="experiment", random_state=42):
        """
        Initialize the AxModel.

        Args:
            search_space: A list of skopt.space objects (Real, Integer, or Categorical).
            experiment_name: A name for the Ax experiment.
            random_state: Random seed for reproducibility.
        """
        self.experiment_name = experiment_name
        self.search_space = search_space
        self.random_state = random_state
        self.ax_client = AxClient(random_seed=random_state)
        self.trained = False

    def _build_parameters(self):
        """
        Build the Ax parameters list from the search_space.
        """
        parameters = []
        for dim in self.search_space:
            if isinstance(dim, Real):
                # For Real dimensions, use a continuous range.
                parameters.append({
                    "name": dim.name,
                    "type": "range",
                    "bounds": list(dim.bounds),
                    "value_type": "float",
                })
            elif isinstance(dim, Integer):
                # For Integer dimensions, use a range and specify value type as int.
                parameters.append({
                    "name": dim.name,
                    "type": "range",
                    "bounds": list(dim.bounds),
                    "value_type": "int",
                })
            elif isinstance(dim, Categorical):
                # For categorical dimensions, use "choice" and list the categories.
                # Here we assume that the categories are strings; if numeric, adjust "value_type" accordingly.
                parameters.append({
                    "name": dim.name,
                    "type": "choice",
                    "values": list(dim.categories),
                    "value_type": "str",
                })
            else:
                raise ValueError(f"Unsupported search space dimension type: {type(dim)}")
        return parameters

    def train(self, exp_df, **kwargs):
        """Train the Ax model using the raw experiment DataFrame."""
        X = exp_df.drop(columns="Output")
        y = exp_df["Output"]
        parameters = self._build_parameters()
        self.ax_client.create_experiment(
            name=self.experiment_name,
            parameters=parameters,
        )
        for i, row in X.iterrows():
            params = row.to_dict()
            outcome = float(y.iloc[i])
            self.ax_client.complete_trial(trial_index=i, raw_data={"objective": outcome})
        self.trained = True

    def predict(self, X, return_std=False, **kwargs):
        """
        For Ax, prediction means asking for the next candidate.
        
        Args:
            X: Not used (the next candidate is computed based on the experiment history).
            return_std: Not applicable; always returns just the candidate.
        
        Returns:
            A dictionary with parameter names and suggested values.
        """
        if not self.trained:
            raise ValueError("The Ax experiment has not been trained with past data yet.")
        parameters, trial_index = self.ax_client.get_next_trial()
        return parameters

    def predict_with_std(self, X):
        """
        Make predictions with standard deviation.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, 'feature_names') and self.feature_names:
                X = pd.DataFrame(X, columns=self.feature_names)
            else:
                raise ValueError("Cannot convert input to DataFrame - feature names unknown")
        
        # Prepare the observations in Ax format
        obs = []
        for _, row in X.iterrows():
            arm_parameters = row.to_dict()
            obs.append(arm_parameters)
        
        # Get the predictions
        means, covariances = self.surrogate.predict(obs)
        
        # Extract standard deviations from covariances
        stds = np.sqrt(np.diag(covariances))
        
        return means, stds

    def evaluate(self, X, y, **kwargs):
        """
        Evaluate the Ax model's performance using stored outcomes.
        In a more complete implementation, you could compute metrics such as RMSE across trials.
        
        Returns:
            A dictionary with evaluation metrics (here empty as a placeholder).
        """
        # Example: Extract and compute statistics from the experiment.
        return {}

    def get_hyperparameters(self):
        """
        Get model hyperparameters.
        
        Returns:
            A dictionary with hyperparameter names and values.
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
            
        try:
            params = {}
            # For Ax models, we can extract some basic info
            if hasattr(self, 'surrogate') and hasattr(self.surrogate, 'model'):
                model_type = type(self.surrogate.model).__name__
                params['model_type'] = model_type
                
                # Try to get some GPEI-specific attributes if available
                if hasattr(self.surrogate.model, 'model'):
                    inner_model = self.surrogate.model.model
                    if hasattr(inner_model, 'covar_module'):
                        params['covar_module'] = str(inner_model.covar_module)
                        
            return params
        except Exception as e:
            return {"error": str(e)}

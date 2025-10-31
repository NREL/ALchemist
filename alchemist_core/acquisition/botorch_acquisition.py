from typing import Dict, List, Optional, Union, Tuple, Any
from alchemist_core.config import get_logger
import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qUpperConfidenceBound,
)
# Import active learning module separately - fix for the import error
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from .base_acquisition import BaseAcquisition

logger = get_logger(__name__)
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.models.botorch_model import BoTorchModel

class BoTorchAcquisition(BaseAcquisition):
    """
    Acquisition function implementation using BoTorch.
    
    Supported acquisition functions:
    - 'ei': Expected Improvement
    - 'logei': Log Expected Improvement (numerically stable)
    - 'pi': Probability of Improvement
    - 'logpi': Log Probability of Improvement (numerically stable)
    - 'ucb': Upper Confidence Bound
    - 'qei': Batch Expected Improvement (for q>1)
    - 'qucb': Batch Upper Confidence Bound (for q>1)
    - 'qipv' or 'qnipv': q-Negative Integrated Posterior Variance (exploratory)
    """
    
    # Valid acquisition function names
    VALID_ACQ_FUNCS = {
        'ei', 'logei', 'pi', 'logpi', 'ucb', 
        'qei', 'qucb', 'qipv', 'qnipv',
        'expectedimprovement', 'probabilityofimprovement', 'upperconfidencebound'
    }
    
    def __init__(
        self, 
        search_space, 
        model=None, 
        acq_func='ucb', 
        maximize=True, 
        random_state=42, 
        acq_func_kwargs=None,
        batch_size=1
    ):
        """
        Initialize the BoTorch acquisition function.
        
        Args:
            search_space: The search space (SearchSpace object)
            model: A trained model (BoTorchModel)
            acq_func: Acquisition function type (see class docstring for options)
            maximize: Whether to maximize (True) or minimize (False) the objective
            random_state: Random state for reproducibility
            acq_func_kwargs: Dictionary of additional arguments for the acquisition function
            batch_size: Number of points to select at once (q)
            
        Raises:
            ValueError: If acq_func is not a valid acquisition function name
        """
        # Validate acquisition function before proceeding
        acq_func_lower = acq_func.lower()
        if acq_func_lower not in self.VALID_ACQ_FUNCS:
            valid_funcs = "', '".join(sorted(['ei', 'logei', 'pi', 'logpi', 'ucb', 'qei', 'qucb', 'qipv']))
            raise ValueError(
                f"Invalid acquisition function '{acq_func}' for BoTorch backend. "
                f"Valid options are: '{valid_funcs}'"
            )
        
        self.search_space_obj = search_space
        self.maximize = maximize
        self.random_state = random_state
        self.acq_func_name = acq_func_lower
        self.batch_size = batch_size
        
        # Process acquisition function kwargs
        self.acq_func_kwargs = acq_func_kwargs or {}
        
        # Set default values if not provided
        if self.acq_func_name == 'ucb' and 'beta' not in self.acq_func_kwargs:
            self.acq_func_kwargs['beta'] = 0.5  # Default UCB exploration parameter
        
        if self.acq_func_name == 'qucb' and 'beta' not in self.acq_func_kwargs:
            self.acq_func_kwargs['beta'] = 0.5  # Default qUCB exploration parameter
            
        if self.acq_func_name in ['qei', 'qucb', 'qipv'] and 'mc_samples' not in self.acq_func_kwargs:
            self.acq_func_kwargs['mc_samples'] = 128  # Default MC samples for batch methods
            
        # Create the acquisition function if model is provided
        self.acq_function = None
        self.model = None
        if model is not None and isinstance(model, BoTorchModel):
            self.update_model(model)
    
    def update_model(self, model):
        """Update the underlying model."""
        if not isinstance(model, BoTorchModel):
            raise ValueError("Model must be a BoTorchModel instance")
        
        self.model = model
        
        # Create the acquisition function based on the specified type
        self._create_acquisition_function()
    
    def _create_acquisition_function(self):
        """Create the appropriate BoTorch acquisition function."""
        if self.model is None or not hasattr(self.model, 'model') or not self.model.is_trained:
            return
        
        # Set torch seed for reproducibility
        torch.manual_seed(self.random_state)
        
        # Get best observed value from the model
        # Important: Use original scale values, not transformed values, because
        # the acquisition function optimization works in original space
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'train_targets'):
            # Check if we have access to original scale targets
            if hasattr(self.model, 'Y_orig') and self.model.Y_orig is not None:
                # Use original scale targets for best_f calculation
                train_Y_orig = self.model.Y_orig.cpu().numpy() if torch.is_tensor(self.model.Y_orig) else self.model.Y_orig
                best_f = float(np.max(train_Y_orig) if self.maximize else np.min(train_Y_orig))
            else:
                # Fallback: use train_targets (may be in transformed space)
                train_Y = self.model.model.train_targets.cpu().numpy()
                best_f = float(np.max(train_Y) if self.maximize else np.min(train_Y))
            best_f = torch.tensor(best_f, dtype=torch.double)
        else:
            best_f = torch.tensor(0.0, dtype=torch.double)
        
        # Create the appropriate acquisition function based on type
        if self.acq_func_name == 'ei':
            # Standard Expected Improvement
            self.acq_function = ExpectedImprovement(
                model=self.model.model,
                best_f=best_f,
                maximize=self.maximize
            )
        elif self.acq_func_name == 'logei':
            # Log Expected Improvement (numerically more stable)
            self.acq_function = LogExpectedImprovement(
                model=self.model.model,
                best_f=best_f,
                maximize=self.maximize
            )
        elif self.acq_func_name == 'pi':
            # Probability of Improvement
            self.acq_function = ProbabilityOfImprovement(
                model=self.model.model,
                best_f=best_f,
                maximize=self.maximize
            )
        elif self.acq_func_name == 'logpi':
            # Log Probability of Improvement
            self.acq_function = LogProbabilityOfImprovement(
                model=self.model.model,
                best_f=best_f,
                maximize=self.maximize
            )
        elif self.acq_func_name == 'ucb':
            # Upper Confidence Bound
            beta = self.acq_func_kwargs.get('beta', 0.5)
            self.acq_function = UpperConfidenceBound(
                model=self.model.model,
                beta=beta,
                maximize=self.maximize
            )
        elif self.acq_func_name == 'qei':
            # Batch Expected Improvement
            mc_samples = self.acq_func_kwargs.get('mc_samples', 128)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=self.random_state)
            
            # Remove the maximize parameter - qEI always maximizes
            # For minimization, we should negate the objectives when training the model
            self.acq_function = qExpectedImprovement(
                model=self.model.model,
                best_f=best_f,
                sampler=sampler
            )
        elif self.acq_func_name == 'qucb':
            # Batch Upper Confidence Bound
            beta = self.acq_func_kwargs.get('beta', 0.5)
            mc_samples = self.acq_func_kwargs.get('mc_samples', 128)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=self.random_state)
            self.acq_function = qUpperConfidenceBound(
                model=self.model.model,
                beta=beta,
                sampler=sampler,
                # Remove maximize parameter here too
            )
        elif self.acq_func_name in ['qipv', 'qnipv']:
            # q-Negative Integrated Posterior Variance (exploratory)
            # Generate MC points for integration over the search space  
            bounds_tensor = self._get_bounds_from_search_space()
            n_mc_points = self.acq_func_kwargs.get('n_mc_points', 500)  # Reduced default
            
            # Generate MC points from uniform distribution within bounds
            lower_bounds, upper_bounds = bounds_tensor[0], bounds_tensor[1]
            mc_points = torch.rand(n_mc_points, len(lower_bounds), dtype=torch.double)
            mc_points = mc_points * (upper_bounds - lower_bounds) + lower_bounds
            
            # Create Integrated Posterior Variance acquisition function
            self.acq_function = qNegIntegratedPosteriorVariance(
                model=self.model.model,
                mc_points=mc_points,
            )
        else:
            # This should never happen due to validation in __init__, but just in case
            raise ValueError(f"Unsupported acquisition function: {self.acq_func_name}")
    
    def select_next(self, candidate_points=None):
        """
        Suggest the next experiment point(s) using BoTorch optimization.
        
        Args:
            candidate_points: Candidate points to evaluate (optional)
            
        Returns:
            Dictionary with the selected point or list of points
        """
        # Ensure we have an acquisition function
        if self.acq_function is None:
            self._create_acquisition_function()
            if self.acq_function is None:
                raise ValueError("Could not create acquisition function - model not properly set")
        
        # Get bounds from the search space
        bounds_tensor = self._get_bounds_from_search_space()
        
        # Identify categorical and integer variables
        categorical_variables = []
        integer_variables = []
        if hasattr(self.search_space_obj, 'get_categorical_variables'):
            categorical_variables = self.search_space_obj.get_categorical_variables()
        if hasattr(self.search_space_obj, 'get_integer_variables'):
            integer_variables = self.search_space_obj.get_integer_variables()
        
        # Set torch seed for reproducibility
        torch.manual_seed(self.random_state)
        
        # If no candidates provided, optimize the acquisition function
        if candidate_points is None:
            # Check if we need batch optimization or single-point optimization
            q = self.batch_size
            
            # For batch acquisition functions, we need a different optimization approach
            is_batch_acq = self.acq_func_name.startswith('q')
            
            # Adjust optimization parameters for qIPV to improve stability and performance
            if self.acq_func_name == 'qipv':
                num_restarts = 20  # More restarts for qIPV
                raw_samples = 100  # Fewer samples per restart
                max_iter = 150     # Fewer iterations
                batch_limit = 5    # Standard batch limit
                options = {
                    "batch_limit": batch_limit,
                    "maxiter": max_iter,
                    "ftol": 1e-3,  # More relaxed convergence criteria
                }
            else:
                # Standard parameters for other acquisition functions
                num_restarts = 20 if is_batch_acq else 10
                raw_samples = 500 if is_batch_acq else 200
                max_iter = 300
                batch_limit = 5
                options = {"batch_limit": batch_limit, "maxiter": max_iter}
            
            # Check if we have categorical variables
            if categorical_variables and len(categorical_variables) > 0:
                # Get categorical dimensions and their possible values
                fixed_features_list = []
                
                # Map variable names to indices
                var_to_idx = {name: i for i, name in enumerate(self.model.feature_names)}
                
                # Identify which dimensions are categorical
                for var_name in categorical_variables:
                    if var_name in var_to_idx:
                        cat_idx = var_to_idx[var_name]
                        
                        # Get possible values for this categorical variable
                        if var_name in self.model.categorical_encodings:
                            cat_values = list(self.model.categorical_encodings[var_name].values())
                            
                            # Create a fixed_features entry for each possible value
                            for val in cat_values:
                                fixed_features = {cat_idx: val}
                                fixed_features_list.append(fixed_features)
                
                try:
                    # Use mixed optimization for categorical variables
                    batch_candidates, batch_acq_values = optimize_acqf_mixed(
                        acq_function=self.acq_function,
                        bounds=bounds_tensor,
                        q=q,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        fixed_features_list=fixed_features_list,
                        options=options,
                    )
                    
                    # Get the best candidate(s)
                    best_candidates = batch_candidates.detach().cpu()
                    
                    # Apply integer constraints if needed
                    if integer_variables:
                        var_to_idx = {name: i for i, name in enumerate(self.model.feature_names)}
                        for var_name in integer_variables:
                            if var_name in var_to_idx:
                                idx = var_to_idx[var_name]
                                best_candidates[:, idx] = torch.round(best_candidates[:, idx])
                    
                    best_candidates = best_candidates.numpy()
                except Exception as e:
                    logger.error(f"Error in optimize_acqf_mixed: {e}")
                    # Fallback to standard optimization
                    batch_candidates, batch_acq_values = optimize_acqf(
                        acq_function=self.acq_function,
                        bounds=bounds_tensor,
                        q=q,
                        num_restarts=num_restarts // 2,  # Reduce for fallback
                        raw_samples=raw_samples // 2,    # Reduce for fallback
                        options=options,
                    )
                    best_candidates = batch_candidates.detach().cpu()
                    
                    # Apply integer constraints if needed
                    if integer_variables:
                        var_to_idx = {name: i for i, name in enumerate(self.model.feature_names)}
                        for var_name in integer_variables:
                            if var_name in var_to_idx:
                                idx = var_to_idx[var_name]
                                best_candidates[:, idx] = torch.round(best_candidates[:, idx])
                    
                    best_candidates = best_candidates.numpy()
            else:
                # For purely continuous variables
                batch_candidates, batch_acq_values = optimize_acqf(
                    acq_function=self.acq_function,
                    bounds=bounds_tensor,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options=options,
                )
                
                best_candidates = batch_candidates.detach().cpu()
                
                # Apply integer constraints if needed
                if integer_variables:
                    var_to_idx = {name: i for i, name in enumerate(self.model.feature_names)}
                    for var_name in integer_variables:
                        if var_name in var_to_idx:
                            idx = var_to_idx[var_name]
                            best_candidates[:, idx] = torch.round(best_candidates[:, idx])
                
                best_candidates = best_candidates.numpy()
        else:
            # If candidates are provided, evaluate them directly
            if isinstance(candidate_points, np.ndarray):
                candidate_tensor = torch.tensor(candidate_points, dtype=torch.double)
            elif isinstance(candidate_points, pd.DataFrame):
                # Encode categorical variables
                candidates_encoded = self.model._encode_categorical_data(candidate_points)
                candidate_tensor = torch.tensor(candidates_encoded.values, dtype=torch.double)
            else:
                candidate_tensor = candidate_points  # Assume it's already a tensor
            
            # Evaluate acquisition function at candidate points
            with torch.no_grad():
                acq_values = self.acq_function(candidate_tensor.unsqueeze(1))
            
            # Find the best candidate
            best_idx = torch.argmax(acq_values)
            best_candidate = candidate_points.iloc[best_idx] if isinstance(candidate_points, pd.DataFrame) else candidate_points[best_idx]
            
            return best_candidate
        
        # If we're returning batch results (q > 1)
        if self.batch_size > 1:
            result_points = []
            
            # Convert each point in the batch to a dictionary with feature names
            for i in range(best_candidates.shape[0]):
                point_dict = {}
                for j, name in enumerate(self.model.feature_names):
                    value = best_candidates[i, j]
                    
                    # If this is a categorical variable, convert back to original value
                    if name in categorical_variables:
                        # Find the original categorical value from the encoding
                        encoding = self.model.categorical_encodings.get(name, {})
                        inv_encoding = {v: k for k, v in encoding.items()}
                        if value in inv_encoding:
                            value = inv_encoding[value]
                        elif int(value) in inv_encoding:
                            value = inv_encoding[int(value)]
                    # If this is an integer variable, ensure it's an integer
                    elif name in integer_variables:
                        value = int(round(float(value)))
                    
                    point_dict[name] = value
                
                result_points.append(point_dict)
            
            return result_points
        
        # For single-point results (q = 1)
        result = {}
        for i, name in enumerate(self.model.feature_names):
            value = best_candidates[0, i]
            
            # If this is a categorical variable, convert back to original value
            if name in categorical_variables:
                # Find the original categorical value from the encoding
                encoding = self.model.categorical_encodings.get(name, {})
                inv_encoding = {v: k for k, v in encoding.items()}
                if value in inv_encoding:
                    value = inv_encoding[value]
                elif int(value) in inv_encoding:
                    value = inv_encoding[int(value)]
            # If this is an integer variable, ensure it's an integer
            elif name in integer_variables:
                value = int(round(float(value)))
            
            result[name] = value
            
        return result
    
    def _get_bounds_from_search_space(self):
        """Extract bounds from the search space."""
        # First try to use the to_botorch_bounds method if available
        if hasattr(self.search_space_obj, 'to_botorch_bounds'):
            bounds_tensor = self.search_space_obj.to_botorch_bounds()
            if isinstance(bounds_tensor, torch.Tensor) and bounds_tensor.dim() == 2 and bounds_tensor.shape[0] == 2:
                return bounds_tensor
        
        # Get feature names from model to ensure proper order
        if not hasattr(self.model, 'feature_names'):
            raise ValueError("Model doesn't have feature_names attribute")
        
        feature_names = self.model.feature_names
        
        # Get categorical variables
        categorical_variables = []
        if hasattr(self.search_space_obj, 'get_categorical_variables'):
            categorical_variables = self.search_space_obj.get_categorical_variables()
        
        # Extract bounds for each feature
        lower_bounds = []
        upper_bounds = []
        
        if hasattr(self.search_space_obj, 'variables'):
            # Create a map for quick lookup
            var_dict = {var['name']: var for var in self.search_space_obj.variables}
            
            for name in feature_names:
                if name in var_dict:
                    var = var_dict[name]
                    if var.get('type') == 'categorical':
                        # For categorical variables, use the appropriate encoding range
                        if hasattr(self.model, 'categorical_encodings') and name in self.model.categorical_encodings:
                            encodings = self.model.categorical_encodings[name]
                            lower_bounds.append(0.0)
                            upper_bounds.append(float(max(encodings.values())))
                        else:
                            # Default fallback for categorical variables
                            lower_bounds.append(0.0)
                            upper_bounds.append(1.0)
                    elif 'min' in var and 'max' in var:
                        lower_bounds.append(float(var['min']))
                        upper_bounds.append(float(var['max']))
                    elif 'bounds' in var:
                        lower_bounds.append(float(var['bounds'][0]))
                        upper_bounds.append(float(var['bounds'][1]))
                else:
                    # Default fallback if variable not found
                    lower_bounds.append(0.0)
                    upper_bounds.append(1.0)
        
        # Validate bounds
        if not lower_bounds or not upper_bounds:
            raise ValueError("Could not extract bounds from search space")
        
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError(f"Inconsistent bounds: got {len(lower_bounds)} lower bounds and {len(upper_bounds)} upper bounds")
        
        if len(lower_bounds) != len(feature_names):
            raise ValueError(f"Dimension mismatch: got {len(lower_bounds)} bounds but model expects {len(feature_names)} features")
        
        return torch.tensor([lower_bounds, upper_bounds], dtype=torch.double)
    
    def update(self, X=None, y=None):
        """
        Update the acquisition function with new observations.
        
        Args:
            X: Features of new observations
            y: Target values of new observations
        """
        # For BoTorch, we typically don't need to explicitly update the acquisition
        # since we create a new one each time with update_model().
        # 
        # However, we need to implement this method to satisfy the BaseAcquisition interface.
        
        if X is not None and y is not None and hasattr(self.model, 'update'):
            # If the model has an update method, use it
            self.model.update(X, y)
            
            # Recreate the acquisition function with the updated model
            self._create_acquisition_function()
        
        return self

    def find_optimum(self, model=None, maximize=None, random_state=None):
        """Find the point where the model predicts the optimal value."""
        if model is not None:
            self.model = model
            
        if maximize is not None:
            self.maximize = maximize
            
        if random_state is not None:
            self.random_state = random_state
    
        # Get bounds from the search space
        bounds_tensor = self._get_bounds_from_search_space()
        
        # Identify categorical and integer variables
        categorical_variables = []
        integer_variables = []
        if hasattr(self.search_space_obj, 'get_categorical_variables'):
            categorical_variables = self.search_space_obj.get_categorical_variables()
        if hasattr(self.search_space_obj, 'get_integer_variables'):
            integer_variables = self.search_space_obj.get_integer_variables()
    
        # Prepare for optimization
        torch.manual_seed(self.random_state)
        
        try:
            # Use a simpler randomized search approach instead of optimize_acqf
            # This avoids the dimension issues in the more complex optimization
            n_samples = 20000  # Large number of random samples 
            best_value = float('-inf') if self.maximize else float('inf')
            best_x = None
            
            # Generate random samples within bounds
            lower_bounds, upper_bounds = bounds_tensor[0], bounds_tensor[1]
            X_samples = torch.rand(n_samples, len(lower_bounds), dtype=torch.double)
            X_samples = X_samples * (upper_bounds - lower_bounds) + lower_bounds
            
            # Round integer variables to nearest integer
            if integer_variables:
                for i, feature_name in enumerate(self.model.feature_names):
                    if feature_name in integer_variables:
                        X_samples[:, i] = torch.round(X_samples[:, i])
            
            # Evaluate model at all samples
            self.model.model.eval()
            with torch.no_grad():
                posterior = self.model.model.posterior(X_samples)
                values = posterior.mean.squeeze()
                
                # If minimizing, negate values for finding maximum
                if not self.maximize:
                    values = -values
                    
                # Find the best value
                best_idx = torch.argmax(values)
                best_x = X_samples[best_idx]
                best_value = values[best_idx].item()
                
            # Convert to numpy
            best_candidate = best_x.cpu().numpy().reshape(1, -1)
        except Exception as e:
            logger.error(f"Error in random search optimization: {e}")
            # Fallback to grid search
            logger.info("Falling back to grid search...")
            
            # Create a simple grid search
            n_points = 10  # Points per dimension
            grid_points = []
            
            # Create grid for each dimension
            for i, feature_name in enumerate(self.model.feature_names):
                if feature_name in integer_variables:
                    # For integer variables, create integer grid
                    min_val = int(lower_bounds[i])
                    max_val = int(upper_bounds[i])
                    if max_val - min_val + 1 <= n_points:
                        # If range is small, use all integer values
                        grid_points.append(torch.arange(min_val, max_val + 1, dtype=torch.double))
                    else:
                        # If range is large, sample n_points integers
                        step = max(1, (max_val - min_val) // (n_points - 1))
                        values = torch.arange(min_val, max_val + 1, step, dtype=torch.double)
                        grid_points.append(values[:n_points])
                else:
                    # For continuous variables, use linspace
                    grid_points.append(torch.linspace(
                        lower_bounds[i], upper_bounds[i], n_points, dtype=torch.double
                    ))
            
            # Create meshgrid
            meshgrid = torch.meshgrid(*grid_points, indexing='ij')
            X_grid = torch.stack([x.reshape(-1) for x in meshgrid], dim=1)
            
            # Evaluate model on grid
            self.model.model.eval()
            with torch.no_grad():
                posterior = self.model.model.posterior(X_grid)
                values = posterior.mean.squeeze()
                
                # If minimizing, negate values
                if not self.maximize:
                    values = -values
                    
                # Find the best value
                best_idx = torch.argmax(values)
                best_x = X_grid[best_idx]
                best_value = values[best_idx].item()
                
            # Convert to numpy
            best_candidate = best_x.cpu().numpy().reshape(1, -1)
            
        # Convert to dictionary and then to DataFrame
        result = {}
        for i, name in enumerate(self.model.feature_names):
            value = best_candidate[0, i]
            
            # If this is a categorical variable, convert back to original value
            if name in categorical_variables:
                # Find the original categorical value from the encoding
                encoding = self.model.categorical_encodings.get(name, {})
                inv_encoding = {v: k for k, v in encoding.items()}
                if value in inv_encoding:
                    value = inv_encoding[value]
                elif int(value) in inv_encoding:
                    value = inv_encoding[int(value)]
            # If this is an integer variable, ensure it's an integer
            elif name in integer_variables:
                value = int(round(value))
        
            result[name] = value
            
        # Convert to DataFrame
        opt_point_df = pd.DataFrame([result])
        
        # Get predicted value and std at optimum
        pred_mean, pred_std = self.model.predict_with_std(opt_point_df)
        
        return {
            'x_opt': opt_point_df,
            'value': float(pred_mean[0]),
            'std': float(pred_std[0])
        }
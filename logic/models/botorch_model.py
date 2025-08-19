import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from .base_model import BaseModel
import warnings
from botorch.models.utils.assorted import InputDataWarning

# Import necessary kernels from GPyTorch
from gpytorch.kernels import MaternKernel, RBFKernel

class BoTorchModel(BaseModel):
    def __init__(self, training_iter=50, random_state=42,
             kernel_options: dict = None, cat_dims: list[int] | None = None, 
             search_space: list = None, input_transform_type: str = "none", 
             output_transform_type: str = "none"):
        """
        Initialize the BoTorchModel with custom options.
        
        Args:
            training_iter: Maximum iterations for model optimization.
            random_state: Random seed for reproducibility.
            kernel_options: Dictionary with kernel options like "cont_kernel_type" and "matern_nu".
            cat_dims: List of column indices that are categorical.
            search_space: Optional search space list.
            input_transform_type: Type of input scaling ("none", "normalize", "standardize")
            output_transform_type: Type of output scaling ("none", "standardize")
        """
        # Suppress BoTorch input scaling warnings since we're implementing transforms explicitly
        warnings.filterwarnings("ignore", category=InputDataWarning)
        
        super().__init__(random_state=random_state, 
                        input_transform_type=input_transform_type,
                        output_transform_type=output_transform_type)
        
        self.training_iter = training_iter
        self.kernel_options = kernel_options or {"cont_kernel_type": "Matern", "matern_nu": 2.5}
        self.cont_kernel_type = self.kernel_options.get("cont_kernel_type", "Matern")
        self.matern_nu = self.kernel_options.get("matern_nu", 2.5)
        self.cat_dims = cat_dims
        self.search_space = search_space
        self.model = None
        self.feature_names = None
        self.categorical_encodings = {}  # Mappings for categorical features
        self.fitted_state_dict = None    # Store the trained model's state
        self.cv_cached_results = None  # Will store y_true and y_pred from cross-validation
        self._is_trained = False  # Initialize training status
    
    def _get_cont_kernel_factory(self):
        """Returns a factory function for the continuous kernel."""
        def factory(batch_shape, ard_num_dims, active_dims):
            if self.cont_kernel_type.lower() == "matern":
                return MaternKernel(
                    nu=self.matern_nu, 
                    ard_num_dims=ard_num_dims, 
                    active_dims=active_dims,
                    batch_shape=batch_shape
                )
            else:  # Default to RBF
                return RBFKernel(
                    ard_num_dims=ard_num_dims, 
                    active_dims=active_dims,
                    batch_shape=batch_shape
                )
        return factory
    
    def _encode_categorical_data(self, X):
        """Encode categorical variables using simple numeric mapping."""
        if not isinstance(X, pd.DataFrame):
            return X
            
        X_encoded = X.copy()
        self.feature_names = list(X.columns)
        
        # Only process columns identified in cat_dims
        if self.cat_dims:
            for idx in self.cat_dims:
                if idx < len(X.columns):
                    col_name = X.columns[idx]
                    # Create mapping if not already created
                    if col_name not in self.categorical_encodings:
                        unique_values = X[col_name].unique()
                        self.categorical_encodings[col_name] = {
                            value: i for i, value in enumerate(unique_values)
                        }
                    # Apply mapping
                    X_encoded[col_name] = X_encoded[col_name].map(
                        self.categorical_encodings[col_name]
                    ).astype(float)
        
        # Ensure all columns are numeric
        for col in X_encoded.columns:
            if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                try:
                    X_encoded[col] = pd.to_numeric(X_encoded[col])
                except:
                    # If conversion fails, treat as categorical
                    if col not in self.categorical_encodings:
                        unique_values = X_encoded[col].unique()
                        self.categorical_encodings[col] = {
                            value: i for i, value in enumerate(unique_values)
                        }
                    X_encoded[col] = X_encoded[col].map(
                        self.categorical_encodings[col]
                    ).astype(float)
        
        return X_encoded
    
    def _create_transforms(self, train_X, train_Y):
        """Create input and output transforms based on transform types."""
        input_transform = None
        outcome_transform = None
        
        # Create input transform
        if self.input_transform_type == "normalize":
            input_transform = Normalize(d=train_X.shape[-1])
        elif self.input_transform_type == "standardize":
            # Note: BoTorch doesn't have a direct Standardize input transform
            # Normalize is equivalent to standardization for inputs
            input_transform = Normalize(d=train_X.shape[-1])
        
        # Create output transform
        if self.output_transform_type == "standardize":
            outcome_transform = Standardize(m=train_Y.shape[-1])
            
        return input_transform, outcome_transform
    
    def train(self, exp_manager, **kwargs):
        """Train the model using an ExperimentManager instance."""
        # Get data with noise values if available
        X, y, noise = exp_manager.get_features_target_and_noise()
        
        if len(X) < 3:
            raise ValueError("Not enough data points to train a Gaussian Process model")
        
        # Store the original feature names before encoding
        self.original_feature_names = X.columns.tolist()
        print(f"Training with {len(self.original_feature_names)} original features: {self.original_feature_names}")
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_data(X)
        
        # Convert to tensors
        train_X = torch.tensor(X_encoded.values, dtype=torch.double)
        train_Y = torch.tensor(y.values, dtype=torch.double).unsqueeze(-1)
        
        # Convert noise values to tensor if available
        if noise is not None:
            train_Yvar = torch.tensor(noise.values, dtype=torch.double).unsqueeze(-1)
            print(f"Using provided noise values for BoTorch model regularization.")
        else:
            train_Yvar = None
        
        # Create transforms
        input_transform, outcome_transform = self._create_transforms(train_X, train_Y)
        
        # Print transform information
        if input_transform is not None:
            print(f"Applied {self.input_transform_type} transform to inputs")
        else:
            print("No input transform applied")
            
        if outcome_transform is not None:
            print(f"Applied {self.output_transform_type} transform to outputs")
        else:
            print("No output transform applied")
        
        # Set random seed
        torch.manual_seed(self.random_state)
        
        # Create and train model
        cont_kernel_factory = self._get_cont_kernel_factory()
        
        # Create model with appropriate parameters based on available data
        if self.cat_dims and len(self.cat_dims) > 0:
            # For models with categorical variables
            if noise is not None:
                self.model = MixedSingleTaskGP(
                    train_X=train_X, 
                    train_Y=train_Y, 
                    train_Yvar=train_Yvar,
                    cat_dims=self.cat_dims,
                    cont_kernel_factory=cont_kernel_factory,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
            else:
                # Don't pass train_Yvar at all when no noise data exists
                self.model = MixedSingleTaskGP(
                    train_X=train_X, 
                    train_Y=train_Y,
                    cat_dims=self.cat_dims,
                    cont_kernel_factory=cont_kernel_factory,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
        else:
            # For continuous-only models
            if noise is not None:
                self.model = SingleTaskGP(
                    train_X=train_X, 
                    train_Y=train_Y, 
                    train_Yvar=train_Yvar,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
            else:
                # Don't pass train_Yvar at all when no noise data exists
                self.model = SingleTaskGP(
                    train_X=train_X, 
                    train_Y=train_Y,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
        
        # Train the model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options={"maxiter": self.training_iter})
        
        # Store the trained state for later use
        self.fitted_state_dict = self.model.state_dict()
        self._is_trained = True  # Mark model as trained
        
        # Store original scale targets for acquisition function calculations
        # This is needed when output transforms are used
        self.Y_orig = train_Y.clone()

        # After model is trained, cache CV results
        if kwargs.get('cache_cv', True):
            # Cache CV results
            self._cache_cross_validation_results(X, y)
        
        return self
    
    def predict(self, X, return_std=False, **kwargs):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_data(X)
        
        # Convert to tensor - handle both DataFrame and numpy array inputs
        if isinstance(X_encoded, pd.DataFrame):
            test_X = torch.tensor(X_encoded.values, dtype=torch.double)
        else:
            # If X_encoded is already a numpy array
            test_X = torch.tensor(X_encoded, dtype=torch.double)
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.likelihood.eval()
        
        # Make predictions
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            
            if return_std:
                std = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1).cpu().numpy()
                return mean, std
                
            return mean

    def predict_with_std(self, X):
        """
        Make predictions with standard deviation.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Process inputs the same way as in predict
        X_encoded = self._encode_categorical_data(X)
        
        # Convert to tensor
        if isinstance(X_encoded, pd.DataFrame):
            X_tensor = torch.tensor(X_encoded.values, dtype=torch.double)
        else:
            # If X_encoded is already a numpy array
            X_tensor = torch.tensor(X_encoded, dtype=torch.double)
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.likelihood.eval()
        
        # Get posterior
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            # Get standard deviation from variance
            variance = posterior.variance.squeeze(-1).cpu().numpy()
            std = np.sqrt(variance)
        
        return mean, std

    @property
    def kernel(self):
        """
        Return a representation of the kernel for visualization purposes.
        This is a compatibility method to make the model work with the visualization system.
        """
        if self.model is None:
            return None
        
        # Create a dict-like object that mimics the necessary parts of a sklearn kernel
        class KernelInfo:
            def __init__(self, model, kernel_type, cat_dims, lengthscales=None):
                self.model = model
                self.kernel_type = kernel_type
                self.cat_dims = cat_dims
                self.lengthscales = lengthscales
                
            def __repr__(self):
                if isinstance(self.model, MixedSingleTaskGP):
                    kernel_str = f"MixedKernel(continuous={self.kernel_type}, categorical=True)"
                else:
                    kernel_str = f"{self.kernel_type}Kernel()"
                return kernel_str
                
            def get_params(self, deep=True):
                """
                Return parameters of this kernel, mimicking scikit-learn's get_params.
                
                Args:
                    deep: If True, will return nested parameters (ignored here but included for compatibility)
                    
                Returns:
                    Dictionary of parameter names mapped to their values
                """
                params = {}
                
                # Base kernel parameters
                if isinstance(self.model, MixedSingleTaskGP):
                    params["kernel"] = "MixedSingleTaskGP"
                    if self.lengthscales is not None:
                        for i, ls in enumerate(self.lengthscales.flatten()):
                            params[f"continuous_dim_{i}_lengthscale"] = float(ls)
                else:
                    params["kernel"] = self.kernel_type
                    if self.lengthscales is not None:
                        for i, ls in enumerate(self.lengthscales.flatten()):
                            params[f"lengthscale_{i}"] = float(ls)
                
                # Add categorical information if applicable
                if self.cat_dims:
                    params["categorical_dimensions"] = self.cat_dims
                
                return params
        
        # Extract lengthscales if available
        lengthscales = None
        try:
            params = self.get_hyperparameters()
            if 'cont_lengthscales' in params:
                lengthscales = params['cont_lengthscales']
            elif 'lengthscale' in params:
                lengthscales = params['lengthscale']
        except:
            pass
        
        return KernelInfo(
            model=self.model,
            kernel_type=self.cont_kernel_type,
            cat_dims=self.cat_dims,
            lengthscales=lengthscales
        )

    def _preprocess_X(self, X):
        """
        Preprocess input data for the model and visualization.
        This is a compatibility method that ensures visualizations work correctly.
        """
        return self._encode_categorical_data(X)
    
    def evaluate(self, experiment_manager, cv_splits=5, debug=False, progress_callback=None, **kwargs):
        """
        Evaluate model performance on increasing subsets of data using cross-validation.
        Uses the same approach as the parity plot to ensure consistent RMSE values.
        """
        exp_df = experiment_manager.get_data()
        
        # Skip this if model not yet trained
        if self.model is None or self.fitted_state_dict is None:
            self.train(experiment_manager)
        
        # Get data - handle noise column if present
        if 'Noise' in exp_df.columns:
            X = exp_df.drop(columns=["Output", "Noise"])
        else:
            X = exp_df.drop(columns=["Output"])
            
        y = exp_df["Output"]
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_data(X)
        
        # Convert to tensors
        full_X = torch.tensor(X_encoded.values, dtype=torch.double)
        full_Y = torch.tensor(y.values, dtype=torch.double).unsqueeze(-1)
        
        # Metrics storage
        rmse_values = []
        mae_values = []
        mape_values = []
        r2_values = []
        n_obs = []
        
        # Calculate total steps for progress
        total_steps = len(range(max(cv_splits+1, 5), len(full_X) + 1))
        current_step = 0
        
        # Evaluate on increasing subsets of data
        for i in range(max(cv_splits+1, 5), len(full_X) + 1):
            if debug:
                print(f"Evaluating with {i} observations")
                
            subset_X = full_X[:i]
            subset_Y = full_Y[:i]
            subset_np_X = subset_X.cpu().numpy()
            
            # Cross-validation results for this subset
            fold_y_trues = []
            fold_y_preds = []
            
            # Use KFold to ensure consistent cross-validation
            kf = KFold(n_splits=min(cv_splits, i-1), shuffle=True, random_state=self.random_state)
            
            # Perform cross-validation for this subset size
            for train_idx, test_idx in kf.split(subset_np_X):
                # Split data
                X_train = subset_X[train_idx]
                y_train = subset_Y[train_idx]
                X_test = subset_X[test_idx]
                y_test = subset_Y[test_idx]
                
                # Create a new model with this fold's training data
                # Need to recreate transforms with the same parameters as the main model
                fold_input_transform, fold_outcome_transform = self._create_transforms(X_train, y_train)
                
                cont_kernel_factory = self._get_cont_kernel_factory()
                if self.cat_dims and len(self.cat_dims) > 0:
                    fold_model = MixedSingleTaskGP(
                        X_train, y_train, 
                        cat_dims=self.cat_dims,
                        cont_kernel_factory=cont_kernel_factory,
                        input_transform=fold_input_transform,
                        outcome_transform=fold_outcome_transform
                    )
                else:
                    fold_model = SingleTaskGP(
                        X_train, y_train,
                        input_transform=fold_input_transform,
                        outcome_transform=fold_outcome_transform
                    )
                
                # Load the trained state - this keeps the hyperparameters without retraining
                fold_model.load_state_dict(self.fitted_state_dict, strict=False)
                
                # Make predictions on test fold
                fold_model.eval()
                fold_model.likelihood.eval()
                
                with torch.no_grad():
                    posterior = fold_model.posterior(X_test)
                    preds = posterior.mean.squeeze(-1)
                    
                    # Store this fold's results
                    fold_y_trues.append(y_test.squeeze(-1))
                    fold_y_preds.append(preds)
            
            # Combine all fold results for this subset size
            all_y_true = torch.cat(fold_y_trues).cpu().numpy()
            all_y_pred = torch.cat(fold_y_preds).cpu().numpy()
            
            # Note: BoTorch models with transforms automatically return predictions 
            # in the original scale, so no manual inverse transform is needed
            # The transforms are handled internally by the BoTorch model
            
            # Calculate metrics using cross-validated predictions
            rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
            mae = mean_absolute_error(all_y_true, all_y_pred)
            
            # Handle division by zero in MAPE calculation more safely
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.nanmean(np.abs((all_y_true - all_y_pred) / (np.abs(all_y_true) + 1e-9))) * 100
                
            r2 = r2_score(all_y_true, all_y_pred)
            
            # Store metrics
            rmse_values.append(rmse)
            mae_values.append(mae)
            mape_values.append(mape)
            r2_values.append(r2)
            n_obs.append(i)
            
            # Update progress
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps)
        
        return {
            "RMSE": rmse_values,
            "MAE": mae_values,
            "MAPE": mape_values,
            "R²": r2_values,
            "n_obs": n_obs
        }
    
    def get_hyperparameters(self):
        """Get model hyperparameters."""
        if not self.is_trained or self.model is None:
            return {"status": "Model not trained"}
            
        try:
            params = {}
            # Extract lengthscales from the model
            if hasattr(self.model, 'covar_module') and hasattr(self.model.covar_module, 'base_kernel'):
                if hasattr(self.model.covar_module.base_kernel, 'lengthscale'):
                    lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().numpy()
                    params['lengthscale'] = lengthscale.tolist() if hasattr(lengthscale, 'tolist') else lengthscale
                    
                if hasattr(self.model.covar_module, 'outputscale'):
                    outputscale = self.model.covar_module.outputscale.detach().numpy()
                    params['outputscale'] = outputscale.tolist() if hasattr(outputscale, 'tolist') else outputscale
                    
            # Include kernel info
            params['kernel_type'] = self.kernel_options.get('cont_kernel_type', 'Unknown')
            if params['kernel_type'] == 'Matern':
                params['nu'] = self.kernel_options.get('matern_nu', None)
                
            return params
        except Exception as e:
            return {"error": str(e)}

    def generate_contour_data(self, x_range, y_range, fixed_values, x_idx=0, y_idx=2):
        """
        Generate contour plot data using the BoTorch model.
        
        Args:
            x_range: Tuple of (min, max) for x-axis values
            y_range: Tuple of (min, max) for y-axis values
            fixed_values: Dict mapping dimension indices to fixed values
            x_idx: Index of the x-axis dimension in the model input
            y_idx: Index of the y-axis dimension in the model input
            
        Returns:
            Tuple of (X, Y, Z) for contour plotting
        """
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Generate grid values directly as tensors
        x_vals = torch.linspace(x_range[0], x_range[1], 100)
        y_vals = torch.linspace(y_range[0], y_range[1], 100)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        
        # Total dimensions in the model
        input_dim = len(self.feature_names) if self.feature_names else 4
        
        # Create placeholder tensors for all dimensions
        grid_tensors = []
        for i in range(input_dim):
            if i == x_idx:
                # This is our x-axis variable
                grid_tensors.append(X.flatten())
            elif i == y_idx:
                # This is our y-axis variable
                grid_tensors.append(Y.flatten())
            elif i in fixed_values:
                # This is a fixed variable
                value = fixed_values[i]
                
                # Handle categorical variables (convert strings to numeric using encoding)
                if isinstance(value, str) and self.feature_names and i < len(self.feature_names):
                    # Get the feature name for this dimension
                    feature_name = self.feature_names[i]
                    
                    # Check if we have an encoding for this feature
                    if feature_name in self.categorical_encodings:
                        # Convert the string value to its numeric encoding
                        if value in self.categorical_encodings[feature_name]:
                            value = float(self.categorical_encodings[feature_name][value])
                        else:
                            # If the value is not in our encoding map, use a default (0)
                            print(f"Warning: Value '{value}' not found in encoding for '{feature_name}'. Using default value 0.")
                            value = 0.0
                    else:
                        # No encoding available, use default
                        print(f"Warning: No encoding found for categorical feature '{feature_name}'. Using default value 0.")
                        value = 0.0
                elif not isinstance(value, (int, float)):
                    # For any other non-numeric types, convert to float if possible
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Cannot convert value '{value}' to float. Using default value 0.")
                        value = 0.0
                        
                # Create tensor with the fixed value
                grid_tensors.append(torch.full_like(X.flatten(), float(value)))
            else:
                # Default fixed value (0)
                grid_tensors.append(torch.zeros_like(X.flatten()))
        
        # Stack tensors to create input grid
        grid_input = torch.stack(grid_tensors, dim=-1).double()
        
        # Get predictions
        self.model.eval()
        self.model.likelihood.eval()
        
        with torch.no_grad():
            posterior = self.model.posterior(grid_input)
            Z = posterior.mean.reshape(X.shape)
        
        return X.numpy(), Y.numpy(), Z.numpy()
    
    # Add this method to BoTorchModel class
    def _cache_cross_validation_results(self, X, y, n_splits=5):
        """
        Perform cross-validation and cache the results for faster parity plots.
        Uses tensors and state_dict for BoTorch models.
        """
        if len(X) < n_splits:
            return  # Not enough data for CV
            
        # Convert pandas/numpy data to tensors if needed
        if isinstance(X, pd.DataFrame):
            X_encoded = self._encode_categorical_data(X)
            X_tensor = torch.tensor(X_encoded.values, dtype=torch.double)
        elif isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.double)
        else:
            X_tensor = X  # Assume it's already a tensor
            
        if isinstance(y, pd.Series) or isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        else:
            y_tensor = y  # Assume it's already a tensor
        
        # Perform cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        y_true_all = []
        y_pred_all = []
        
        # Need to convert tensor back to numpy for KFold
        X_np = X_tensor.cpu().numpy()
        
        for train_idx, test_idx in kf.split(X_np):
            # Split data
            X_train = X_tensor[train_idx]
            y_train = y_tensor[train_idx]
            X_test = X_tensor[test_idx]
            y_test = y_tensor[test_idx]
            
            # Create transforms for this CV fold
            input_transform, outcome_transform = self._create_transforms(X_train, y_train)
            
            # Create a new model with the subset data and same transforms as main model
            cont_kernel_factory = self._get_cont_kernel_factory()
            if self.cat_dims and len(self.cat_dims) > 0:
                cv_model = MixedSingleTaskGP(
                    X_train, y_train, 
                    cat_dims=self.cat_dims,
                    cont_kernel_factory=cont_kernel_factory,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
            else:
                cv_model = SingleTaskGP(
                    X_train, y_train,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform
                )
            
            # Load the trained state - this should now work properly with transforms
            cv_model.load_state_dict(self.fitted_state_dict, strict=False)
            
            # Make predictions
            cv_model.eval()
            cv_model.likelihood.eval()
            
            with torch.no_grad():
                posterior = cv_model.posterior(X_test)
                preds = posterior.mean.squeeze(-1)
                
                # Store results
                y_true_all.append(y_test.squeeze(-1))
                y_pred_all.append(preds)
        
        # Concatenate all results and convert to numpy
        y_true_all = torch.cat(y_true_all).cpu().numpy()
        y_pred_all = torch.cat(y_pred_all).cpu().numpy()
        
        # Note: For BoTorch models, output transforms are handled internally by the model
        # The predictions are already in the original scale due to BoTorch's transform handling
        
        # Cache the results
        self.cv_cached_results = {
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
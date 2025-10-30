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
from alchemist_core.config import get_logger
import warnings
from botorch.models.utils.assorted import InputDataWarning

# Import necessary kernels from GPyTorch
from gpytorch.kernels import MaternKernel, RBFKernel

logger = get_logger(__name__)

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
        
        # Calibration attributes
        self.calibration_enabled = False
        self.calibration_factor = 1.0  # Multiplicative factor for std (s = std(z))
    
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
        logger.info(f"Training with {len(self.original_feature_names)} original features: {self.original_feature_names}")
        
        # Encode categorical variables
        X_encoded = self._encode_categorical_data(X)
        
        # Convert to tensors
        train_X = torch.tensor(X_encoded.values, dtype=torch.double)
        train_Y = torch.tensor(y.values, dtype=torch.double).unsqueeze(-1)
        
        # Convert noise values to tensor if available
        if noise is not None:
            train_Yvar = torch.tensor(noise.values, dtype=torch.double).unsqueeze(-1)
            logger.info(f"Using provided noise values for BoTorch model regularization.")
        else:
            train_Yvar = None
        
        # Create transforms
        input_transform, outcome_transform = self._create_transforms(train_X, train_Y)
        
        # Print transform information
        if input_transform is not None:
            logger.info(f"Applied {self.input_transform_type} transform to inputs")
        else:
            logger.info("No input transform applied")
            
        if outcome_transform is not None:
            logger.info(f"Applied {self.output_transform_type} transform to outputs")
        else:
            logger.info("No output transform applied")
        
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
        
        # Compute calibration factors if requested
        if kwargs.get('calibrate_uncertainty', True) and self.cv_cached_results is not None:
            self._compute_calibration_factors()
        
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
                
                # Apply calibration to standard deviation if enabled
                if self.calibration_enabled:
                    std = std * self.calibration_factor
                
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
                        # Handle both list and tensor types
                        ls_values = self.lengthscales if isinstance(self.lengthscales, list) else self.lengthscales.flatten()
                        for i, ls in enumerate(ls_values):
                            params[f"continuous_dim_{i}_lengthscale"] = float(ls)
                else:
                    params["kernel"] = self.kernel_type
                    if self.lengthscales is not None:
                        # Handle both list and tensor types
                        ls_values = self.lengthscales if isinstance(self.lengthscales, list) else self.lengthscales.flatten()
                        for i, ls in enumerate(ls_values):
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
                logger.info(f"Evaluating with {i} observations")
                
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
            
            # Add debugging information about model structure
            model_type = type(self.model).__name__
            params['model_type'] = model_type
            
            # Debug model structure
            if hasattr(self.model, 'covar_module'):
                covar_type = type(self.model.covar_module).__name__
                params['covar_module_type'] = covar_type
                
                # Handle different covariance module structures
                covar_module = self.model.covar_module
                
                # Try multiple access patterns to get lengthscale and outputscale
                lengthscale_found = False
                outputscale_found = False
                
                # Pattern 1: AdditiveKernel (for MixedSingleTaskGP with categorical variables)
                if hasattr(covar_module, 'kernels'):
                    # This is an AdditiveKernel - iterate through sub-kernels
                    all_lengthscales = []
                    kernel_info = []
                    
                    def extract_continuous_lengthscales(kernel):
                        """Extract lengthscales from only the continuous kernel components."""
                        lengthscales = []
                        continuous_kernels = []
                        
                        # Check for direct lengthscale from continuous kernels only
                        kernel_type = type(kernel).__name__
                        if kernel_type in ['MaternKernel', 'RBFKernel', 'PeriodicKernel'] and hasattr(kernel, 'lengthscale') and kernel.lengthscale is not None:
                            ls = kernel.lengthscale.detach().numpy()
                            lengthscales.extend(ls.flatten().tolist())
                            continuous_kernels.append(f"{kernel_type}({len(ls.flatten())} dims)")
                            return lengthscales, continuous_kernels
                        
                        # Check base_kernel (for ScaleKernel wrapping continuous kernels)
                        if hasattr(kernel, 'base_kernel') and kernel.base_kernel is not None:
                            base_lengthscales, base_kernels = extract_continuous_lengthscales(kernel.base_kernel)
                            lengthscales.extend(base_lengthscales)
                            continuous_kernels.extend(base_kernels)
                        
                        # Check for sub-kernels (for AdditiveKernel, ProductKernel)
                        if hasattr(kernel, 'kernels'):
                            for sub_kernel in kernel.kernels:
                                sub_lengthscales, sub_kernels = extract_continuous_lengthscales(sub_kernel)
                                lengthscales.extend(sub_lengthscales)
                                continuous_kernels.extend(sub_kernels)
                        
                        return lengthscales, continuous_kernels
                    
                    # Extract only continuous lengthscales and clean kernel info
                    all_continuous_kernels = []
                    for i, kernel in enumerate(covar_module.kernels):
                        kernel_lengthscales, kernel_types = extract_continuous_lengthscales(kernel)
                        all_lengthscales.extend(kernel_lengthscales)
                        all_continuous_kernels.extend(kernel_types)
                    
                    # Store clean kernel info
                    if all_continuous_kernels:
                        kernel_info = all_continuous_kernels
                    
                    if all_lengthscales:
                        params['lengthscale'] = all_lengthscales
                        lengthscale_found = True
                    
                    # Add info about the additive kernel structure
                    params['additive_kernels'] = kernel_info
                
                # Pattern 2: Direct access to base_kernel.lengthscale (most common)
                if not lengthscale_found and hasattr(covar_module, 'base_kernel') and hasattr(covar_module.base_kernel, 'lengthscale'):
                    lengthscale = covar_module.base_kernel.lengthscale.detach().numpy()
                    params['lengthscale'] = lengthscale.flatten().tolist()  # Flatten for display
                    lengthscale_found = True
                
                # Pattern 3: For mixed models, try data_covar_module
                if not lengthscale_found and hasattr(covar_module, 'data_covar_module'):
                    data_covar = covar_module.data_covar_module
                    if hasattr(data_covar, 'base_kernel') and hasattr(data_covar.base_kernel, 'lengthscale'):
                        lengthscale = data_covar.base_kernel.lengthscale.detach().numpy()
                        params['lengthscale'] = lengthscale.flatten().tolist()
                        lengthscale_found = True
                
                # Pattern 4: Direct access to lengthscale (for some kernel types)
                if not lengthscale_found and hasattr(covar_module, 'lengthscale') and covar_module.lengthscale is not None:
                    lengthscale = covar_module.lengthscale.detach().numpy()
                    params['lengthscale'] = lengthscale.flatten().tolist()
                    lengthscale_found = True
                
                # Get outputscale - try multiple patterns
                if hasattr(covar_module, 'outputscale'):
                    outputscale = covar_module.outputscale.detach().numpy()
                    params['outputscale'] = outputscale.tolist() if hasattr(outputscale, 'tolist') else float(outputscale)
                    outputscale_found = True
                elif hasattr(covar_module, 'data_covar_module') and hasattr(covar_module.data_covar_module, 'outputscale'):
                    outputscale = covar_module.data_covar_module.outputscale.detach().numpy()
                    params['outputscale'] = outputscale.tolist() if hasattr(outputscale, 'tolist') else float(outputscale)
                    outputscale_found = True
                
                # Add feature names for lengthscale interpretation
                if 'lengthscale' in params and hasattr(self, 'original_feature_names'):
                    continuous_features = []
                    if self.cat_dims:
                        # Filter out categorical dimensions
                        for i, feature_name in enumerate(self.original_feature_names):
                            if i not in self.cat_dims:
                                continuous_features.append(feature_name)
                    else:
                        # All features are continuous
                        continuous_features = self.original_feature_names
                    
                    # For mixed models, we may have duplicate lengthscales due to complex kernel structure
                    # Try to map only to the actual continuous features we know about
                    if len(continuous_features) > 0:
                        # Take only the first set of lengthscales that match our continuous features
                        num_continuous = len(continuous_features)
                        if len(params['lengthscale']) >= num_continuous:
                            # Use only the first n lengthscales corresponding to our continuous features
                            params['continuous_features'] = continuous_features
                            params['primary_lengthscales'] = params['lengthscale'][:num_continuous]
                
                # Try to extract noise parameter
                if hasattr(self.model, 'likelihood') and hasattr(self.model.likelihood, 'noise'):
                    noise = self.model.likelihood.noise.detach().numpy()
                    params['noise'] = noise.tolist() if hasattr(noise, 'tolist') else float(noise)
                    
            # Include kernel configuration info
            params['kernel_type'] = self.kernel_options.get('cont_kernel_type', 'Unknown')
            if params['kernel_type'] == 'Matern':
                params['nu'] = self.kernel_options.get('matern_nu', None)
                
            # Add transform information
            if hasattr(self.model, 'input_transform') and self.model.input_transform is not None:
                params['input_transform'] = type(self.model.input_transform).__name__
            else:
                params['input_transform'] = 'None'
                
            if hasattr(self.model, 'outcome_transform') and self.model.outcome_transform is not None:
                params['outcome_transform'] = type(self.model.outcome_transform).__name__  
            else:
                params['outcome_transform'] = 'None'
                
            return params
        except Exception as e:
            # More detailed error reporting
            import traceback
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "model_type": type(self.model).__name__ if self.model else "None"
            }
            return error_details

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
                            logger.warning(f"Value '{value}' not found in encoding for '{feature_name}'. Using default value 0.")
                            value = 0.0
                    else:
                        # No encoding available, use default
                        logger.warning(f"No encoding found for categorical feature '{feature_name}'. Using default value 0.")
                        value = 0.0
                elif not isinstance(value, (int, float)):
                    # For any other non-numeric types, convert to float if possible
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert value '{value}' to float. Using default value 0.")
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
        y_std_all = []
        
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
                # Get standard deviation from variance
                stds = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
                
                # Store results
                y_true_all.append(y_test.squeeze(-1))
                y_pred_all.append(preds)
                y_std_all.append(stds)
        
        # Concatenate all results and convert to numpy
        y_true_all = torch.cat(y_true_all).cpu().numpy()
        y_pred_all = torch.cat(y_pred_all).cpu().numpy()
        y_std_all = torch.cat(y_std_all).cpu().numpy()
        
        # Note: For BoTorch models, output transforms are handled internally by the model
        # The predictions are already in the original scale due to BoTorch's transform handling
        
        # Cache the results
        self.cv_cached_results = {
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_std': y_std_all
        }

    def _compute_calibration_factors(self):
        """
        Compute calibration factor from CV results.
        The calibration_factor is the std of z-scores (standardized residuals).
        This factor will be used to scale predicted std in future predictions.
        Also creates a calibrated copy of CV results for plotting.
        """
        if self.cv_cached_results is None:
            logger.warning("No CV results available for calibration.")
            return
        
        y_true = self.cv_cached_results['y_true']
        y_pred = self.cv_cached_results['y_pred']
        y_std = self.cv_cached_results['y_std']
        
        # Compute standardized residuals (z-scores)
        z_scores = (y_true - y_pred) / y_std
        
        # Calibration factor = std(z)
        self.calibration_factor = np.std(z_scores, ddof=1)
        self.calibration_enabled = True
        
        # Create calibrated copy of CV results for plotting
        self.cv_cached_results_calibrated = {
            'y_true': y_true.copy(),
            'y_pred': y_pred.copy(),
            'y_std': y_std * self.calibration_factor  # Apply calibration
        }
        
        # Print calibration info
        logger.info(f"\n{'='*60}")
        logger.info("UNCERTAINTY CALIBRATION")
        logger.info(f"{'='*60}")
        logger.info(f"Calibration factor (s): {self.calibration_factor:.4f}")
        logger.info(f"  - Future σ predictions will be multiplied by {self.calibration_factor:.4f}")
        logger.info(f"  - Note: Acquisition functions use uncalibrated uncertainties")
        
        if self.calibration_factor < 0.8:
            logger.info("  ⚠ Model appears under-confident (s < 1)")
            logger.info("     Predicted uncertainties will be DECREASED")
        elif self.calibration_factor > 1.2:
            logger.info("  ⚠ Model appears over-confident (s > 1)")
            logger.info("     Predicted uncertainties will be INCREASED")
        else:
            logger.info("  ✓ Uncertainty appears well-calibrated")
        
        logger.info(f"{'='*60}\n")
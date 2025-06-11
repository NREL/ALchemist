from .base_model import BaseModel
from logic.search_space import SearchSpace
from logic.experiment_manager import ExperimentManager
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C

class SklearnModel(BaseModel):
    def __init__(self, kernel_options: dict, n_restarts_optimizer=30, random_state=42, optimizer="L-BFGS-B"):
        """
        Initialize the SklearnModel with kernel options.
        
        Args:
            kernel_options: Dictionary with keys:
                - "kernel_type": one of "RBF", "Matern", "RationalQuadratic" 
                - If "Matern" is selected, a key "matern_nu" should be provided.
            n_restarts_optimizer: Number of restarts for the optimizer.
            random_state: Random state for reproducibility.
            optimizer: Optimization method for hyperparameter tuning.
        """
        super().__init__(random_state=random_state)
        self.kernel_options = kernel_options
        self.n_restarts_optimizer = n_restarts_optimizer
        self.optimizer = optimizer
        self.model = None
        self.optimized_kernel = None
        self.encoder = None  # For one-hot encoding
        self.categorical_variables = []
        self.cv_cached_results = None  # Will store y_true and y_pred from cross-validation

    def _custom_optimizer(self, obj_func, initial_theta, bounds, args=(), **kwargs):
        result = scipy.optimize.minimize(
            obj_func,
            initial_theta,
            bounds=bounds if self.optimizer not in ['CG', 'BFGS'] else None,
            method=self.optimizer,
            jac=True,
            args=args,
            **kwargs
        )
        return result.x, result.fun

    def _build_kernel(self, X):
        """Build the kernel using training data X to initialize length scales."""
        kernel_type = self.kernel_options.get("kernel_type", "RBF")
        # Compute initial length scales as the mean of the data along each dimension.
        ls_init = np.mean(X, axis=0)
        ls_bounds = [(1e-5, l * 1e5) for l in ls_init]
        constant = C()
        if kernel_type == "RBF":
            kernel = constant * RBF(length_scale=ls_init, length_scale_bounds=ls_bounds)
        elif kernel_type == "Matern":
            matern_nu = self.kernel_options.get("matern_nu", 1.5)
            kernel = constant * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=matern_nu)
        elif kernel_type == "RationalQuadratic":
            kernel = constant * RationalQuadratic()
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        return kernel

    def _preprocess_data(self, experiment_manager):
        """Preprocess the data for scikit-learn with one-hot encoding for categoricals."""
        # Get data with noise values if available
        X, y, noise = experiment_manager.get_features_target_and_noise()
        categorical_variables = experiment_manager.search_space.get_categorical_variables()
        self.categorical_variables = categorical_variables
        
        # Store noise values for later use in the model (use None if no noise provided)
        self.alpha = noise.values if noise is not None else None
        print(f"{'Using provided noise values for regularization' if noise is not None else 'No noise values provided - using scikit-learn default regularization'}")
        
        # Separate categorical and numerical columns
        categorical_df = X[categorical_variables] if categorical_variables else None
        numerical_df = X.drop(columns=categorical_variables) if categorical_variables else X

        # One-hot-encode categorical variables if they exist, using drop='first' for skopt compatibility
        if categorical_df is not None and not categorical_df.empty:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop='first')
            encoded_categorical = self.encoder.fit_transform(categorical_df)
            encoded_categorical_df = pd.DataFrame(
                encoded_categorical,
                columns=self.encoder.get_feature_names_out(categorical_variables),
                index=X.index
            )
            # Merge numerical and encoded categorical data
            processed_X = pd.concat([numerical_df, encoded_categorical_df], axis=1)
        else:
            # If no categorical variables, use only numerical data
            processed_X = numerical_df
            self.encoder = None

        # Save the feature names for debugging dimensional mismatches
        self.feature_names = processed_X.columns.tolist()
        print(f"Model trained with {len(self.feature_names)} features: {self.feature_names}")
        
        return processed_X.values, y.values

    def _preprocess_X(self, X):
        """Preprocess X for prediction (apply the same transformations as in training)."""
        if isinstance(X, pd.DataFrame):
            if self.categorical_variables and self.encoder:
                categorical_X = X[self.categorical_variables] if len(self.categorical_variables) > 0 else None
                numerical_X = X.drop(columns=self.categorical_variables) if len(self.categorical_variables) > 0 else X
                
                if categorical_X is not None and not categorical_X.empty:
                    encoded_categorical = self.encoder.transform(categorical_X)
                    encoded_categorical_df = pd.DataFrame(
                        encoded_categorical,
                        columns=self.encoder.get_feature_names_out(self.categorical_variables),
                        index=X.index
                    )
                    # Merge numerical and encoded categorical data
                    processed_X = pd.concat([numerical_X, encoded_categorical_df], axis=1).values
                else:
                    processed_X = numerical_X.values
            else:
                processed_X = X.values
        else:
            # Assume it's already preprocessed if not a DataFrame
            processed_X = X
            
        return processed_X

    def train(self, experiment_manager, **kwargs):
        """Train the model using the ExperimentManager."""
        X, y = self._preprocess_data(experiment_manager)
        self.kernel = self._build_kernel(X)
        
        # Create base parameters dictionary
        params = {
            "kernel": self.kernel,
            "n_restarts_optimizer": self.n_restarts_optimizer,
            "random_state": self.random_state,
            "optimizer": self._custom_optimizer
        }
        
        # Only add alpha parameter when noise values are available
        if self.alpha is not None:
            params["alpha"] = self.alpha
        
        # Create model with appropriate parameters
        self.model = GaussianProcessRegressor(**params)
        
        # Store the raw training data for possible reuse with skopt
        self.X_train_ = X
        self.y_train_ = y
        
        self.model.fit(X, y)
        self.optimized_kernel = self.model.kernel_
        self._is_trained = True
        
        # After model is trained, cache CV results
        if kwargs.get('cache_cv', True):
            self._cache_cross_validation_results(X, y)

    def predict(self, X, return_std=False, **kwargs):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            return_std: Whether to return standard deviations
            
        Returns:
            If return_std is False: numpy array of predictions
            If return_std is True: tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        X_processed = self._preprocess_X(X)
        return self.model.predict(X_processed, return_std=return_std)

    def predict_with_std(self, X):
        """
        Make predictions with standard deviation.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        X_processed = self._preprocess_X(X)
        return self.model.predict(X_processed, return_std=True)

    def evaluate(self, experiment_manager, cv_splits=5, debug=False, progress_callback=None, **kwargs):
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        X, y = self._preprocess_data(experiment_manager)
        
        if debug:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            rmse_values, mae_values, mape_values, r2_values = [], [], [], []
            for i in range(5, len(X_train) + 1):
                subset_X_train = X_train[:i]
                subset_y_train = y_train[:i]
                eval_model = GaussianProcessRegressor(
                    kernel=self.optimized_kernel,
                    optimizer=None,
                    random_state=self.random_state
                )
                eval_model.fit(subset_X_train, subset_y_train)
                y_pred = eval_model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = eval_model.score(X_test, y_test)
                rmse_values.append(rmse)
                mae_values.append(mae)
                mape_values.append(mape)
                r2_values.append(r2)
                if progress_callback:
                    progress_callback(i / len(X_train))
            return {"RMSE": rmse_values, "MAE": mae_values, "MAPE": mape_values, "R²": r2_values}
        else:
            kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
            scoring = {
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "mape": "neg_mean_absolute_percentage_error"
            }
            rmse_values, mae_values, mape_values, r2_values = [], [], [], []
            for i in range(5, len(X) + 1):
                subset_X = X[:i]
                subset_y = y[:i]
                if i >= 10:
                    scoring["r2"] = "r2"
                eval_model = GaussianProcessRegressor(
                    kernel=self.optimized_kernel,
                    optimizer=None,
                    random_state=self.random_state
                )
                cv_results = cross_validate(
                    eval_model,
                    subset_X,
                    subset_y,
                    cv=kf,
                    scoring=scoring,
                    n_jobs=-1
                )
                rmse = -np.mean(cv_results["test_rmse"])  # Cross-validation error
                mae = -np.mean(cv_results["test_mae"])
                mape = -np.mean(cv_results["test_mape"])
                rmse_values.append(rmse)
                mae_values.append(mae)
                mape_values.append(mape)
                if i >= 10:
                    r2 = np.mean(cv_results["test_r2"])
                    r2_values.append(r2)
                else:
                    r2_values.append(None)
                if progress_callback:
                    progress_callback(i / len(X))
            return {"RMSE": rmse_values, "MAE": mae_values, "MAPE": mape_values, "R²": r2_values}

    def get_hyperparameters(self):
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.kernel_.get_params()
    

    def _cache_cross_validation_results(self, X, y, n_splits=5):
        """
        Perform cross-validation and cache the results for faster parity plots.
        """
        if len(X) < n_splits:
            return  # Not enough data for CV
            
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in kf.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create a copy of the model with the same hyperparameters
            cv_model = GaussianProcessRegressor(
                kernel=self.optimized_kernel,
                optimizer=None,  # Don't re-optimize
                random_state=self.random_state
            )
            
            # Train on this fold's training data
            cv_model.fit(X_train, y_train)
            
            # Predict on test data
            y_pred = cv_model.predict(X_test)
            
            # Store results
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        
        # Cache the results
        self.cv_cached_results = {
            'y_true': np.array(y_true_all),
            'y_pred': np.array(y_pred_all)
        }

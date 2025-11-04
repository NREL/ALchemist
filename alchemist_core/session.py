"""
Optimization Session API - High-level interface for Bayesian optimization workflows.

This module provides the main entry point for using ALchemist as a headless library.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
import pandas as pd
import numpy as np
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.events import EventEmitter
from alchemist_core.config import get_logger

logger = get_logger(__name__)


class OptimizationSession:
    """
    High-level interface for Bayesian optimization workflows.
    
    This class orchestrates the complete optimization loop:
    1. Define search space
    2. Load/add experimental data
    3. Train surrogate model
    4. Run acquisition to suggest next experiments
    5. Iterate
    
    Example:
        >>> from alchemist_core import OptimizationSession
        >>> 
        >>> # Create session with search space
        >>> session = OptimizationSession()
        >>> session.add_variable('temperature', 'real', bounds=(300, 500))
        >>> session.add_variable('pressure', 'real', bounds=(1, 10))
        >>> session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
        >>> 
        >>> # Load experimental data
        >>> session.load_data('experiments.csv', target_column='yield')
        >>> 
        >>> # Train model
        >>> session.train_model(backend='botorch', kernel='Matern')
        >>> 
        >>> # Suggest next experiment
        >>> next_point = session.suggest_next(strategy='EI', goal='maximize')
        >>> print(next_point)
    """
    
    def __init__(self, search_space: Optional[SearchSpace] = None, 
                 experiment_manager: Optional[ExperimentManager] = None,
                 event_emitter: Optional[EventEmitter] = None):
        """
        Initialize optimization session.
        
        Args:
            search_space: Pre-configured SearchSpace object (optional)
            experiment_manager: Pre-configured ExperimentManager (optional)
            event_emitter: EventEmitter for progress notifications (optional)
        """
        self.search_space = search_space if search_space is not None else SearchSpace()
        self.experiment_manager = experiment_manager if experiment_manager is not None else ExperimentManager()
        self.events = event_emitter if event_emitter is not None else EventEmitter()
        
        # Link search_space to experiment_manager
        self.experiment_manager.set_search_space(self.search_space)
        
        # Model and acquisition state
        self.model = None
        self.model_backend = None
        self.acquisition = None
        
        # Configuration
        self.config = {
            'random_state': 42,
            'verbose': True
        }
        
        logger.info("OptimizationSession initialized")
    
    # ============================================================
    # Search Space Management
    # ============================================================
    
    def add_variable(self, name: str, var_type: str, **kwargs) -> None:
        """
        Add a variable to the search space.
        
        Args:
            name: Variable name
            var_type: Type ('real', 'integer', 'categorical')
            **kwargs: Type-specific parameters:
                - For 'real'/'integer': bounds=(min, max) or min=..., max=...
                - For 'categorical': categories=[list of values] or values=[list]
        
        Example:
            >>> session.add_variable('temp', 'real', bounds=(300, 500))
            >>> session.add_variable('catalyst', 'categorical', categories=['A', 'B'])
        """
        # Convert user-friendly API to internal format
        params = kwargs.copy()
        
        # Handle 'bounds' parameter for real/integer
        if 'bounds' in params and var_type.lower() in ['real', 'integer']:
            min_val, max_val = params.pop('bounds')
            params['min'] = min_val
            params['max'] = max_val
        
        # Handle 'categories' parameter for categorical
        if 'categories' in params and var_type.lower() == 'categorical':
            params['values'] = params.pop('categories')
        
        self.search_space.add_variable(name, var_type, **params)
        
        # Update the search_space reference in experiment_manager
        self.experiment_manager.set_search_space(self.search_space)
        
        logger.info(f"Added variable '{name}' ({var_type}) to search space")
        self.events.emit('variable_added', {'name': name, 'type': var_type})
    
    def load_search_space(self, filepath: str) -> None:
        """
        Load search space from JSON or CSV file.
        
        Args:
            filepath: Path to search space definition file
        """
        self.search_space = SearchSpace.from_json(filepath)
        logger.info(f"Loaded search space from {filepath}")
        self.events.emit('search_space_loaded', {'filepath': filepath})
    
    def get_search_space_summary(self) -> Dict[str, Any]:
        """
        Get summary of current search space.
        
        Returns:
            Dictionary with variable information
        """
        variables = []
        for var in self.search_space.variables:
            var_summary = {
                'name': var['name'],
                'type': var['type']
            }
            
            # Convert min/max to bounds for real/integer
            if var['type'] in ['real', 'integer']:
                if 'min' in var and 'max' in var:
                    var_summary['bounds'] = [var['min'], var['max']]
                else:
                    var_summary['bounds'] = None
            else:
                var_summary['bounds'] = None
            
            # Convert values to categories for categorical
            if var['type'] == 'categorical':
                var_summary['categories'] = var.get('values')
            else:
                var_summary['categories'] = None
            
            # Include optional fields
            if 'unit' in var:
                var_summary['unit'] = var['unit']
            if 'description' in var:
                var_summary['description'] = var['description']
            
            variables.append(var_summary)
        
        return {
            'n_variables': len(self.search_space.variables),
            'variables': variables,
            'categorical_variables': self.search_space.get_categorical_variables()
        }
    
    # ============================================================
    # Data Management
    # ============================================================
    
    def load_data(self, filepath: str, target_column: str = 'Output',
                  noise_column: Optional[str] = None) -> None:
        """
        Load experimental data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target/output column (default: 'Output')
            noise_column: Optional column with measurement noise/uncertainty
        
        Example:
            >>> session.load_data('experiments.csv', target_column='yield')
        """
        # Load the CSV
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Rename target column to 'Output' if different
        if target_column != 'Output' and target_column in df.columns:
            df = df.rename(columns={target_column: 'Output'})
        
        # Rename noise column to 'Noise' if specified
        if noise_column and noise_column in df.columns:
            df = df.rename(columns={noise_column: 'Noise'})
        
        # Save to temporary file and load via ExperimentManager
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp:
            df.to_csv(tmp.name, index=False)
            temp_path = tmp.name
        
        try:
            self.experiment_manager = ExperimentManager.from_csv(
                temp_path,
                self.search_space
            )
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        n_experiments = len(self.experiment_manager.df)
        logger.info(f"Loaded {n_experiments} experiments from {filepath}")
        self.events.emit('data_loaded', {'n_experiments': n_experiments, 'filepath': filepath})
    
    def add_experiment(self, inputs: Dict[str, Any], output: float, 
                      noise: Optional[float] = None) -> None:
        """
        Add a single experiment to the dataset.
        
        Args:
            inputs: Dictionary mapping variable names to values
            output: Target/output value
            noise: Optional measurement uncertainty
        
        Example:
            >>> session.add_experiment(
            ...     inputs={'temperature': 350, 'catalyst': 'A'},
            ...     output=0.85
            ... )
        """
        # Use ExperimentManager's add_experiment method
        self.experiment_manager.add_experiment(
            point_dict=inputs,
            output_value=output,
            noise_value=noise
        )
        
        logger.info(f"Added experiment: {inputs} → {output}")
        self.events.emit('experiment_added', {'inputs': inputs, 'output': output})
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current experimental data.
        
        Returns:
            Dictionary with data statistics
        """
        df = self.experiment_manager.get_data()
        if df is None or df.empty:
            return {'n_experiments': 0, 'has_data': False}
        
        X, y = self.experiment_manager.get_features_and_target()
        return {
            'n_experiments': len(y),
            'has_data': True,
            'has_noise': self.experiment_manager.has_noise_data(),
            'target_stats': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            },
            'feature_names': list(X.columns)
        }
    
    # ============================================================
    # Model Training
    # ============================================================
    
    def train_model(self, backend: str = 'sklearn', kernel: str = 'Matern',
                   kernel_params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Train surrogate model on current data.
        
        Args:
            backend: 'sklearn' or 'botorch'
            kernel: Kernel type ('RBF', 'Matern', 'RationalQuadratic')
            kernel_params: Additional kernel parameters (e.g., {'nu': 2.5} for Matern)
            **kwargs: Backend-specific parameters
        
        Returns:
            Dictionary with training results and hyperparameters
        
        Example:
            >>> results = session.train_model(backend='botorch', kernel='Matern')
            >>> print(results['metrics'])
        """
        df = self.experiment_manager.get_data()
        if df is None or df.empty:
            raise ValueError("No experimental data available. Use load_data() or add_experiment() first.")
        
        self.model_backend = backend.lower()
        
        # Normalize kernel name to match expected case
        kernel_name_map = {
            'rbf': 'RBF',
            'matern': 'Matern',
            'rationalquadratic': 'RationalQuadratic',
            'rational_quadratic': 'RationalQuadratic'
        }
        kernel = kernel_name_map.get(kernel.lower(), kernel)
        
        # Extract calibration_enabled before passing kwargs to model constructor
        calibration_enabled = kwargs.pop('calibration_enabled', False)
        
        # Import appropriate model class
        if self.model_backend == 'sklearn':
            from alchemist_core.models.sklearn_model import SklearnModel
            
            # Build kernel options
            kernel_options = {'kernel_type': kernel}
            if kernel_params:
                kernel_options.update(kernel_params)
            
            self.model = SklearnModel(
                kernel_options=kernel_options,
                random_state=self.config['random_state'],
                **kwargs
            )
            
        elif self.model_backend == 'botorch':
            from alchemist_core.models.botorch_model import BoTorchModel
            
            # Build kernel options - BoTorch uses 'cont_kernel_type' not 'kernel_type'
            kernel_options = {'cont_kernel_type': kernel}
            if kernel_params:
                # Add matern_nu if provided
                if 'nu' in kernel_params:
                    kernel_options['matern_nu'] = kernel_params['nu']
                # Add any other kernel params
                for k, v in kernel_params.items():
                    if k != 'nu':  # Already handled above
                        kernel_options[k] = v
            
            self.model = BoTorchModel(
                kernel_options=kernel_options,
                random_state=self.config['random_state'],
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'sklearn' or 'botorch'")
        
        # Train model
        logger.info(f"Training {backend} model with {kernel} kernel...")
        self.events.emit('training_started', {'backend': backend, 'kernel': kernel})
        
        self.model.train(self.experiment_manager)
        
        # Apply calibration if requested (sklearn only)
        if calibration_enabled and self.model_backend == 'sklearn':
            if hasattr(self.model, '_compute_calibration_factors'):
                self.model._compute_calibration_factors()
                logger.info("Uncertainty calibration enabled")
        
        # Get hyperparameters
        hyperparams = self.model.get_hyperparameters()
        
        # Convert hyperparameters to JSON-serializable format
        # (kernel objects can't be serialized directly)
        json_hyperparams = {}
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                json_hyperparams[key] = value
            elif isinstance(value, np.ndarray):
                json_hyperparams[key] = value.tolist()
            else:
                # Convert complex objects to their string representation
                json_hyperparams[key] = str(value)
        
        # Compute metrics from CV results if available
        metrics = {}
        if hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_true = self.model.cv_cached_results['y_true']
            y_pred = self.model.cv_cached_results['y_pred']
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        results = {
            'backend': backend,
            'kernel': kernel,
            'hyperparameters': json_hyperparams,
            'metrics': metrics,
            'success': True
        }
        
        logger.info(f"Model trained successfully. R²: {metrics.get('r2', 'N/A')}")
        self.events.emit('training_completed', results)
        
        return results
    
    def get_model_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of trained model.
        
        Returns:
            Dictionary with model information, or None if no model trained
        """
        if self.model is None:
            return None
        
        # Compute metrics if available
        metrics = {}
        if hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results is not None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            y_true = self.model.cv_cached_results['y_true']
            y_pred = self.model.cv_cached_results['y_pred']
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        # Get hyperparameters and make them JSON-serializable
        hyperparams = self.model.get_hyperparameters()
        json_hyperparams = {}
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                json_hyperparams[key] = value
            elif isinstance(value, np.ndarray):
                json_hyperparams[key] = value.tolist()
            else:
                # Convert complex objects to their string representation
                json_hyperparams[key] = str(value)
        
        return {
            'backend': self.model_backend,
            'hyperparameters': json_hyperparams,
            'metrics': metrics,
            'is_trained': True
        }
    
    # ============================================================
    # Acquisition and Suggestions
    # ============================================================
    
    def suggest_next(self, strategy: str = 'EI', goal: str = 'maximize',
                    n_suggestions: int = 1, **kwargs) -> pd.DataFrame:
        """
        Suggest next experiment(s) using acquisition function.
        
        Args:
            strategy: Acquisition strategy ('EI', 'PI', 'UCB', 'qEI', etc.)
            goal: 'maximize' or 'minimize'
            n_suggestions: Number of suggestions (batch acquisition)
            **kwargs: Strategy-specific parameters
        
        Returns:
            DataFrame with suggested experiment(s)
        
        Example:
            >>> next_point = session.suggest_next(strategy='EI', goal='maximize')
            >>> print(next_point)
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")
        
        # Import appropriate acquisition class
        if self.model_backend == 'sklearn':
            from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition
            
            self.acquisition = SkoptAcquisition(
                search_space=self.search_space.to_skopt(),
                model=self.model,  # Pass the full SklearnModel wrapper, not just .model
                acq_func=strategy.lower(),
                maximize=(goal.lower() == 'maximize'),
                random_state=self.config['random_state']
            )
            
            # Update acquisition with existing experimental data (un-encoded)
            X, y = self.experiment_manager.get_features_and_target()
            self.acquisition.update(X, y)
            
        elif self.model_backend == 'botorch':
            from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
            
            self.acquisition = BoTorchAcquisition(
                model=self.model,
                search_space=self.search_space,
                acq_func=strategy,
                maximize=(goal.lower() == 'maximize'),
                batch_size=n_suggestions
            )
        
        logger.info(f"Running acquisition: {strategy} ({goal})")
        self.events.emit('acquisition_started', {'strategy': strategy, 'goal': goal})
        
        # Get suggestion
        next_point = self.acquisition.select_next()
        
        # Robustly handle output type and convert to DataFrame
        if isinstance(next_point, pd.DataFrame):
            suggestion_dict = next_point.to_dict('records')[0]
            result_df = next_point
        elif isinstance(next_point, list):
            # Get variable names from search space
            var_names = [var['name'] for var in self.search_space.variables]
            
            # Check if it's a list of dicts or a list of values
            if len(next_point) > 0 and isinstance(next_point[0], dict):
                # List of dicts
                result_df = pd.DataFrame(next_point)
                suggestion_dict = next_point[0]
            else:
                # List of values - create dict with variable names
                suggestion_dict = dict(zip(var_names, next_point))
                result_df = pd.DataFrame([suggestion_dict])
        else:
            # Fallback: wrap in DataFrame
            result_df = pd.DataFrame([next_point])
            suggestion_dict = result_df.to_dict('records')[0]
        
        logger.info(f"Suggested point: {suggestion_dict}")
        self.events.emit('acquisition_completed', {'suggestion': suggestion_dict})
        
        return result_df    # ============================================================
    # Predictions
    # ============================================================
    
    def predict(self, inputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions at new points.
        
        Args:
            inputs: DataFrame with input features
        
        Returns:
            Tuple of (predictions, uncertainties)
        
        Example:
            >>> test_points = pd.DataFrame({
            ...     'temperature': [350, 400],
            ...     'catalyst': ['A', 'B']
            ... })
            >>> predictions, uncertainties = session.predict(test_points)
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")
        
        # Call model's predict with return_std=True to get both predictions and uncertainties
        if self.model_backend == 'sklearn':
            return self.model.predict(inputs, return_std=True)
        elif self.model_backend == 'botorch':
            # BoTorch model's predict also needs return_std=True to return (mean, std)
            return self.model.predict(inputs, return_std=True)
        else:
            # Fallback - try with return_std
            try:
                return self.model.predict(inputs, return_std=True)
            except TypeError:
                # If return_std not supported, just return predictions with zero std
                preds = self.model.predict(inputs)
                return preds, np.zeros_like(preds)
    
    # ============================================================
    # Event Handling
    # ============================================================
    
    def on(self, event: str, callback: Callable) -> None:
        """
        Register event listener.
        
        Args:
            event: Event name
            callback: Callback function
        
        Example:
            >>> def on_training_done(data):
            ...     print(f"Training completed with R² = {data['metrics']['r2']}")
            >>> session.on('training_completed', on_training_done)
        """
        self.events.on(event, callback)
    
    # ============================================================
    # Configuration
    # ============================================================
    
    def set_config(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Configuration parameters (random_state, verbose, etc.)
        
        Example:
            >>> session.set_config(random_state=123, verbose=False)
        """
        self.config.update(kwargs)
        logger.info(f"Updated configuration: {kwargs}")

"""
Optimization Session API - High-level interface for Bayesian optimization workflows.

This module provides the main entry point for using ALchemist as a headless library.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union, Literal
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.events import EventEmitter
from alchemist_core.config import get_logger
from alchemist_core.audit_log import AuditLog, SessionMetadata, AuditEntry

# Optional matplotlib import for visualization methods
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    Figure = None  # Type hint placeholder

# Import visualization functions (delegates to visualization module)
try:
    from alchemist_core.visualization import (
        create_parity_plot,
        create_contour_plot,
        create_slice_plot,
        create_metrics_plot,
        create_qq_plot,
        create_calibration_plot,
        check_matplotlib
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

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
        > from alchemist_core import OptimizationSession
        > 
        > # Create session with search space
        > session = OptimizationSession()
        > session.add_variable('temperature', 'real', bounds=(300, 500))
        > session.add_variable('pressure', 'real', bounds=(1, 10))
        > session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
        > 
        > # Load experimental data
        > session.load_data('experiments.csv', target_column='yield')
        > 
        > # Train model
        > session.train_model(backend='botorch', kernel='Matern')
        > 
        > # Suggest next experiment
        > next_point = session.suggest_next(strategy='EI', goal='maximize')
        > print(next_point)
    """
    
    def __init__(self, search_space: Optional[SearchSpace] = None, 
                 experiment_manager: Optional[ExperimentManager] = None,
                 event_emitter: Optional[EventEmitter] = None,
                 session_metadata: Optional[SessionMetadata] = None):
        """
        Initialize optimization session.
        
        Args:
            search_space: Pre-configured SearchSpace object (optional)
            experiment_manager: Pre-configured ExperimentManager (optional)
            event_emitter: EventEmitter for progress notifications (optional)
            session_metadata: Pre-configured session metadata (optional)
        """
        self.search_space = search_space if search_space is not None else SearchSpace()
        self.experiment_manager = experiment_manager if experiment_manager is not None else ExperimentManager()
        self.events = event_emitter if event_emitter is not None else EventEmitter()
        
        # Session metadata and audit log
        self.metadata = session_metadata if session_metadata is not None else SessionMetadata.create()
        self.audit_log = AuditLog()
        
        # Link search_space to experiment_manager
        self.experiment_manager.set_search_space(self.search_space)
        
        # Model and acquisition state
        self.model = None
        self.model_backend = None
        self.acquisition = None
        
        # Staged experiments (for workflow management)
        self.staged_experiments = []  # List of experiment dicts awaiting evaluation
        self.last_suggestions = []  # Most recent acquisition suggestions (for UI)
        
        # Configuration
        self.config = {
            'random_state': 42,
            'verbose': True,
            'auto_train': False,  # Auto-train model after adding experiments
            'auto_train_threshold': 5  # Minimum experiments before auto-train
        }
        
        logger.info(f"OptimizationSession initialized: {self.metadata.session_id}")
    
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
            > session.add_variable('temp', 'real', bounds=(300, 500))
            > session.add_variable('catalyst', 'categorical', categories=['A', 'B'])
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
    
    def load_data(self, filepath: str, target_columns: Union[str, List[str]] = 'Output',
                  noise_column: Optional[str] = None) -> None:
        """
        Load experimental data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_columns: Target column name(s). Can be:
                - String for single-objective: 'yield'
                - List for multi-objective: ['yield', 'selectivity']
                Default: 'Output'
            noise_column: Optional column with measurement noise/uncertainty
        
        Examples:
            Single-objective:
            >>> session.load_data('experiments.csv', target_columns='yield')
            >>> session.load_data('experiments.csv', target_columns=['yield'])  # also works
            
            Multi-objective (future):
            >>> session.load_data('experiments.csv', target_columns=['yield', 'selectivity'])
        
        Note:
            If the CSV doesn't have columns matching target_columns, an error will be raised.
            Target columns will be preserved with their original names internally.
        """
        # Load the CSV
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Normalize target_columns to list
        if isinstance(target_columns, str):
            target_columns_list = [target_columns]
        else:
            target_columns_list = list(target_columns)
        
        # Validate that all target columns exist
        missing_cols = [col for col in target_columns_list if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Target column(s) {missing_cols} not found in CSV file. "
                f"Available columns: {list(df.columns)}. "
                f"Please specify the correct target column name(s) using the target_columns parameter."
            )
        
        # Warn if 'Output' column exists but user specified different target(s)
        if 'Output' in df.columns and 'Output' not in target_columns_list:
            logger.warning(
                f"CSV contains 'Output' column but you specified {target_columns_list}. "
                f"Using {target_columns_list} as specified."
            )
        
        # Store the target column names for ExperimentManager
        target_col_internal = target_columns_list
        
        # Rename noise column to 'Noise' if specified and different
        if noise_column and noise_column in df.columns and noise_column != 'Noise':
            df = df.rename(columns={noise_column: 'Noise'})
        
        # Save to temporary file and load via ExperimentManager
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as tmp:
            df.to_csv(tmp.name, index=False)
            temp_path = tmp.name
        
        try:
            # Create ExperimentManager with the specified target column(s)
            self.experiment_manager = ExperimentManager(
                search_space=self.search_space,
                target_columns=target_col_internal
            )
            self.experiment_manager.load_from_csv(temp_path)
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        n_experiments = len(self.experiment_manager.df)
        logger.info(f"Loaded {n_experiments} experiments from {filepath}")
        self.events.emit('data_loaded', {'n_experiments': n_experiments, 'filepath': filepath})
    
    def add_experiment(self, inputs: Dict[str, Any], output: float, 
                      noise: Optional[float] = None, iteration: Optional[int] = None,
                      reason: Optional[str] = None) -> None:
        """
        Add a single experiment to the dataset.
        
        Args:
            inputs: Dictionary mapping variable names to values
            output: Target/output value
            noise: Optional measurement uncertainty
            iteration: Iteration number (auto-assigned if None)
            reason: Reason for this experiment (e.g., 'Manual', 'Expected Improvement')
        
        Example:
            > session.add_experiment(
            ...     inputs={'temperature': 350, 'catalyst': 'A'},
            ...     output=0.85,
            ...     reason='Manual'
            ... )
        """
        # Use ExperimentManager's add_experiment method
        self.experiment_manager.add_experiment(
            point_dict=inputs,
            output_value=output,
            noise_value=noise,
            iteration=iteration,
            reason=reason
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
    # Staged Experiments (Workflow Management)
    # ============================================================
    
    def add_staged_experiment(self, inputs: Dict[str, Any]) -> None:
        """
        Add an experiment to the staging area (awaiting evaluation).
        
        Staged experiments are typically suggested by acquisition functions
        but not yet evaluated. They can be retrieved, evaluated externally,
        and then added to the dataset with add_experiment().
        
        Args:
            inputs: Dictionary mapping variable names to values
            
        Example:
            > # Generate suggestions and stage them
            > suggestions = session.suggest_next(n_suggestions=3)
            > for point in suggestions.to_dict('records'):
            >     session.add_staged_experiment(point)
            > 
            > # Later, evaluate and add
            > staged = session.get_staged_experiments()
            > for point in staged:
            >     output = run_experiment(**point)
            >     session.add_experiment(point, output=output)
            > session.clear_staged_experiments()
        """
        self.staged_experiments.append(inputs)
        logger.debug(f"Staged experiment: {inputs}")
        self.events.emit('experiment_staged', {'inputs': inputs})
    
    def get_staged_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all staged experiments awaiting evaluation.
        
        Returns:
            List of experiment input dictionaries
        """
        return self.staged_experiments.copy()
    
    def clear_staged_experiments(self) -> int:
        """
        Clear all staged experiments.
        
        Returns:
            Number of experiments cleared
        """
        count = len(self.staged_experiments)
        self.staged_experiments.clear()
        if count > 0:
            logger.info(f"Cleared {count} staged experiments")
            self.events.emit('staged_experiments_cleared', {'count': count})
        return count
    
    def move_staged_to_experiments(self, outputs: List[float], 
                                   noises: Optional[List[float]] = None,
                                   iteration: Optional[int] = None,
                                   reason: Optional[str] = None) -> int:
        """
        Evaluate staged experiments and add them to the dataset in batch.
        
        Convenience method that pairs staged inputs with outputs and adds
        them all to the experiment manager, then clears the staging area.
        
        Args:
            outputs: List of output values (must match length of staged experiments)
            noises: Optional list of measurement uncertainties
            iteration: Iteration number for all experiments (auto-assigned if None)
            reason: Reason for these experiments (e.g., 'Expected Improvement')
            
        Returns:
            Number of experiments added
            
        Example:
            > # Stage some experiments
            > session.add_staged_experiment({'x': 1.0, 'y': 2.0})
            > session.add_staged_experiment({'x': 3.0, 'y': 4.0})
            > 
            > # Evaluate them
            > outputs = [run_experiment(**point) for point in session.get_staged_experiments()]
            > 
            > # Add to dataset and clear staging
            > session.move_staged_to_experiments(outputs, reason='LogEI')
        """
        if len(outputs) != len(self.staged_experiments):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) must match "
                f"number of staged experiments ({len(self.staged_experiments)})"
            )
        
        if noises is not None and len(noises) != len(self.staged_experiments):
            raise ValueError(
                f"Number of noise values ({len(noises)}) must match "
                f"number of staged experiments ({len(self.staged_experiments)})"
            )
        
        # Add each experiment
        for i, inputs in enumerate(self.staged_experiments):
            noise = noises[i] if noises is not None else None
            self.add_experiment(
                inputs=inputs,
                output=outputs[i],
                noise=noise,
                iteration=iteration,
                reason=reason
            )
        
        count = len(self.staged_experiments)
        self.clear_staged_experiments()
        
        logger.info(f"Moved {count} staged experiments to dataset")
        return count
    
    # ============================================================
    # Initial Design Generation
    # ============================================================
    
    def generate_initial_design(
        self,
        method: str = "lhs",
        n_points: int = 10,
        random_seed: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate initial experimental design (Design of Experiments).
        
        Creates a set of experimental conditions to evaluate before starting
        Bayesian optimization. This does NOT add the experiments to the session -
        you must evaluate them and add the results using add_experiment().
        
        Supported methods:
        - 'random': Uniform random sampling
        - 'lhs': Latin Hypercube Sampling (recommended, good space-filling properties)
        - 'sobol': Sobol quasi-random sequences (low discrepancy)
        - 'halton': Halton sequences
        - 'hammersly': Hammersly sequences (low discrepancy)
        
        Args:
            method: Sampling strategy to use
            n_points: Number of points to generate
            random_seed: Random seed for reproducibility
            **kwargs: Additional method-specific parameters:
                - lhs_criterion: For LHS method ("maximin", "correlation", "ratio")
        
        Returns:
            List of dictionaries with variable names and values (no outputs)
        
        Example:
            > # Generate initial design
            > points = session.generate_initial_design('lhs', n_points=10)
            > 
            > # Run experiments and add results
            > for point in points:
            >     output = run_experiment(**point)  # Your experiment function
            >     session.add_experiment(point, output=output)
            > 
            > # Now ready to train model
            > session.train_model()
        """
        if len(self.search_space.variables) == 0:
            raise ValueError(
                "No variables defined in search space. "
                "Use add_variable() to define variables before generating initial design."
            )
        
        from alchemist_core.utils.doe import generate_initial_design
        
        points = generate_initial_design(
            search_space=self.search_space,
            method=method,
            n_points=n_points,
            random_seed=random_seed,
            **kwargs
        )
        
        # Store sampler info in config for audit trail
        self.config['initial_design_method'] = method
        self.config['initial_design_n_points'] = len(points)
        
        logger.info(f"Generated {len(points)} initial design points using {method} method")
        self.events.emit('initial_design_generated', {
            'method': method,
            'n_points': len(points)
        })
        
        # Add a lightweight audit data_locked entry for the initial design metadata
        try:
            extra = {'initial_design_method': method, 'initial_design_n_points': len(points)}
            # Create an empty dataframe snapshot of the planned points
            import pandas as pd
            planned_df = pd.DataFrame(points)
            self.audit_log.lock_data(planned_df, notes=f"Initial design ({method})", extra_parameters=extra)
        except Exception:
            # Audit logging should not block design generation
            logger.debug("Failed to add initial design to audit log")

        return points
    
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
            > results = session.train_model(backend='botorch', kernel='Matern')
            > print(results['metrics'])
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
        
        # Validate and map transform types based on backend
        # BoTorch uses: 'normalize', 'standardize'
        # Sklearn uses: 'minmax', 'standard', 'robust', 'none'
        if self.model_backend == 'sklearn':
            # Map BoTorch transform types to sklearn equivalents
            transform_map = {
                'normalize': 'minmax',      # BoTorch normalize → sklearn minmax
                'standardize': 'standard',  # BoTorch standardize → sklearn standard
                'none': 'none'
            }
            if 'input_transform_type' in kwargs:
                original = kwargs['input_transform_type']
                kwargs['input_transform_type'] = transform_map.get(original, original)
                if original != kwargs['input_transform_type']:
                    logger.debug(f"Mapped input transform '{original}' → '{kwargs['input_transform_type']}' for sklearn")
            if 'output_transform_type' in kwargs:
                original = kwargs['output_transform_type']
                kwargs['output_transform_type'] = transform_map.get(original, original)
                if original != kwargs['output_transform_type']:
                    logger.debug(f"Mapped output transform '{original}' → '{kwargs['output_transform_type']}' for sklearn")
        
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
            
            # Apply sensible defaults for BoTorch if not explicitly overridden
            # Input normalization and output standardization are critical for performance
            if 'input_transform_type' not in kwargs:
                kwargs['input_transform_type'] = 'normalize'
                logger.debug("Auto-applying input normalization for BoTorch model")
            if 'output_transform_type' not in kwargs:
                kwargs['output_transform_type'] = 'standardize'
                logger.debug("Auto-applying output standardization for BoTorch model")
            
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
            
            # Identify categorical variable indices for BoTorch
            # Only compute if not already provided in kwargs (e.g., from UI)
            if 'cat_dims' not in kwargs:
                cat_dims = []
                categorical_var_names = self.search_space.get_categorical_variables()
                if categorical_var_names:
                    # Get the column order from search space
                    all_var_names = self.search_space.get_variable_names()
                    cat_dims = [i for i, name in enumerate(all_var_names) if name in categorical_var_names]
                    logger.debug(f"Categorical dimensions for BoTorch: {cat_dims} (variables: {categorical_var_names})")
                kwargs['cat_dims'] = cat_dims if cat_dims else None
            
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
        
        # Extract kernel name and parameters
        kernel_name = 'unknown'
        if self.model_backend == 'sklearn':
            # First try kernel_options
            if hasattr(self.model, 'kernel_options') and 'kernel_type' in self.model.kernel_options:
                kernel_name = self.model.kernel_options['kernel_type']
                # Add nu parameter for Matern kernels
                if kernel_name == 'Matern' and 'matern_nu' in self.model.kernel_options:
                    json_hyperparams['matern_nu'] = self.model.kernel_options['matern_nu']
            # Then try trained kernel
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'kernel_'):
                kernel_obj = self.model.model.kernel_
                # Navigate through Product/Sum kernels to find base kernel
                if hasattr(kernel_obj, 'k2'):  # Product kernel (Constant * BaseKernel)
                    base_kernel = kernel_obj.k2
                else:
                    base_kernel = kernel_obj
                
                kernel_class = type(base_kernel).__name__
                if 'Matern' in kernel_class:
                    kernel_name = 'Matern'
                    # Extract nu parameter if available
                    if hasattr(base_kernel, 'nu'):
                        json_hyperparams['matern_nu'] = float(base_kernel.nu)
                elif 'RBF' in kernel_class:
                    kernel_name = 'RBF'
                elif 'RationalQuadratic' in kernel_class:
                    kernel_name = 'RationalQuadratic'
                else:
                    kernel_name = kernel_class
        elif self.model_backend == 'botorch':
            if hasattr(self.model, 'cont_kernel_type'):
                kernel_name = self.model.cont_kernel_type
            elif 'kernel_type' in json_hyperparams:
                kernel_name = json_hyperparams['kernel_type']
        
        return {
            'backend': self.model_backend,
            'kernel': kernel_name,
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
            > next_point = session.suggest_next(strategy='EI', goal='maximize')
            > print(next_point)
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
                random_state=self.config['random_state'],
                acq_func_kwargs=kwargs  # Pass xi, kappa, etc. to acquisition function
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
        
        # Store suggestions for UI/API access
        self.last_suggestions = result_df.to_dict('records')
        
        # Cache suggestion info for audit log
        self._last_acquisition_info = {
            'strategy': strategy,
            'goal': goal,
            'parameters': kwargs
        }
        
        return result_df
    
    def find_optimum(self, goal: str = 'maximize', n_grid_points: int = 10000) -> Dict[str, Any]:
        """
        Find the point where the model predicts the optimal value.
        
        Uses a grid search approach to find the point with the best predicted
        value (maximum or minimum) across the search space. This is useful for
        identifying the model's predicted optimum independent of acquisition
        function suggestions.
        
        Args:
            goal: 'maximize' or 'minimize' - which direction to optimize
            n_grid_points: Target number of grid points for search (default: 10000)
        
        Returns:
            Dictionary with:
                - 'x_opt': DataFrame with optimal point (single row)
                - 'value': Predicted value at optimum
                - 'std': Uncertainty (standard deviation) at optimum
        
        Example:
            >>> # Find predicted maximum
            >>> result = session.find_optimum(goal='maximize')
            >>> print(f"Optimum at: {result['x_opt']}")
            >>> print(f"Predicted value: {result['value']:.2f} ± {result['std']:.2f}")
            
            >>> # Find predicted minimum
            >>> result = session.find_optimum(goal='minimize')
            
            >>> # Use finer grid for more accuracy
            >>> result = session.find_optimum(goal='maximize', n_grid_points=50000)
        
        Note:
            - Requires a trained model
            - Uses the same grid-based approach as regret plot for consistency
            - Handles categorical variables correctly through proper encoding
            - Grid size is target value; actual number depends on dimensionality
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")
        
        # Generate prediction grid in ORIGINAL variable space (not encoded)
        grid = self._generate_prediction_grid(n_grid_points)
        
        # Use model's predict method which handles encoding internally
        means, stds = self.predict(grid)
        
        # Find argmax or argmin
        if goal.lower() == 'maximize':
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)
        
        # Extract the optimal point (already in original variable space)
        opt_point_df = grid.iloc[[best_idx]].reset_index(drop=True)
        
        result = {
            'x_opt': opt_point_df,
            'value': float(means[best_idx]),
            'std': float(stds[best_idx])
        }
        
        logger.info(f"Found optimum: {result['x_opt'].to_dict('records')[0]}")
        logger.info(f"Predicted value: {result['value']:.4f} ± {result['std']:.4f}")
        
        return result
    
    # ============================================================
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
            > test_points = pd.DataFrame({
            ...     'temperature': [350, 400],
            ...     'catalyst': ['A', 'B']
            ... })
            > predictions, uncertainties = session.predict(test_points)
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
            > def on_training_done(data):
            ...     print(f"Training completed with R² = {data['metrics']['r2']}")
            > session.on('training_completed', on_training_done)
        """
        self.events.on(event, callback)
    
    # ============================================================
    # Configuration
    # ============================================================
    
    def set_config(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        
        Example:
            > session.set_config(random_state=123, verbose=False)
        """
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    # ============================================================
    # Audit Log & Session Management
    # ============================================================
    
    def lock_data(self, notes: str = "", extra_parameters: Optional[Dict[str, Any]] = None) -> AuditEntry:
        """
        Lock in current experimental data configuration.
        
        Creates an immutable audit log entry capturing the current data state.
        This should be called when you're satisfied with your experimental dataset
        and ready to proceed with modeling.
        
        Args:
            notes: Optional user notes about this data configuration
            
        Returns:
            Created AuditEntry
            
        Example:
            > session.add_experiment({'temp': 100, 'pressure': 5}, output=85.2)
            > session.lock_data(notes="Initial screening dataset")
        """
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Get current experimental data
        df = self.experiment_manager.get_data()
        
        # Lock data in audit log
        entry = self.audit_log.lock_data(
            experiment_data=df,
            notes=notes,
            extra_parameters=extra_parameters
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked data: {len(df)} experiments")
        self.events.emit('data_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def lock_model(self, notes: str = "") -> AuditEntry:
        """
        Lock in current trained model configuration.
        
        Creates an immutable audit log entry capturing the trained model state.
        This should be called when you're satisfied with your model performance
        and ready to use it for acquisition.
        
        Args:
            notes: Optional user notes about this model
            
        Returns:
            Created AuditEntry
            
        Raises:
            ValueError: If no model has been trained
            
        Example:
            > session.train_model(backend='sklearn', kernel='matern')
            > session.lock_model(notes="Best cross-validation performance")
        """
        if self.model is None:
            raise ValueError("No trained model available. Use train_model() first.")
        
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Get model info
        model_info = self.get_model_summary()
        
        # Extract hyperparameters
        hyperparameters = model_info.get('hyperparameters', {})
        
        # Get kernel name from model_info (which extracts it properly)
        kernel_name = model_info.get('kernel', 'unknown')
        
        # Get CV metrics if available - use model_info metrics which are already populated
        cv_metrics = model_info.get('metrics', None)
        if cv_metrics and all(k in cv_metrics for k in ['rmse', 'r2']):
            # Metrics already in correct format from get_model_summary
            pass
        elif hasattr(self.model, 'cv_cached_results') and self.model.cv_cached_results:
            # Fallback to direct access
            cv_metrics = {
                'rmse': float(self.model.cv_cached_results.get('rmse', 0)),
                'r2': float(self.model.cv_cached_results.get('r2', 0)),
                'mae': float(self.model.cv_cached_results.get('mae', 0))
            }
        else:
            cv_metrics = None
        
        # Get current iteration number
        # Use the next iteration number for the model lock so model+acquisition share the same iteration
        iteration = self.experiment_manager._current_iteration + 1
        
        # Include scaler information if available in hyperparameters
        try:
            if hasattr(self.model, 'input_transform_type'):
                hyperparameters['input_transform_type'] = self.model.input_transform_type
            if hasattr(self.model, 'output_transform_type'):
                hyperparameters['output_transform_type'] = self.model.output_transform_type
        except Exception:
            pass

        # Try to extract Matern nu for sklearn models if not already present
        try:
            if self.model_backend == 'sklearn' and 'matern_nu' not in hyperparameters:
                # Try to navigate fitted kernel object for sklearn GaussianProcessRegressor
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'kernel_'):
                    kernel_obj = self.model.model.kernel_
                    base_kernel = getattr(kernel_obj, 'k2', kernel_obj)
                    if hasattr(base_kernel, 'nu'):
                        hyperparameters['matern_nu'] = float(base_kernel.nu)
        except Exception:
            pass

        entry = self.audit_log.lock_model(
            backend=self.model_backend,
            kernel=kernel_name,
            hyperparameters=hyperparameters,
            cv_metrics=cv_metrics,
            iteration=iteration,
            notes=notes
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked model: {self.model_backend}/{model_info.get('kernel')}, iteration {iteration}")
        self.events.emit('model_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def lock_acquisition(self, strategy: str, parameters: Dict[str, Any],
                        suggestions: List[Dict[str, Any]], notes: str = "") -> AuditEntry:
        """
        Lock in acquisition function decision and suggested experiments.
        
        Creates an immutable audit log entry capturing the acquisition decision.
        This should be called when you've reviewed the suggestions and are ready
        to run the recommended experiments.
        
        Args:
            strategy: Acquisition strategy name ('EI', 'PI', 'UCB', etc.)
            parameters: Acquisition function parameters (xi, kappa, etc.)
            suggestions: List of suggested experiment dictionaries
            notes: Optional user notes about this decision
            
        Returns:
            Created AuditEntry
            
        Example:
            > suggestions = session.suggest_next(strategy='EI', n_suggestions=3)
            > session.lock_acquisition(
            ...     strategy='EI',
            ...     parameters={'xi': 0.01, 'goal': 'maximize'},
            ...     suggestions=suggestions,
            ...     notes="Top 3 candidates for next batch"
            ... )
        """
        # Set search space in audit log (once)
        if self.audit_log.search_space_definition is None:
            self.audit_log.set_search_space(self.search_space.variables)
        
        # Increment iteration counter first so this acquisition is logged as the next iteration
        self.experiment_manager._current_iteration += 1
        iteration = self.experiment_manager._current_iteration

        entry = self.audit_log.lock_acquisition(
            strategy=strategy,
            parameters=parameters,
            suggestions=suggestions,
            iteration=iteration,
            notes=notes
        )
        
        self.metadata.update_modified()
        logger.info(f"Locked acquisition: {strategy}, {len(suggestions)} suggestions")
        self.events.emit('acquisition_locked', {'entry': entry.to_dict()})
        
        return entry
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get complete audit log as list of dictionaries.
        
        Returns:
            List of audit entry dictionaries
        """
        return self.audit_log.to_dict()
    
    def export_audit_markdown(self) -> str:
        """
        Export audit log as markdown for publications.
        
        Returns:
            Markdown-formatted audit trail
        """
        # Pass session metadata to markdown exporter so user-entered metadata appears
        try:
            metadata_dict = self.metadata.to_dict()
        except Exception:
            metadata_dict = None

        return self.audit_log.to_markdown(session_metadata=metadata_dict)
    
    def save_session(self, filepath: str):
        """
        Save complete session state to JSON file.
        
        Saves all session data including:
        - Session metadata (name, description, tags)
        - Search space definition
        - Experimental data
        - Trained model state (if available)
        - Complete audit log
        
        Args:
            filepath: Path to save session file (.json extension recommended)
            
        Example:
            > session.save_session("~/ALchemist_Sessions/catalyst_study_nov2025.json")
        """
        filepath = Path(filepath)
        
        # Update audit log's experimental data snapshot to reflect current state
        # This ensures the data table in the audit log markdown is always up-to-date
        current_data = self.experiment_manager.get_data()
        if current_data is not None and len(current_data) > 0:
            self.audit_log.experiment_data = current_data.copy()
        
        # Prepare session data
        session_data = {
            'version': '1.0.0',
            'metadata': self.metadata.to_dict(),
            'audit_log': self.audit_log.to_dict(),
            'search_space': {
                'variables': self.search_space.variables
            },
            'experiments': {
                'data': self.experiment_manager.get_data().to_dict(orient='records'),
                'n_total': len(self.experiment_manager.df)
            },
            'config': self.config
        }
        
        # Add model state if available
        if self.model is not None:
            model_info = self.get_model_summary()
            
            # Get kernel name from model_info which properly extracts it
            kernel_name = model_info.get('kernel', 'unknown')
            
            # Extract kernel parameters if available
            kernel_params = {}
            if self.model_backend == 'sklearn' and hasattr(self.model, 'model'):
                kernel_obj = self.model.model.kernel
                # Extract kernel-specific parameters
                if hasattr(kernel_obj, 'get_params'):
                    kernel_params = kernel_obj.get_params()
            elif self.model_backend == 'botorch':
                # For BoTorch, parameters are in hyperparameters
                hyperparams = model_info.get('hyperparameters', {})
                if 'matern_nu' in hyperparams:
                    kernel_params['nu'] = hyperparams['matern_nu']
            
            session_data['model_config'] = {
                'backend': self.model_backend,
                'kernel': kernel_name,
                'kernel_params': kernel_params,
                'hyperparameters': model_info.get('hyperparameters', {}),
                'metrics': model_info.get('metrics', {})
            }
        
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        self.metadata.update_modified()
        logger.info(f"Saved session to {filepath}")
        self.events.emit('session_saved', {'filepath': str(filepath)})

    def export_session_json(self) -> str:
        """
        Export current session state as a JSON string (no filesystem side-effects for caller).

        Returns:
            JSON string of session data
        """
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            # Use existing save_session logic to write a complete JSON
            self.save_session(tmp_path)

        try:
            with open(tmp_path, 'r') as f:
                content = f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return content
    
    def load_session(self, filepath: str = None, retrain_on_load: bool = True) -> 'OptimizationSession':
        """
        Load session from JSON file.
        
        This method works both as a static method (creating a new session) and as an
        instance method (loading into existing session):
        
        Static usage (returns new session):
            > session = OptimizationSession.load_session("my_session.json")
        
        Instance usage (loads into existing session):
            > session = OptimizationSession()
            > session.load_session("my_session.json")
            > # session.experiment_manager.df is now populated
        
        Args:
            filepath: Path to session file (required when called as static method,
                     can be self when called as instance method)
            retrain_on_load: Whether to retrain model if config exists (default: True)
            
        Returns:
            OptimizationSession (new or modified instance)
        """
        # Detect if called as instance method or static method
        # When called as static method: self is actually the filepath string
        # When called as instance method: self is an OptimizationSession instance
        if isinstance(self, OptimizationSession):
            # Instance method: load into this session
            if filepath is None:
                raise ValueError("filepath is required when calling as instance method")
            
            # Load from static implementation
            loaded_session = OptimizationSession._load_session_impl(filepath, retrain_on_load)
            
            # Copy all attributes from loaded session to this instance
            self.search_space = loaded_session.search_space
            self.experiment_manager = loaded_session.experiment_manager
            self.metadata = loaded_session.metadata
            self.audit_log = loaded_session.audit_log
            self.config = loaded_session.config
            self.model = loaded_session.model
            self.model_backend = loaded_session.model_backend
            self.acquisition = loaded_session.acquisition
            self.staged_experiments = loaded_session.staged_experiments
            self.last_suggestions = loaded_session.last_suggestions
            
            # Don't copy events emitter - keep the original
            logger.info(f"Loaded session data into current instance from {filepath}")
            self.events.emit('session_loaded', {'filepath': str(filepath)})
            
            return self
        else:
            # Static method: self is actually the filepath, retrain_on_load is in filepath param
            actual_filepath = self
            actual_retrain = filepath if filepath is not None else True
            return OptimizationSession._load_session_impl(actual_filepath, actual_retrain)
    
    @staticmethod
    def _load_session_impl(filepath: str, retrain_on_load: bool = True) -> 'OptimizationSession':
        """
        Internal implementation for loading session from file.
        This always creates and returns a new session.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        # Check version compatibility
        version = session_data.get('version', '1.0.0')
        if not version.startswith('1.'):
            logger.warning(f"Session file version {version} may not be fully compatible")
        
        # Create session
        session = OptimizationSession()
        
        # Restore metadata
        if 'metadata' in session_data:
            session.metadata = SessionMetadata.from_dict(session_data['metadata'])
        
        # Restore audit log
        if 'audit_log' in session_data:
            session.audit_log.from_dict(session_data['audit_log'])
        
        # Restore search space
        if 'search_space' in session_data:
            for var in session_data['search_space']['variables']:
                session.search_space.add_variable(
                    var['name'],
                    var['type'],
                    **{k: v for k, v in var.items() if k not in ['name', 'type']}
                )
        
        # Restore experimental data
        if 'experiments' in session_data and session_data['experiments']['data']:
            df = pd.DataFrame(session_data['experiments']['data'])
            
            # Metadata columns to exclude from inputs
            metadata_cols = {'Output', 'Noise', 'Iteration', 'Reason'}
            
            # Add experiments one by one
            for _, row in df.iterrows():
                # Only include actual input variables, not metadata
                inputs = {col: row[col] for col in df.columns if col not in metadata_cols}
                output = row.get('Output')
                noise = row.get('Noise') if pd.notna(row.get('Noise')) else None
                iteration = row.get('Iteration') if pd.notna(row.get('Iteration')) else None
                reason = row.get('Reason') if pd.notna(row.get('Reason')) else None
                
                session.add_experiment(inputs, output, noise=noise, iteration=iteration, reason=reason)
        
        # Restore config
        if 'config' in session_data:
            session.config.update(session_data['config'])
        
        # Auto-retrain model if configuration exists (optional)
        if 'model_config' in session_data and retrain_on_load:
            model_config = session_data['model_config']
            logger.info(f"Auto-retraining model: {model_config['backend']} with {model_config.get('kernel', 'default')} kernel")
            
            try:
                # Trigger model training with saved configuration
                session.train_model(
                    backend=model_config['backend'],
                    kernel=model_config.get('kernel', 'Matern'),
                    kernel_params=model_config.get('kernel_params', {})
                )
                logger.info("Model retrained successfully")
                session.events.emit('model_retrained', {'backend': model_config['backend']})
            except Exception as e:
                logger.warning(f"Failed to retrain model: {e}")
                session.events.emit('model_retrain_failed', {'error': str(e)})
        
        logger.info(f"Loaded session from {filepath}")
        session.events.emit('session_loaded', {'filepath': str(filepath)})
        
        return session
    
    def update_metadata(self, name: Optional[str] = None, 
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       author: Optional[str] = None):
        """
        Update session metadata.
        
        Args:
            name: New session name (optional)
            description: New description (optional)
            tags: New tags (optional)
            
        Example:
            > session.update_metadata(
            ...     name="Catalyst Screening - Final",
            ...     description="Optimized Pt/Pd ratios",
            ...     tags=["catalyst", "platinum", "palladium", "final"]
            ... )
        """
        if name is not None:
            self.metadata.name = name
        if description is not None:
            self.metadata.description = description
        if author is not None:
            # Backwards compatible: store author if provided
            setattr(self.metadata, 'author', author)
        if tags is not None:
            self.metadata.tags = tags
        
        self.metadata.update_modified()
        logger.info("Updated session metadata")
        self.events.emit('metadata_updated', self.metadata.to_dict())
    
    # ============================================================
    # Legacy Configuration
    # ============================================================
    
    def set_config(self, **kwargs) -> None:
        """
        Update session configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        
        Example:
            > session.set_config(random_state=123, verbose=False)
        """
        self.config.update(kwargs)
        logger.info(f"Updated config: {kwargs}")
    
    # ============================================================
    # Visualization Methods (Notebook Support)
    # ============================================================
    
    def _check_matplotlib(self) -> None:
        """Check if matplotlib is available for plotting."""
        if _HAS_VISUALIZATION:
            check_matplotlib()  # Use visualization module's check
        elif not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization methods. "
                "Install with: pip install matplotlib"
            )
    
    def _check_model_trained(self) -> None:
        """Check if model is trained before plotting."""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_model() before creating visualizations."
            )
    
    def _check_cv_results(self, use_calibrated: bool = False) -> Dict[str, np.ndarray]:
        """
        Get CV results from model, handling both calibrated and uncalibrated.
        
        Args:
            use_calibrated: Whether to use calibrated results if available
            
        Returns:
            Dictionary with y_true, y_pred, y_std arrays
        """
        self._check_model_trained()
        
        # Check for calibrated results first if requested
        if use_calibrated and hasattr(self.model, 'cv_cached_results_calibrated'):
            if self.model.cv_cached_results_calibrated is not None:
                return self.model.cv_cached_results_calibrated
        
        # Fall back to uncalibrated results
        if hasattr(self.model, 'cv_cached_results'):
            if self.model.cv_cached_results is not None:
                return self.model.cv_cached_results
        
        raise ValueError(
            "No CV results available. Model must be trained with cross-validation."
        )
    
    def plot_parity(
        self,
        use_calibrated: bool = False,
        sigma_multiplier: float = 1.96,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        show_metrics: bool = True,
        show_error_bars: bool = True
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create parity plot of actual vs predicted values from cross-validation.
        
        This plot shows how well the model's predictions match the actual experimental
        values, with optional error bars indicating prediction uncertainty.
        
        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            sigma_multiplier: Error bar size (1.96 = 95% CI, 1.0 = 68% CI, 2.58 = 99% CI)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated with metrics)
            show_metrics: Include RMSE, MAE, R² in title
            show_error_bars: Display uncertainty error bars
        
        Returns:
            matplotlib Figure object (displays inline in Jupyter)
        
        Example:
            >>> fig = session.plot_parity()
            >>> fig.show()  # In notebooks, displays automatically
            
            >>> # With custom styling
            >>> fig = session.plot_parity(
            ...     sigma_multiplier=2.58,  # 99% confidence interval
            ...     figsize=(10, 8),
            ...     dpi=150
            ... )
            >>> fig.savefig('parity.png', bbox_inches='tight')
        
        Note:
            Requires model to be trained with cross-validation (default behavior).
            Error bars are only shown if model provides uncertainty estimates.
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        # Get CV results
        cv_results = self._check_cv_results(use_calibrated)
        y_true = cv_results['y_true']
        y_pred = cv_results['y_pred']
        y_std = cv_results.get('y_std', None)
        
        # Delegate to visualization module
        fig, ax = create_parity_plot(
            y_true=y_true,
            y_pred=y_pred,
            y_std=y_std,
            sigma_multiplier=sigma_multiplier,
            figsize=figsize,
            dpi=dpi,
            title=title,
            show_metrics=show_metrics,
            show_error_bars=show_error_bars
        )
        
        logger.info("Generated parity plot")
        return fig
    
    def plot_slice(
        self,
        x_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        n_points: int = 100,
        show_uncertainty: Union[bool, List[float]] = True,
        show_experiments: bool = True,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 1D slice plot showing model predictions along one variable.
        
        Visualizes how the model's prediction changes as one variable is varied
        while all other variables are held constant. Shows prediction mean and
        optional uncertainty bands.
        
        Args:
            x_var: Variable name to vary along X axis (must be 'real' or 'integer')
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            n_points: Number of points to evaluate along the slice
            show_uncertainty: Show uncertainty bands. Can be:
                - True: Show ±1σ and ±2σ bands (default)
                - False: No uncertainty bands
                - List[float]: Custom sigma values, e.g., [1.0, 2.0, 3.0] for ±1σ, ±2σ, ±3σ
            show_experiments: Plot experimental data points as scatter
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: auto-generated)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # With custom uncertainty bands (±1σ, ±2σ, ±3σ)
            >>> fig = session.plot_slice(
            ...     'temperature',
            ...     fixed_values={'pressure': 5.0, 'catalyst': 'Pt'},
            ...     show_uncertainty=[1.0, 2.0, 3.0]
            ... )
            >>> fig.savefig('slice.png', dpi=300)
        
        Note:
            - Model must be trained before plotting
            - Uncertainty bands require model to support std predictions
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable info
        var_names = self.search_space.get_variable_names()
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        
        # Get x variable definition
        x_var_def = next(v for v in self.search_space.variables if v['name'] == x_var)
        
        if x_var_def['type'] not in ['real', 'integer']:
            raise ValueError(f"Variable '{x_var}' must be 'real' or 'integer' type for slice plot")
        
        # Create range for x variable
        x_min, x_max = x_var_def['min'], x_var_def['max']
        x_values = np.linspace(x_min, x_max, n_points)
        
        # Build prediction data with fixed values
        slice_data = {x_var: x_values}
        
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name == x_var:
                continue
            
            if var_name in fixed_values:
                slice_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    slice_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    slice_data[var_name] = var['values'][0]
        
        # Create DataFrame with correct column order
        if hasattr(self.model, 'original_feature_names') and self.model.original_feature_names:
            column_order = self.model.original_feature_names
        else:
            column_order = self.search_space.get_variable_names()
        
        slice_df = pd.DataFrame(slice_data, columns=column_order)
        
        # Get predictions with uncertainty
        predictions, std = self.predict(slice_df)
        
        # Prepare experimental data for plotting
        exp_x = None
        exp_y = None
        if show_experiments and len(self.experiment_manager.df) > 0:
            df = self.experiment_manager.df
            
            # Filter points that match the fixed values
            mask = pd.Series([True] * len(df))
            for var_name, fixed_val in fixed_values.items():
                if var_name in df.columns:
                    # For numerical values, allow small tolerance
                    if isinstance(fixed_val, (int, float)):
                        mask &= np.abs(df[var_name] - fixed_val) < 1e-6
                    else:
                        mask &= df[var_name] == fixed_val
            
            if mask.any():
                filtered_df = df[mask]
                exp_x = filtered_df[x_var].values
                exp_y = filtered_df['Output'].values
        
        # Generate title if not provided
        if title is None:
            if fixed_values:
                fixed_str = ', '.join([f'{k}={v}' for k, v in fixed_values.items()])
                title = f"1D Slice: {x_var}\n({fixed_str})"
            else:
                title = f"1D Slice: {x_var}"
        
        # Delegate to visualization module
        # Handle show_uncertainty parameter conversion
        sigma_bands = None
        if show_uncertainty is not False:
            if isinstance(show_uncertainty, bool):
                # Default: [1.0, 2.0]
                sigma_bands = [1.0, 2.0] if show_uncertainty else None
            else:
                # Custom list of sigma values
                sigma_bands = show_uncertainty
        
        fig, ax = create_slice_plot(
            x_values=x_values,
            predictions=predictions,
            x_var=x_var,
            std=std,
            sigma_bands=sigma_bands,
            exp_x=exp_x,
            exp_y=exp_y,
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info(f"Generated 1D slice plot for {x_var}")
        return fig
    
    def plot_contour(
        self,
        x_var: str,
        y_var: str,
        fixed_values: Optional[Dict[str, Any]] = None,
        grid_resolution: int = 50,
        show_experiments: bool = True,
        show_suggestions: bool = False,
        cmap: str = 'viridis',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create 2D contour plot of model predictions over a variable space.
        
        Visualizes the model's predicted response surface by varying two variables
        while holding others constant. Useful for understanding variable interactions
        and identifying optimal regions.
        
        Args:
            x_var: Variable name for X axis (must be 'real' type)
            y_var: Variable name for Y axis (must be 'real' type)
            fixed_values: Dict of {var_name: value} for other variables.
                         If not provided, uses midpoint for real/integer,
                         first category for categorical.
            grid_resolution: Grid density (NxN points)
            show_experiments: Plot experimental data points as scatter
            show_suggestions: Plot last suggested points (if available)
            cmap: Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'plasma')
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: "Contour Plot of Model Predictions")
        
        Returns:
            matplotlib Figure object (displays inline in Jupyter)
        
        Example:
            >>> # Basic contour plot
            >>> fig = session.plot_contour('temperature', 'pressure')
            
            >>> # With fixed values for other variables
            >>> fig = session.plot_contour(
            ...     'temperature', 'pressure',
            ...     fixed_values={'catalyst': 'Pt', 'flow_rate': 50},
            ...     cmap='coolwarm',
            ...     grid_resolution=100
            ... )
            >>> fig.savefig('contour.png', dpi=300, bbox_inches='tight')
        
        Note:
            - Requires at least 2 'real' type variables
            - Model must be trained before plotting
            - Categorical variables are automatically encoded using model's encoding
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        if fixed_values is None:
            fixed_values = {}
        
        # Get variable names
        var_names = self.search_space.get_variable_names()
        
        # Validate variables exist
        if x_var not in var_names:
            raise ValueError(f"Variable '{x_var}' not in search space")
        if y_var not in var_names:
            raise ValueError(f"Variable '{y_var}' not in search space")
        
        # Get variable info (search_space.variables is a list)
        x_var_info = next(v for v in self.search_space.variables if v['name'] == x_var)
        y_var_info = next(v for v in self.search_space.variables if v['name'] == y_var)
        
        if x_var_info['type'] != 'real':
            raise ValueError(f"X variable '{x_var}' must be 'real' type, got '{x_var_info['type']}'")
        if y_var_info['type'] != 'real':
            raise ValueError(f"Y variable '{y_var}' must be 'real' type, got '{y_var_info['type']}'")
        
        # Get bounds
        x_bounds = (x_var_info['min'], x_var_info['max'])
        y_bounds = (y_var_info['min'], y_var_info['max'])
        
        # Create meshgrid
        x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
        X_grid, Y_grid = np.meshgrid(x, y)
        
        # Build prediction dataframe with ALL variables in proper order
        # Start with grid variables
        grid_data = {
            x_var: X_grid.ravel(),
            y_var: Y_grid.ravel()
        }
        
        # Add fixed values for other variables
        for var in self.search_space.variables:
            var_name = var['name']
            if var_name in [x_var, y_var]:
                continue
            
            if var_name in fixed_values:
                grid_data[var_name] = fixed_values[var_name]
            else:
                # Use default value
                if var['type'] in ['real', 'integer']:
                    grid_data[var_name] = (var['min'] + var['max']) / 2
                elif var['type'] == 'categorical':
                    grid_data[var_name] = var['values'][0]
        
        # Create DataFrame with columns in the same order as original training data
        # This is critical for model preprocessing to work correctly
        if hasattr(self.model, 'original_feature_names') and self.model.original_feature_names:
            # Use the model's stored column order
            column_order = self.model.original_feature_names
        else:
            # Fall back to search space order
            column_order = self.search_space.get_variable_names()
        
        grid_df = pd.DataFrame(grid_data, columns=column_order)
        
        # Get predictions - use Session's predict method for consistency
        predictions, _ = self.predict(grid_df)
        
        # Reshape to grid
        predictions_grid = predictions.reshape(X_grid.shape)
        
        # Prepare experimental data for overlay
        exp_x = None
        exp_y = None
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns:
                exp_x = exp_df[x_var].values
                exp_y = exp_df[y_var].values
        
        # Prepare suggestion data for overlay
        sugg_x = None
        sugg_y = None
        if show_suggestions and len(self.last_suggestions) > 0:
            # last_suggestions is a DataFrame
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            
            if x_var in sugg_df.columns and y_var in sugg_df.columns:
                sugg_x = sugg_df[x_var].values
                sugg_y = sugg_df[y_var].values
        
        # Delegate to visualization module
        fig, ax, cbar = create_contour_plot(
            x_grid=X_grid,
            y_grid=Y_grid,
            predictions_grid=predictions_grid,
            x_var=x_var,
            y_var=y_var,
            exp_x=exp_x,
            exp_y=exp_y,
            suggest_x=sugg_x,
            suggest_y=sugg_y,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            title=title or "Contour Plot of Model Predictions"
        )
        
        logger.info(f"Generated contour plot for {x_var} vs {y_var}")
        # Return figure only for backwards compatibility (colorbar accessible via fig/ax)
        return fig
    
    def plot_metrics(
        self,
        metric: Literal['rmse', 'mae', 'r2', 'mape'] = 'rmse',
        cv_splits: int = 5,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        use_cached: bool = True
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot cross-validation metrics as a function of training set size.
        
        Shows how model performance improves as more experimental data is added.
        This evaluates the model at each training set size from 5 observations up to
        the current total, providing insight into data efficiency and whether more
        experiments are needed.
        
        Args:
            metric: Which metric to plot ('rmse', 'mae', 'r2', or 'mape')
            cv_splits: Number of cross-validation folds (default: 5)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            use_cached: Use cached metrics if available (default: True)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Plot RMSE vs number of experiments
            >>> fig = session.plot_metrics('rmse')
            
            >>> # Plot R² to see improvement
            >>> fig = session.plot_metrics('r2')
            
            >>> # Force recomputation of metrics
            >>> fig = session.plot_metrics('rmse', use_cached=False)
        
        Note:
            Calls model.evaluate() if metrics not cached, which can be computationally
            expensive for large datasets. Set use_cached=False to force recomputation.
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        # Need at least 5 observations for CV
        n_total = len(self.experiment_manager.df)
        if n_total < 5:
            raise ValueError(f"Need at least 5 observations for metrics plot (have {n_total})")
        
        # Check for cached metrics first
        cache_key = f'_cached_cv_metrics_{cv_splits}'
        if use_cached and hasattr(self.model, cache_key):
            cv_metrics = getattr(self.model, cache_key)
            logger.info(f"Using cached CV metrics for {metric.upper()}")
        else:
            # Call model's evaluate method to get metrics over training sizes
            logger.info(f"Computing {metric.upper()} over training set sizes (this may take a moment)...")
            cv_metrics = self.model.evaluate(
                self.experiment_manager,
                cv_splits=cv_splits,
                debug=False
            )
            # Cache the results
            setattr(self.model, cache_key, cv_metrics)
        
        # Extract the requested metric
        metric_key_map = {
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'mape': 'MAPE'
        }
        
        if metric not in metric_key_map:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(metric_key_map.keys())}")
        
        metric_key = metric_key_map[metric]
        metric_values = cv_metrics.get(metric_key, [])
        
        if not metric_values:
            raise RuntimeError(f"Model did not return {metric_key} values from evaluate()")
        
        # X-axis: training set sizes (starts at 5)
        x_range = np.arange(5, len(metric_values) + 5)
        metric_array = np.array(metric_values)
        
        # Delegate to visualization module
        fig, ax = create_metrics_plot(
            training_sizes=x_range,
            metric_values=metric_array,
            metric_name=metric,
            figsize=figsize,
            dpi=dpi
        )
        
        logger.info(f"Generated {metric} metrics plot with {len(metric_values)} points")
        return fig
    
    def plot_qq(
        self,
        use_calibrated: bool = False,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create Q-Q (quantile-quantile) plot for model residuals normality check.
        
        Visualizes whether the model's prediction errors (residuals) follow a normal
        distribution. Points should lie close to the diagonal line if residuals are
        normally distributed, which is an assumption of Gaussian Process models.
        
        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: "Q-Q Plot: Residuals Normality Check")
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Check if residuals are normally distributed
            >>> fig = session.plot_qq()
            >>> fig.savefig('qq_plot.png')
            
            >>> # Use calibrated predictions if available
            >>> fig = session.plot_qq(use_calibrated=True)
        
        Note:
            - Requires model to be trained with cross-validation
            - Significant deviations from the diagonal suggest non-normal residuals
            - Useful for diagnosing model assumptions and identifying outliers
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        # Get CV results
        cv_results = self._check_cv_results(use_calibrated)
        y_true = cv_results['y_true']
        y_pred = cv_results['y_pred']
        y_std = cv_results.get('y_std', None)
        
        # Compute standardized residuals (z-scores)
        residuals = y_true - y_pred
        if y_std is not None and len(y_std) > 0:
            z_scores = residuals / y_std
        else:
            # Fallback: standardize by residual standard deviation
            z_scores = residuals / np.std(residuals)
        
        # Delegate to visualization module
        fig, ax = create_qq_plot(
            z_scores=z_scores,
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info("Generated Q-Q plot for residuals")
        return fig
    
    def plot_calibration(
        self,
        use_calibrated: bool = False,
        n_bins: int = 10,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Create calibration plot showing reliability of uncertainty estimates.
        
        Compares predicted confidence intervals to actual coverage. For well-calibrated
        models, a 68% confidence interval should contain ~68% of true values, 95% should
        contain ~95%, etc. This plot helps diagnose if the model's uncertainty estimates
        are too narrow (overconfident) or too wide (underconfident).
        
        Args:
            use_calibrated: Use calibrated uncertainty estimates if available
            n_bins: Number of bins for grouping predictions (default: 10)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom title (default: "Calibration Plot: Uncertainty Reliability")
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Check if uncertainty estimates are reliable
            >>> fig = session.plot_calibration()
            >>> fig.savefig('calibration_plot.png')
            
            >>> # With more bins for finer resolution
            >>> fig = session.plot_calibration(n_bins=20)
        
        Note:
            - Requires model to be trained with cross-validation and provide uncertainties
            - Points above diagonal = model is underconfident (intervals too wide)
            - Points below diagonal = model is overconfident (intervals too narrow)
            - Well-calibrated models have points close to the diagonal
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        # Get CV results
        cv_results = self._check_cv_results(use_calibrated)
        y_true = cv_results['y_true']
        y_pred = cv_results['y_pred']
        y_std = cv_results.get('y_std', None)
        
        if y_std is None:
            raise ValueError(
                "Model does not provide uncertainty estimates (y_std). "
                "Calibration plot requires uncertainty predictions."
            )
        
        # Compute calibration curve data
        from scipy import stats
        
        # Compute empirical coverage for a range of nominal probabilities
        nominal_probs = np.arange(0.10, 1.00, 0.05)
        empirical_coverage = []
        
        for prob in nominal_probs:
            # Convert probability to sigma multiplier
            sigma = stats.norm.ppf((1 + prob) / 2)
            
            # Compute empirical coverage at this sigma level
            lower_bound = y_pred - sigma * y_std
            upper_bound = y_pred + sigma * y_std
            within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            empirical_coverage.append(np.mean(within_interval))
        
        empirical_coverage = np.array(empirical_coverage)
        
        # Delegate to visualization module
        fig, ax = create_calibration_plot(
            nominal_probs=nominal_probs,
            empirical_coverage=empirical_coverage,
            figsize=figsize,
            dpi=dpi,
            title=title or "Calibration Plot: Uncertainty Reliability"
        )
        
        logger.info("Generated calibration plot for uncertainty estimates")
        return fig
    
    def plot_regret(
        self,
        goal: Literal['maximize', 'minimize'] = 'maximize',
        include_predictions: bool = True,
        show_cumulative: bool = False,
        backend: Optional[str] = None,
        kernel: Optional[str] = None,
        n_grid_points: int = 1000,
        sigma_bands: Optional[List[float]] = None,
        start_iteration: int = 5,
        reuse_hyperparameters: bool = True,
        use_calibrated_uncertainty: bool = False,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot optimization progress (regret curve).
        
        Shows the best value found as a function of iteration number. The curve
        displays cumulative best results and all observed values, providing insight
        into optimization convergence.
        
        A flattening curve indicates the optimization is converging (no further
        improvements being found). This is useful for determining when to stop
        an optimization campaign.
        
        Optionally overlays the model's predicted best value (max posterior mean)
        with uncertainty bands, showing where the model believes the optimum lies.
        
        Args:
            goal: 'maximize' or 'minimize' - which direction to optimize
            include_predictions: Whether to overlay max(posterior mean) with uncertainty bands
            backend: Model backend ('sklearn' or 'botorch'). Uses session default if None.
            kernel: Kernel type ('RBF', 'Matern', etc.). Uses session default if None.
            n_grid_points: Number of points to evaluate for finding max posterior mean
            sigma_bands: List of sigma values for uncertainty bands (e.g., [1.0, 2.0])
            start_iteration: First iteration to compute predictions (needs enough data)
            reuse_hyperparameters: Reuse final model's hyperparameters (faster, default True)
            use_calibrated_uncertainty: If True, apply calibration to uncertainties. If False,
                use raw GP uncertainties. Default False recommended for convergence assessment
                since raw uncertainties better reflect model's internal convergence. Set True
                for realistic prediction intervals that account for model miscalibration.
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom plot title (auto-generated if None)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # For a maximization problem
            >>> fig = session.plot_regret(goal='maximize')
            >>> fig.savefig('optimization_progress.png')
            
            >>> # With custom uncertainty bands (±1σ, ±2σ)
            >>> fig = session.plot_regret(goal='maximize', sigma_bands=[1.0, 2.0])
            
            >>> # For a minimization problem
            >>> fig = session.plot_regret(goal='minimize')
        
        Note:
            - Requires at least 2 experiments
            - Also known as "simple regret" or "incumbent trajectory"
            - Best used to visualize overall optimization progress
        """
        self._check_matplotlib()
        
        # Check we have experiments
        n_exp = len(self.experiment_manager.df)
        if n_exp < 2:
            raise ValueError(f"Need at least 2 experiments for regret plot (have {n_exp})")
        
        # Get observed values and create iteration array (1-based for user clarity)
        # Use first target column (single-objective optimization)
        target_col = self.experiment_manager.target_columns[0]
        observed_values = self.experiment_manager.df[target_col].values
        iterations = np.arange(1, n_exp + 1)  # 1-based: [1, 2, 3, ..., n]
        
        # Compute posterior predictions if requested
        predicted_means = None
        predicted_stds = None
        
        if include_predictions and n_exp >= start_iteration:
            try:
                predicted_means, predicted_stds = self._compute_posterior_predictions(
                    goal=goal,
                    backend=backend,
                    kernel=kernel,
                    n_grid_points=n_grid_points,
                    start_iteration=start_iteration,
                    reuse_hyperparameters=reuse_hyperparameters,
                    use_calibrated_uncertainty=use_calibrated_uncertainty
                )
            except Exception as e:
                logger.warning(f"Could not compute posterior predictions: {e}. Plotting observations only.")
        
        # Import visualization function
        from alchemist_core.visualization.plots import create_regret_plot
        
        # Delegate to visualization module
        fig, ax = create_regret_plot(
            iterations=iterations,
            observed_values=observed_values,
            show_cumulative=show_cumulative,
            goal=goal,
            predicted_means=predicted_means,
            predicted_stds=predicted_stds,
            sigma_bands=sigma_bands,
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info(f"Generated regret plot with {n_exp} experiments")
        return fig
    
    def _generate_prediction_grid(self, n_grid_points: int) -> pd.DataFrame:
        """
        Generate grid of test points across search space for predictions.
        
        Args:
            n_grid_points: Target number of grid points (actual number depends on dimensionality)
        
        Returns:
            DataFrame with columns for each variable
        """
        grid_1d = []
        var_names = []
        
        for var in self.search_space.variables:
            var_names.append(var['name'])
            
            if var['type'] == 'real':
                # Continuous: linspace
                n_per_dim = int(n_grid_points ** (1/len(self.search_space.variables)))
                grid_1d.append(np.linspace(var['min'], var['max'], n_per_dim))
            elif var['type'] == 'integer':
                # Integer: range of integers
                n_per_dim = int(n_grid_points ** (1/len(self.search_space.variables)))
                grid_1d.append(np.linspace(var['min'], var['max'], n_per_dim).astype(int))
            else:
                # Categorical: use actual category values
                grid_1d.append(var['values'])
        
        # Generate test points using Cartesian product
        from itertools import product
        X_test_tuples = list(product(*grid_1d))
        
        # Convert to DataFrame with proper variable names and types
        grid = pd.DataFrame(X_test_tuples, columns=var_names)
        
        # Ensure correct dtypes for categorical variables
        for var in self.search_space.variables:
            if var['type'] == 'categorical':
                grid[var['name']] = grid[var['name']].astype(str)
        
        return grid
    
    def _compute_posterior_predictions(
        self,
        goal: str,
        backend: Optional[str],
        kernel: Optional[str],
        n_grid_points: int,
        start_iteration: int,
        reuse_hyperparameters: bool,
        use_calibrated_uncertainty: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute max(posterior mean) and corresponding std at each iteration.
        
        Helper method for regret plot to overlay model predictions with uncertainty.
        
        IMPORTANT: When reuse_hyperparameters=True, this uses the final model's 
        hyperparameters for ALL iterations by creating fresh GP models with those
        hyperparameters and subsets of data. This avoids numerical instability from
        repeated MLE optimization.
        
        Returns:
            Tuple of (predicted_means, predicted_stds) arrays, same length as n_experiments
        """
        n_exp = len(self.experiment_manager.df)
        
        # Initialize arrays (NaN for iterations before start_iteration)
        predicted_means = np.full(n_exp, np.nan)
        predicted_stds = np.full(n_exp, np.nan)
        
        # Determine backend and kernel
        if backend is None:
            if self.model is None or not self.model.is_trained:
                raise ValueError("No trained model in session. Train a model first or specify backend/kernel.")
            backend = self.model_backend
        
        if kernel is None:
            if self.model is None or not self.model.is_trained:
                raise ValueError("No trained model in session. Train a model first or specify backend/kernel.")
            if backend == 'sklearn':
                kernel = self.model.kernel_options.get('kernel_type', 'RBF')
            elif backend == 'botorch':
                # BoTorchModel stores kernel type in cont_kernel_type
                kernel = getattr(self.model, 'cont_kernel_type', 'Matern')
        
        # Extract optimized state_dict for botorch or kernel params for sklearn
        optimized_state_dict = None
        optimized_kernel_params = None
        if reuse_hyperparameters and self.model is not None and self.model.is_trained:
            if backend == 'sklearn':
                optimized_kernel_params = self.model.optimized_kernel.get_params()
            elif backend == 'botorch':
                # Store the fitted state dict from the final model
                optimized_state_dict = self.model.fitted_state_dict
        
        # Generate grid for predictions
        grid = self._generate_prediction_grid(n_grid_points)
        
        # Get full dataset
        full_df = self.experiment_manager.df
        target_col = self.experiment_manager.target_columns[0]
        
        # Suppress INFO logging for temp sessions to avoid spam
        import logging
        original_session_level = logger.level
        original_model_level = logging.getLogger('alchemist_core.models.botorch_model').level
        logger.setLevel(logging.WARNING)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(logging.WARNING)
        
        # Loop through iterations
        for i in range(start_iteration, n_exp + 1):
            try:
                # Create temporary session with subset of data
                temp_session = OptimizationSession()
                
                # Directly assign search space to avoid logging spam
                temp_session.search_space = self.search_space
                temp_session.experiment_manager.set_search_space(self.search_space)
                
                # Add subset of experiments
                for idx in range(i):
                    row = full_df.iloc[idx]
                    inputs = {var['name']: row[var['name']] for var in self.experiment_manager.search_space.variables}
                    temp_session.add_experiment(inputs, output=row[target_col])
                
                # Train model on subset using SAME approach for all iterations
                if backend == 'sklearn':
                    # Create model instance
                    from alchemist_core.models.sklearn_model import SklearnModel
                    temp_model = SklearnModel(kernel_options={'kernel_type': kernel})
                    
                    if reuse_hyperparameters and optimized_kernel_params is not None:
                        # Override n_restarts to disable optimization
                        temp_model.n_restarts_optimizer = 0
                        temp_model._custom_optimizer = None
                        # Store the optimized kernel to use
                        from sklearn.base import clone
                        temp_model._reuse_kernel = clone(self.model.optimized_kernel)
                    
                    # Attach model and train
                    temp_session.model = temp_model
                    temp_session.model_backend = 'sklearn'
                    
                    # Train WITHOUT recomputing calibration (if reusing hyperparameters)
                    if reuse_hyperparameters:
                        temp_model.train(temp_session.experiment_manager, calibrate_uncertainty=False)
                        # Transfer calibration factor from final model
                        if hasattr(self.model, 'calibration_factor'):
                            temp_model.calibration_factor = self.model.calibration_factor
                            # Enable calibration only if user requested calibrated uncertainties
                            temp_model.calibration_enabled = use_calibrated_uncertainty
                    else:
                        temp_model.train(temp_session.experiment_manager)
                    
                    # Verify model was trained
                    if not temp_model.is_trained:
                        raise ValueError(f"Model training failed at iteration {i}")
                    if temp_session.model is None:
                        raise ValueError(f"temp_session.model is None after training at iteration {i}")
                    
                elif backend == 'botorch':
                    # For BoTorch: create a fresh model and load the fitted hyperparameters
                    from alchemist_core.models.botorch_model import BoTorchModel
                    import torch
                    
                    # Create model instance with same configuration as original model
                    kernel_opts = {'cont_kernel_type': kernel}
                    if hasattr(self.model, 'matern_nu'):
                        kernel_opts['matern_nu'] = self.model.matern_nu
                    
                    temp_model = BoTorchModel(
                        kernel_options=kernel_opts,
                        input_transform_type=self.model.input_transform_type if hasattr(self.model, 'input_transform_type') else 'normalize',
                        output_transform_type=self.model.output_transform_type if hasattr(self.model, 'output_transform_type') else 'standardize'
                    )
                    
                    # Train model on subset (this creates the GP with subset of data)
                    # Disable calibration computation if reusing hyperparameters
                    if reuse_hyperparameters:
                        temp_model.train(temp_session.experiment_manager, calibrate_uncertainty=False)
                    else:
                        temp_model.train(temp_session.experiment_manager)
                    
                    # Apply optimized hyperparameters from final model to trained subset model
                    # Only works for simple kernel structures (no categorical variables)
                    if reuse_hyperparameters and optimized_state_dict is not None:
                        try:
                            with torch.no_grad():
                                # Extract hyperparameters from final model
                                # This only works for ScaleKernel(base_kernel), not AdditiveKernel
                                final_lengthscale = self.model.model.covar_module.base_kernel.lengthscale.detach().clone()
                                final_outputscale = self.model.model.covar_module.outputscale.detach().clone()
                                final_noise = self.model.model.likelihood.noise.detach().clone()
                                
                                # Set hyperparameters in temp model (trained on subset)
                                temp_model.model.covar_module.base_kernel.lengthscale = final_lengthscale
                                temp_model.model.covar_module.outputscale = final_outputscale
                                temp_model.model.likelihood.noise = final_noise
                        except AttributeError:
                            # If kernel structure is complex (e.g., has categorical variables),
                            # skip hyperparameter reuse - fall back to each iteration's own optimization
                            pass
                    
                    # Transfer calibration factor from final model (even if hyperparameters couldn't be transferred)
                    # This ensures last iteration matches final model exactly
                    if reuse_hyperparameters and hasattr(self.model, 'calibration_factor'):
                        temp_model.calibration_factor = self.model.calibration_factor
                        # Enable calibration only if user requested calibrated uncertainties
                        temp_model.calibration_enabled = use_calibrated_uncertainty
                    
                    # Attach to session
                    temp_session.model = temp_model
                    temp_session.model_backend = 'botorch'
                
                # Predict on grid using temp_session.predict (consistent for all iterations)
                result = temp_session.predict(grid)
                if result is None:
                    raise ValueError(f"predict() returned None at iteration {i}")
                means, stds = result
                
                # Find max mean (or min for minimization)
                if goal.lower() == 'maximize':
                    best_idx = np.argmax(means)
                else:
                    best_idx = np.argmin(means)
                
                predicted_means[i - 1] = means[best_idx]
                predicted_stds[i - 1] = stds[best_idx]
                
            except Exception as e:
                import traceback
                logger.warning(f"Failed to compute predictions for iteration {i}: {e}")
                logger.debug(traceback.format_exc())
                # Leave as NaN
        
        # Restore original logging levels
        logger.setLevel(original_session_level)
        logging.getLogger('alchemist_core.models.botorch_model').setLevel(original_model_level)
        
        return predicted_means, predicted_stds
    
    def plot_probability_of_improvement(
        self,
        goal: Literal['maximize', 'minimize'] = 'maximize',
        backend: Optional[str] = None,
        kernel: Optional[str] = None,
        n_grid_points: int = 1000,
        start_iteration: int = 5,
        reuse_hyperparameters: bool = True,
        xi: float = 0.01,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm]
        """
        Plot maximum probability of improvement over optimization iterations.
        
        Retroactively computes how the probability of finding a better solution
        evolved during optimization. At each iteration:
        1. Trains GP on observations up to that point (reusing hyperparameters)
        2. Computes PI across the search space using native acquisition functions
        3. Records the maximum PI value
        
        Uses native PI implementations:
        - sklearn backend: skopt.acquisition.gaussian_pi
        - botorch backend: botorch.acquisition.ProbabilityOfImprovement
        
        Decreasing max(PI) indicates the optimization is converging and has
        less potential for improvement remaining.
        
        Args:
            goal: 'maximize' or 'minimize' - optimization direction
            backend: Model backend to use (defaults to session's model_backend)
            kernel: Kernel type for GP (defaults to session's kernel type)
            n_grid_points: Number of points to sample search space
            start_iteration: Minimum observations before computing PI (default: 5)
            reuse_hyperparameters: If True, use final model's optimized hyperparameters
                                   for all iterations (much faster, recommended)
            xi: PI parameter controlling improvement threshold (default: 0.01)
            figsize: Figure size as (width, height) in inches
            dpi: Dots per inch for figure resolution
            title: Custom plot title (auto-generated if None)
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # After running optimization
            >>> fig = session.plot_probability_of_improvement(goal='maximize')
            >>> fig.savefig('pi_convergence.png')
        
        Note:
            - Requires at least `start_iteration` experiments
            - Use fewer n_grid_points for faster computation
            - PI values near 0 suggest little room for improvement
            - Reusing hyperparameters (default) is much faster and usually sufficient
            - Uses rigorous acquisition function implementations (not approximations)
        """
        self._check_matplotlib()
        
        # Check we have enough experiments
        n_exp = len(self.experiment_manager.df)
        if n_exp < start_iteration:
            raise ValueError(
                f"Need at least {start_iteration} experiments for PI plot "
                f"(have {n_exp}). Lower start_iteration if needed."
            )
        
        # Default to session's model configuration if not specified
        if backend is None:
            if self.model_backend is None:
                raise ValueError(
                    "No backend specified and session has no trained model. "
                    "Either train a model first or specify backend parameter."
                )
            backend = self.model_backend
        
        if kernel is None:
            if self.model is None:
                raise ValueError(
                    "No kernel specified and session has no trained model. "
                    "Either train a model first or specify kernel parameter."
                )
            # Extract kernel type from trained model
            if self.model_backend == 'sklearn' and hasattr(self.model, 'optimized_kernel'):
                # sklearn model
                kernel_obj = self.model.optimized_kernel
                if 'RBF' in str(type(kernel_obj)):
                    kernel = 'RBF'
                elif 'Matern' in str(type(kernel_obj)):
                    kernel = 'Matern'
                elif 'RationalQuadratic' in str(type(kernel_obj)):
                    kernel = 'RationalQuadratic'
                else:
                    kernel = 'RBF'  # fallback
            elif self.model_backend == 'botorch' and hasattr(self.model, 'cont_kernel_type'):
                # botorch model - use the stored kernel type
                kernel = self.model.cont_kernel_type
            else:
                # Final fallback if we can't determine kernel
                kernel = 'Matern'
        
        # Get optimized hyperparameters if reusing them
        optimized_kernel_params = None
        if reuse_hyperparameters and self.model is not None:
            if backend.lower() == 'sklearn' and hasattr(self.model, 'optimized_kernel'):
                # Extract the optimized kernel parameters
                optimized_kernel_params = self.model.optimized_kernel
                logger.info(f"Reusing optimized kernel hyperparameters from trained model")
            # Note: botorch hyperparameter reuse would go here if needed
        
        # Get data
        target_col = self.experiment_manager.target_columns[0]
        X_all, y_all = self.experiment_manager.get_features_and_target()
        
        # Generate grid of test points across search space
        X_test = self._generate_prediction_grid(n_grid_points)
        
        logger.info(f"Computing PI convergence from iteration {start_iteration} to {n_exp}...")
        logger.info(f"Using {len(X_test)} test points across search space")
        logger.info(f"Using native PI acquisition functions (xi={xi})")
        if reuse_hyperparameters and optimized_kernel_params is not None:
            logger.info("Using optimized hyperparameters from final model (faster)")
        else:
            logger.info("Optimizing hyperparameters at each iteration (slower but more accurate)")
        
        # Compute max PI at each iteration
        iterations = []
        max_pi_values = []
        
        for i in range(start_iteration, n_exp + 1):
            # Get data up to iteration i
            X_train = X_all.iloc[:i]
            y_train = y_all[:i]
            
            # Create temporary session for this iteration
            temp_session = OptimizationSession(
                search_space=self.search_space,
                experiment_manager=ExperimentManager(search_space=self.search_space)
            )
            temp_session.experiment_manager.df = self.experiment_manager.df.iloc[:i].copy()
            
            # Train model with optimized hyperparameters if available
            try:
                if reuse_hyperparameters and optimized_kernel_params is not None and backend.lower() == 'sklearn':
                    # For sklearn: directly access model and set optimized kernel
                    from alchemist_core.models.sklearn_model import SklearnModel
                    
                    # Create model instance with kernel options
                    model_kwargs = {
                        'kernel_options': {'kernel_type': kernel},
                        'n_restarts_optimizer': 0  # Don't optimize since we're using fixed hyperparameters
                    }
                    temp_model = SklearnModel(**model_kwargs)
                    
                    # Preprocess data
                    X_processed, y_processed = temp_model._preprocess_data(temp_session.experiment_manager)
                    
                    # Import sklearn's GP
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    
                    # Create GP with the optimized kernel and optimizer=None to keep it fixed
                    gp_params = {
                        'kernel': optimized_kernel_params,
                        'optimizer': None,  # Keep hyperparameters fixed
                        'random_state': temp_model.random_state
                    }
                    
                    # Only add alpha if we have noise values
                    if temp_model.alpha is not None:
                        gp_params['alpha'] = temp_model.alpha
                    
                    temp_model.model = GaussianProcessRegressor(**gp_params)
                    
                    # Fit model (only computes GP weights, not hyperparameters)
                    temp_model.model.fit(X_processed, y_processed)
                    temp_model._is_trained = True
                    
                    # Set the model in the session
                    temp_session.model = temp_model
                    temp_session.model_backend = 'sklearn'
                else:
                    # Standard training with hyperparameter optimization
                    temp_session.train_model(backend=backend, kernel=kernel)
            except Exception as e:
                logger.warning(f"Failed to train model at iteration {i}: {e}")
                continue
            
            # Compute PI using native acquisition functions
            try:
                if backend.lower() == 'sklearn':
                    # Use skopt's gaussian_pi function
                    from skopt.acquisition import gaussian_pi
                    
                    # For maximization, negate y values so skopt treats it as minimization
                    if goal.lower() == 'maximize':
                        y_opt = -y_train.max()
                    else:
                        y_opt = y_train.min()
                    
                    # Preprocess X_test using the model's preprocessing pipeline
                    # This handles categorical encoding and scaling
                    X_test_processed = temp_session.model._preprocess_X(X_test)
                    
                    # Compute PI for all test points using skopt's implementation
                    # Note: gaussian_pi expects model with predict(X, return_std=True)
                    pi_values = gaussian_pi(
                        X=X_test_processed,
                        model=temp_session.model.model,  # sklearn GP model
                        y_opt=y_opt,
                        xi=xi
                    )
                    
                    max_pi = float(np.max(pi_values))
                    
                elif backend.lower() == 'botorch':
                    # Use BoTorch's ProbabilityOfImprovement
                    import torch
                    from botorch.acquisition import ProbabilityOfImprovement
                    
                    # Determine best value seen so far
                    if goal.lower() == 'maximize':
                        best_f = float(y_train.max())
                    else:
                        best_f = float(y_train.min())
                    
                    # Encode categorical variables if present
                    X_test_encoded = temp_session.model._encode_categorical_data(X_test)
                    
                    # Convert to torch tensor
                    X_tensor = torch.from_numpy(X_test_encoded.values).to(
                        dtype=temp_session.model.model.train_inputs[0].dtype,
                        device=temp_session.model.model.train_inputs[0].device
                    )
                    
                    # Create PI acquisition function
                    if goal.lower() == 'maximize':
                        pi_acq = ProbabilityOfImprovement(
                            model=temp_session.model.model,
                            best_f=best_f,
                            maximize=True
                        )
                    else:
                        pi_acq = ProbabilityOfImprovement(
                            model=temp_session.model.model,
                            best_f=best_f,
                            maximize=False
                        )
                    
                    # Evaluate PI on all test points
                    temp_session.model.model.eval()
                    with torch.no_grad():
                        pi_values = pi_acq(X_tensor.unsqueeze(-2))  # Add batch dimension
                    
                    max_pi = float(pi_values.max().item())
                    
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                    
            except Exception as e:
                logger.warning(f"Failed to compute PI at iteration {i}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
            
            # Record max PI
            iterations.append(i)
            max_pi_values.append(max_pi)
            
            if i % 5 == 0 or i == n_exp:
                logger.info(f"  Iteration {i}/{n_exp}: max(PI) = {max_pi:.4f}")
        
        if not iterations:
            raise RuntimeError("Failed to compute PI for any iterations")
        
        # Import visualization function
        from alchemist_core.visualization.plots import create_probability_of_improvement_plot
        
        # Create plot
        fig, ax = create_probability_of_improvement_plot(
            iterations=np.array(iterations),
            max_pi_values=np.array(max_pi_values),
            figsize=figsize,
            dpi=dpi,
            title=title
        )
        
        logger.info(f"Generated PI convergence plot with {len(iterations)} points")
        return fig

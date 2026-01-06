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
    
    @staticmethod
    def load_session(filepath: str, retrain_on_load: bool = True) -> 'OptimizationSession':
        """
        Load session from JSON file.
        
        Args:
            filepath: Path to session file
            
        Returns:
            OptimizationSession with restored state
            
        Example:
            > session = OptimizationSession.load_session("my_session.json")
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
        if not _HAS_MATPLOTLIB:
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
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = float('nan')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot data with optional error bars
        if show_error_bars and y_std is not None:
            yerr = sigma_multiplier * y_std
            ax.errorbar(y_true, y_pred, yerr=yerr, 
                       fmt='o', alpha=0.7, capsize=3, capthick=1,
                       elinewidth=1, markersize=5)
        else:
            ax.scatter(y_true, y_pred, alpha=0.7)
        
        # Add parity line (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parity line')
        
        # Set labels
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        
        # Create title
        if title is None and show_metrics:
            ci_labels = {
                1.0: "68% CI",
                1.96: "95% CI",
                2.0: "95.4% CI",
                2.58: "99% CI",
                3.0: "99.7% CI"
            }
            ci_label = ci_labels.get(sigma_multiplier, f"{sigma_multiplier}σ")
            
            if show_error_bars and y_std is not None:
                title = (f"Cross-Validation Parity Plot\n"
                        f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n"
                        f"Error bars: ±{sigma_multiplier}σ ({ci_label})")
            else:
                title = (f"Cross-Validation Parity Plot\n"
                        f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        if title:
            ax.set_title(title)
        
        ax.legend()
        fig.tight_layout()
        
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
    ) -> Figure: # pyright: ignore[reportInvalidTypeForm] # pyright: ignore[reportInvalidTypeForm]
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
            >>> fig.savefig('slice.png', dpi=300), 'catalyst': 'Pt'},
            ...     show_uncertainty=False
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Style settings
        mean_color = "#0B3C5D"          # dark blue-teal, prints well
        exp_face   = "#E07A00"          # orange (colorblind-friendly vs blue)
        grid_alpha = 0.25
        
        # Plot uncertainty bands (BEFORE mean line), with readable nesting
        if show_uncertainty and std is not None:
            # Determine which sigma values to plot
            if isinstance(show_uncertainty, bool):
                # Default: show ±1σ and ±2σ
                sigma_values = [1.0, 2.0]
            else:
                # Custom sigma values provided
                sigma_values = sorted(show_uncertainty)
            
            # Plot largest band first (background) - reverse order for z-stacking
            sig_desc = sorted(sigma_values, reverse=True)
            n = len(sig_desc)
            
            # Sequential colormap: same hue, different lightness
            cmap = plt.get_cmap("Blues")
            
            for i, sigma in enumerate(sig_desc):
                # i=0 is largest sigma (should be most transparent)
                # i=n-1 is smallest sigma (should be most opaque)
                t = i / max(1, n - 1)  # 0..1, where 0=largest σ, 1=smallest σ
                
                # Choose lighter tones for larger sigma, darker for smaller sigma
                face = cmap(0.3 + 0.3 * t)  # 0.30 (largest σ) to 0.60 (smallest σ) in Blues
                edge = plt.matplotlib.colors.to_rgba(mean_color, 0.55)
                
                # Sigmoid-based alpha as function of sigma value
                # alpha = 1 - 1 / (1 + exp(-sigma + 2))
                # Smaller sigma → higher alpha (more opaque)
                alpha = 1.0 - 1.0 / (1.0 + np.exp(-sigma + 2.0))
                
                ax.fill_between(
                    x_values,
                    predictions - sigma * std,
                    predictions + sigma * std,
                    facecolor=plt.matplotlib.colors.to_rgba(face, alpha),
                    edgecolor=edge,
                    linewidth=0.9,
                    label=f'±{sigma:.1f}σ',
                    zorder=1
                )
        
        # Mean prediction on top
        ax.plot(
            x_values,
            predictions,
            color=mean_color,
            linewidth=2.6,
            label='Prediction',
            zorder=3
        )
        
        # Plot experimental points
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
                ax.scatter(
                    filtered_df[x_var],
                    filtered_df['Output'],
                    s=70,
                    facecolor=exp_face,
                    edgecolor='black',
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=4,
                    label=f'Experiments (n={mask.sum()})'
                )
        
        # Labels and grid
        ax.set_xlabel(x_var)
        ax.set_ylabel('Predicted Output')
        ax.grid(True, alpha=grid_alpha)
        ax.set_axisbelow(True)
        
        # Create legend with improved ordering: Prediction, bands (small->large), Experiments
        handles, labels = ax.get_legend_handles_labels()
        def sort_key(lbl):
            if lbl == 'Prediction':
                return (0, 0)
            if lbl.startswith('±'):
                # Sort smaller sigma first in legend
                val = float(lbl.split('±')[1].split('σ')[0])
                return (1, val)
            if lbl.startswith('Experiments'):
                return (2, 0)
            return (3, 0)
        
        order = sorted(range(len(labels)), key=lambda k: sort_key(labels[k]))
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=True)
        
        if title is None:
            # Generate title showing fixed values
            if fixed_values:
                fixed_str = ', '.join([f'{k}={v}' for k, v in fixed_values.items()])
                title = f"1D Slice: {x_var}\n({fixed_str})"
            else:
                title = f"1D Slice: {x_var}"
        
        ax.set_title(title)
        fig.tight_layout()
        
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot contour
        contour = ax.contourf(X_grid, Y_grid, predictions_grid, levels=20, cmap=cmap)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Predicted Output')
        
        # Plot experimental data points
        if show_experiments and not self.experiment_manager.df.empty:
            exp_df = self.experiment_manager.df
            if x_var in exp_df.columns and y_var in exp_df.columns:
                ax.scatter(
                    exp_df[x_var], 
                    exp_df[y_var], 
                    c='white', 
                    edgecolors='black', 
                    s=50, 
                    label='Experiments',
                    zorder=10
                )
        
        # Plot suggested points
        if show_suggestions and len(self.last_suggestions) > 0:
            # last_suggestions is a DataFrame
            if isinstance(self.last_suggestions, pd.DataFrame):
                sugg_df = self.last_suggestions
            else:
                sugg_df = pd.DataFrame(self.last_suggestions)
            
            if x_var in sugg_df.columns and y_var in sugg_df.columns:
                ax.scatter(
                    sugg_df[x_var],
                    sugg_df[y_var],
                    c='red',
                    marker='D',
                    s=80,
                    label='Suggestions',
                    zorder=11
                )
        
        # Add legend if needed
        if (show_experiments or show_suggestions) and ax.get_legend_handles_labels()[0]:
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(title or "Contour Plot of Model Predictions")
        
        fig.tight_layout()
        
        logger.info(f"Generated contour plot for {x_var} vs {y_var}")
        return fig
    
    def plot_metrics(
        self,
        metric: Literal['rmse', 'mae', 'r2', 'mape'] = 'rmse',
        cv_splits: int = 5,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100
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
        
        Returns:
            matplotlib Figure object
        
        Example:
            >>> # Plot RMSE vs number of experiments
            >>> fig = session.plot_metrics('rmse')
            
            >>> # Plot R² to see improvement
            >>> fig = session.plot_metrics('r2')
        
        Note:
            This calls model.evaluate() which computes CV metrics for each training
            set size. This can be computationally expensive for large datasets.
        """
        self._check_matplotlib()
        self._check_model_trained()
        
        # Need at least 5 observations for CV
        n_total = len(self.experiment_manager.df)
        if n_total < 5:
            raise ValueError(f"Need at least 5 observations for metrics plot (have {n_total})")
        
        # Call model's evaluate method to get metrics over training sizes
        logger.info(f"Computing {metric.upper()} over training set sizes (this may take a moment)...")
        cv_metrics = self.model.evaluate(
            self.experiment_manager,
            cv_splits=cv_splits,
            debug=False
        )
        
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
        x_range = range(5, len(metric_values) + 5)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot as line with markers
        ax.plot(x_range, metric_values, marker='o', linewidth=2, 
                markersize=6, color='#2E86AB')
        
        # Labels
        metric_labels = {
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'mape': 'MAPE (%)'
        }
        ax.set_xlabel("Number of Observations")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric_labels[metric]} vs Number of Observations")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        logger.info(f"Generated {metric} metrics plot with {len(metric_values)} points")
        return fig


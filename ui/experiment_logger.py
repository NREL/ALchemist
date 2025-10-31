"""
UI-Layer Experiment Logging

This module provides logging utilities for the ALchemist UI layer.
It creates detailed log files for experiment workflows, recording:
- Model training details and hyperparameters
- Acquisition strategy results
- Experiment data summaries

This is separate from alchemist_core logging, which uses Python's logging module
for console/file output. ExperimentLogger creates human-readable experiment reports.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


class ExperimentLogger:
    """
    Logger class for tracking and exporting experiment details.
    Records model training, acquisition suggestions, and experiment results.
    
    This creates formatted log files in the logs/ directory with experiment
    details, separate from the console logging provided by alchemist_core.
    """
    def __init__(self, log_dir="logs"):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.current_experiment_id = None
        self.current_log_file = None
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def start_experiment(self, experiment_name=None):
        """Start a new experiment and create a log file"""
        # Generate experiment ID using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.current_experiment_id = f"{experiment_name}_{timestamp}"
        else:
            self.current_experiment_id = f"experiment_{timestamp}"
        
        # Create log file
        log_path = os.path.join(self.log_dir, f"{self.current_experiment_id}.log")
        self.current_log_file = open(log_path, 'w')
        
        # Write header
        self.log("=====================================")
        self.log(f"EXPERIMENT: {self.current_experiment_id}")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=====================================")
        self.log("")
        
        return self.current_experiment_id
    
    def log(self, message, print_to_console=True):
        """Log a message to the current log file and optionally print to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if self.current_log_file:
            self.current_log_file.write(log_message + "\n")
            self.current_log_file.flush()  # Ensure it's written immediately
        
        if print_to_console:
            print(message)  # Simple print for UI layer
    
    def log_model_training(self, model_data):
        """Log model training details"""
        self.log("MODEL TRAINING")
        self.log(f"Backend: {model_data.get('backend', 'Unknown')}")
        self.log(f"Kernel: {model_data.get('kernel', 'Unknown')}")
        
        # Log hyperparameters
        self.log("Hyperparameters:")
        hyperparams = model_data.get('hyperparameters', {})
        for name, value in hyperparams.items():
            self.log(f"  {name}: {value}")
        
        # Log performance metrics
        self.log("Performance Metrics:")
        metrics = model_data.get('metrics', {})
        for name, value in metrics.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                self.log(f"  {name}: {value[-1]:.4f}")
            elif isinstance(value, (int, float)):
                self.log(f"  {name}: {value:.4f}")
            else:
                self.log(f"  {name}: {value}")
        
        self.log("")
    
    def log_acquisition(self, result_data):
        """Log details of an acquisition strategy result."""
        self.log("\nACQUISITION STRATEGY")
        self.log(f"Strategy: {result_data.get('strategy_type', 'Unknown')}")
        self.log(f"Goal: {'Maximize' if result_data.get('maximize', True) else 'Minimize'}")
        
        # Log strategy parameters
        params = result_data.get('strategy_params', {})
        if params:
            for param_name, param_value in params.items():
                self.log(f"  {param_name}: {param_value}")
        
        # Handle batch acquisition results
        batch_size = result_data.get('batch_size', 1)
        is_batch = result_data.get('is_batch', False)
        
        # Special handling for batch results
        if is_batch and batch_size > 1 and 'all_points' in result_data:
            self.log(f"Selected {batch_size} points:")
            points = result_data.get('all_points', [])
            
            # Log each point in the batch
            for i, point in enumerate(points):
                if i >= batch_size:
                    break
                    
                self.log(f"Point {i+1}:")
                # Log coordinates
                for key, value in point.items():
                    if key not in ['value', 'std']:
                        self.log(f"  {key}: {value}")
                
                # Log prediction if available
                if 'value' in point:
                    self.log(f"  Predicted Value: {point['value']:.4f}")
                if 'std' in point:
                    self.log(f"  Prediction Std: {point['std']:.4f}")
                    ci_low = point['value'] - 1.96 * point['std']
                    ci_high = point['value'] + 1.96 * point['std']
                    self.log(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        else:
            # Standard single-point logging (unchanged)
            # Get point data
            point_df = result_data.get('point_df')
            
            self.log("Suggested Point:")
            if point_df is not None:
                for col in point_df.columns:
                    # Handle multi-row DataFrames (just show the first point)
                    self.log(f"  {col}: {point_df[col].iloc[0]}")

            # Log prediction
            pred_value = result_data.get('value')
            pred_std = result_data.get('std')
            
            # Safe formatting for potential numpy/array values
            if pred_value is not None:
                if isinstance(pred_value, (np.ndarray, list)):
                    pred_value = float(pred_value[0])  # Take first value if it's an array
                self.log(f"Predicted Value: {pred_value:.4f}")
                
            if pred_std is not None:
                if isinstance(pred_std, (np.ndarray, list)):
                    pred_std = float(pred_std[0])  # Take first value if it's an array
                self.log(f"Prediction Std: {pred_std:.4f}")
                
                # Calculate and log confidence interval
                if pred_value is not None:
                    ci_low = pred_value - 1.96 * pred_std
                    ci_high = pred_value + 1.96 * pred_std
                    self.log(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        
        self.log("")
    
    def log_experiment_data(self, exp_df):
        """Log the current experiment data"""
        self.log("EXPERIMENT DATA")
        self.log(f"Number of data points: {len(exp_df)}")
        
        # Log column names
        self.log(f"Variables: {', '.join(exp_df.columns.tolist())}")
        
        # Log summary statistics for output
        if 'Output' in exp_df.columns:
            output = exp_df['Output'].dropna()
            if len(output) > 0:
                self.log(f"Output Min: {output.min():.4f}")
                self.log(f"Output Max: {output.max():.4f}")
                self.log(f"Output Mean: {output.mean():.4f}")
                self.log(f"Output Std: {output.std():.4f}")
        
        self.log("")
    
    def end_experiment(self):
        """End the current experiment and close the log file"""
        if self.current_log_file:
            self.log("=====================================")
            self.log(f"EXPERIMENT ENDED: {self.current_experiment_id}")
            self.log(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("=====================================")
            
            self.current_log_file.close()
            self.current_log_file = None
            self.current_experiment_id = None

import customtkinter as ctk
import numpy as np
import pandas as pd
from customtkinter import filedialog
from datetime import datetime

class ResultNotificationWindow:
    def __init__(self, parent, result_data, model_data):
        """
        Display a notification window with detailed information about a suggested point.
        
        Args:
            parent: Parent widget
            result_data: Dictionary with acquisition results
            model_data: Dictionary with model information
        """
        self.window = ctk.CTkToplevel(parent)
        self.window.title("Suggested Next Experiment")
        self.window.geometry("600x500")
        self.window.lift()
        self.window.focus_force()
        self.window.grab_set()
        
        # Create a tabbed interface for organization
        self.tab_view = ctk.CTkTabview(self.window)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tab_view.add("Point Details")
        self.tab_view.add("Model Info")
        self.tab_view.add("Strategy Info")
        
        # Fill point details tab
        self._create_point_details_tab(result_data)
        
        # Fill model info tab
        self._create_model_info_tab(model_data)
        
        # Fill strategy info tab
        self._create_strategy_info_tab(result_data)
        
        # Add export and close buttons
        button_frame = ctk.CTkFrame(self.window)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        export_button = ctk.CTkButton(
            button_frame, 
            text="Export to CSV", 
            command=lambda: self._export_to_csv(result_data, model_data)
        )
        export_button.pack(side="left", padx=10, pady=5)
        
        close_button = ctk.CTkButton(
            button_frame, 
            text="Close", 
            command=self.window.destroy
        )
        close_button.pack(side="right", padx=10, pady=5)
    
    def _create_point_details_tab(self, result_data):
        """Create a tab showing the details of the suggested point(s)."""
        # IMPORTANT FIX: Use the correct tab reference
        points_tab = self.tab_view.tab("Point Details")
        
        point_df = result_data.get('point_df')
        if point_df is None:
            # Error case
            ctk.CTkLabel(points_tab, text="Error retrieving point data").pack(pady=10)
            return
        
        # Check if this is a batch result (multiple points)
        is_batch = result_data.get('is_batch', False)
        batch_size = result_data.get('batch_size', 1)
        
        # Get predicted values
        pred_value = result_data.get('value')
        pred_std = result_data.get('std')
        
        if is_batch and batch_size > 1:
            # Create a notebook for multiple points
            points_notebook = ctk.CTkTabview(points_tab)
            points_notebook.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create a tab for each point in the batch
            for i in range(batch_size):
                # Create tab with more descriptive name
                tab_name = f"Point {i+1}"
                points_notebook.add(tab_name)
                
                # Create content frame
                point_frame = ctk.CTkFrame(points_notebook.tab(tab_name))
                point_frame.pack(fill="both", expand=True, padx=5, pady=5)
                
                # Add a title for this batch point
                ctk.CTkLabel(
                    point_frame,
                    text=f"Batch Point {i+1} of {batch_size}",
                    font=("Arial", 12, "bold")
                ).pack(pady=5)
                
                # Add point details
                for col in point_df.columns:
                    row_frame = ctk.CTkFrame(point_frame)
                    row_frame.pack(fill="x", pady=2)
                    ctk.CTkLabel(row_frame, text=f"{col}:", width=100, anchor="w").pack(side="left", padx=5)
                    value = point_df[col].iloc[i]
                    ctk.CTkLabel(row_frame, text=f"{value}", anchor="w").pack(side="left", padx=5)
                
                # Add predicted value and std if available
                if pred_value is not None:
                    value_frame = ctk.CTkFrame(point_frame)
                    value_frame.pack(fill="x", pady=2)
                    ctk.CTkLabel(value_frame, text="Predicted Value:", width=100, anchor="w").pack(side="left", padx=5)
                    p_val = pred_value[i] if isinstance(pred_value, (list, np.ndarray)) else pred_value
                    # Handle numpy float types
                    p_val_float = float(p_val)
                    ctk.CTkLabel(value_frame, text=f"{p_val_float:.4f}", anchor="w").pack(side="left", padx=5)
                    
                if pred_std is not None:
                    std_frame = ctk.CTkFrame(point_frame)
                    std_frame.pack(fill="x", pady=2)
                    ctk.CTkLabel(std_frame, text="Prediction Std:", width=100, anchor="w").pack(side="left", padx=5)
                    p_std = pred_std[i] if isinstance(pred_std, (list, np.ndarray)) else pred_std
                    # Handle numpy float types
                    p_std_float = float(p_std)
                    ctk.CTkLabel(std_frame, text=f"{p_std_float:.4f}", anchor="w").pack(side="left", padx=5)
                    
                    # Add confidence interval
                    ci_frame = ctk.CTkFrame(point_frame)
                    ci_frame.pack(fill="x", pady=2)
                    ctk.CTkLabel(ci_frame, text="95% CI:", width=100, anchor="w").pack(side="left", padx=5)
                    ci_low = p_val_float - 1.96 * p_std_float
                    ci_high = p_val_float + 1.96 * p_std_float
                    ctk.CTkLabel(ci_frame, text=f"[{ci_low:.4f}, {ci_high:.4f}]", anchor="w").pack(side="left", padx=5)
        else:
            # Original code for single point results
            # Point coordinates frame
            coords_frame = ctk.CTkFrame(points_tab)
            coords_frame.pack(fill="x", padx=10, pady=10)
            
            # Title for coordinates
            ctk.CTkLabel(
                coords_frame, 
                text="Suggested Experiment Coordinates", 
                font=("Arial", 16, "bold")
            ).pack(pady=5)
            
            # Create scrollable frame for coordinates
            coords_scroll = ctk.CTkScrollableFrame(coords_frame, height=150)
            coords_scroll.pack(fill="x", padx=10, pady=5)
            
            # Add each coordinate
            point_df = result_data.get('point_df')
            if point_df is not None:
                for col in point_df.columns:
                    value = point_df[col].values[0]
                    # Format value based on type
                    if isinstance(value, (int, float, np.number)):
                        value_str = f"{float(value):.4f}"
                    else:
                        value_str = str(value)
                        
                    row_frame = ctk.CTkFrame(coords_scroll)
                    row_frame.pack(fill="x", pady=2)
                    ctk.CTkLabel(row_frame, text=col, width=150, anchor="w").pack(side="left", padx=5)
                    ctk.CTkLabel(row_frame, text=value_str, anchor="w").pack(side="left", padx=5, fill="x", expand=True)
            else:
                # Display a message when no point data is available
                ctk.CTkLabel(coords_scroll, text="No coordinates available", text_color="gray").pack(pady=20)
            
            # Prediction information
            pred_frame = ctk.CTkFrame(points_tab)
            pred_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                pred_frame, 
                text="Predicted Outcome", 
                font=("Arial", 16, "bold")
            ).pack(pady=5)
            
            # Display predicted value and uncertainty
            pred_value = result_data.get('value')
            pred_std = result_data.get('std')
            goal = "Maximum" if result_data.get('maximize', True) else "Minimum"
            
            value_frame = ctk.CTkFrame(pred_frame)
            value_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(value_frame, text=f"Predicted {goal} Value:", width=150, anchor="w").pack(side="left", padx=5)
            
            # Handle None values
            if pred_value is not None:
                # Convert to float to handle numpy types
                pred_value_float = float(pred_value)
                ctk.CTkLabel(value_frame, text=f"{pred_value_float:.4f}", anchor="w").pack(side="left", padx=5)
            else:
                ctk.CTkLabel(value_frame, text="Not available", text_color="gray", anchor="w").pack(side="left", padx=5)
            
            if pred_std is not None:
                # Convert to float to handle numpy types
                pred_std_float = float(pred_std)
                
                std_frame = ctk.CTkFrame(pred_frame)
                std_frame.pack(fill="x", padx=10, pady=5)
                
                ctk.CTkLabel(std_frame, text="Prediction Uncertainty:", width=150, anchor="w").pack(side="left", padx=5)
                ctk.CTkLabel(std_frame, text=f"±{pred_std_float:.4f}", anchor="w").pack(side="left", padx=5)
                
                ci_frame = ctk.CTkFrame(pred_frame)
                ci_frame.pack(fill="x", padx=10, pady=5)
                
                ci_low = pred_value_float - 1.96 * pred_std_float
                ci_high = pred_value_float + 1.96 * pred_std_float
                
                ctk.CTkLabel(ci_frame, text="95% Confidence Interval:", width=150, anchor="w").pack(side="left", padx=5)
                ctk.CTkLabel(ci_frame, text=f"[{ci_low:.4f}, {ci_high:.4f}]", anchor="w").pack(side="left", padx=5)
            
    def _create_model_info_tab(self, model_data):
        """Create the model info tab content"""
        # Use the correct tab reference
        tab = self.tab_view.tab("Model Info")
        
        # Main frame for model info
        model_frame = ctk.CTkFrame(tab)
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Backend info
        ctk.CTkLabel(
            model_frame, 
            text="Model Configuration", 
            font=("Arial", 16, "bold")
        ).pack(pady=5)
        
        # Create scrollable frame for model details
        model_scroll = ctk.CTkScrollableFrame(model_frame, height=350)
        model_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Basic model info
        self._add_info_row(model_scroll, "Backend", model_data.get('backend', 'Unknown'))
        self._add_info_row(model_scroll, "Kernel", model_data.get('kernel', 'Unknown'))
        
        # Add hyperparameters section
        ctk.CTkLabel(
            model_scroll, 
            text="Hyperparameters", 
            font=("Arial", 14)
        ).pack(pady=5, anchor="w")
        
        # Add each hyperparameter
        hyperparams = model_data.get('hyperparameters', {})
        if isinstance(hyperparams, dict):
            for param_name, param_value in hyperparams.items():
                # Skip internal parameters used for formatting
                if param_name in ['continuous_features', 'covar_module_type', 'additive_kernels', 'primary_lengthscales']:
                    continue
                
                # Special handling for lengthscales to make them more readable
                if param_name == 'lengthscale' and isinstance(param_value, (list, np.ndarray)):
                    # For mixed models, prefer primary_lengthscales if available
                    display_lengthscales = hyperparams.get('primary_lengthscales', param_value)
                    continuous_features = hyperparams.get('continuous_features', [])
                    
                    if len(display_lengthscales) == 1:
                        # Isotropic kernel
                        param_str = f"{float(display_lengthscales[0]):.4f} (isotropic)"
                    else:
                        # ARD kernel - show primary lengthscales for continuous features
                        if continuous_features and len(continuous_features) == len(display_lengthscales):
                            # Show individual lengthscales with feature names
                            param_str = f"ARD Continuous Features:"
                            for i, (feature, ls) in enumerate(zip(continuous_features, display_lengthscales)):
                                self._add_info_row(model_scroll, f"  └─ {feature}", f"{float(ls):.4f}")
                            
                            # Also show summary if we have additional lengthscales
                            if len(param_value) > len(display_lengthscales):
                                extra_count = len(param_value) - len(display_lengthscales)
                                param_str += f" (+{extra_count} additional kernel parameters)"
                        else:
                            # Fallback to showing all lengthscales
                            lengthscale_strs = [f"{float(ls):.4f}" for ls in display_lengthscales]
                            param_str = f"[{', '.join(lengthscale_strs)}] (ARD)"
                            
                            # Add kernel structure info if available
                            kernel_types = hyperparams.get('additive_kernels', [])
                            if kernel_types:
                                param_str += f" from {kernel_types}"
                
                elif isinstance(param_value, (list, np.ndarray)):
                    if len(param_value) == 1:
                        # Handle single-element arrays
                        val = param_value[0]
                        if isinstance(val, (tuple, list)):
                            param_str = str(val)
                        else:
                            param_str = f"{float(val):.6f}"
                    else:
                        # Handle multi-element arrays - check if elements are tuples/lists
                        try:
                            param_str = f"[{', '.join([f'{float(v):.4f}' for v in param_value])}]"
                        except (TypeError, ValueError):
                            # If conversion fails (e.g., tuples in list), use string representation
                            param_str = str(param_value)
                elif isinstance(param_value, (int, float, np.number)):
                    param_str = f"{float(param_value):.6f}"
                else:
                    param_str = str(param_value)
                
                self._add_info_row(model_scroll, param_name, param_str)
                
        # Add performance metrics section
        ctk.CTkLabel(
            model_scroll, 
            text="Model Performance", 
            font=("Arial", 14)
        ).pack(pady=5, anchor="w")
        
        metrics = model_data.get('metrics', {})
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (list, np.ndarray)):
                    if len(metric_value) > 0:
                        # Use the most recent value
                        self._add_info_row(model_scroll, metric_name, f"{float(metric_value[-1]):.4f}")
                elif isinstance(metric_value, (int, float, np.number)):
                    self._add_info_row(model_scroll, metric_name, f"{float(metric_value):.4f}")
                else:
                    self._add_info_row(model_scroll, metric_name, str(metric_value))
    
    def _create_strategy_info_tab(self, result_data):
        """Create the strategy info tab content"""
        # Use the correct tab reference
        tab = self.tab_view.tab("Strategy Info")
        
        # Strategy frame
        strategy_frame = ctk.CTkFrame(tab)
        strategy_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            strategy_frame, 
            text="Acquisition Strategy", 
            font=("Arial", 16, "bold")
        ).pack(pady=5)
        
        # Create scrollable frame for strategy details
        strategy_scroll = ctk.CTkScrollableFrame(strategy_frame, height=350)
        strategy_scroll.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Strategy type
        self._add_info_row(
            strategy_scroll, 
            "Strategy Type", 
            result_data.get('strategy_type', 'Unknown')
        )
        
        # Optimization goal
        self._add_info_row(
            strategy_scroll, 
            "Optimization Goal", 
            "Maximize" if result_data.get('maximize', True) else "Minimize"
        )
        
        # Strategy parameters
        ctk.CTkLabel(
            strategy_scroll, 
            text="Strategy Parameters", 
            font=("Arial", 14)
        ).pack(pady=5, anchor="w")
        
        params = result_data.get('strategy_params', {})
        if isinstance(params, dict):
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float, np.number)):
                    param_str = f"{float(param_value):.4f}"
                else:
                    param_str = str(param_value)
                self._add_info_row(strategy_scroll, param_name, param_str)
                
        # Strategy description
        desc = result_data.get('strategy_description', '')
        if desc:
            ctk.CTkLabel(
                strategy_scroll, 
                text="Strategy Description", 
                font=("Arial", 14)
            ).pack(pady=5, anchor="w")
            
            ctk.CTkLabel(
                strategy_scroll, 
                text=desc,
                wraplength=450,
                justify="left"
            ).pack(pady=5, anchor="w", fill="x")
    
    def _add_info_row(self, parent, label, value):
        """Helper to add an information row with label and value"""
        row_frame = ctk.CTkFrame(parent)
        row_frame.pack(fill="x", pady=2)
        
        ctk.CTkLabel(row_frame, text=label, width=150, anchor="w").pack(side="left", padx=5)
        
        # For long values, we need to handle differently
        if isinstance(value, str) and len(value) > 40:
            ctk.CTkLabel(
                row_frame, 
                text=value, 
                anchor="w",
                wraplength=350,
                justify="left"
            ).pack(side="left", padx=5, fill="x", expand=True)
        else:
            ctk.CTkLabel(
                row_frame, 
                text=value, 
                anchor="w"
            ).pack(side="left", padx=5, fill="x", expand=True)
    
    def _export_to_csv(self, result_data, model_data):
        """Export the results to a CSV file"""
        import csv
        from datetime import datetime
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Result Details"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header row
                writer.writerow(["Result Export", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                
                # Point details
                writer.writerow(["SUGGESTED POINT DETAILS"])
                point_df = result_data.get('point_df')
                if point_df is not None:
                    for col in point_df.columns:
                        writer.writerow([col, point_df[col].values[0]])
                
                writer.writerow([])
                writer.writerow(["PREDICTION DETAILS"])
                writer.writerow(["Predicted Value", result_data.get('value')])
                if result_data.get('std') is not None:
                    writer.writerow(["Prediction Std", result_data.get('std')])
                    ci_low = result_data.get('value') - 1.96 * result_data.get('std')
                    ci_high = result_data.get('value') + 1.96 * result_data.get('std')
                    writer.writerow(["95% CI Low", ci_low])
                    writer.writerow(["95% CI High", ci_high])
                
                writer.writerow([])
                writer.writerow(["MODEL DETAILS"])
                writer.writerow(["Backend", model_data.get('backend')])
                writer.writerow(["Kernel", model_data.get('kernel')])
                
                writer.writerow([])
                writer.writerow(["HYPERPARAMETERS"])
                hyperparams = model_data.get('hyperparameters', {})
                if isinstance(hyperparams, dict):
                    for param_name, param_value in hyperparams.items():
                        writer.writerow([param_name, param_value])
                
                writer.writerow([])
                writer.writerow(["ACQUISITION STRATEGY"])
                writer.writerow(["Strategy", result_data.get('strategy_type')])
                writer.writerow(["Goal", "Maximize" if result_data.get('maximize', True) else "Minimize"])
                
                params = result_data.get('strategy_params', {})
                if isinstance(params, dict):
                    for param_name, param_value in params.items():
                        writer.writerow([param_name, param_value])
            
            print(f"Results exported to {file_path}")
            
        except Exception as e:
            print(f"Error exporting results: {e}")


class CalibrationWarningDialog:
    """Dialog to warn users about poor model calibration and suggest acquisition parameter adjustments."""
    
    def __init__(self, parent, calibration_factor, backend="scikit-learn"):
        """
        Display a warning dialog about poor calibration.
        
        Args:
            parent: Parent widget
            calibration_factor: The computed calibration factor (std of z-scores)
            backend: Model backend ("scikit-learn" or "botorch")
        """
        self.window = ctk.CTkToplevel(parent)
        self.window.title("⚠ Model Calibration Warning")
        self.window.geometry("550x400")
        self.window.lift()
        self.window.focus_force()
        self.window.grab_set()
        
        # Determine if over-confident or under-confident
        if calibration_factor > 1.2:
            self.issue_type = "over-confident"
            self.direction = "increase"
        else:  # calibration_factor < 0.8
            self.issue_type = "under-confident"
            self.direction = "decrease"
        
        self.calibration_factor = calibration_factor
        self.backend = backend
        
        # Main frame
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Warning icon and title
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 15))
        
        warning_label = ctk.CTkLabel(
            title_frame,
            text="⚠",
            font=("Arial", 48),
            text_color="#FFA500"
        )
        warning_label.pack(side="left", padx=(0, 15))
        
        title_text = ctk.CTkLabel(
            title_frame,
            text=f"Model is {self.issue_type.title()}",
            font=("Arial", 24, "bold")
        )
        title_text.pack(side="left")
        
        # Calibration info
        info_frame = ctk.CTkFrame(main_frame)
        info_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            info_frame,
            text=f"Calibration Factor: {calibration_factor:.4f}",
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        explanation = (
            f"The model's predicted uncertainties are too {'narrow' if self.issue_type == 'over-confident' else 'wide'}.\n"
            f"This means the model is {'underestimating' if self.issue_type == 'over-confident' else 'overestimating'} "
            f"how uncertain it should be about predictions."
        )
        
        ctk.CTkLabel(
            info_frame,
            text=explanation,
            font=("Arial", 12),
            wraplength=480,
            justify="left"
        ).pack(pady=10, padx=10)
        
        # Recommendation frame
        rec_frame = ctk.CTkFrame(main_frame)
        rec_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            rec_frame,
            text="💡 Recommended Actions:",
            font=("Arial", 14, "bold"),
            anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 5))
        
        # Action 1: Calibration checkbox
        action1 = (
            "✓ Keep 'Calibrate Uncertainty' checked\n"
            "  (Predictions shown to you will be corrected)"
        )
        ctk.CTkLabel(
            rec_frame,
            text=action1,
            font=("Arial", 11),
            anchor="w",
            justify="left"
        ).pack(fill="x", padx=20, pady=2)
        
        # Action 2: Adjust acquisition parameters
        if self.issue_type == "over-confident":
            # Over-confident: increase exploration
            action2 = (
                f"⚠ {self.direction.title()} acquisition exploration parameters:\n"
                f"  • For EI/PI: Increase ξ (xi) to ~0.05-0.10\n"
                f"  • For UCB: Increase κ (kappa) to ~2.5-3.0\n"
                f"  (This compensates for underestimated uncertainties)"
            )
        else:
            # Under-confident: decrease exploration
            action2 = (
                f"⚠ {self.direction.title()} acquisition exploration parameters:\n"
                f"  • For EI/PI: Decrease ξ (xi) to ~0.001-0.005\n"
                f"  • For UCB: Decrease κ (kappa) to ~1.0-1.5\n"
                f"  (This compensates for overestimated uncertainties)"
            )
        
        ctk.CTkLabel(
            rec_frame,
            text=action2,
            font=("Arial", 11),
            anchor="w",
            justify="left",
            text_color="#FFA500"
        ).pack(fill="x", padx=20, pady=10)
        
        # Note about acquisition
        note_text = (
            "Note: Acquisition functions use uncalibrated uncertainties, so manual\n"
            "parameter adjustment helps maintain proper exploration/exploitation balance."
        )
        ctk.CTkLabel(
            rec_frame,
            text=note_text,
            font=("Arial", 10, "italic"),
            anchor="w",
            justify="left",
            text_color="gray"
        ).pack(fill="x", padx=20, pady=(5, 10))
        
        # Button frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))
        
        close_button = ctk.CTkButton(
            button_frame,
            text="Got it",
            command=self.window.destroy,
            width=120
        )
        close_button.pack(side="right")


def show_calibration_warning(parent, calibration_factor, backend="scikit-learn"):
    """
    Show calibration warning dialog if calibration is poor.
    
    Args:
        parent: Parent widget
        calibration_factor: The computed calibration factor
        backend: Model backend
        
    Returns:
        True if warning was shown, False otherwise
    """
    if calibration_factor < 0.8 or calibration_factor > 1.2:
        CalibrationWarningDialog(parent, calibration_factor, backend)
        return True
    return False

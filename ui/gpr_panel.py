# ==========================
# IMPORTS
# ==========================
import customtkinter as ctk
import matplotlib.pyplot as plt
import pandas as pd
import threading
import traceback

from alchemist_core.data.experiment_manager import ExperimentManager
from ui.visualizations import Visualizations

plt.rcParams['savefig.dpi'] = 600

# ==========================
# CLASS: GaussianProcessPanel
# ==========================
class GaussianProcessPanel(ctk.CTkFrame):
    def __init__(self, parent, main_app=None):
        super().__init__(parent)

        # ==========================
        # INITIALIZATION
        # ==========================
        # In non-tabbed mode, parent is the main app
        # In tabbed mode, parent is the tab and main_app is provided separately
        self.main_app = main_app if main_app is not None else parent
        self.master = parent  # Keep master as parent for proper Tkinter behavior
        
        # State variables
        self.advanced_enabled = False  # Track the state of advanced options
        
        self.search_space = None  # Initialize search_space
        self.gpr_model = None     # Initialize model
        self.exp_df = None        # Initialize experiment data
        self.visualizations = None
        self.categorical_variables = []  # Store categorical variable names
        self.cv_results_text = None

        # Configure panel layout - set strong constraints
        self.configure(width=280, height=600)  # Reduced width from 300
        # Ensure these constraints are applied
        self.pack_propagate(False)

        # ==========================
        # UI ELEMENTS
        # ==========================
        # Title Label
        ctk.CTkLabel(self, text='Gaussian Process Model', font=('Arial', 16)).pack(pady=5)

        # Backend selection segmented button
        self.backend_options = ["scikit-learn", "botorch"]  # Removed "ax" from options
        self.backend_var = ctk.StringVar(value="scikit-learn")
        self.backend_segmented = ctk.CTkSegmentedButton(
            self,
            values=self.backend_options,
            variable=self.backend_var,
            command=self.load_backend_options
        )
        self.backend_segmented.pack(pady=5)

        # Advanced options container
        self.advanced_frame = ctk.CTkFrame(self)
        self.advanced_frame.pack(fill="x", pady=5)

        # Advanced options toggle switch with state tracking
        self.advanced_var = ctk.BooleanVar(self.advanced_frame, value=False)
        self.advanced_switch = ctk.CTkSwitch(
            self.advanced_frame,
            text="Enable Advanced Options",
            variable=self.advanced_var,
            command=self.toggle_advanced_options
        )
        self.advanced_switch.pack(pady=5)

        # Subframes for backend kernel options
        self.sklearn_frame = ctk.CTkFrame(self.advanced_frame)
        self.botorch_frame = ctk.CTkFrame(self.advanced_frame)
        # Removed: self.ax_frame = ctk.CTkFrame(self.advanced_frame)
        self.sklearn_frame.pack(fill="x", expand=True)  # Initially pack scikit-learn frame
        self.create_sklearn_widgets()
        self.create_botorch_widgets()
        # Removed: self.create_ax_widgets()  # Create Ax widgets

        # Buttons for training and visualizations
        self.train_button = ctk.CTkButton(self, text="Train Model", command=self.train_model_threaded)
        self.train_button.pack(pady=10)
        self.visualize_button = ctk.CTkButton(
            self,
            text="Show Analysis Plots",
            command=self.open_visualization_window,
            state="disabled"  # Disabled until model is trained
        )
        self.visualize_button.pack(pady=10)
        
        # Initialize advanced options state
        self.toggle_advanced_options()

    # ==========================
    # WIDGET CREATION METHODS
    # ==========================
    def create_sklearn_widgets(self):
        """Create scikit-learn kernel customization widgets inside the sklearn_frame."""
        self.kernel_label = ctk.CTkLabel(self.sklearn_frame, text='Kernel Selection:')
        self.kernel_label.pack(pady=2)
        self.kernel_var = ctk.StringVar(value="RBF")
        self.kernel_menu = ctk.CTkOptionMenu(
            self.sklearn_frame,
            values=["RBF", "Matern", "RationalQuadratic"],
            variable=self.kernel_var,
            command=self.update_nu_visibility,
            state="disabled"  # Disabled by default
        )
        self.kernel_menu.pack(pady=5)
        self.nu_label = ctk.CTkLabel(self.sklearn_frame, text='Matern nu:', text_color="grey")  # Grey by default
        self.nu_var = ctk.StringVar(value="1.5")
        self.nu_menu = ctk.CTkOptionMenu(
            self.sklearn_frame,
            values=["0.5", "1.5", "2.5", "inf"],
            variable=self.nu_var,
            state="disabled"  # Disabled by default
        )
        self.auto_kernel_label = ctk.CTkLabel(
            self.sklearn_frame,
            text="Note: Kernel hyperparameters will be automatically optimized.",
            text_color='grey',
            wraplength=250
        )
        self.auto_kernel_label.pack(pady=2)
        self.optimizer_label = ctk.CTkLabel(self.sklearn_frame, text='Optimizer:', text_color="grey")  # Grey by default
        self.optimizer_label.pack(pady=2)
        self.optimizer_var = ctk.StringVar(value="L-BFGS-B")
        self.optimizer_menu = ctk.CTkOptionMenu(
            self.sklearn_frame,
            values=["CG", "BFGS", "L-BFGS-B", "TNC"],
            variable=self.optimizer_var,
            state="disabled"  # Disabled by default
        )
        self.optimizer_menu.pack(pady=5)

        # === Input/Output Scaling Options ===
        # Input scaling
        self.sk_input_scale_label = ctk.CTkLabel(self.sklearn_frame, text='Input Scaling:', text_color="grey")
        self.sk_input_scale_label.pack(pady=2)
        self.sk_input_scale_var = ctk.StringVar(value="none")
        self.sk_input_scale_menu = ctk.CTkOptionMenu(
            self.sklearn_frame,
            values=["none", "minmax", "standard", "robust"],
            variable=self.sk_input_scale_var,
            state="disabled"
        )
        self.sk_input_scale_menu.pack(pady=5)

        # Output scaling
        self.sk_output_scale_label = ctk.CTkLabel(self.sklearn_frame, text='Output Scaling:', text_color="grey")
        self.sk_output_scale_label.pack(pady=2)
        self.sk_output_scale_var = ctk.StringVar(value="none")
        self.sk_output_scale_menu = ctk.CTkOptionMenu(
            self.sklearn_frame,
            values=["none", "minmax", "standard", "robust"],
            variable=self.sk_output_scale_var,
            state="disabled"
        )
        self.sk_output_scale_menu.pack(pady=5)

        # Calibrate uncertainty checkbox (advanced option, disabled by default)
        self.sk_calibrate_uncertainty_var = ctk.BooleanVar(value=False)
        self.sk_calibrate_uncertainty_checkbox = ctk.CTkCheckBox(
            self.sklearn_frame,
            text="Calibrate Uncertainty",
            variable=self.sk_calibrate_uncertainty_var,
            state="disabled"
        )
        self.sk_calibrate_uncertainty_checkbox.pack(pady=5)

        # Store advanced widgets and labels for toggling
        self.advanced_widgets = [
            self.kernel_menu,
            self.optimizer_menu,
            self.nu_menu,
            self.sk_input_scale_menu,
            self.sk_output_scale_menu,
            self.sk_calibrate_uncertainty_checkbox
        ]
        self.advanced_labels = [
            self.kernel_label,
            self.optimizer_label,
            self.nu_label,
            self.sk_input_scale_label,
            self.sk_output_scale_label
        ]

    def create_botorch_widgets(self):
        """Create BoTorch kernel customization widgets inside the botorch_frame."""
        self.bt_kernel_label = ctk.CTkLabel(self.botorch_frame, text="Continuous Kernel:")
        self.bt_kernel_label.pack(pady=2)
        self.bt_kernel_var = ctk.StringVar(value="Matern")
        self.bt_kernel_menu = ctk.CTkOptionMenu(
            self.botorch_frame,
            values=["RBF", "Matern"],
            variable=self.bt_kernel_var,
            command=self.update_bt_nu_visibility,
            state="disabled"
        )
        self.bt_kernel_menu.pack(pady=5)
        self.bt_nu_label = ctk.CTkLabel(self.botorch_frame, text="Matern nu:")
        self.bt_nu_var = ctk.StringVar(value="2.5")
        self.bt_nu_menu = ctk.CTkOptionMenu(
            self.botorch_frame,
            values=["0.5", "1.5", "2.5"],
            variable=self.bt_nu_var,
            state="disabled"
        )
        if self.bt_kernel_var.get() == "Matern":
            self.bt_nu_label.pack(pady=2)
            self.bt_nu_menu.pack(pady=5)
        
        # Add an informational label about BoTorch
        self.bt_info_label = ctk.CTkLabel(
            self.botorch_frame,
            text="BoTorch uses sensible defaults for training parameters.",
            text_color="grey",
            wraplength=250
        )
        self.bt_info_label.pack(pady=5)

        # === Input/Output Scaling Options ===
        # Input scaling
        self.bt_input_scale_label = ctk.CTkLabel(self.botorch_frame, text='Input Scaling:')
        self.bt_input_scale_label.pack(pady=2)
        self.bt_input_scale_var = ctk.StringVar(value="none")
        self.bt_input_scale_menu = ctk.CTkOptionMenu(
            self.botorch_frame,
            values=["none", "normalize", "standardize"],
            variable=self.bt_input_scale_var,
            state="disabled"
        )
        self.bt_input_scale_menu.pack(pady=5)

        # Output scaling
        self.bt_output_scale_label = ctk.CTkLabel(self.botorch_frame, text='Output Scaling:')
        self.bt_output_scale_label.pack(pady=2)
        self.bt_output_scale_var = ctk.StringVar(value="none")
        self.bt_output_scale_menu = ctk.CTkOptionMenu(
            self.botorch_frame,
            values=["none", "standardize"],
            variable=self.bt_output_scale_var,
            state="disabled"
        )
        self.bt_output_scale_menu.pack(pady=5)

        # Calibrate uncertainty checkbox (advanced option, disabled by default)
        self.bt_calibrate_uncertainty_var = ctk.BooleanVar(value=False)
        self.bt_calibrate_uncertainty_checkbox = ctk.CTkCheckBox(
            self.botorch_frame,
            text="Calibrate Uncertainty",
            variable=self.bt_calibrate_uncertainty_var,
            state="disabled"
        )
        self.bt_calibrate_uncertainty_checkbox.pack(pady=5)

        # Store BoTorch advanced widgets for toggling
        self.bt_advanced_widgets = [
            self.bt_kernel_menu,
            self.bt_nu_menu,
            self.bt_input_scale_menu,
            self.bt_output_scale_menu,
            self.bt_calibrate_uncertainty_checkbox
        ]
        self.bt_advanced_labels = [
            self.bt_kernel_label,
            self.bt_nu_label,
            self.bt_input_scale_label,
            self.bt_output_scale_label
        ]

    # ==========================
    # BACKEND AND ACQUISITION OPTIONS
    # ==========================
    def load_backend_options(self, event=None):
        """Switch kernel options based on selected backend and update acquisition options."""
        self.sklearn_frame.pack_forget()
        self.botorch_frame.pack_forget()
        # self.ax_frame.pack_forget()
        
        backend = self.backend_var.get()
        if backend == "scikit-learn":
            self.sklearn_frame.pack(fill="x", expand=True)
            # Apply current advanced options state
            self.toggle_advanced_options()
        elif backend == "botorch":
            self.botorch_frame.pack(fill="x", expand=True)
            # Apply current advanced options state
            self.toggle_advanced_options()
        # elif backend == "ax":
        #     self.ax_frame.pack(fill="x", expand=True)
        
        # Update acquisition panel if it exists
        if hasattr(self.main_app, 'acquisition_panel'):
            self.main_app.acquisition_panel.update_for_backend(backend)

    def load_acquisition_options(self):
        """Show acquisition function options based on the selected backend."""
        self.acq_sklearn_frame.pack_forget()
        self.acq_botorch_frame.pack_forget()
        self.acq_ax_frame.pack_forget()
        backend = self.backend_var.get()
        if backend == "scikit-learn":
            self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "botorch":
            self.acq_botorch_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "ax":
            self.acq_ax_frame.pack(fill="x", expand=True, padx=10, pady=5)

    def toggle_advanced_options(self):
        """Enable or disable advanced options for all backends."""
        # Store the current state for retrieval when switching layouts
        self.advanced_enabled = self.advanced_var.get()
        state = "normal" if self.advanced_enabled else "disabled"
        
        # Apply toggle to scikit-learn advanced widgets and labels
        for widget in self.advanced_widgets:
            widget.configure(state=state)
        for label in self.advanced_labels:
            label.configure(text_color="white" if state == "normal" else "grey")

        # Apply toggle to BoTorch advanced widgets and labels
        if hasattr(self, 'bt_advanced_widgets'):
            for widget in self.bt_advanced_widgets:
                widget.configure(state=state)
        if hasattr(self, 'bt_advanced_labels'):
            for label in self.bt_advanced_labels:
                label.configure(text_color="white" if state == "normal" else "grey")

        # Update visibility of Matern nu options based on current backend
        backend = self.backend_var.get()
        if backend == "scikit-learn":
            self.update_nu_visibility()
        elif backend == "botorch":
            self.update_bt_nu_visibility()

    def update_nu_visibility(self, event=None):
        if self.kernel_var.get() == "Matern":
            if not self.nu_label.winfo_ismapped():
                self.nu_label.pack(pady=2, before=self.auto_kernel_label)
            if not self.nu_menu.winfo_ismapped():
                self.nu_menu.pack(pady=5, before=self.auto_kernel_label)
        else:
            self.nu_label.pack_forget()
            self.nu_menu.pack_forget()

    def update_bt_nu_visibility(self, event=None):
        """Update visibility of Matern nu for BoTorch"""
        if self.bt_kernel_var.get() == "Matern":
            if not self.bt_nu_label.winfo_ismapped():
                self.bt_nu_label.pack(pady=2, before=self.bt_info_label)
            if not self.bt_nu_menu.winfo_ismapped():
                self.bt_nu_menu.pack(pady=5, before=self.bt_info_label)
        else:
            self.bt_nu_label.pack_forget()
            self.bt_nu_menu.pack_forget()

    # ==========================
    # MODEL TRAINING
    # ==========================
    def train_model_threaded(self):
        self.loading_window = ctk.CTkToplevel(self)
        self.loading_window.title("Training Model")
        window_width = 300
        window_height = 100
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        position_x = (screen_width // 2) - (window_width // 2)
        position_y = (screen_height // 2) - (window_height // 2)
        self.loading_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.loading_window.lift()
        self.loading_window.focus_force()
        self.loading_window.grab_set()
        self.progress_var = ctk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(self.loading_window, variable=self.progress_var, progress_color='green')
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_label = ctk.CTkLabel(self.loading_window, text="Training model and performing cross-validation...")
        self.progress_label.pack(pady=10)
        self.training_thread = threading.Thread(target=self.train_model_with_progress)
        self.training_thread.start()
        self.check_training_progress()

    def train_model_with_progress(self):
        def progress_callback(progress):
            self.progress_var.set(progress)
        self.train_model(progress_callback=progress_callback)

    def check_training_progress(self):
        if self.training_thread.is_alive():
            self.after(100, self.check_training_progress)
        else:
            self.loading_window.destroy()

    def train_model(self, progress_callback=None, debug=False):
        """Train the model based on the selected backend and evaluate it."""
        if self.main_app.exp_df is None or len(self.main_app.exp_df) < 5:
            print("Error: Not enough data to train the model (need at least 5 observations).")
            return

        # Get backend selection
        backend = self.backend_var.get()

        # Fixed values
        random_state = 42
        n_restarts = 30

        try:
            # ============================================================
            # Session-based training path
            # ============================================================
            print("Training model using OptimizationSession API...")
            
            # Ensure session has current data
            self.main_app._sync_data_to_session()
            
            # Build kernel options
            if backend == "scikit-learn":
                kernel = self.kernel_var.get()
                kernel_params = {}
                if kernel == "Matern":
                    kernel_params['nu'] = float(self.nu_var.get())
                
                # Train using session API
                results = self.main_app.session.train_model(
                    backend='sklearn',
                    kernel=kernel,
                    kernel_params=kernel_params,
                    n_restarts_optimizer=n_restarts,
                    optimizer=self.optimizer_var.get(),
                    input_transform_type=self.sk_input_scale_var.get(),
                    output_transform_type=self.sk_output_scale_var.get()
                )
                
            elif backend == "botorch":
                kernel = self.bt_kernel_var.get()
                kernel_params = {}
                if kernel == "Matern":
                    kernel_params['nu'] = float(self.bt_nu_var.get())
                
                # Get categorical dimensions
                categorical_variables = self.main_app.search_space_manager.get_categorical_variables()
                cat_dims = [
                    list(self.main_app.exp_df.columns).index(var)
                    for var in categorical_variables
                    if var in self.main_app.exp_df.columns
                ]
                
                # Train using session API
                results = self.main_app.session.train_model(
                    backend='botorch',
                    kernel=kernel,
                    kernel_params=kernel_params,
                    cat_dims=cat_dims,
                    training_iter=100,
                    input_transform_type=self.bt_input_scale_var.get(),
                    output_transform_type=self.bt_output_scale_var.get()
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
            
            # Get the trained model from session
            model = self.main_app.session.model
            
            # Store in main_app for compatibility
            self.main_app.gpr_model = model
            
            # Get detailed per-fold CV metrics for visualization
            # The session API only returns aggregated metrics, but we need per-fold for plots
            cv_metrics = model.evaluate(
                self.main_app.session.experiment_manager,
                cv_splits=5,
                debug=debug,
                progress_callback=progress_callback
            )
            
            # Store per-fold metrics in the main app for visualization
            self.main_app.rmse_values = cv_metrics.get("RMSE", [])
            self.main_app.mae_values = cv_metrics.get("MAE", [])
            self.main_app.mape_values = cv_metrics.get("MAPE", [])
            self.main_app.r2_values = cv_metrics.get("R²", [])
            
            # Store hyperparameters
            self.main_app.learned_hyperparameters = results.get('hyperparameters', {})
            
            # Check for calibration warnings and show popup if needed
            if hasattr(model, 'calibration_enabled') and model.calibration_enabled:
                from ui.notifications import show_calibration_warning
                show_calibration_warning(self, model.calibration_factor, backend)
            
            # Get aggregated metrics from session results for console output
            session_metrics = results.get('metrics', {})
            
            print(f"Model trained successfully with {backend} backend")
            print(f"  R² = {session_metrics.get('r2', 'N/A'):.3f}")
            print(f"  RMSE = {session_metrics.get('rmse', 'N/A'):.3f}")
            print("Learned hyperparameters:", self.main_app.learned_hyperparameters)
            
            # Initialize visualizations
            self.visualizations = Visualizations(
                parent=self,
                search_space=self.main_app.search_space,
                gpr_model=self.main_app.gpr_model,
                exp_df=self.main_app.exp_df,
                encoded_X=self.main_app.exp_df.drop(columns='Output'),
                encoded_y=self.main_app.exp_df['Output']
            )
            self.visualizations.rmse_values = self.main_app.rmse_values
            self.visualizations.mae_values = self.main_app.mae_values
            self.visualizations.mape_values = self.main_app.mape_values
            self.visualizations.r2_values = self.main_app.r2_values
            self.visualize_button.configure(state="normal")
            
            # Enable acquisition panel
            if hasattr(self.main_app, 'acquisition_panel'):
                self.main_app.acquisition_panel.enable()
        
        except Exception as e:
            print(f"Error training model: {e}")
            traceback.print_exc()

    # ==========================
    # VISUALIZATIONS
    # ==========================
    def initialize_visualizations(self):
        self.visualizations = Visualizations(
            parent=self,
            search_space=self.main_app.search_space,
            gpr_model=self.main_app.gpr_model,
            exp_df=self.main_app.exp_df,
            encoded_X=self.main_app.exp_df.drop(columns='Output'),
            encoded_y=self.main_app.exp_df['Output']
        )
        self.visualizations.rmse_values = self.main_app.rmse_values
        self.visualizations.mae_values = self.main_app.mae_values
        self.visualizations.mape_values = self.main_app.mape_values
        self.visualizations.r2_values = self.main_app.r2_values
        self.visualize_button.configure(state="normal")

    def open_visualization_window(self):
        if self.visualizations is not None:
            self.visualizations.open_window()
        else:
            print("Error: Visualizations have not been initialized.")

    # ==========================
    # UTILITY METHODS
    # ==========================
    def get_hyperparameters(self):
        if not hasattr(self.main_app, 'gpr_model') or self.main_app.gpr_model is None:
            print("Error: Model is not trained.")
            return None
        return self.main_app.gpr_model.get_hyperparameters()

    def update_search_space(self, search_space, categorical_variables):
        """Update the search space and categorical variables."""
        # Store the search space manager reference
        self.search_space_manager = search_space
        
        # Always create a skopt-compatible version for functions that need iteration
        if hasattr(search_space, 'to_skopt'):
            self.search_space = search_space.to_skopt()
        else:
            self.search_space = search_space
            
        self.categorical_variables = categorical_variables
        
        # Also store in main_app for tabbed mode
        if hasattr(self, 'main_app'):
            # Store both formats in the main app
            self.main_app.search_space_manager = search_space  # Store the manager object
            
            # Always ensure main_app.search_space is the skopt-compatible version
            if hasattr(search_space, 'to_skopt'):
                self.main_app.search_space = search_space.to_skopt()
            else:
                self.main_app.search_space = search_space
                
        print("Updated search space:", self.search_space_manager)
        print("Categorical variables:", self.categorical_variables)

    def reset_model(self):
        """Reset the model completely, useful when columns change."""
        # Clear the model instance
        self.model_instance = None
        
        # Reset visualization button state
        self.visualize_button.configure(state="disabled")
        
        # Enable training button
        self.train_button.configure(state="normal")
        
        # Inform user
        print("Model has been reset. Please train again to apply changes.")

    def prepare_experiment_data(self):
        """Prepare experiment data for model training"""
        # Create a clean experiment manager with current data
        experiment_manager = ExperimentManager(self.main_app.search_space_manager)
        
        # Add current experiment data to the manager
        exp_df = self.main_app.exp_df.copy()
        
        # Simply note whether noise data exists (without implying defaults)
        if 'Noise' not in exp_df.columns:
            print("No noise column found. Models will use their internal regularization.")
        
        experiment_manager.add_experiments_batch(exp_df)
        return experiment_manager

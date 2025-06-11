import customtkinter as ctk
import numpy as np
import pandas as pd

from ui.notifications import ResultNotificationWindow

class AcquisitionPanel(ctk.CTkScrollableFrame):  # Changed from CTkFrame to CTkScrollableFrame
    def __init__(self, parent, main_app):
        super().__init__(parent, height=600, width=300)  # Added height and width parameters
        
        self.main_app = main_app  # Reference to main application
        
        # Title at the very top
        ctk.CTkLabel(self, text='Acquisition Functions', font=('Arial', 16)).pack(pady=5)
        
        # Create frames for different backends
        self.acq_sklearn_frame = ctk.CTkFrame(self)
        self.acq_botorch_frame = ctk.CTkFrame(self)
        
        # Create the widgets for each backend
        self.create_sklearn_widgets()
        self.create_botorch_widgets()
        
        # Initial load - always starts with scikit-learn
        self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
        # Run strategy button right after the acquisition functions (moved up)
        self.run_button = ctk.CTkButton(
            self, 
            text="Run Acquisition Strategy",
            command=self.run_selected_strategy,
            state="disabled"  # Disabled by default until model is trained
        )
        self.run_button.pack(pady=10, padx=10, fill="x")
        
        # Add separator
        separator = ctk.CTkFrame(self, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=15)
        
        # Add "Model Maximum" section
        self.create_model_maximum_section()
        
    def create_sklearn_widgets(self):
        """Create acquisition function widgets for scikit-learn logic."""
        self.acq_sklearn_var = ctk.StringVar(value="Expected Improvement (EI)")
        ctk.CTkLabel(self.acq_sklearn_frame, text="Acquisition Strategy:").pack(pady=2)
        self.acq_sklearn_menu = ctk.CTkOptionMenu(
            self.acq_sklearn_frame,
            values=["Expected Improvement (EI)", "Upper Confidence Bound (UCB)", 
                   "Probability of Improvement (PI)", "GP Hedge (Auto-balance)"],
            variable=self.acq_sklearn_var,
            command=self.update_acq_param_visibility
        )
        self.acq_sklearn_menu.pack(pady=5)
        
        # Add maximize/minimize segmented button
        ctk.CTkLabel(self.acq_sklearn_frame, text="Optimization Goal:").pack(pady=2)
        self.goal_options = ["Maximize", "Minimize"]
        self.goal_var = ctk.StringVar(value="Maximize")
        self.goal_segmented = ctk.CTkSegmentedButton(
            self.acq_sklearn_frame,
            values=self.goal_options,
            variable=self.goal_var,
        )
        self.goal_segmented.pack(pady=5)
        
        # Add acquisition function parameter widgets
        ctk.CTkLabel(self.acq_sklearn_frame, text="Acquisition Parameters:").pack(pady=2)
        
        # xi parameter frame (for EI and PI) - Updated to go down to 0.0001
        self.xi_frame = ctk.CTkFrame(self.acq_sklearn_frame)
        self.xi_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.xi_frame, text="ξ (xi):").pack(side="left", padx=5)
        self.xi_value_label = ctk.CTkLabel(self.xi_frame, text="0.01")
        self.xi_value_label.pack(side="right", padx=5)
        self.xi_slider = ctk.CTkSlider(
            self.xi_frame, 
            from_=0.0001,  # Changed from 0.001 to 0.0001
            to=0.1, 
            number_of_steps=999,  # Increased steps for finer control
            command=self.update_xi_value
        )
        self.xi_slider.pack(fill="x", padx=5)
        self.xi_slider.set(0.01)  # Default value
        
        # kappa parameter frame (for UCB)
        self.kappa_frame = ctk.CTkFrame(self.acq_sklearn_frame)
        self.kappa_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.kappa_frame, text="κ (kappa):").pack(side="left", padx=5)
        self.kappa_value_label = ctk.CTkLabel(self.kappa_frame, text="1.96")
        self.kappa_value_label.pack(side="right", padx=5)
        self.kappa_slider = ctk.CTkSlider(
            self.kappa_frame, 
            from_=0.1, 
            to=5.0, 
            number_of_steps=49,
            command=self.update_kappa_value
        )
        self.kappa_slider.pack(fill="x", padx=5)
        self.kappa_slider.set(1.96)  # Default value
        
        # Parameter info/help labels
        self.xi_info_label = ctk.CTkLabel(
            self.acq_sklearn_frame, 
            text="ξ (xi): Higher values favor exploration over exploitation", 
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        )
        self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))

        self.kappa_info_label = ctk.CTkLabel(
            self.acq_sklearn_frame, 
            text="κ (kappa): Higher values increase exploration", 
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        )
        self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))

        # Set initial visibility
        self.update_acq_param_visibility(self.acq_sklearn_var.get())

    def create_botorch_widgets(self):
        """Create acquisition function widgets for BoTorch logic."""
        # Create a frame for the segmented button that will always be visible
        self.botorch_type_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_type_frame.pack(fill="x", expand=True, pady=5)
        
        # Create tabs for regular, batch, and exploratory acquisition functions in this separate frame
        ctk.CTkLabel(self.botorch_type_frame, text="Acquisition Type:").pack(pady=2)
        self.botorch_acq_type_var = ctk.StringVar(value="Regular")
        self.botorch_acq_type_seg = ctk.CTkSegmentedButton(
            self.botorch_type_frame,
            values=["Regular", "Batch", "Exploratory"],
            variable=self.botorch_acq_type_var,
            command=self.update_botorch_acq_type
        )
        self.botorch_acq_type_seg.pack(pady=5)
        
        # Create subframes for different acquisition function types
        self.botorch_acq_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_batch_frame = ctk.CTkFrame(self.acq_botorch_frame)
        self.botorch_exploratory_frame = ctk.CTkFrame(self.acq_botorch_frame)
        
        # Regular acquisition functions
        self.acq_botorch_var = ctk.StringVar(value="Expected Improvement")  # Changed default
        ctk.CTkLabel(self.botorch_acq_frame, text="Acquisition Function:").pack(pady=2)
        self.acq_botorch_menu = ctk.CTkOptionMenu(
            self.botorch_acq_frame,
            values=[
                "Expected Improvement",
                "Log Expected Improvement",
                "Probability of Improvement",
                "Log Probability of Improvement",
                "Upper Confidence Bound"
            ],
            variable=self.acq_botorch_var,
            command=self.update_botorch_description
        )
        self.acq_botorch_menu.pack(pady=5)
        
        # Batch acquisition functions
        self.acq_botorch_batch_var = ctk.StringVar(value="q-Expected Improvement")
        ctk.CTkLabel(self.botorch_batch_frame, text="Batch Acquisition Function:").pack(pady=2)
        self.acq_botorch_batch_menu = ctk.CTkOptionMenu(
            self.botorch_batch_frame,
            values=[
                "q-Expected Improvement",
                "q-Upper Confidence Bound"
            ],
            variable=self.acq_botorch_batch_var,
            command=self.update_botorch_description
        )
        self.acq_botorch_batch_menu.pack(pady=5)
        
        # Batch size selection
        ctk.CTkLabel(self.botorch_batch_frame, text="Batch Size (q):").pack(pady=2)
        self.botorch_batch_size_var = ctk.IntVar(value=2)
        batch_sizes = [str(i) for i in range(2, 11)]  # 2 to 10
        self.botorch_batch_size_menu = ctk.CTkOptionMenu(
            self.botorch_batch_frame,
            values=batch_sizes,
            command=lambda v: self.botorch_batch_size_var.set(int(v))
        )
        self.botorch_batch_size_menu.pack(pady=5)
        
        # Beta parameter for q-UCB
        self.botorch_beta_frame = ctk.CTkFrame(self.botorch_batch_frame)
        self.botorch_beta_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(self.botorch_beta_frame, text="β (beta):").pack(side="left", padx=5)
        self.botorch_beta_value_label = ctk.CTkLabel(self.botorch_beta_frame, text="0.5")
        self.botorch_beta_value_label.pack(side="right", padx=5)
        self.botorch_beta_slider = ctk.CTkSlider(
            self.botorch_beta_frame, 
            from_=0.1, 
            to=2.0, 
            number_of_steps=19,
            command=self.update_botorch_beta_value
        )
        self.botorch_beta_slider.pack(fill="x", padx=5)
        self.botorch_beta_slider.set(0.5)  # Default value
        
        # Exploratory acquisition functions
        ctk.CTkLabel(
            self.botorch_exploratory_frame, 
            text="Integrated Posterior Variance",
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        ctk.CTkLabel(
            self.botorch_exploratory_frame,
            text="This purely exploratory acquisition function selects points to reduce model uncertainty, "
                 "without considering optimization goals.",
            wraplength=250,
            justify="left"
        ).pack(pady=5)
        
        # Number of MC points for exploratory acquisition
        ctk.CTkLabel(self.botorch_exploratory_frame, text="MC Integration Points:").pack(pady=2)
        self.botorch_mc_points_var = ctk.IntVar(value=1000)
        
        def validate_mc_points(action, value_if_allowed):
            """Validate MC points entry to prevent empty values"""
            if value_if_allowed == "":
                return False
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False

        # Use validation for the MC points entry
        vcmd = (self.register(validate_mc_points), '%d', '%P')
        self.botorch_mc_points_entry = ctk.CTkEntry(
            self.botorch_exploratory_frame,
            textvariable=self.botorch_mc_points_var,
            width=100,
            validate="key",
            validatecommand=vcmd
        )
        self.botorch_mc_points_entry.pack(pady=5)
        
        # Add explanation for MC points
        ctk.CTkLabel(
            self.botorch_exploratory_frame, 
            text="Higher values improve accuracy but increase computation time.\nTypically 500-2000 points works well.",
            font=("Arial", 10), 
            text_color="gray",
            wraplength=250
        ).pack(pady=(0, 5))
        
        # Add maximize/minimize option for regular and batch acquisitions
        ctk.CTkLabel(self.acq_botorch_frame, text="Optimization Goal:").pack(pady=2)
        self.botorch_goal_options = ["Maximize", "Minimize"]
        self.botorch_goal_var = ctk.StringVar(value="Maximize")
        self.botorch_goal_segmented = ctk.CTkSegmentedButton(
            self.acq_botorch_frame,
            values=self.botorch_goal_options,
            variable=self.botorch_goal_var,
        )
        self.botorch_goal_segmented.pack(pady=5)
        
        # Add description of selected acquisition function
        descrip_frame = ctk.CTkFrame(self.acq_botorch_frame)
        descrip_frame.pack(fill="x", padx=10, pady=5)
        
        # Strategy descriptions dictionary
        self.strategy_info = {
            "Expected Improvement": (
                "Expected Improvement (EI) balances exploration and exploitation by calculating "
                "the expected improvement over the current best observed value."
            ),
            "Log Expected Improvement": (
                "LogEI is a numerically stable version of EI. It balances exploration and exploitation "
                "by calculating the log of the expected improvement over the current best value."
            ),
            "Probability of Improvement": (
                "Probability of Improvement (PI) selects points based on the probability that they "
                "will improve over the current best observed value."
            ),
            "Log Probability of Improvement": (
                "LogPI is a numerically stable version of PI. It calculates the log of the probability "
                "of improving over the current best value."
            ),
            "Upper Confidence Bound": (
                "Upper Confidence Bound (UCB) balances exploration and exploitation by selecting "
                "points where the upper confidence bound is highest (for maximization)."
            ),
            "q-Expected Improvement": (
                "q-Expected Improvement (qEI) is a batch version of EI that selects multiple points "
                "simultaneously while accounting for the interactions between selections."
            ),
            "q-Upper Confidence Bound": (
                "q-Upper Confidence Bound (qUCB) is a batch version of UCB that selects multiple points "
                "simultaneously while balancing exploration and exploitation."
            ),
            "Integrated Posterior Variance": (
                "q-Negative Integrated Posterior Variance (qNIPV) selects points to reduce overall model "
                "uncertainty. This is a purely exploratory function for active learning, not optimization."
            )
        }
        
        self.acq_descrip_label = ctk.CTkLabel(
            descrip_frame,
            text=self.strategy_info["Expected Improvement"],
            wraplength=250,
            justify="left",
            text_color="light gray"
        )
        self.acq_descrip_label.pack(pady=5, fill="x")
        
        # Show the regular acquisition frame by default
        self.botorch_acq_frame.pack(fill="x", expand=True, pady=5)

    def update_botorch_acq_type(self, selection):
        """Update the visible botorch acquisition function type frame."""
        # Hide all content frames first
        self.botorch_acq_frame.pack_forget()
        self.botorch_batch_frame.pack_forget()
        self.botorch_exploratory_frame.pack_forget()
        
        # Show the selected frame
        if selection == "Regular":
            self.botorch_acq_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description(self.acq_botorch_var.get())
        elif selection == "Batch":
            self.botorch_batch_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description(self.acq_botorch_batch_var.get())
        elif selection == "Exploratory":
            self.botorch_exploratory_frame.pack(fill="x", expand=True, pady=5)
            self.update_botorch_description("Integrated Posterior Variance")

    def update_botorch_description(self, selection):
        """Update the description text for the selected BoTorch acquisition function."""
        self.acq_descrip_label.configure(text=self.strategy_info.get(selection, ""))
        
        # Show/hide beta slider for UCB functions
        if selection == "q-Upper Confidence Bound":
            self.botorch_beta_frame.pack(fill="x", pady=5)
        else:
            # Only hide if we're in batch mode
            if self.botorch_acq_type_var.get() == "Batch":
                self.botorch_beta_frame.pack_forget()

    def update_botorch_beta_value(self, value):
        """Update the beta parameter value label for BoTorch UCB functions."""
        self.botorch_beta_value_label.configure(text=f"{value:.2f}")

    def create_ax_widgets(self):
        """Create acquisition function widgets for Ax logic."""
        info_text = (
            "Ax is designed to work autonomously with minimal configuration.\n\n"
            "It automatically selects appropriate acquisition functions and optimization strategies "
            "based on your problem characteristics and available data."
        )
        ctk.CTkLabel(
            self.acq_ax_frame, 
            text=info_text,
            wraplength=250,
            justify="left"
        ).pack(pady=10, padx=10)
        
        # Add maximize/minimize option
        ctk.CTkLabel(self.acq_ax_frame, text="Optimization Goal:").pack(pady=2)
        self.ax_goal_options = ["Maximize", "Minimize"]
        self.ax_goal_var = ctk.StringVar(value="Maximize")
        self.ax_goal_segmented = ctk.CTkSegmentedButton(
            self.acq_ax_frame,
            values=self.ax_goal_options,
            variable=self.ax_goal_var,
        )
        self.ax_goal_segmented.pack(pady=5)

    def load_backend_options(self, event=None):
        """Show acquisition options based on selected logic."""
        self.acq_sklearn_frame.pack_forget()
        self.acq_botorch_frame.pack_forget()
        
        backend = self.backend_var.get()
        if backend == "scikit-learn":
            self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "botorch":
            self.acq_botorch_frame.pack(fill="x", expand=True, padx=10, pady=5)
        # Removed the Ax backend option
        
        # Sync the backend selection with the model panel
        if hasattr(self.main_app, 'model_frame') and self.main_app.model_frame is not None:
            self.main_app.model_frame.backend_var.set(backend)
            self.main_app.model_frame.load_backend_options()

    def toggle_gp_hedge(self):
        """Toggle between manual strategy selection and automatic GP hedge."""
        is_gp_hedge = self.use_gp_hedge_var.get()
        
        # Enable/disable the strategy dropdown
        self.acq_sklearn_menu.configure(state="disabled" if is_gp_hedge else "normal")
        
        # Show/hide parameter sliders appropriately
        if is_gp_hedge:
            # GP Hedge can use both parameters
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
        else:
            # Regular strategy selection - update based on current selection
            self.update_acq_param_visibility()

    def update_xi_value(self, value):
        """Update the xi parameter value label."""
        self.xi_value_label.configure(text=f"{value:.3f}")

    def update_kappa_value(self, value):
        """Update the kappa parameter value label."""
        self.kappa_value_label.configure(text=f"{value:.2f}")

    def update_acq_param_visibility(self, selection=None):
        """Show/hide acquisition function parameters based on selection."""
        if selection is None:
            selection = self.acq_sklearn_var.get()
            
        if selection in ["Expected Improvement (EI)", "Probability of Improvement (PI)"]:
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack_forget()
            self.kappa_info_label.pack_forget()
        elif selection == "Upper Confidence Bound (UCB)":
            self.xi_frame.pack_forget()
            self.xi_info_label.pack_forget()
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
        elif selection == "GP Hedge (Auto-balance)":
            # GP Hedge shows both parameters
            self.xi_frame.pack(fill="x", pady=5)
            self.xi_info_label.pack(after=self.xi_frame, pady=(0, 5))
            self.kappa_frame.pack(fill="x", pady=5)
            self.kappa_info_label.pack(after=self.kappa_frame, pady=(0, 5))
    
    def update_for_backend(self, backend):
        """Update acquisition options based on GPR backend selection."""
        self.acq_sklearn_frame.pack_forget()
        self.acq_botorch_frame.pack_forget()
        
        if backend == "scikit-learn":
            self.acq_sklearn_frame.pack(fill="x", expand=True, padx=10, pady=5)
        elif backend == "botorch":
            self.acq_botorch_frame.pack(fill="x", expand=True, padx=10, pady=5)
        # Removed the Ax backend option

    def run_selected_strategy(self):
        """Execute the selected acquisition strategy."""
        # Access GPR model from main app
        if not hasattr(self.main_app, 'gpr_model') or self.main_app.gpr_model is None:
            print("Error: Model not trained. Please train the model first.")
            return
        
        # Get backend from the main app's model panel
        if hasattr(self.main_app, 'model_frame'):
            backend = self.main_app.model_frame.backend_var.get()
        else:
            backend = "scikit-learn"  # Default
        
        try:
            # Strategy description mapping for scikit-learn
            sklearn_strategy_descriptions = {
                "Expected Improvement (EI)": (
                    "Expected Improvement (EI) balances exploration and exploitation by "
                    "selecting points with the highest expected improvement over the current best value. "
                    "Higher ξ values favor exploration over exploitation."
                ),
                "Upper Confidence Bound (UCB)": (
                    "Upper Confidence Bound (UCB) balances exploration and exploitation by "
                    "selecting points with the highest upper bound of the confidence interval. "
                    "Higher κ values lead to more exploration."
                ),
                "Probability of Improvement (PI)": (
                    "Probability of Improvement (PI) selects points with the highest probability of "
                    "improving over the current best value. Higher ξ values favor exploration over exploitation."
                ),
                "GP Hedge (Auto-balance)": (
                    "GP Hedge automatically balances between multiple acquisition strategies "
                    "(Expected Improvement, Upper Confidence Bound, and Probability of Improvement) "
                    "by adaptively selecting the best-performing strategy during optimization."
                )
            }
            
            if backend == "scikit-learn":
                strategy = self.acq_sklearn_var.get()
                maximize = self.goal_var.get() == "Maximize"
                
                # Import acquisition framework
                from logic.acquisition.skopt_acquisition import SkoptAcquisition
                
                # Check if GP Hedge is selected
                is_gp_hedge = strategy == "GP Hedge (Auto-balance)"
                
                if is_gp_hedge:
                    acq_func = "gp_hedge"
                    acq_func_kwargs = {
                        "xi": float(self.xi_slider.get()),
                        "kappa": float(self.kappa_slider.get())
                    }
                else:
                    acq_func_map = {
                        "Expected Improvement (EI)": "ei",
                        "Upper Confidence Bound (UCB)": "ucb",
                        "Probability of Improvement (PI)": "pi"
                    }
                    acq_func = acq_func_map.get(strategy, "ei")
                    
                    # Get acquisition function parameters
                    acq_func_kwargs = {}
                    if acq_func in ["ei", "pi"]:
                        acq_func_kwargs["xi"] = float(self.xi_slider.get())
                    elif acq_func == "ucb":
                        acq_func_kwargs["kappa"] = float(self.kappa_slider.get())
                
                # Create acquisition function using trained model
                acquisition = SkoptAcquisition(
                    search_space=self.main_app.search_space,
                    model=self.main_app.gpr_model,
                    acq_func=acq_func,
                    maximize=maximize,
                    acq_func_kwargs=acq_func_kwargs,
                    random_state=42
                )
                
                # Update with existing data
                acquisition.update(
                    self.main_app.exp_df.drop(columns='Output'),
                    self.main_app.exp_df['Output']
                )
                
                # Get next point
                next_point = acquisition.select_next()
                
                # Convert to DataFrame for consistency
                # Get only the feature columns (excluding Output and Noise)
                feature_cols = [col for col in self.main_app.exp_df.columns if col not in ['Output', 'Noise']]
                next_point_df = pd.DataFrame([next_point], columns=feature_cols)

                # If the main data has a Noise column, add a default noise value to the result
                if 'Noise' in self.main_app.exp_df.columns:
                    next_point_df['Noise'] = 1e-6  # Default small noise value

                # Get predicted value and std at this point
                if hasattr(self.main_app.gpr_model, 'predict_with_std'):
                    # Important: Make a copy without the Noise column for prediction
                    pred_df = next_point_df.drop(columns=['Noise']) if 'Noise' in next_point_df.columns else next_point_df
                    pred_value, pred_std = self.main_app.gpr_model.predict_with_std(pred_df)
                    pred_value = pred_value[0]
                    pred_std = pred_std[0]
                else:
                    pred_df = next_point_df.drop(columns=['Noise']) if 'Noise' in next_point_df.columns else next_point_df
                    pred_value = self.main_app.gpr_model.predict(pred_df)[0]
                    pred_std = None
                
                # Store the next point
                self.main_app.next_point = next_point_df
                
                # Update the plot
                self.main_app.update_pool_plot()
                
                # Prepare result data for the notification window
                result_data = {
                    'point_df': next_point_df,
                    'value': pred_value,
                    'std': pred_std,
                    'maximize': maximize,
                    'strategy_type': strategy,
                    'strategy_params': acq_func_kwargs,
                    'strategy_description': sklearn_strategy_descriptions.get(strategy, "")
                }
                
                # Prepare model data for the notification window
                model_data = {
                    'backend': backend,
                    'kernel': str(self.main_app.gpr_model.kernel if hasattr(self.main_app.gpr_model, 'kernel') else "Unknown"),
                    'hyperparameters': self.main_app.gpr_model.get_hyperparameters(),
                    'metrics': {
                        'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                        'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
                    }
                }
                
                # Log the acquisition if a logger exists
                if hasattr(self.main_app, 'experiment_logger'):
                    self.main_app.experiment_logger.log_acquisition(result_data)
                
                # Show the notification window
                notification = ResultNotificationWindow(self, result_data, model_data)
                
                # Print strategy details
                param_str = ""
                if acq_func in ["ei", "pi"]:
                    param_str = f", ξ={acq_func_kwargs['xi']:.3f}"
                elif acq_func == "ucb":
                    param_str = f", κ={acq_func_kwargs['kappa']:.2f}"
                elif acq_func == "gp_hedge":
                    param_str = f", ξ={acq_func_kwargs['xi']:.3f}, κ={acq_func_kwargs['kappa']:.2f}"
                    
                print(f"Strategy '{strategy}' ({'maximizing' if maximize else 'minimizing'}{param_str}) executed successfully.")
                
            elif backend == "botorch":
                maximize = self.botorch_goal_var.get() == "Maximize"
                acq_type = self.botorch_acq_type_var.get()
                
                # Determine the acquisition function and its parameters based on the selected type
                if acq_type == "Regular":
                    strategy = self.acq_botorch_var.get()
                    acq_func_map = {
                        "Expected Improvement": "ei",
                        "Log Expected Improvement": "logei",
                        "Probability of Improvement": "pi",
                        "Log Probability of Improvement": "logpi",
                        "Upper Confidence Bound": "ucb"
                    }
                    acq_func = acq_func_map.get(strategy, "logei")
                    batch_size = 1  # Regular acquisition functions use q=1
                    
                    # Set parameters for UCB if selected
                    acq_func_kwargs = {}
                    if acq_func == "ucb":
                        acq_func_kwargs = {"beta": 0.5}  # Default value for regular UCB
                    
                elif acq_type == "Batch":
                    strategy = self.acq_botorch_batch_var.get()
                    acq_func_map = {
                        "q-Expected Improvement": "qei",
                        "q-Upper Confidence Bound": "qucb"
                    }
                    acq_func = acq_func_map.get(strategy, "qei")
                    batch_size = self.botorch_batch_size_var.get()
                    
                    # Set parameters for qUCB if selected
                    acq_func_kwargs = {"mc_samples": 128}  # Default MC samples
                    if acq_func == "qucb":
                        acq_func_kwargs["beta"] = float(self.botorch_beta_slider.get())
                    
                elif acq_type == "Exploratory":
                    strategy = "Integrated Posterior Variance"
                    acq_func = "qipv"
                    batch_size = 1  # Usually 1 for exploratory acquisition
                    # Set parameters for qNIPV
                    acq_func_kwargs = {
                        "n_mc_points": self.botorch_mc_points_var.get(),
                        "mc_samples": 128  # Default MC samples
                    }
                
                # Import BoTorch acquisition framework
                from logic.acquisition.botorch_acquisition import BoTorchAcquisition
                
                # Create acquisition function using trained model
                acquisition = BoTorchAcquisition(
                    search_space=self.main_app.search_space_manager,
                    model=self.main_app.gpr_model,
                    acq_func=acq_func,
                    maximize=maximize,
                    acq_func_kwargs=acq_func_kwargs,
                    random_state=42,
                    batch_size=batch_size
                )
                
                try:
                    # Get next point(s)
                    next_points = acquisition.select_next()
                    
                    # Handle both single point and batch point returns
                    if isinstance(next_points, list):
                        # Multiple points returned (batch acquisition)
                        next_point_dfs = [pd.DataFrame([pt], columns=list(pt.keys())) for pt in next_points]
                        next_point_df = pd.concat(next_point_dfs, ignore_index=True)
                        batch_result = True
                    else:
                        # Single point returned
                        next_point_df = pd.DataFrame([next_points], columns=list(next_points.keys()))
                        batch_result = False
                    
                    # Get predicted values and std at these points
                    if hasattr(self.main_app.gpr_model, 'predict_with_std'):
                        pred_values, pred_stds = self.main_app.gpr_model.predict_with_std(next_point_df)
                        # If batch, keep as arrays; if single point, get first value
                        if not batch_result:
                            pred_value = pred_values[0]
                            pred_std = pred_stds[0]
                        else:
                            pred_value = pred_values  # Keep as array for batch
                            pred_std = pred_stds      # Keep as array for batch
                    else:
                        pred_value = self.main_app.gpr_model.predict(next_point_df)
                        pred_std = None
                    
                    # Store the next point(s)
                    self.main_app.next_point = next_point_df
                    
                    # Update the plot
                    self.main_app.update_pool_plot()
                    
                    # Format strategy name for display
                    if batch_result:
                        display_strategy = f"BoTorch Batch: {strategy} (q={batch_size})"
                    else:
                        display_strategy = f"BoTorch: {strategy}"
                    
                    # Get strategy description from the info dictionary
                    strategy_description = self.strategy_info.get(strategy, "")
                    
                    # Print the result for debugging
                    print("ACQUISITION STRATEGY")
                    print(f"Strategy: {display_strategy}")
                    print(f"Goal: {'Maximize' if maximize else 'Minimize'}")
                    print(f"Batch size: {batch_size}")
                    
                    if batch_result:
                        print(f"Selected {len(next_points)} points:")
                        for i, point in enumerate(next_points):
                            print(f"Point {i+1}:")
                            for k, v in point.items():
                                print(f"  {k}: {v}")
                            if i < len(pred_values):
                                print(f"  Predicted Value: {pred_values[i]:.4f}")
                                print(f"  Prediction Std: {pred_stds[i]:.4f}")
                                print(f"  95% CI: [{pred_values[i] - 1.96*pred_stds[i]:.4f}, {pred_values[i] + 1.96*pred_stds[i]:.4f}]")
                    else:
                        print("Selected point:")
                        for k, v in next_points.items():
                            print(f"  {k}: {v}")
                        print(f"Predicted Value: {pred_value:.4f}")
                        print(f"Prediction Std: {pred_std:.4f}")
                        print(f"95% CI: [{pred_value - 1.96*pred_std:.4f}, {pred_value + 1.96*pred_std:.4f}]")
                
                    # Prepare result data for the notification window
                    result_data = {
                        'point_df': next_point_df,
                        'value': pred_value,
                        'std': pred_std,
                        'maximize': maximize,
                        'strategy_type': display_strategy,
                        'strategy_params': acq_func_kwargs,
                        'strategy_description': strategy_description,
                        'is_batch': batch_result,
                        'batch_size': batch_size if batch_result else 1
                    }
                    
                    # Prepare model data
                    model_data = {
                        'backend': 'botorch',
                        'kernel': self.main_app.gpr_model.get_hyperparameters().get('kernel', 'Unknown'),
                        'hyperparameters': self.main_app.gpr_model.get_hyperparameters(),
                        'metrics': {
                            'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                            'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
                        }
                    }
                    
                    # Create notification window
                    notification = ResultNotificationWindow(self, result_data, model_data)
                    
                    # Log the acquisition if a logger exists
                    if hasattr(self.main_app, 'experiment_logger'):
                        self.main_app.experiment_logger.log_acquisition(result_data)
                
                except Exception as e:
                    # If there's an error, let's show a notification with the error information
                    import traceback
                    print(f"Error with BoTorch acquisition: {e}")
                    traceback.print_exc()
                    
                    # Create a placeholder result with error info
                    result_data = {
                        'point_df': None,
                        'value': 0.0,  # Use default values to avoid formatting errors
                        'std': 0.0,
                        'maximize': maximize,
                        'strategy_type': f"BoTorch: {strategy}",
                        'strategy_params': acq_func_kwargs,
                        'strategy_description': f"Error: {str(e)}\n\nPlease check the console for details."
                    }
                    
                    # Show the notification window with error information
                    ResultNotificationWindow(self, result_data, model_data={})
        
        except Exception as e:
            print(f"Error executing strategy: {e}")
            import traceback
            traceback.print_exc()
        '''
        elif backend == "ax":
            maximize = self.ax_goal_var.get() == "Maximize"
            
            # Implementation for Ax will go here in the future
            print("Ax acquisition strategies not fully implemented yet.")
            
            # For now, show a notification with basic info
            result_data = {
                'point_df': None,
                'value': None,
                'std': None,
                'maximize': maximize,
                'strategy_type': "Ax Bayesian Optimization",
                'strategy_params': {},
                'strategy_description': "Ax acquisition strategies will be fully implemented in a future update."
            }
            
            model_data = {
                'backend': backend,
                'kernel': "Ax model",
                'hyperparameters': self.main_app.gpr_model.get_hyperparameters() if hasattr(self.main_app.gpr_model, 'get_hyperparameters') else {},
                'metrics': {}
            }
            
            ResultNotificationWindow(self, result_data, model_data)
            '''
            
    def create_model_maximum_section(self):
        """Create a section for finding the model's predicted optimum."""
        # Create a frame for the model optimum section
        opt_frame = ctk.CTkFrame(self)
        opt_frame.pack(fill="x", expand=True, padx=10, pady=5)
        
        # Add title
        ctk.CTkLabel(opt_frame, text="Model Prediction Optimum", font=('Arial', 14)).pack(pady=5)
        
        # Add explanatory text
        ctk.CTkLabel(
            opt_frame, 
            text="Find the point where the model predicts the optimal value.",
            wraplength=250,
            justify="left"
        ).pack(pady=5)
        
        # Add maximize/minimize option
        ctk.CTkLabel(opt_frame, text="Optimization Goal:").pack(pady=2)
        self.opt_goal_options = ["Maximize", "Minimize"]
        self.opt_goal_var = ctk.StringVar(value="Maximize")
        self.opt_goal_segmented = ctk.CTkSegmentedButton(
            opt_frame,
            values=self.opt_goal_options,
            variable=self.opt_goal_var,
        )
        self.opt_goal_segmented.pack(pady=5)
        
        # Add button to find the optimum
        self.find_opt_button = ctk.CTkButton(
            opt_frame,
            text="Find Model Optimum",
            command=self.find_model_optimum,
            state="disabled"  # Disabled until model is trained
        )
        self.find_opt_button.pack(pady=10, fill="x")
        
        # Warning label about model reliability
        ctk.CTkLabel(
            opt_frame,
            text="Note: This relies entirely on the model's prediction, "
                 "not on acquisition functions that balance exploration and exploitation.",
            wraplength=250,
            text_color="gray",
            justify="left",
            font=("Arial", 10)
        ).pack(pady=5)

    def find_model_optimum(self):
        """Find the point where the model predicts the optimal value."""
        if not hasattr(self.main_app, 'gpr_model') or self.main_app.gpr_model is None:
            print("Error: Model not trained. Please train the model first.")
            return
            
        try:
            # Get whether to maximize or minimize
            maximize = self.opt_goal_var.get() == "Maximize"
            
            # Get the backend from the model panel
            backend = self.main_app.model_frame.backend_var.get()
            
            if backend == "scikit-learn":
                # Import acquisition framework
                from logic.acquisition.skopt_acquisition import SkoptAcquisition
                
                # Create acquisition instance
                acquisition = SkoptAcquisition(
                    search_space=self.main_app.search_space,
                    model=self.main_app.gpr_model,
                    maximize=maximize,
                    random_state=42
                )
                
                # Find the optimum
                print(f"Searching for model's predicted {'maximum' if maximize else 'minimum'}...")
                result = acquisition.find_optimum(
                    model=self.main_app.gpr_model,
                    maximize=maximize,
                    random_state=42
                )
                
            elif backend == "botorch":
                # Import BoTorch acquisition framework
                from logic.acquisition.botorch_acquisition import BoTorchAcquisition
                
                # Create acquisition instance
                acquisition = BoTorchAcquisition(
                    search_space=self.main_app.search_space_manager,
                    model=self.main_app.gpr_model,
                    acq_func="ucb",  # This doesn't matter for find_optimum
                    maximize=maximize,
                    random_state=42
                )
                
                # Find the optimum using BoTorch
                result = self._find_botorch_optimum(acquisition, maximize)
                
            else:
                print(f"Finding model optimum not implemented for {backend} backend")
                return
                
            # Extract results
            opt_point_df = result['x_opt']
            
            # Add this check:
            if 'Noise' in self.main_app.exp_df.columns and 'Noise' not in opt_point_df.columns:
                opt_point_df['Noise'] = 1e-6  # Add default noise
            
            # Get predicted value and std at optimum (ensure we drop Noise for prediction)
            pred_df = opt_point_df.drop(columns=['Noise']) if 'Noise' in opt_point_df.columns else opt_point_df
            opt_value = float(result['value'])
            opt_std = float(result['std']) if result.get('std') is not None else None
            
            # Store the optimum point
            self.main_app.next_point = opt_point_df
            
            # Update the plot
            self.main_app.update_pool_plot()
            
            # Prepare result data for the notification window
            result_data = {
                'point_df': opt_point_df,
                'value': opt_value,
                'std': opt_std,
                'maximize': maximize,
                'strategy_type': "Model Optimum Finder",
                'strategy_params': {'random_state': 42},
                'strategy_description': (
                    "This strategy directly finds the point where the model "
                    "predicts the best value, without considering exploration. "
                    "It's best used when you're confident in your model's accuracy "
                    "and want to exploit its predictions."
                )
            }
            
            # Get metrics if available
            metrics = {
                'RMSE': self.main_app.rmse_values[-1] if hasattr(self.main_app, 'rmse_values') and self.main_app.rmse_values else None,
                'R²': self.main_app.r2_values[-1] if hasattr(self.main_app, 'r2_values') and self.main_app.r2_values else None
            }
            
            # Use the backend-agnostic model data collector
            from ui.utils import get_model_data
            model_data = {
                'backend': backend,
                'kernel': str(self.main_app.gpr_model.kernel if hasattr(self.main_app.gpr_model, 'kernel') else "Unknown"),
                'hyperparameters': self.main_app.gpr_model.get_hyperparameters(),
                'metrics': metrics
            }
            
            # Log the acquisition if a logger exists
            if hasattr(self.main_app, 'experiment_logger'):
                self.main_app.experiment_logger.log_acquisition(result_data)
            
            # Show the notification window
            ResultNotificationWindow(self, result_data, model_data)
            
        except Exception as e:
            print(f"Error finding model optimum: {e}")
            import traceback
            traceback.print_exc()

    def _find_botorch_optimum(self, acquisition, maximize):
        """Find the optimum point using BoTorch."""
        # Get model from the acquisition
        model = acquisition.model
        
        # Use the select_next method with the appropriate acquisition function
        # This is a simplification - in a real implementation you might want to try multiple starts
        next_point = acquisition.select_next()
        
        # Convert to DataFrame for predictions
        if isinstance(next_point, dict):
            next_point_df = pd.DataFrame([next_point])
        else:
            next_point_df = pd.DataFrame([next_point], columns=model.feature_names)
        
        # Get predicted value and std
        pred_mean, pred_std = model.predict_with_std(next_point_df)
        
        # Return in the same format as SkoptAcquisition.find_optimum
        return {
            'x_opt': next_point_df,
            'value': float(pred_mean[0]),
            'std': float(pred_std[0])
        }
    
    def enable(self):
        """
        Enable the acquisition panel buttons after model training is complete.
        Called by the GPR panel after successful model training.
        """
        # Enable the run button
        self.run_button.configure(state="normal")
        
        # Enable the find optimum button
        self.find_opt_button.configure(state="normal")
        
        print("Acquisition panel activated - you can now run acquisition strategies or find model optimum")
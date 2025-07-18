import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import r2_score
from skopt.space import Real, Integer, Categorical
from skopt.learning.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import threading
from ui.custom_widgets import CTkSpinbox

plt.rcParams['savefig.dpi'] = 600


class Visualizations:
    def __init__(self, parent, search_space, gpr_model, exp_df, encoded_X, encoded_y):
        self.parent = parent  # Store the parent widget
        self.search_space = search_space
        
        # Pre-convert search space to skopt if it's a SearchSpace object
        if hasattr(search_space, 'to_skopt'):
            self.skopt_space = search_space.to_skopt()
        else:
            self.skopt_space = search_space
            
        self.gpr_model = gpr_model
        self.exp_df = exp_df
        self.encoded_X = encoded_X
        self.encoded_y = encoded_y

        # Initialize metrics as None; they will be set later
        self.rmse_values = None
        self.mae_values = None
        self.mape_values = None
        self.r2_values = None

        # Delay Matplotlib figure initialization
        self.fig = None
        self.ax = None
        self.canvas = None
        self.colorbar = None

        self.customization_options = {
            'title': "Contour Plot of Model Predictions",
            'xlabel': "X Axis",
            'ylabel': "Y Axis",
            'colormap': "viridis",
            'font': "Arial",
            'title_fontsize': 12,
            'label_fontsize': 10,
            'title_fontweight': "normal",
            'label_fontweight': "normal",
            'tick_style': "x and y",  # Set default to "x and y"
            'tick_direction': "out",  # Set default to "out"
            'show_experiments': False,  # Disabled by default
            'show_next_point': False,    # Disabled by default
            'x_number_format': "auto",  # X-axis number formatting
            'y_number_format': "auto",  # Y-axis number formatting
            'colorbar_format': "auto"   # Colorbar number formatting
        }
        self.last_plot_method = None

    def _get_number_formatter(self, format_type):
        """Create a matplotlib formatter based on the format type."""
        from matplotlib.ticker import FuncFormatter, ScalarFormatter, PercentFormatter
        import matplotlib.ticker as ticker
        
        if format_type == "auto":
            return None  # Use matplotlib's default
        elif format_type == "decimal":
            return FuncFormatter(lambda x, p: f"{x:.2f}")
        elif format_type == "scientific":
            return ticker.ScientificFormatter(precision=2)
        elif format_type == "percent":
            # Handle both 0-1 range and 0-100 range automatically
            def percent_formatter(x, p):
                if abs(x) <= 1.0:  # Assume 0-1 range
                    return f"{x*100:.1f}%"
                else:  # Assume already in percentage
                    return f"{x:.1f}%"
            return FuncFormatter(percent_formatter)
        elif format_type == "integer":
            return FuncFormatter(lambda x, p: f"{int(round(x))}")
        elif format_type == "custom":
            return FuncFormatter(lambda x, p: f"{x:.3g}")  # General format with 3 significant digits
        else:
            return None

    def initialize_figure(self):
        """Initialize the Matplotlib figure and axes."""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))

    def open_window(self):
        """Open the visualization window."""
        self.initialize_figure()  # Ensure the figure is initialized

        if self.search_space is None:
            print("Error: Search space is not defined.")
            return

        # Convert search space to skopt dimensions if it's a SearchSpace object
        skopt_space = self.search_space
        if hasattr(self.search_space, 'to_skopt'):
            skopt_space = self.search_space.to_skopt()
    
        # Now use skopt_space instead of self.search_space
        real_dims = [dim.name for dim in self.skopt_space if isinstance(dim, Real)]
        if not real_dims or len(real_dims) < 2:
            print("Error: Need at least two Real dimensions for contour plotting.")
            return

        # Proceed with the rest of the method
        visualization_window = ctk.CTkToplevel(self.parent)
        visualization_window.title("Model Visualizations")
        visualization_window.geometry("900x650")
        visualization_window.lift()
        visualization_window.focus_force()
        visualization_window.grab_set()

        # Top control frame for other visualizations (metrics, parity, hyperparameters)
        control_frame = ctk.CTkFrame(visualization_window)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.error_metric_var = ctk.StringVar(value="RMSE")
        self.error_metric_menu = ctk.CTkOptionMenu(
            control_frame,
            values=["RMSE", "MAE", "MAPE", "R2"],
            variable=self.error_metric_var
        )
        self.error_metric_menu.pack(side="left", padx=10)
        self.plot_metrics_button = ctk.CTkButton(control_frame, text="Plot Metrics", command=self.plot_metrics)
        self.plot_metrics_button.pack(side="left", padx=10)
        self.plot_parity_button = ctk.CTkButton(control_frame, text="Plot Parity", command=self.plot_parity)
        self.plot_parity_button.pack(side="left", padx=10)

        # Main container holds the plot and the contour controls side by side.
        main_container = ctk.CTkFrame(visualization_window)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Plot frame on the left with reduced height.
        self.plot_frame = ctk.CTkFrame(main_container, height=450)
        self.plot_frame.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        self.metrics_toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.metrics_toolbar.config(background='#2E2E2E')
        self.metrics_toolbar._message_label.config(background='#2E2E2E')
        for button in self.metrics_toolbar.winfo_children():
            button.config(background='#D0D0D0')
        self.metrics_toolbar.winfo_children()[-1].config(background='#2E2E2E')
        self.metrics_toolbar.winfo_children()[-2].config(background='#2E2E2E')
        self.metrics_toolbar.update()

        # Scrollable contour controls frame on the right.
        self.contour_controls_frame = ctk.CTkScrollableFrame(main_container, width=250, height=600)
        self.contour_controls_frame.pack(side="right", fill="y", padx=5, pady=5)

        # Title for the contour controls
        ctk.CTkLabel(self.contour_controls_frame, text="Contour Plot Options", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        # Two dropdowns for selecting the contour plot axes.
        ctk.CTkLabel(self.contour_controls_frame, text="X-Axis Variable:").pack(pady=(5, 2))
        self.contour_axis_x_dropdown = ctk.CTkOptionMenu(
            self.contour_controls_frame, values=real_dims, variable=ctk.StringVar(value=real_dims[0])
        )
        self.contour_axis_x_dropdown.pack(pady=(0, 5))

        ctk.CTkLabel(self.contour_controls_frame, text="Y-Axis Variable:").pack(pady=(5, 2))
        self.contour_axis_y_dropdown = ctk.CTkOptionMenu(
            self.contour_controls_frame, values=real_dims, variable=ctk.StringVar(value=real_dims[1])
        )
        self.contour_axis_y_dropdown.pack(pady=(0, 5))

        # REMOVE: The toggle switches for next point and experimental points
        # They will be added to the customization dialog instead

        # Bind changes in axis selection to update fixed controls.
        self.contour_axis_x_dropdown.configure(command=self.update_fixed_controls)
        self.contour_axis_y_dropdown.configure(command=self.update_fixed_controls)

        # Container for the fixed-value controls for the non-axis dimensions.
        self.fixed_controls_container = ctk.CTkFrame(self.contour_controls_frame)
        self.fixed_controls_container.pack(fill="x", pady=5)
        self.fixed_controls = {}

        # Initially populate the fixed controls.
        self.update_fixed_controls()

        # Button to generate the contour plot.
        self.contour_plot_button = ctk.CTkButton(self.contour_controls_frame, text="Plot Contour", command=self.plot_contour)
        self.contour_plot_button.pack(pady=10)

        self.customize_plot_button = ctk.CTkButton(self.contour_controls_frame, text="Customize Plot", command=self.customize_plot)
        self.customize_plot_button.pack(side='bottom', pady=10)

        # Frame at the bottom to display hyperparameters
        hyperparameters_frame = ctk.CTkFrame(visualization_window)
        hyperparameters_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        hyperparameters = self.get_hyperparameters()
        if hyperparameters:
            hyperparameters_str = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
            ctk.CTkLabel(hyperparameters_frame, text="Learned Hyperparameters", font=('Arial', 16)).pack(pady=5)
            ctk.CTkLabel(hyperparameters_frame, text=hyperparameters_str, justify='left').pack(pady=5)

    def update_fixed_controls(self, event=None):
        """Dynamically create controls to fix values for dimensions not selected as axes."""
        # Clear any existing fixed controls.
        for widget in self.fixed_controls_container.winfo_children():
            widget.destroy()
        self.fixed_controls.clear()

        axis_x = self.contour_axis_x_dropdown.get()
        axis_y = self.contour_axis_y_dropdown.get()

        # Convert search space to skopt dimensions if it's a SearchSpace object
        skopt_space = self.search_space
        if hasattr(self.search_space, 'to_skopt'):
            skopt_space = self.search_space.to_skopt()

        # For every dimension in the search space that is not one of the chosen axes, add a fixed value control.
        for dim in skopt_space:
            if dim.name in [axis_x, axis_y]:
                continue

            label = ctk.CTkLabel(self.fixed_controls_container, text=f"Fixed {dim.name}:")
            label.pack(pady=2)

            if isinstance(dim, Real):
                slider = ctk.CTkSlider(
                    self.fixed_controls_container,
                    from_=dim.bounds[0],
                    to=dim.bounds[1]
                )
                slider.set((dim.bounds[0] + dim.bounds[1]) / 2)
                slider.pack(pady=5)
                self.fixed_controls[dim.name] = slider

            elif isinstance(dim, Integer):
                spinbox = CTkSpinbox(
                    self.fixed_controls_container,
                    step_size=1
                )
                spinbox.set(0)
                spinbox.pack(pady=5)
                self.fixed_controls[dim.name] = spinbox

            elif isinstance(dim, Categorical):
                option = ctk.CTkOptionMenu(
                    self.fixed_controls_container,
                    values=dim.categories
                )
                option.set(dim.categories[0])
                option.pack(pady=5)
                self.fixed_controls[dim.name] = option

    def plot_contour(self):
        """Generate a 2D contour plot for the two selected Real dimensions using fixed values for the others."""
        if self.exp_df is None or len(self.exp_df) < 2:
            print("Error: Not enough observations to plot contour.")
            return

        model = self.gpr_model
        if model is None:
            print("Error: Model is not trained.")
            return

        # Clear previous plot and colorbar.
        self.clear_current_plot()

        # Get the selected axes from the dropdowns.
        axis_x_name = self.contour_axis_x_dropdown.get()
        axis_y_name = self.contour_axis_y_dropdown.get()

        # Convert search space to skopt dimensions if needed
        skopt_space = self.search_space
        if hasattr(self.search_space, 'to_skopt'):
            skopt_space = self.search_space.to_skopt()

        # Find the corresponding dimension objects.
        axis_x_dim = next(dim for dim in skopt_space if dim.name == axis_x_name)
        axis_y_dim = next(dim for dim in skopt_space if dim.name == axis_y_name)

        # Check if this is a BoTorch model
        if hasattr(model, 'generate_contour_data') and callable(model.generate_contour_data):
            # For BoTorch models, use the specialized method
            # First, find indices of the selected dimensions
            feature_names = model.feature_names
            if feature_names:
                x_idx = feature_names.index(axis_x_name)
                y_idx = feature_names.index(axis_y_name)
                
                # Get fixed values for other dimensions
                fixed_values = {}
                for i, dim_name in enumerate(feature_names):
                    if dim_name not in [axis_x_name, axis_y_name] and dim_name in self.fixed_controls:
                        fixed_values[i] = self.fixed_controls[dim_name].get()
                
                # Get contour data
                X1, X2, y_pred = model.generate_contour_data(
                    x_range=axis_x_dim.bounds,
                    y_range=axis_y_dim.bounds,
                    fixed_values=fixed_values,
                    x_idx=x_idx,
                    y_idx=y_idx
                )
            else:
                # Fallback if feature names aren't available
                X1, X2 = np.meshgrid(
                    np.linspace(axis_x_dim.bounds[0], axis_x_dim.bounds[1], 100),
                    np.linspace(axis_y_dim.bounds[0], axis_y_dim.bounds[1], 100)
                )
                y_pred = np.zeros_like(X1)  # Placeholder
        else:
            # For non-BoTorch models, use existing code
            # Create a grid for the two selected dimensions.
            x1 = np.linspace(axis_x_dim.bounds[0], axis_x_dim.bounds[1], 100)
            x2 = np.linspace(axis_y_dim.bounds[0], axis_y_dim.bounds[1], 100)
            X1, X2 = np.meshgrid(x1, x2)
            
            # Create a dataframe with all variables needed for prediction
            # Start with the grid variables (the ones we're plotting)
            grid_df = pd.DataFrame({axis_x_name: X1.ravel(), axis_y_name: X2.ravel()})
            
            # Get list of all variables in the search space
            all_dims = skopt_space
            
            # Add fixed values for all other variables
            for dim in all_dims:
                if dim.name not in [axis_x_name, axis_y_name]:
                    if dim.name in self.fixed_controls:
                        control = self.fixed_controls[dim.name]
                        grid_df[dim.name] = control.get()
                    else:
                        # Use default mid-range value if no control exists
                        if hasattr(dim, 'bounds'):
                            grid_df[dim.name] = (dim.bounds[0] + dim.bounds[1]) / 2
                        elif hasattr(dim, 'categories'):
                            grid_df[dim.name] = dim.categories[0]
            
            # Check if the model has a _preprocess_X method (our custom models should have this)
            if hasattr(model, '_preprocess_X') and callable(model._preprocess_X):
                # Use the model's preprocessing method
                processed_grid = model._preprocess_X(grid_df)
                y_pred = model.predict(processed_grid)
            else:
                # No preprocessing needed
                y_pred = model.predict(grid_df)
            
            # Reshape predictions to the grid shape
            y_pred = y_pred.reshape(X1.shape)

        # Use the colormap from user customizations if set; otherwise default to 'viridis'.
        cmap = (self.customization_options.get('colormap') if hasattr(self, 'customization_options') and 
                self.customization_options.get('colormap') else 'viridis')
        contour = self.ax.contourf(X1, X2, y_pred, levels=20, cmap=cmap)
        self.colorbar = self.fig.colorbar(contour, ax=self.ax)
        
        # Plot experimental data points only if customization option is True
        if (self.exp_df is not None and 
            not self.exp_df.empty and 
            self.customization_options.get('show_experiments', False)):
            if axis_x_name in self.exp_df.columns and axis_y_name in self.exp_df.columns:
                self.ax.scatter(
                    self.exp_df[axis_x_name], 
                    self.exp_df[axis_y_name], 
                    c='white', 
                    edgecolors='black', 
                    s=50, 
                    label='Experiments'
                )
        
        # Plot the next point if it exists and the customization option is True
        main_app = self.parent.main_app if hasattr(self.parent, 'main_app') else self.parent
        if (hasattr(main_app, 'next_point') and 
            main_app.next_point is not None and 
            self.customization_options.get('show_next_point', False)):
            next_point = main_app.next_point
            if axis_x_name in next_point.columns and axis_y_name in next_point.columns:
                self.ax.scatter(
                    next_point[axis_x_name], 
                    next_point[axis_y_name], 
                    c='red', 
                    marker='D', 
                    s=80, 
                    label='Next Point',
                    zorder=10  # Ensure it's on top
                )
                
        # Add legend if there are labeled elements and they are being shown
        legend_needed = False
        if (self.exp_df is not None and 
            not self.exp_df.empty and 
            self.customization_options.get('show_experiments', False) and
            axis_x_name in self.exp_df.columns and axis_y_name in self.exp_df.columns):
            legend_needed = True
        
        main_app = self.parent.main_app if hasattr(self.parent, 'main_app') else self.parent
        if (hasattr(main_app, 'next_point') and 
            main_app.next_point is not None and 
            self.customization_options.get('show_next_point', False) and
            axis_x_name in main_app.next_point.columns and axis_y_name in main_app.next_point.columns):
            legend_needed = True
            
        if legend_needed and not self.ax.get_legend():
            self.ax.legend()

        # Set default labels as defined by this plot only if not customized by user
        if not self.customization_options.get('xlabel_user_set'):
            self.ax.set_xlabel(axis_x_name)
            self.customization_options['xlabel'] = axis_x_name
        if not self.customization_options.get('ylabel_user_set'):
            self.ax.set_ylabel(axis_y_name)
            self.customization_options['ylabel'] = axis_y_name
        if not self.customization_options.get('title_user_set'):
            self.ax.set_title("Contour Plot of Model Predictions")
            self.customization_options['title'] = "Contour Plot of Model Predictions"
        
        self.fig.tight_layout()

        # Store the current plot settings in customization options (only if not user-set)
        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()

        # Apply any user customizations (only if they exist).
        self.apply_customizations_to_axes()
        self.canvas.draw()

        # Remember that this is the last plot method called.
        self.last_plot_method = self.plot_contour

    def plot_metrics(self):
        """Generate the plots for model evaluation metrics using precomputed values."""
        required_metrics = ['rmse_values', 'mae_values', 'mape_values', 'r2_values']
        if not all(hasattr(self, attr) for attr in required_metrics):
            print("Error: Metrics have not been computed. Please train the model first.")
            return

        rmse_values = self.rmse_values
        mae_values = self.mae_values
        mape_values = self.mape_values
        r2_values = self.r2_values
        error_metric = self.error_metric_var.get()
        x_range = range(5, len(rmse_values) + 5)

        self.clear_current_plot()

        if error_metric == "RMSE":
            self.ax.plot(x_range, rmse_values, marker='o')
            if not self.customization_options.get('title_user_set'):
                self.ax.set_title("RMSE vs Number of Observations")
                self.customization_options['title'] = "RMSE vs Number of Observations"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("RMSE")
                self.customization_options['ylabel'] = "RMSE"
        elif error_metric == "MAE":
            self.ax.plot(x_range, mae_values, marker='o')
            if not self.customization_options.get('title_user_set'):
                self.ax.set_title("MAE vs Number of Observations")
                self.customization_options['title'] = "MAE vs Number of Observations"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("MAE")
                self.customization_options['ylabel'] = "MAE"
        elif error_metric == "MAPE":
            self.ax.plot(x_range, mape_values, marker='o')
            if not self.customization_options.get('title_user_set'):
                self.ax.set_title("MAPE vs Number of Observations")
                self.customization_options['title'] = "MAPE vs Number of Observations"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("MAPE (%)")
                self.customization_options['ylabel'] = "MAPE (%)"
        elif error_metric == "R2":
            self.ax.plot(x_range, r2_values, marker='o')
            if not self.customization_options.get('title_user_set'):
                self.ax.set_title("R² vs Number of Observations")
                self.customization_options['title'] = "R² vs Number of Observations"
            if not self.customization_options.get('ylabel_user_set'):
                self.ax.set_ylabel("R²")
                self.customization_options['ylabel'] = "R²"
        else:
            print("Error: Unknown error metric selected.")
            return

        if not self.customization_options.get('xlabel_user_set'):
            self.ax.set_xlabel("Number of Observations")
            self.customization_options['xlabel'] = "Number of Observations"
        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()
        self.fig.tight_layout()
        self.apply_customizations_to_axes()
        self.canvas.draw()

        self.last_plot_method = self.plot_metrics

    def plot_parity(self, debug=False):
        """Generate and display a parity plot of actual vs predicted values using cross-validation."""
        if self.exp_df is None or len(self.exp_df) < 5:  # Need at least 5 points for 5-fold CV
            print("Error: Not enough observations for a parity plot (need at least 5).")
            return

        # Clear the current plot
        self.clear_current_plot()
        
        # Check if the model has cached CV results
        if hasattr(self.gpr_model, 'cv_cached_results') and self.gpr_model.cv_cached_results is not None:
            # Use cached results
            cv_results = self.gpr_model.cv_cached_results
            y_true_all = cv_results['y_true']
            y_pred_all = cv_results['y_pred']
            
            print("Using cached cross-validation results")
        else:
            # If no cached results, perform cross-validation on the fly
            print("No cached results found, performing cross-validation...")
            
            # Get the raw data from experiment manager
            X = self.exp_df.drop(columns=["Output"])
            y = self.exp_df["Output"]
            
            # Determine the backend in use
            backend = self.gpr_model.__class__.__name__
            
            # Set up K-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_true_all = []
            y_pred_all = []
            
            # Perform cross-validation based on backend type
            if backend == "SklearnModel":
                # For scikit-learn models
                for train_idx, test_idx in kf.split(X):
                    # Split data
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]
                    
                    # Create a copy of the model with the same hyperparameters
                    optimized_kernel = self.gpr_model.optimized_kernel
                    cv_model = GaussianProcessRegressor(
                        kernel=optimized_kernel,
                        optimizer=None,  # Don't re-optimize
                        random_state=self.gpr_model.random_state
                    )
                    
                    # Process data the same way as the original model
                    if hasattr(self.gpr_model, '_preprocess_X'):
                        # Use the simpler preprocessing
                        X_train_processed = self.gpr_model._preprocess_X(X_train)
                        cv_model.fit(X_train_processed, y_train.values)
                        X_test_processed = self.gpr_model._preprocess_X(X_test)
                        y_pred = cv_model.predict(X_test_processed)
                    else:
                        # Fallback to direct fitting
                        cv_model.fit(X_train.values, y_train.values)
                        y_pred = cv_model.predict(X_test.values)
                    
                    # Store actual and predicted values
                    y_true_all.extend(y_test.values)
                    y_pred_all.extend(y_pred)
                        
            elif backend == "BoTorchModel":
                # For BoTorch models
                for train_idx, test_idx in kf.split(X):
                    # Split data
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]
                    
                    # Create temporary experiment manager with the training data
                    from logic.experiment_manager import ExperimentManager
                    temp_exp_manager = ExperimentManager()
                    train_df = pd.concat([X_train, y_train.rename("Output")], axis=1)
                    temp_exp_manager.df = train_df
                    
                    # Create a new model instance with the same parameters
                    from logic.models.botorch_model import BoTorchModel
                    temp_model = BoTorchModel(
                        training_iter=self.gpr_model.training_iter,
                        random_state=self.gpr_model.random_state,
                        kernel_options=self.gpr_model.kernel_options,
                        cat_dims=self.gpr_model.cat_dims
                    )
                    
                    # Train model without optimizing (just to set up structure)
                    temp_model.train(temp_exp_manager)
                    
                    # Load the optimized state from the full model
                    if hasattr(self.gpr_model, 'fitted_state_dict') and self.gpr_model.fitted_state_dict is not None:
                        temp_model.model.load_state_dict(self.gpr_model.fitted_state_dict, strict=False)
                    
                    # Make predictions on test set
                    y_pred = temp_model.predict(X_test)
                    
                    # Store results
                    y_true_all.extend(y_test.values)
                    y_pred_all.extend(y_pred)
            else:
                print(f"Error: Unsupported backend type '{backend}' for parity plot.")
                return
        
        # Convert to numpy arrays
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        try:
            r2 = r2_score(y_true_all, y_pred_all)
        except:
            r2 = float('nan')
        
        # Create parity plot
        self.ax.scatter(y_true_all, y_pred_all, alpha=0.7)
        
        # Add parity line (y=x)
        min_val = min(np.min(y_true_all), np.min(y_pred_all))
        max_val = max(np.max(y_true_all), np.max(y_pred_all))
        self.ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Set labels and title only if not customized by user
        title_str = f"Cross-Validation Parity Plot\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
        if not self.customization_options.get('title_user_set'):
            self.ax.set_title(title_str)
            self.customization_options['title'] = title_str
        if not self.customization_options.get('xlabel_user_set'):
            self.ax.set_xlabel("Actual Values")
            self.customization_options['xlabel'] = "Actual Values"
        if not self.customization_options.get('ylabel_user_set'):
            self.ax.set_ylabel("Predicted Values")
            self.customization_options['ylabel'] = "Predicted Values"
        
        # Store axis limits
        self.customization_options['xlim'] = self.ax.get_xlim()
        self.customization_options['ylim'] = self.ax.get_ylim()
        
        # Apply customizations and draw
        self.apply_customizations_to_axes()
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Remember last plot method
        self.last_plot_method = self.plot_parity

    
    def customize_plot(self, event=None):
        """Open a window to customize plot parameters, prepopulated with current settings."""
        customization_window = ctk.CTkToplevel(self.parent)
        customization_window.title("Customize Plot")
        customization_window.geometry("300x650")  # Increased height for number formatting controls
        customization_window.resizable(False, False)
        customization_window.lift()
        customization_window.focus_force()
        customization_window.grab_set()

        # Retrieve current settings
        current_title = self.ax.get_title()
        current_xlabel = self.ax.get_xlabel()
        current_ylabel = self.ax.get_ylabel()
        current_cmap = "viridis"
        if self.ax.collections:
            try:
                current_cmap = self.ax.collections[0].get_cmap().name
            except Exception:
                current_cmap = "viridis"
        current_font = self.customization_options.get('font', 'Arial') if hasattr(self, 'customization_options') else "Arial"
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_title_fontsize = int(self.ax.title.get_fontsize())
        current_label_fontsize = int(self.ax.xaxis.label.get_fontsize())
        current_title_fontstyle = 'bold' if self.ax.title.get_fontweight() == 'bold' else 'normal'
        current_label_fontstyle = 'bold' if self.ax.xaxis.label.get_fontweight() == 'bold' else 'normal'

        frame = ctk.CTkScrollableFrame(customization_window)
        frame.pack(pady=5, padx=5, fill="both", expand=True)

        ctk.CTkLabel(frame, text="Plot Title:").pack(pady=5)
        title_entry = ctk.CTkEntry(frame)
        title_entry.insert(0, current_title)
        title_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Label:").pack(pady=5)
        x_label_entry = ctk.CTkEntry(frame)
        x_label_entry.insert(0, current_xlabel)
        x_label_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Label:").pack(pady=5)
        y_label_entry = ctk.CTkEntry(frame)
        y_label_entry.insert(0, current_ylabel)
        y_label_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Colormap:").pack(pady=5)
        colormap_var = ctk.StringVar(value=current_cmap)
        colormap_menu = ctk.CTkOptionMenu(frame, values=plt.colormaps(), variable=colormap_var)
        colormap_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Font:").pack(pady=5)
        font_var = ctk.StringVar(value=current_font)
        fonts = [
            "DejaVu Sans",
            "DejaVu Serif",
            "Arial",
            "Times New Roman",
            "Courier New",
            "Verdana",
            "Georgia",
            "Palatino Linotype",
            "STIXGeneral",
            "Segoe UI"
        ]
        font_menu = ctk.CTkOptionMenu(frame, values=fonts, variable=font_var)
        font_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Limit Min:").pack(pady=5)
        x_min_entry = ctk.CTkEntry(frame)
        x_min_entry.insert(0, str(current_xlim[0]))
        x_min_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="X-axis Limit Max:").pack(pady=5)
        x_max_entry = ctk.CTkEntry(frame)
        x_max_entry.insert(0, str(current_xlim[1]))
        x_max_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Limit Min:").pack(pady=5)
        y_min_entry = ctk.CTkEntry(frame)
        y_min_entry.insert(0, str(current_ylim[0]))
        y_min_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Y-axis Limit Max:").pack(pady=5)
        y_max_entry = ctk.CTkEntry(frame)
        y_max_entry.insert(0, str(current_ylim[1]))
        y_max_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Title Font Size:").pack(pady=5)
        title_fontsize_entry = ctk.CTkEntry(frame)
        title_fontsize_entry.insert(0, str(current_title_fontsize))
        title_fontsize_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Label Font Size:").pack(pady=5)
        label_fontsize_entry = ctk.CTkEntry(frame)
        label_fontsize_entry.insert(0, str(current_label_fontsize))
        label_fontsize_entry.pack(pady=5)

        ctk.CTkLabel(frame, text="Title Font Style:").pack(pady=5)
        title_fontstyle_var = ctk.StringVar(value=current_title_fontstyle)
        title_fontstyle_menu = ctk.CTkOptionMenu(frame, values=["normal", "bold"], variable=title_fontstyle_var)
        title_fontstyle_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Label Font Style:").pack(pady=5)
        label_fontstyle_var = ctk.StringVar(value=current_label_fontstyle)
        label_fontstyle_menu = ctk.CTkOptionMenu(frame, values=["normal", "bold"], variable=label_fontstyle_var)
        label_fontstyle_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Tick Style:").pack(pady=5)
        tick_style_var = ctk.StringVar(value=self.customization_options.get('tick_style', "x and y"))
        tick_style_menu = ctk.CTkOptionMenu(frame, values=["all", "x and y", "none"], variable=tick_style_var)
        tick_style_menu.pack(pady=5)

        ctk.CTkLabel(frame, text="Tick Direction:").pack(pady=5)
        tick_direction_var = ctk.StringVar(value=self.customization_options.get('tick_direction', "out"))
        tick_direction_menu = ctk.CTkOptionMenu(frame, values=["in", "out"], variable=tick_direction_var)
        tick_direction_menu.pack(pady=5)

        # Add a section title for number formatting options
        ctk.CTkLabel(frame, text="Number Formatting", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        
        # Add a small help text
        help_text = "Format options: auto, decimal (0.00), scientific (1e-3), percent (50%), integer (1), custom (3 sig figs)"
        help_label = ctk.CTkLabel(frame, text=help_text, wraplength=280, font=("Arial", 9))
        help_label.pack(pady=(0, 10))
        
        # X-axis number formatting
        ctk.CTkLabel(frame, text="X-Axis Number Format:").pack(pady=5)
        x_format_var = ctk.StringVar(value=self.customization_options.get('x_number_format', "auto"))
        x_format_options = ["auto", "decimal", "scientific", "percent", "integer", "custom"]
        x_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=x_format_var)
        x_format_menu.pack(pady=5)
        
        # Y-axis number formatting
        ctk.CTkLabel(frame, text="Y-Axis Number Format:").pack(pady=5)
        y_format_var = ctk.StringVar(value=self.customization_options.get('y_number_format', "auto"))
        y_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=y_format_var)
        y_format_menu.pack(pady=5)
        
        # Colorbar number formatting
        ctk.CTkLabel(frame, text="Colorbar Number Format:").pack(pady=5)
        colorbar_format_var = ctk.StringVar(value=self.customization_options.get('colorbar_format', "auto"))
        colorbar_format_menu = ctk.CTkOptionMenu(frame, values=x_format_options, variable=colorbar_format_var)
        colorbar_format_menu.pack(pady=5)

        # Add a section title for data display options
        ctk.CTkLabel(frame, text="Data Display Options", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        
        # Add toggle for experimental points
        show_exp_points_var = ctk.BooleanVar(
            value=self.customization_options.get('show_experiments', False)
        )
        show_exp_switch = ctk.CTkSwitch(
            frame, 
            text="Show Experimental Points",
            variable=show_exp_points_var
        )
        show_exp_switch.pack(pady=5)
        
        # Add toggle for next point
        show_next_point_var = ctk.BooleanVar(
            value=self.customization_options.get('show_next_point', False)
        )
        show_next_switch = ctk.CTkSwitch(
            frame, 
            text="Show Next Point",
            variable=show_next_point_var
        )
        show_next_switch.pack(pady=5)

        button_frame = ctk.CTkFrame(customization_window)
        button_frame.pack(pady=10, fill="x")

        def apply_customizations():
            try:
                # Store user customizations
                user_customizations = {
                    'title': title_entry.get(),
                    'xlabel': x_label_entry.get(),
                    'ylabel': y_label_entry.get(),
                    'colormap': colormap_var.get(),
                    'font': font_var.get(),
                    'xlim': [float(x_min_entry.get()), float(x_max_entry.get())],
                    'ylim': [float(y_min_entry.get()), float(y_max_entry.get())],
                    'title_fontsize': int(title_fontsize_entry.get()),
                    'label_fontsize': int(label_fontsize_entry.get()),
                    'title_fontstyle': title_fontstyle_var.get(),
                    'label_fontstyle': label_fontstyle_var.get(),
                    'tick_style': tick_style_var.get(),
                    'tick_direction': tick_direction_var.get(),
                    'show_experiments': show_exp_points_var.get(),
                    'show_next_point': show_next_point_var.get(),
                    'x_number_format': x_format_var.get(),
                    'y_number_format': y_format_var.get(),
                    'colorbar_format': colorbar_format_var.get(),
                    # Set flags to indicate these were set by user
                    'title_user_set': True,
                    'xlabel_user_set': True,
                    'ylabel_user_set': True
                }
                
                # Update the customization options with user values
                self.customization_options.update(user_customizations)
                
                # Apply customizations directly without regenerating the plot
                self.apply_customizations_to_axes()
                
                # If we're changing data display options (experiments/next point), 
                # we need to redraw the plot to show/hide those elements
                if (self.last_plot_method == self.plot_contour and 
                    ('show_experiments' in user_customizations or 'show_next_point' in user_customizations)):
                    self.last_plot_method()
                
                # Briefly change button text to show success
                apply_button.configure(text="Applied!")
                customization_window.after(1000, lambda: apply_button.configure(text="Apply"))
                
            except Exception as e:
                print(f"Error applying customizations: {e}")
                # Change button text to show error
                apply_button.configure(text="Error!")
                customization_window.after(2000, lambda: apply_button.configure(text="Apply"))

        apply_button = ctk.CTkButton(button_frame, text="Apply", command=apply_customizations)
        apply_button.pack(side="left", padx=5, pady=10)
        
        close_button = ctk.CTkButton(button_frame, text="Close", command=customization_window.destroy)
        close_button.pack(side="right", padx=5, pady=10)

    def apply_customizations_to_axes(self):
        """
        Apply user customization options stored in self.customization_options to the axes.
        Only properties that have been explicitly set by the user (i.e. non-empty) will override the default.
        """
        custom = getattr(self, 'customization_options', {})
        font = custom.get('font', None)
        if custom.get('title'):
            self.ax.set_title(custom['title'], fontname=font if font else None, fontsize=custom.get('title_fontsize', None), fontweight=custom.get('title_fontstyle', None))
        if custom.get('xlabel'):
            self.ax.set_xlabel(custom['xlabel'], fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        if custom.get('ylabel'):
            self.ax.set_ylabel(custom['ylabel'], fontname=font if font else None, fontsize=custom.get('label_fontsize', None), fontweight=custom.get('label_fontstyle', None))
        if font:
            for label in (self.ax.get_xticklabels() + self.ax.get_yticklabels()):
                label.set_fontname(font)
                if custom.get('label_fontstyle') == 'bold':
                    label.set_fontweight('bold')
        if custom.get('xlim'):
            self.ax.set_xlim(custom['xlim'])
        if custom.get('ylim'):
            self.ax.set_ylim(custom['ylim'])
        
        # Apply tick styling (use defaults if not explicitly set)
        tick_style = custom.get('tick_style', 'x and y')
        tick_direction = custom.get('tick_direction', 'out')
        if tick_style == 'all':
            self.ax.tick_params(axis='both', which='both', direction=tick_direction,
                                bottom=True, top=True, left=True, right=True)
        elif tick_style == 'x and y':
            self.ax.tick_params(axis='x', which='both', direction=tick_direction,
                                bottom=True, top=False)
            self.ax.tick_params(axis='y', which='both', direction=tick_direction,
                                left=True, right=False)
        elif tick_style == 'none':
            self.ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        if custom.get('colormap') and self.ax.collections:
            try:
                self.ax.collections[0].set_cmap(custom['colormap'])
            except Exception as e:
                print("Error updating colormap:", e)
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            self.colorbar.ax.tick_params(labelsize=10)
            if font:
                for label in self.colorbar.ax.get_yticklabels():
                    label.set_fontname(font)
                    if custom.get('label_fontstyle') == 'bold':
                        label.set_fontweight('bold')
            
            # Apply colorbar number formatting
            colorbar_format = custom.get('colorbar_format', 'auto')
            colorbar_formatter = self._get_number_formatter(colorbar_format)
            if colorbar_formatter:
                self.colorbar.ax.yaxis.set_major_formatter(colorbar_formatter)
        
        # Apply axis number formatting (apply defaults if not set)
        x_format = custom.get('x_number_format', 'auto')
        x_formatter = self._get_number_formatter(x_format)
        if x_formatter:
            self.ax.xaxis.set_major_formatter(x_formatter)
        
        y_format = custom.get('y_number_format', 'auto')
        y_formatter = self._get_number_formatter(y_format)
        if y_formatter:
            self.ax.yaxis.set_major_formatter(y_formatter)
        
        # Apply tight layout to prevent label cutoff
        self.fig.tight_layout()
        self.canvas.draw()


    def get_hyperparameters(self):
        """Extract the hyperparameters from the trained model."""
        if self.gpr_model is None:
            print("Error: Model is not trained.")
            return None

        # Access the kernel from the SklearnModel
        kernel = self.gpr_model.kernel
        if kernel is None:
            print("Error: Kernel is not defined in the model.")
            return None

        # Extract hyperparameters
        hyperparameters = kernel.get_params()
        lengthscale = hyperparameters.get('k2__length_scale', 'N/A')

        return {
            'Kernel': kernel,
            'Lengthscale': lengthscale
        }

    def clear_current_plot(self):
        """Clears the current axes and removes any existing colorbar."""
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception as e:
                print("Error removing previous colorbar:", e)
            self.colorbar = None
        self.ax.clear()


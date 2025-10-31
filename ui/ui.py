import customtkinter as ctk
from customtkinter import filedialog
import mplcursors
from tabulate import tabulate
import tksheet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skopt.space import Categorical, Integer, Real
import numpy as np
from skopt.sampler import Lhs, Sobol, Hammersly
import tkinter as tk

from ui.variables_setup import SpaceSetupWindow
from ui.gpr_panel import GaussianProcessPanel
from ui.acquisition_panel import AcquisitionPanel

# DEPRECATED: Pool visualization (will be removed in v0.3.0)
from ui.pool_viz import generate_pool, plot_pool

# Deprecated imports - these functions are no longer used in the modern UI
# from logic.clustering import cluster_pool
# from logic.emoc import select_EMOC
# from logic.optimization import select_optimize

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.data.experiment_manager import ExperimentManager

# UI-layer utilities
from ui.experiment_logger import ExperimentLogger

# Import new session API
from alchemist_core.session import OptimizationSession
from alchemist_core.events import EventEmitter

plt.rcParams['savefig.dpi'] = 600


# ============================================================
# UI Helper Functions
# ============================================================

# ============================================================
# Main Application
# ============================================================

class ALchemistApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self._configure_window()

        # Create the menu bar
        self._create_menu_bar()

        # State variables for data management using our new classes
        self.search_space_manager = SearchSpace()
        self.experiment_manager = ExperimentManager()
        self.next_point = None  # Keep as DataFrame for visualization
        
        # Legacy variables for compatibility during transition
        self.var_df = None
        self.exp_df = pd.DataFrame()
        self.search_space = None
        
        # NEW: Create OptimizationSession for session-based API
        # This provides a parallel code path alongside the existing direct logic calls
        self.session = OptimizationSession()
        
        # Connect session events to UI updates
        self.session.events.on('progress', self._on_session_progress)
        self.session.events.on('model_trained', self._on_session_model_trained)
        self.session.events.on('suggestions_ready', self._on_session_suggestions)

        # Build essential UI sections
        self._create_vertical_frame()
        self._create_variable_management_frame()
        self._create_experiment_management_frame()
        self._create_visualization_frame()
        
        # Start in tabbed layout by default
        self.using_tabs = True
        
        # Create tabbed interface
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.pack(side='right', fill='both', padx=10, pady=10)
        self.tab_view.configure(width=300)
        
        # Add tabs
        self.tab_view.add("Model")
        self.tab_view.add("Acquisition")
        
        # Set the default tab
        self.tab_view.set("Model")
        
        # Create panels inside tabs
        self.model_frame = GaussianProcessPanel(self.tab_view.tab("Model"), self)
        self.model_frame.pack(fill='both', expand=True)
        
        self.acquisition_panel = AcquisitionPanel(self.tab_view.tab("Acquisition"), self)
        self.acquisition_panel.pack(fill='both', expand=True)
        
        # Set initial UI state based on data load
        self._update_ui_state()
        
        # Initialize the experiment logger
        self.experiment_logger = ExperimentLogger()
        self.experiment_logger.start_experiment("ALchemist_Experiment")
    
    def _configure_window(self):
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')
        self.title('Active Learning Experiment Planner')
        self.geometry('1450x800')
        self.minsize(1300, 600)  # Increase minimum width to accommodate all panels
        self.protocol('WM_DELETE_WINDOW', self._quit)

    def _create_menu_bar(self):
        menu_bar = tk.Menu(self)
        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Help", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        # Documentation menu
        doc_menu = tk.Menu(menu_bar, tearoff=0)
        doc_menu.add_command(label="User Guide", command=self.show_user_guide)
        doc_menu.add_command(label="API Reference", command=self.show_api_reference)
        menu_bar.add_cascade(label="Documentation", menu=doc_menu)
        # Preferences menu
        pref_menu = tk.Menu(menu_bar, tearoff=0)
        pref_menu.add_command(label="Settings", command=self.show_settings)
        pref_menu.add_command(label="Toggle Tabbed Layout", command=self.toggle_tabbed_layout)
        pref_menu.add_command(label="Toggle Noise Column", command=self.toggle_noise_column)
        pref_menu.add_separator()
    # Removed: Toggle Session API menu item (session API is now always enabled)
        menu_bar.add_cascade(label="Preferences", menu=pref_menu)
        self.config(menu=menu_bar)

    def show_help(self):
        tk.messagebox.showinfo("Help", "This is the help dialog.")

    def show_about(self):
        tk.messagebox.showinfo("About", "This is the about dialog.")

    def show_user_guide(self):
        tk.messagebox.showinfo("User Guide", "This is the user guide.")

    def show_api_reference(self):
        tk.messagebox.showinfo("API Reference", "This is the API reference.")

    def show_settings(self):
        tk.messagebox.showinfo("Settings", "This is the settings dialog.")

    def _quit(self):
        # Cancel all pending "after" tasks
        for task_id in self.tk.call('after', 'info'):
            self.after_cancel(task_id)
    
        # Now safely destroy the window
        self.quit()
        self.destroy()
    
    # ============================================================
    # Session Event Handlers
    # ============================================================
    
    def _on_session_progress(self, event_data):
        """Handle progress events from the session."""
        message = event_data.get('message', '')
        # UI components can listen to this for progress updates
    
    def _on_session_model_trained(self, event_data):
        """Handle model training completion from session."""
        metrics = event_data.get('metrics', {})
        print(f"Session: Model trained successfully")
        print(f"  R² = {metrics.get('mean_R²', 'N/A'):.3f}")
        print(f"  RMSE = {metrics.get('mean_RMSE', 'N/A'):.3f}")
        # Sync session model to main_app.gpr_model for visualization compatibility
        self.gpr_model = self.session.model
    
    def _on_session_suggestions(self, event_data):
        """Handle acquisition suggestions from session."""
        suggestions = event_data.get('suggestions', None)
        if suggestions is not None:
            print(f"Session: {len(suggestions)} new experiments suggested")
            # Sync to main_app.next_point for visualization compatibility
            self.next_point = suggestions

    def _create_vertical_frame(self):
        # LEFT COLUMN: Fixed-width frame for variable and experiment management.
        self.vertical_frame = ctk.CTkFrame(self, width=450)
        self.vertical_frame.pack(side='left', fill='y', padx=10, pady=10)
        self.vertical_frame.pack_propagate(False)  # Prevent automatic resizing

    def _create_variable_management_frame(self):
        self.frame_vars = ctk.CTkFrame(self.vertical_frame)
        self.frame_vars.pack(side='top', fill='both', padx=5, pady=5)

        ctk.CTkLabel(self.frame_vars, text='Variable Management', font=('Arial', 16)).pack(pady=5)
        self.var_sheet = tksheet.Sheet(self.frame_vars, height=200, header=['Variables', 'Type', 'Min', 'Max', 'Values'])
        self.var_sheet.pack(fill='both', expand=True, padx=5, pady=5)
        self.var_sheet.set_all_column_widths()
        self.var_sheet.enable_bindings()

        self.frame_vars_buttons = ctk.CTkFrame(self.frame_vars)
        self.frame_vars_buttons.pack(fill='x', pady=5)
        self.load_var_button = ctk.CTkButton(self.frame_vars_buttons, text='Load Variables', command=self.load_variables)
        self.load_var_button.pack(side='left', padx=5, pady=5)
        def open_space_setup():
            self.var_space_editor = SpaceSetupWindow(self)
            self.var_space_editor.grab_set()
        self.gen_var_button = ctk.CTkButton(self.frame_vars_buttons, text='Generate Variables File', command=open_space_setup)
        self.gen_var_button.pack(side='left', padx=5, pady=5)

    def _create_experiment_management_frame(self):
        self.frame_exp = ctk.CTkFrame(self.vertical_frame)
        self.frame_exp.pack(side='top', fill='both', padx=5, pady=5)

        ctk.CTkLabel(self.frame_exp, text='Experiment Data', font=('Arial', 16)).pack(pady=5)
        self.exp_sheet = tksheet.Sheet(self.frame_exp)
        self.exp_sheet.pack(fill='both', expand=True, padx=5, pady=5)
        self.exp_sheet.enable_bindings()

        self.frame_exp_buttons_top = ctk.CTkFrame(self.frame_exp)
        self.frame_exp_buttons_top.pack(fill='x', pady=5)
        self.load_exp_button = ctk.CTkButton(self.frame_exp_buttons_top, text='Load Experiments', command=self.load_experiments, state='disabled')
        self.load_exp_button.pack(side='left', padx=5, pady=5)
        self.save_exp_button = ctk.CTkButton(self.frame_exp_buttons_top, text='Save Experiments', command=self.save_experiments)
        self.save_exp_button.pack(side='left', padx=5, pady=5)

        self.frame_exp_buttons_bottom = ctk.CTkFrame(self.frame_exp)
        self.frame_exp_buttons_bottom.pack(fill='x', pady=5)
        self.gen_template_button = ctk.CTkButton(self.frame_exp_buttons_bottom, text='Generate Initial Points', command=self.generate_initial_points, state='disabled')
        self.gen_template_button.pack(side='left', padx=5, pady=5)
        self.add_point_button = ctk.CTkButton(self.frame_exp_buttons_bottom, text='Add Point', command=self.add_point)
        self.add_point_button.pack(side='left', padx=5, pady=5)

    def _create_visualization_frame(self):
        # MIDDLE COLUMN: Visualization frame expands but maintains aspect ratio
        self.frame_viz = ctk.CTkFrame(self)
        self.frame_viz.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        # Set size constraints
        self.frame_viz.pack_propagate(False)
        self.frame_viz.configure(width=500, height=600)  # Fixed width and height for better aspect ratio

        ctk.CTkLabel(self.frame_viz, text='Visualization', font=('Arial', 16)).pack(pady=5)

        # Frame for variable dropdowns and clustering switch
        self.frame_viz_options = ctk.CTkFrame(self.frame_viz)
        self.frame_viz_options.pack(pady=5)

        # Add variable dropdowns
        self._create_variables_dropdown()

        # Clustering switch
        self.cluster_switch = ctk.CTkSwitch(self.frame_viz_options, text='Clustering', command=self.update_pool_plot, state='disabled')
        self.cluster_switch.pack(side='left', padx=5, pady=5)

        # Visualization canvas - use square figure for better aspect ratio
        self.fig, self.ax = plt.subplots(figsize=(5, 5))  # Square figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_viz)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_viz)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

    def _create_variables_dropdown(self):
        '''Creates dropdowns for selecting variables for 2D visualization.'''
        # Always use the skopt-compatible version for iteration
        if self.search_space is None:
            variables = ['Variable 1', 'Variable 2']
        else:
            # Make sure we're using an iterable version
            if hasattr(self.search_space, 'to_skopt'):
                skopt_space = self.search_space.to_skopt()
            else:
                skopt_space = self.search_space
            variables = [dim.name for dim in skopt_space]

        # Dropdown for the first variable
        self.var1_dropdown = ctk.CTkComboBox(self.frame_viz_options, values=variables, command=self.update_pool_plot)
        self.var1_dropdown.set(variables[0] if variables else 'Variable 1')
        self.var1_dropdown.pack(side='left', padx=5, pady=5)

        # Dropdown for the second variable
        self.var2_dropdown = ctk.CTkComboBox(self.frame_viz_options, values=variables, command=self.update_pool_plot)
        self.var2_dropdown.set(variables[1] if len(variables) > 1 else 'Variable 2')
        self.var2_dropdown.pack(side='left', padx=5, pady=5)

    def _create_model_frame(self):
        # RIGHT COLUMN: Model frame (GPR) placed on the right.
        self.model_frame = GaussianProcessPanel(self)
        self.model_frame.pack(side='right', fill='both', padx=10, pady=10)
        # Ensure the panel doesn't collapse below minimum width
        self.model_frame.configure(width=300, height=600)
        self.model_frame.pack_propagate(False)  # Prevent automatic resizing

    def _create_acquisition_frame(self):
        # NEW PANEL: Acquisition function panel on far right
        self.acq_frame = ctk.CTkFrame(self)
        self.acq_frame.pack(side='right', fill='both', padx=(5, 10), pady=10)
        self.acq_frame.configure(width=280)
        self.acq_frame.pack_propagate(False)
        
        # Create the acquisition panel (no title in the frame as the panel has its own)
        self.acquisition_panel = AcquisitionPanel(self.acq_frame, self)
        self.acquisition_panel.pack(fill='both', expand=True, padx=5, pady=5)

    def _update_ui_state(self):
        """Updates the UI state based on the loaded data."""
        if self.search_space is not None:
            self.load_exp_button.configure(state='normal')
            self.gen_template_button.configure(state='normal')
            variables = [dim.name for dim in self.search_space]
            if list(self.var1_dropdown.cget('values')) != list(variables):
                self.var1_dropdown.configure(values=variables)
                self.var1_dropdown.set(variables[0])
                self.var2_dropdown.configure(values=variables)
                self.var2_dropdown.set(variables[1])
        else:
            self.load_exp_button.configure(state='disabled')
            self.gen_template_button.configure(state='disabled')
        if self.exp_df is not None:
            self.cluster_switch.configure(state='normal')
        else:
            self.cluster_switch.configure(state='disabled')

    def load_variables(self):
        """Loads a search space from a file using a file dialog."""
        file_path = filedialog.askopenfilename(
            title='Select Variable Space File',
            filetypes=[('JSON Files', '*.json'), ('CSV Files', '*.csv')]
        )
        if file_path:
            try:
                # Determine file type and load accordingly
                if file_path.lower().endswith('.json'):
                    # Load JSON file
                    self.search_space_manager.load_from_json(file_path)
                elif file_path.lower().endswith('.csv'):
                    # Load CSV file using the same logic as variables_setup.py
                    data = self._load_variables_from_csv(file_path)
                    self.search_space_manager.from_dict(data)
                else:
                    raise ValueError("Unsupported file format. Please use .json or .csv files.")
                
                # CRITICAL: Update the legacy variable with skopt format
                self.search_space = self.search_space_manager.to_skopt()
                
                # Update the experiment manager with the new search space
                self.experiment_manager.set_search_space(self.search_space_manager)

                # Update the variable sheet with the loaded search space
                data = []
                for var in self.search_space_manager.variables:
                    if var["type"] == "categorical":
                        # For categorical variables, include the categories
                        row = [
                            var["name"],               # Variable Name
                            'Categorical',             # Type of the variable
                            '',                        # No min for categorical
                            '',                        # No max for categorical
                            ', '.join(map(str, var["values"]))  # List the possible categories as a string
                        ]
                    else:
                        # For Integer and Real variables
                        row = [
                            var["name"],               # Variable Name
                            var["type"].capitalize(),  # Type of the variable ('Integer' or 'Real')
                            var["min"],                # Minimum Value
                            var["max"],                # Maximum Value
                            ''                         # No values for Integer/Real
                        ]
                    data.append(row)

                # Insert the data into the tksheet
                self.var_sheet.set_sheet_data(data)
                self.var_sheet.set_all_column_widths()

                # Update the experiment sheet headers
                variables = self.search_space_manager.get_variable_names()
                exp_sheet_headers = variables + ['Output']
                self.exp_sheet.set_header_data(exp_sheet_headers)

                # Update the model frame with the search space and categorical variables
                self.model_frame.update_search_space(
                    self.search_space_manager,  # Pass the manager object
                    self.search_space_manager.get_categorical_variables()
                )

                # Update the UI state
                self._update_ui_state()
                print('Search space loaded successfully.')
                
                # Ensure we're using the skopt-compatible version for pool generation
                self.pool = generate_pool(self.search_space, lhs_iterations=20)

                # Reset kmeans and update plot
                self.kmeans = None
                self.update_pool_plot()
            except Exception as e:
                print('Error loading search space:', e)
    
    def _load_variables_from_csv(self, file_path):
        """Load variables from CSV file using the same logic as variables_setup.py"""
        import csv
        data = []
        with open(file_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                typ = row.get("Type", "").strip()
                variable_name = row.get("Variable", "").strip()
                
                # Skip rows with empty variable names
                if not variable_name:
                    continue
                    
                if typ == "Real":
                    try:
                        min_val = float(row.get("Min", "0").strip() or "0")
                        max_val = float(row.get("Max", "1").strip() or "1")
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid min/max values for Real variable '{variable_name}'. Using defaults 0, 1.")
                        min_val, max_val = 0.0, 1.0
                    d = {
                        "name": variable_name,
                        "type": "real",  # lowercase for SearchSpace compatibility
                        "min": min_val,
                        "max": max_val
                    }
                elif typ == "Integer":
                    try:
                        min_val = int(float(row.get("Min", "0").strip() or "0"))
                        max_val = int(float(row.get("Max", "1").strip() or "1"))
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid min/max values for Integer variable '{variable_name}'. Using defaults 0, 1.")
                        min_val, max_val = 0, 1
                    d = {
                        "name": variable_name,
                        "type": "integer",  # lowercase for SearchSpace compatibility
                        "min": min_val,
                        "max": max_val
                    }
                elif typ == "Categorical":
                    values_str = row.get("Values", "").strip()
                    if values_str:
                        values = [v.strip() for v in values_str.split(",") if v.strip()]
                    else:
                        values = []
                    
                    if not values:
                        print(f"Warning: No values found for Categorical variable '{variable_name}'. Skipping.")
                        continue
                        
                    d = {
                        "name": variable_name,
                        "type": "categorical",  # lowercase for SearchSpace compatibility
                        "values": values
                    }
                else:
                    print(f"Warning: Unknown variable type '{typ}' for variable '{variable_name}'. Skipping.")
                    continue
                data.append(d)
        return data

    def load_experiments(self, file_path=None):
        '''Loads experimental data from a CSV file using a file dialog.'''
        if file_path is None:
            file_path = filedialog.askopenfilename(title='Select Experiments CSV', filetypes=[('CSV Files', '*.csv')])
        
        if file_path:
            try:
                # Load experiments using the ExperimentManager
                self.experiment_manager.load_from_csv(file_path)
                
                # Update the main DataFrame from the experiment manager
                self.exp_df = self.experiment_manager.get_data()
                
                # Get the headers and data for the tksheet
                headers = self.exp_df.columns.tolist()
                data = self.exp_df.values.tolist()
                
                # Update the experiment sheet
                self.exp_sheet.set_sheet_data(data)
                self.exp_sheet.set_header_data(headers)
                self.exp_sheet.set_all_column_widths()
                
                # Log the data loading
                print(f"Loaded {len(self.exp_df)} experiment points from {file_path}")
                if 'Noise' in self.exp_df.columns:
                    print("Notice: Noise column detected. This will be used for model regularization if available.")
                
                # Enable UI elements that require experiment data
                self._update_ui_state()
                
                # Reset any existing model
                if hasattr(self, 'model_frame') and self.model_frame is not None:
                    self.model_frame.reset_model()
                    
                # Update plot if available
                self.update_pool_plot()
            except Exception as e:
                print(f"Error loading experiments: {e}")
                import traceback
                traceback.print_exc()

    def update_exp_df_from_sheet(self):
        '''Updates the exp_df DataFrame with the current data from the exp_sheet.'''
        sheet_data = self.exp_sheet.get_sheet_data(get_header=False)
        self.exp_df = pd.DataFrame(sheet_data, columns=self.exp_sheet.headers())

    def save_experiments(self):
        '''Saves the experimental data to a CSV file using a file dialog.'''
        self.update_exp_df_from_sheet()  # Update the DataFrame with the current data from the sheet
        if self.exp_df is not None:
            file_path = filedialog.asksaveasfilename(
                title='Save Experiments CSV',
                defaultextension='.csv',
                filetypes=[('CSV Files', '*.csv')]
            )
            if file_path:
                try:
                    self.exp_df.to_csv(file_path, index=False)
                    print('Experiments saved successfully.')
                except Exception as e:
                    print('Error saving experiments:', e)
        else:
            print('No experimental data to save.')

    def generate_template(self):
        '''Generates a blank template with 10 starter points based on loaded variables.'''
        if self.var_df is not None:
            num_points = 10
            # Assume the variable names are in a column called 'Variables'
            if 'Variables' in self.var_df.columns:
                var_names = self.var_df['Variables'].tolist()
            else:
                var_names = self.var_df.columns.tolist()

            # Create a DataFrame with a column for each variable and an 'Output' column
            data = {var: [None] * num_points for var in var_names}
            data['Output'] = [None] * num_points
            self.exp_df = pd.DataFrame(data)
            self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
            self._update_ui_state()
            print('Experiment template generated.')
        else:
            print('Please load variables before generating a template.')

    def _get_skopt_space(self):
        """Helper to always return the skopt-compatible version of the search space"""
        if hasattr(self.search_space, 'to_skopt'):
            return self.search_space.to_skopt()
        return self.search_space

    def update_pool_plot(self, event=None):
        self.ax.cla()

        var1 = self.var1_dropdown.get()
        var2 = self.var2_dropdown.get()

        if self.cluster_switch.get():
            # DEPRECATED: cluster_pool functionality is deprecated
            print("WARNING: Clustering visualization is deprecated")
            # Compute kmeans if it hasn't been computed already.
            # if not hasattr(self, 'kmeans') or self.kmeans is None:
            #     # Use skopt-compatible version
            #     skopt_space = self._get_skopt_space()
            #     _, _, self.kmeans = cluster_pool(self.pool, self.exp_df, skopt_space, add_cluster=False)
            # # If a next point exists, enable the highlighting of the largest empty cluster.
            # add_cluster_flag = True if self.next_point is not None else False
            # plot_pool(self.pool, var1, var2, self.ax, kmeans=self.kmeans,
            #         add_cluster=add_cluster_flag, experiments=self.exp_df)
            # Fallback to non-clustered visualization
            plot_pool(self.pool, var1, var2, self.ax, kmeans=None, experiments=self.exp_df)
        else:
            plot_pool(self.pool, var1, var2, self.ax, kmeans=None, experiments=self.exp_df)

        if self.exp_df is not None and not self.exp_df.empty:
            self.ax.plot(self.exp_df[var1], self.exp_df[var2], 'go', markeredgecolor='k')

        if self.next_point is not None:

            if hasattr(self, 'tooltip'):
                self.tooltip.remove()

            # Plot the points
            self.ax.plot(self.next_point[var1], self.next_point[var2],
                        'bD', markeredgecolor='k', markersize=10)
            scatter = self.ax.scatter(self.next_point[var1], self.next_point[var2])
            
            # Create the tooltip
            self.tooltip = mplcursors.cursor(scatter, hover=True)
            
            # Format the numeric values to 1 decimal place, but leave strings as they are
            next_point_formatted = self.next_point.T.apply(lambda x: x.map(lambda v: f'{v:.1f}' if isinstance(v, (int, float)) else v))
            
            # Create the tooltip text with formatted values
            tooltip_text = tabulate(next_point_formatted, tablefmt='plain')

            # Set up the tooltip appearance
            self.tooltip.connect('add', lambda sel: sel.annotation.set_bbox(
                dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')))
            
            # Set the tooltip text to the formatted text
            self.tooltip.connect('add', lambda sel: sel.annotation.set_text(tooltip_text))

        self.canvas.draw()



    def next_explore_point(self):
        # DEPRECATED: This method uses legacy clustering and EMOC acquisition
        # Use the modern AcquisitionPanel with session API instead
        print("WARNING: next_explore_point() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI for next point selection")
        return
        # # Use skopt-compatible version
        # skopt_space = self._get_skopt_space()
        # # cluster_pool now returns the new clustering (with an added cluster) and kmeans.
        # labels, largest_empty_cluster, kmeans = cluster_pool(self.pool, self.exp_df, skopt_space, add_cluster=True)
        # # Update the stored kmeans object.
        # self.kmeans = kmeans
        #
        # largest_empty_cluster_points = self.pool[labels == largest_empty_cluster]
        #
        # X = self.exp_df.drop(columns='Output')
        # y = self.exp_df['Output']
        #
        # self.next_point = select_EMOC(largest_empty_cluster_points, X, y, self.search_space, model=self.gpr_model, verbose=False)
        # self.update_pool_plot()



    
    def pool_mode(self):
        self.update_pool_plot()
        print('Pool mode activated.')
    
    def explore_mode(self):
        '''Placeholder for exploration mode functionality.'''
        print('Exploration mode activated.')
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Explore mode visualization',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes)
        self.canvas.draw()

    def optimize_mode(self, use_dataframe=True):
        '''DEPRECATED: Use AcquisitionPanel with OptimizationSession API instead.'''
        print("WARNING: optimize_mode() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI for optimization")
        return
        # '''Optimizes the next experiment point based on the loaded data.'''
        # if self.exp_df is not None and self.search_space is not None:
        #     if use_dataframe:
        #         next_point = select_optimize(self.search_space, self.exp_df, base_estimator=self.gpr_model)
        #     else:
        #         if hasattr(self, 'encoded_X') and hasattr(self, 'gpr_model'):
        #             X = self.encoded_X.values
        #             y = self.exp_df['Output'].values
        #             next_point = select_optimize(self.search_space, (X, y), base_estimator=self.gpr_model)
        #         else:
        #             print('Encoded data or GPR model not found. Please train the model first.')
        #             return
        #
        #     # Convert the next_point to a DataFrame for consistency
        #     next_point_df = pd.DataFrame([next_point], columns=self.exp_df.drop(columns='Output').columns)
        #     
        #     # Store the next point for visualization
        #     self.next_point = next_point_df
        #     
        #     # Update the plot with the new point
        #     self.update_pool_plot()
        #     
        #     print('Optimization mode activated.')
        # else:
        #     print('Please load experiments and variables before optimizing.')


    def generate_initial_points(self):
        """Opens a window to select the sampling strategy and number of points."""
        if not self.search_space:
            print('Please load variables before generating initial points.')
            return

        self.initial_points_window = ctk.CTkToplevel(self)
        self.initial_points_window.title("Generate Initial Points")
        self.initial_points_window.geometry("300x200")
        self.initial_points_window.grab_set()

        ctk.CTkLabel(self.initial_points_window, text="Select Strategy:").pack(pady=5)
        self.strategy_var = ctk.StringVar(value="random")
        strategies = ["random", "LHS", "Sobol", "Halton", "Hammersly"]
        self.strategy_dropdown = ctk.CTkComboBox(self.initial_points_window, values=strategies, variable=self.strategy_var)
        self.strategy_dropdown.pack(pady=5)

        ctk.CTkLabel(self.initial_points_window, text="Number of Points:").pack(pady=5)
        self.num_points_entry = ctk.CTkEntry(self.initial_points_window)
        self.num_points_entry.insert(0, "10")
        self.num_points_entry.pack(pady=5)

        ctk.CTkButton(self.initial_points_window, text="Generate", command=self._generate_points).pack(pady=10)

    def _generate_points(self):
        """Generates initial points based on the selected strategy and number of points."""
        strategy = self.strategy_var.get()
        try:
            num_points = int(self.num_points_entry.get())
        except ValueError:
            print("Invalid number of points.")
            return

        if not self.search_space:
            print('Search space is not loaded.')
            return

        # Generate samples based on the chosen strategy
        if strategy == "random":
            # Manually sample for each dimension based on its type
            samples_list = []
            for dim in self.search_space:
                if isinstance(dim, Categorical):
                    samples = np.random.choice(dim.categories, size=num_points)
                elif isinstance(dim, Integer):
                    # np.random.randint is [low, high), so add 1 to include the upper bound
                    samples = np.random.randint(dim.low, dim.high + 1, size=num_points)
                elif isinstance(dim, Real):
                    samples = np.random.uniform(dim.low, dim.high, size=num_points)
                else:
                    raise ValueError(f"Unknown dimension type: {type(dim)}")
                samples_list.append(samples)
            # Combine the samples into a 2D numpy array
            samples = np.column_stack(samples_list)
        else:
            # Use the appropriate skopt sampler
            if strategy == "LHS":
                sampler = Lhs(lhs_type="classic", criterion="maximin")
            elif strategy == "Sobol":
                sampler = Sobol()
            elif strategy == "Halton":
                sampler = Hammersly()
            elif strategy == "Hammersly":
                sampler = Hammersly()
            else:
                print("Unknown sampling strategy.")
                return

            samples = sampler.generate(self.search_space, num_points)
            # Convert list of samples to a NumPy array for slicing
            samples = np.array(samples)

        # Build a DataFrame with the generated points and an 'Output' column.
        data = {dim.name: samples[:, i].tolist() for i, dim in enumerate(self.search_space)}
        data['Output'] = [None] * num_points
        self.exp_df = pd.DataFrame(data)
        self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())
        self._update_ui_state()
        print('Initial points generated.')
        self.initial_points_window.destroy()

    def add_point(self):
        '''Opens a window to add a new experiment point.'''
        if not self.search_space:
            print('Please load variables before adding a point.')
            return

        self.add_point_window = ctk.CTkToplevel(self)
        self.add_point_window.title("Add New Point")
        self.add_point_window.geometry("400x450")  # Made taller for the new field
        self.add_point_window.grab_set()

        self.var_entries = {}
        for var in self.search_space:
            ctk.CTkLabel(self.add_point_window, text=var.name).pack(pady=5)
            entry = ctk.CTkEntry(self.add_point_window)
            entry.pack(pady=5)
            self.var_entries[var.name] = entry

        ctk.CTkLabel(self.add_point_window, text='Output').pack(pady=5)
        self.output_entry = ctk.CTkEntry(self.add_point_window)
        self.output_entry.pack(pady=5)
        
        # Add noise field
        ctk.CTkLabel(self.add_point_window, text='Noise (optional)').pack(pady=5)
        self.noise_entry = ctk.CTkEntry(self.add_point_window)
        self.noise_entry.pack(pady=5)
        
        # Add info tooltip about noise
        ctk.CTkLabel(
            self.add_point_window, 
            text='Noise value represents measurement uncertainty\nand helps prevent overfitting.',
            font=('Arial', 10),
            text_color='grey'
        ).pack(pady=0)

        self.add_point_button_frame = ctk.CTkFrame(self.add_point_window)
        self.add_point_button_frame.pack(pady=5)

        self.save_checkbox = ctk.CTkCheckBox(self.add_point_button_frame, text='Save to file')
        self.save_checkbox.select()
        self.save_checkbox.pack(side='left', padx=5, pady=5)

        self.retrain_checkbox = ctk.CTkCheckBox(self.add_point_button_frame, text='Retrain model')
        self.retrain_checkbox.select()
        self.retrain_checkbox.pack(side='left', padx=5, pady=5)

        ctk.CTkButton(self.add_point_window, text='Save & Close', command=self.save_new_point).pack(pady=10)

    def save_new_point(self):
        '''Saves the new point to the tksheet, exp_df, and optionally to a file.'''
        new_point = {var: entry.get() for var, entry in self.var_entries.items()}
        new_point['Output'] = self.output_entry.get()
        
        # Add noise if provided
        noise_value = self.noise_entry.get().strip()
        if noise_value:
            try:
                new_point['Noise'] = float(noise_value)
            except ValueError:
                print(f"Invalid noise value '{noise_value}'. Using default.")
                new_point['Noise'] = 1e-6
        elif 'Noise' in self.exp_df.columns:
            # If noise column exists but no value provided, use default
            new_point['Noise'] = 1e-6

        # Add the new point to the exp_df
        new_point_df = pd.DataFrame([new_point])
        self.exp_df = pd.concat([self.exp_df, new_point_df], ignore_index=True)

        # Update the tksheet
        self.exp_sheet.set_sheet_data(self.exp_df.values.tolist())

        # Save to file if checkbox is checked
        if self.save_checkbox.get():
            if hasattr(self, 'exp_file_path') and self.exp_file_path:
                self.exp_df.to_csv(self.exp_file_path, index=False)
                self.load_experiments(self.exp_file_path)
            else:
                file_path = filedialog.asksaveasfilename(
                    title='Save Experiments CSV',
                    defaultextension='.csv',
                    filetypes=[('CSV Files', '*.csv')]
                )
                if file_path:
                    self.exp_df.to_csv(file_path, index=False)
        
        if self.retrain_checkbox.get():
            self.retrain_model()

        self.add_point_window.destroy()

    def retrain_model(self):
        print('Retraining model with new data...')
        self.model_frame.train_model_threaded()

    def run_selected_strategy(self):
        """DEPRECATED: Use AcquisitionPanel.run_selected_strategy() instead."""
        print("WARNING: run_selected_strategy() is deprecated and no longer functional")
        print("Please use the Acquisition Panel in the UI")
        return
        # """Executes the selected acquisition strategy."""
        # strategy = self.strategy_var.get()
        # try:
        #     if strategy == "Expected Improvement (EI)":
        #         self.next_point = select_EMOC(
        #             self.pool,
        #             self.exp_df.drop(columns='Output'),
        #             self.exp_df['Output'],
        #             self.search_space,
        #             model=self.gpr_model
        #         )
        #     elif strategy == "Upper Confidence Bound (UCB)":
        #         self.next_point = select_optimize(
        #             self.search_space,
        #             self.exp_df,
        #             base_estimator=self.gpr_model,
        #             acq_func="ucb"
        #         )
        #     elif strategy == "Probability of Improvement (PI)":
        #         self.next_point = select_optimize(
        #             self.search_space,
        #             self.exp_df,
        #             base_estimator=self.gpr_model,
        #             acq_func="pi"
        #         )
        #     elif strategy == "Thompson Sampling":
        #         # Implement Thompson Sampling logic here
        #         print("Thompson Sampling is not yet implemented.")
        #         return
        #     elif strategy == "Entropy Search":
        #         # Implement Entropy Search logic here
        #         print("Entropy Search is not yet implemented.")
        #         return
        #     elif strategy == "Custom Strategy":
        #         # Allow the user to define a custom strategy
        #         print("Custom Strategy is not yet implemented.")
        #         return
        #     elif strategy == "EMOC (Exploration)":
        #         # Comment out import and implementation
        #         # from logic.acquisition.emoc_acquisition import EMOCAcquisition
        #         
        #         # Placeholder message
        #         print("EMOC acquisition function not implemented in this version.")
        #         return
        #         
        #         # # Create acquisition function using trained model
        #         # acquisition = EMOCAcquisition(
        #         #     search_space=self.main_app.search_space,
        #         #     model=self.main_app.gpr_model,
        #         #     random_state=42
        #         # )
        #         # 
        #         # # Update with existing data
        #         # if hasattr(acquisition, 'update'):
        #         #     acquisition.update(
        #         #         self.main_app.exp_df.drop(columns='Output'),
        #         #         self.main_app.exp_df['Output']
        #         #     )
        #         # 
        #         # # Generate a pool if needed
        #         # if not hasattr(self.main_app, 'pool') or self.main_app.pool is None:
        #         #     from logic.pool import generate_pool
        #         #     self.main_app.pool = generate_pool(
        #         #         self.main_app.search_space, 
        #         #         self.main_app.exp_df, 
        #         #         pool_size=5000
        #         #     )
        #         # 
        #         # # Get next point
        #         # next_point = acquisition.select_next(self.main_app.pool)
        #         # 
        #         # # acq_func_kwargs for result data
        #         # acq_func_kwargs = {}
        #         
        #     elif strategy == "GandALF (Clustering + EMOC)":
        #         # Comment out import and implementation
        #         # from logic.acquisition.gandalf_acquisition import GandALFAcquisition
        #         
        #         # Placeholder message
        #         print("GandALF acquisition function not implemented in this version.")
        #         return
        #         
        #         # # Create acquisition instance
        #         # acquisition = GandALFAcquisition(
        #         #     search_space=self.main_app.search_space,
        #         #     model=self.main_app.gpr_model,
        #         #     random_state=42
        #         # )
        #         # 
        #         # # Update with existing data
        #         # acquisition.update(
        #         #     self.main_app.exp_df.drop(columns='Output'),
        #         #     self.main_app.exp_df['Output']
        #         # )
        #         # 
        #         # # Generate a pool if needed
        #         # if not hasattr(self.main_app, 'pool') or self.main_app.pool is None:
        #         #     from logic.pool import generate_pool
        #         #     self.main_app.pool = generate_pool(
        #         #         self.main_app.search_space, 
        #         #         self.main_app.exp_df,
        #         #         pool_size=5000
        #         #     )
        #         # 
        #         # # Get next point
        #         # next_point = acquisition.select_next(self.main_app.pool)
        #         # 
        #         # # acq_func_kwargs for result data
        #         # acq_func_kwargs = {'clustering': True}
        #     else:
        #         print("Unknown strategy selected.")
        #         return
        # 
        #     self.update_pool_plot()
        #     print(f"Strategy '{strategy}' executed successfully.")
        # except Exception as e:
        #     print(f"Error executing strategy '{strategy}': {e}")

    def toggle_tabbed_layout(self):
        """Toggle between side-by-side and tabbed layout"""
        if self.using_tabs:
            # Switch to side-by-side layout
            self.using_tabs = False
            
            # Store trained model and UI state for transfer
            trained_model = getattr(self, 'gpr_model', None)
            visualizations = None
            acq_enabled = False
            advanced_enabled = False
            current_backend = "scikit-learn"
            
            if hasattr(self, 'model_frame'):
                if hasattr(self.model_frame, 'visualizations'):
                    visualizations = self.model_frame.visualizations
                if hasattr(self.model_frame, 'advanced_var'):
                    advanced_enabled = self.model_frame.advanced_var.get()
                if hasattr(self.model_frame, 'backend_var'):
                    current_backend = self.model_frame.backend_var.get()
            
            if hasattr(self, 'acquisition_panel'):
                acq_enabled = self.acquisition_panel.run_button.cget("state") == "normal"
            
            # Remove the tabbed interface
            if hasattr(self, 'tab_view'):
                self.tab_view.destroy()
            
            # Create side-by-side layout
            self._create_model_frame()
            self._create_acquisition_frame()
            
            # Transfer the model and state
            if trained_model:
                self.model_frame.gpr_model = trained_model
                if visualizations:
                    self.model_frame.visualizations = visualizations
                    self.model_frame.visualize_button.configure(state="normal")
            
            # Set advanced options state
            self.model_frame.advanced_var.set(advanced_enabled)
            self.model_frame.toggle_advanced_options()
            
            # Set backend
            self.model_frame.backend_var.set(current_backend)
            self.model_frame.load_backend_options()
            
            # Set acquisition panel state
            if acq_enabled:
                self.acquisition_panel.enable()
            
            print("Switched to side-by-side layout")
        else:
            # Switch to tabbed layout
            self.using_tabs = True
            
            # Store trained model and UI state for transfer
            trained_model = getattr(self, 'gpr_model', None)
            visualizations = None
            acq_enabled = False
            advanced_enabled = False
            current_backend = "scikit-learn"
            
            if hasattr(self, 'model_frame'):
                if hasattr(self.model_frame, 'visualizations'):
                    visualizations = self.model_frame.visualizations
                if hasattr(self.model_frame, 'advanced_var'):
                    advanced_enabled = self.model_frame.advanced_var.get()
                if hasattr(self.model_frame, 'backend_var'):
                    current_backend = self.model_frame.backend_var.get()
            
            if hasattr(self, 'acquisition_panel'):
                acq_enabled = self.acquisition_panel.run_button.cget("state") == "normal"
            
            # Unpack existing frames
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'acq_frame'):
                self.acq_frame.pack_forget()
                
            # Create tabbed interface
            self.tab_view = ctk.CTkTabview(self)
            self.tab_view.pack(side='right', fill='both', padx=10, pady=10)
            self.tab_view.configure(width=300)
            
            # Add tabs
            self.tab_view.add("Model")
            self.tab_view.add("Acquisition")
            
            # Set the default tab
            self.tab_view.set("Model")
            
            # Create panels inside tabs
            self.model_frame = GaussianProcessPanel(self.tab_view.tab("Model"), self)
            self.model_frame.pack(fill='both', expand=True)
            
            self.acquisition_panel = AcquisitionPanel(self.tab_view.tab("Acquisition"), self)
            self.acquisition_panel.pack(fill='both', expand=True)
            
            # Transfer the model and state
            if trained_model:
                self.model_frame.gpr_model = trained_model
                if visualizations:
                    self.model_frame.visualizations = visualizations
                    self.model_frame.visualize_button.configure(state="normal")
            
            # Set advanced options state
            self.model_frame.advanced_var.set(advanced_enabled)
            self.model_frame.toggle_advanced_options()
            
            # Set backend
            self.model_frame.backend_var.set(current_backend)
            self.model_frame.load_backend_options()
            
            # Set acquisition panel state
            if acq_enabled:
                self.acquisition_panel.enable()
            
            print("Switched to tabbed layout for small screens")
            
    def switch_tab(self, tab_name):
        """Switch between model and acquisition tabs"""
        if tab_name == "Model":
            if hasattr(self, 'acquisition_panel'):
                self.acquisition_panel.pack_forget()
            if hasattr(self, 'model_frame'):
                self.model_frame.pack(in_=self.right_panel, fill='both', expand=True)
        else:  # Acquisition
            if hasattr(self, 'model_frame'):
                self.model_frame.pack_forget()
            if hasattr(self, 'acquisition_panel'):
                self.acquisition_panel.pack(in_=self.right_panel, fill='both', expand=True)

    def toggle_noise_column(self):
        """Show or hide the noise column from view without deleting the data"""
        # Update dataframe from UI
        self.update_exp_df_from_sheet()
        
        has_noise = 'Noise' in self.exp_df.columns
        
        if has_noise:
            # Instead of removing, just hide from view
            visible_df = self.exp_df.drop(columns=['Noise'])
            print("Noise column hidden from view (data is preserved).")
            self.noise_column_hidden = True
        else:
            # Add noise column if it doesn't exist
            if hasattr(self, 'noise_column_hidden') and self.noise_column_hidden:
                # Restore from backup if we were hiding it
                visible_df = self.exp_df
                self.noise_column_hidden = False
                print("Noise column restored to view.")
            else:
                # Add new noise column with default value
                self.exp_df['Noise'] = 1e-6
                visible_df = self.exp_df
                self.noise_column_hidden = False
                print("Noise column added with default value 1e-6.")
        
        # Update UI with visible columns (not modifying actual data)
        self.exp_sheet.set_sheet_data(visible_df.values.tolist())
        self.exp_sheet.set_header_data(visible_df.columns.tolist())
        self.exp_sheet.set_all_column_widths()
        
        # No need to reset model since the actual data structure isn't changing
        print("Note: Toggle only affects display. Model training will use noise if present.")
    
    # Removed: toggle_session_api method (session API is always enabled)
    
    def _sync_data_to_session(self):
        """Synchronize current UI state (variables and experiments) to the session."""
        try:
            # Sync search space from search_space_manager
            if hasattr(self, 'search_space_manager') and len(self.search_space_manager.variables) > 0:
                # Import the core SearchSpace to avoid confusion
                from alchemist_core.data.search_space import SearchSpace as CoreSearchSpace
                
                # Clear session search space and rebuild
                self.session.search_space = CoreSearchSpace()
                
                # Variables is a LIST of dictionaries, not a dict
                for var_dict in self.search_space_manager.variables:
                    var_name = var_dict['name']
                    var_type = var_dict['type']
                    
                    if var_type in ['real', 'integer']:
                        self.session.add_variable(
                            var_name, 
                            var_type, 
                            bounds=(var_dict['min'], var_dict['max'])
                        )
                    elif var_type == 'categorical':
                        self.session.add_variable(
                            var_name,
                            var_type,
                            categories=var_dict['values']
                        )
                print(f"Synced {len(self.session.search_space.variables)} variables to session")
            
            # Sync experiment data
            if hasattr(self, 'exp_df') and len(self.exp_df) > 0:
                # Copy experiment data to session's experiment manager
                # IMPORTANT: ExperimentManager uses 'df' attribute, not 'experiments_df'
                self.session.experiment_manager.df = self.exp_df.copy()
                
                # Set search space in experiment manager
                self.session.experiment_manager.set_search_space(self.session.search_space)
                
                print(f"Synced {len(self.exp_df)} experiments to session")
                
        except Exception as e:
            print(f"Error syncing data to session: {e}")
            import traceback
            traceback.print_exc()
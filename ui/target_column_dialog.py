"""
Target Column Selection Dialog

Allows users to select which column(s) in their CSV should be treated as optimization targets.
Supports both single-objective and multi-objective optimization.
"""

import customtkinter as ctk
from typing import List, Optional, Tuple
import tkinter as tk


class TargetColumnDialog(ctk.CTkToplevel):
    """
    Dialog for selecting target columns when loading experimental data.
    
    Features:
    - Single/Multi-objective mode toggle
    - Column selection (dropdown for single, checkboxes for multi)
    - Validation before confirming
    """
    
    def __init__(self, parent, available_columns: List[str], default_column: str = None):
        """
        Initialize the target column selection dialog.
        
        Args:
            parent: Parent window
            available_columns: List of column names available in the CSV
            default_column: Default column to select (if it exists in available_columns)
        """
        super().__init__(parent)
        
        self.title("Select Target Column(s)")
        self.geometry("500x550")
        self.resizable(True, True)
        self.minsize(450, 450)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Store data
        self.available_columns = available_columns
        self.default_column = default_column if default_column in available_columns else None
        self.result = None  # Will store selected column(s) when confirmed
        
        # UI state
        self.mode = "single"  # "single" or "multi"
        self.checkbox_vars = {}  # For multi-objective mode
        
        self._create_ui()
        
        # Center the dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
        
    def _create_ui(self):
        """Create the dialog UI elements."""
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            header_frame,
            text="Select Target Column(s)",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            header_frame,
            text="Choose which column(s) to optimize:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(anchor="w", pady=(5, 0))
        
        # Mode selector (Single vs Multi-objective)
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            mode_frame,
            text="Optimization Mode:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(side="left", padx=(10, 20))
        
        self.mode_var = ctk.StringVar(value="single")
        
        self.single_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Single-Objective",
            variable=self.mode_var,
            value="single",
            command=self._on_mode_change
        )
        self.single_radio.pack(side="left", padx=10)
        
        self.multi_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Multi-Objective",
            variable=self.mode_var,
            value="multi",
            command=self._on_mode_change
        )
        self.multi_radio.pack(side="left", padx=10)
        
        # Column selection area (content changes based on mode)
        self.selection_frame = ctk.CTkFrame(self)
        self.selection_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self._update_selection_ui()
        
        # Buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(10, 20))
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=100
        ).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="Confirm",
            command=self._on_confirm,
            width=100
        ).pack(side="right")
        
    def _on_mode_change(self):
        """Handle mode change between single and multi-objective."""
        self.mode = self.mode_var.get()
        self._update_selection_ui()
        
    def _update_selection_ui(self):
        """Update the column selection UI based on current mode."""
        # Clear existing widgets
        for widget in self.selection_frame.winfo_children():
            widget.destroy()
            
        if self.mode == "single":
            self._create_single_objective_ui()
        else:
            self._create_multi_objective_ui()
            
    def _create_single_objective_ui(self):
        """Create UI for single-objective mode (dropdown)."""
        ctk.CTkLabel(
            self.selection_frame,
            text="Select target column:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        # Dropdown menu
        self.column_var = ctk.StringVar(value=self.default_column or self.available_columns[0])
        
        self.column_dropdown = ctk.CTkOptionMenu(
            self.selection_frame,
            variable=self.column_var,
            values=self.available_columns,
            width=400
        )
        self.column_dropdown.pack(padx=20, pady=10)
        
        # Info text
        info_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="💡 Tip: This column will be maximized or minimized during optimization.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=400,
            justify="left"
        ).pack(anchor="w")
        
    def _create_multi_objective_ui(self):
        """Create UI for multi-objective mode (checkboxes)."""
        ctk.CTkLabel(
            self.selection_frame,
            text="Select target columns (2 or more):",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", padx=20, pady=(20, 10))
        
        # Scrollable frame for checkboxes
        checkbox_frame = ctk.CTkScrollableFrame(
            self.selection_frame,
            height=200
        )
        checkbox_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create checkboxes for each column
        self.checkbox_vars = {}
        for col in self.available_columns:
            var = ctk.BooleanVar(value=False)
            self.checkbox_vars[col] = var
            
            checkbox = ctk.CTkCheckBox(
                checkbox_frame,
                text=col,
                variable=var
            )
            checkbox.pack(anchor="w", pady=5, padx=10)
            
        # Info text
        info_frame = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=(10, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="💡 Tip: Multi-objective optimization finds trade-offs between objectives.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=400,
            justify="left"
        ).pack(anchor="w")
        
    def _on_confirm(self):
        """Handle confirm button click."""
        if self.mode == "single":
            # Single-objective: return selected column as string
            selected = self.column_var.get()
            if selected:
                self.result = selected
                self.destroy()
        else:
            # Multi-objective: return list of selected columns
            selected = [col for col, var in self.checkbox_vars.items() if var.get()]
            if len(selected) < 2:
                # Show error - need at least 2 objectives
                error_dialog = ctk.CTkToplevel(self)
                error_dialog.title("Invalid Selection")
                error_dialog.geometry("350x150")
                error_dialog.transient(self)
                error_dialog.grab_set()
                
                ctk.CTkLabel(
                    error_dialog,
                    text="⚠️ Multi-Objective Mode",
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack(pady=(20, 10))
                
                ctk.CTkLabel(
                    error_dialog,
                    text="Please select at least 2 target columns\nfor multi-objective optimization.",
                    font=ctk.CTkFont(size=12)
                ).pack(pady=10)
                
                ctk.CTkButton(
                    error_dialog,
                    text="OK",
                    command=error_dialog.destroy,
                    width=100
                ).pack(pady=10)
                
                # Center error dialog
                error_dialog.update_idletasks()
                x = self.winfo_x() + (self.winfo_width() // 2) - (error_dialog.winfo_width() // 2)
                y = self.winfo_y() + (self.winfo_height() // 2) - (error_dialog.winfo_height() // 2)
                error_dialog.geometry(f"+{x}+{y}")
                return
                
            self.result = selected
            self.destroy()
            
    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.destroy()
        
    def get_result(self) -> Optional[str | List[str]]:
        """
        Get the user's selection.
        
        Returns:
            String for single-objective, list for multi-objective, or None if cancelled
        """
        return self.result


def show_target_column_dialog(parent, available_columns: List[str], 
                              default_column: str = None) -> Optional[str | List[str]]:
    """
    Show target column selection dialog and return user's choice.
    
    Args:
        parent: Parent window
        available_columns: List of column names available in the CSV
        default_column: Default column to select (if it exists)
        
    Returns:
        Selected column(s) or None if cancelled
    """
    dialog = TargetColumnDialog(parent, available_columns, default_column)
    parent.wait_window(dialog)
    return dialog.get_result()

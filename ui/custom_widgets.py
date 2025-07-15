"""
Custom UI widgets for ALchemist application.

This module contains custom widgets that extend CustomTkinter functionality
for specific use cases in the ALchemist application.
"""

import tkinter
from typing import Union, Callable
import customtkinter as ctk


class CTkSpinbox(ctk.CTkFrame):
    """
    A spinbox widget for CustomTkinter with increment/decrement buttons.
    
    Combines an entry field with +/- buttons for numeric input with step controls.
    """
    
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 step_size: Union[int, float] = 1,
                 command: Callable = None,
                 min_value: Union[int, float] = None,
                 max_value: Union[int, float] = None,
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.command = command
        self.min_value = min_value
        self.max_value = max_value

        self.configure(fg_color=("gray78", "gray28"))  # set frame color

        self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
        self.grid_columnconfigure(1, weight=1)  # entry expands

        self.subtract_button = ctk.CTkButton(self, text="-", width=height-6, height=height-6,
                                           command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = ctk.CTkEntry(self, width=width-(2*height), height=height-6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = ctk.CTkButton(self, text="+", width=height-6, height=height-6,
                                      command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
        self.entry.insert(0, "0.0")

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            current_value = float(self.entry.get())
            new_value = current_value + self.step_size
            
            # Check max_value constraint
            if self.max_value is not None and new_value > self.max_value:
                new_value = self.max_value
                
            self.entry.delete(0, "end")
            self.entry.insert(0, str(new_value))
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            current_value = float(self.entry.get())
            new_value = current_value - self.step_size
            
            # Check min_value constraint
            if self.min_value is not None and new_value < self.min_value:
                new_value = self.min_value
                
            self.entry.delete(0, "end")
            self.entry.insert(0, str(new_value))
        except ValueError:
            return

    def get(self) -> Union[float, None]:
        try:
            return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: Union[int, float]):
        # Validate against min/max constraints
        if self.min_value is not None and value < self.min_value:
            value = self.min_value
        if self.max_value is not None and value > self.max_value:
            value = self.max_value
            
        self.entry.delete(0, "end")
        self.entry.insert(0, str(float(value)))

    def configure_limits(self, min_value: Union[int, float] = None, max_value: Union[int, float] = None):
        """Update the min/max value constraints."""
        self.min_value = min_value
        self.max_value = max_value

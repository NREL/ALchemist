"""
UI utilities and helper functions for ALchemist.

This module contains utility functions that extend
the base CustomTkinter functionality.
"""

import customtkinter as ctk
from typing import Union, Callable
from ui.custom_widgets import CTkSpinbox

# Re-export for convenience
__all__ = ['CTkSpinbox', 'create_labeled_spinbox']


def create_labeled_spinbox(parent, label_text: str, **spinbox_kwargs) -> tuple:
    """
    Create a labeled spinbox widget.
    
    Returns:
        tuple: (label_widget, spinbox_widget)
    """
    label = ctk.CTkLabel(parent, text=label_text)
    spinbox = CTkSpinbox(parent, **spinbox_kwargs)
    return label, spinbox

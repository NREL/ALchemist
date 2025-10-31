"""
ALchemist UI Layer

This package contains the graphical user interface for ALchemist, built with
CustomTkinter. The UI layer is cleanly separated from the core library
(alchemist_core) to enable:

1. Independent development of UI and core features
2. Future UI migrations (e.g., web-based interface)
3. Headless operation via alchemist_core Session API

Import Structure:
-----------------
The UI layer should ONLY import from:
- alchemist_core: For all core functionality (data, models, acquisition, session)
- ui: For UI-specific utilities and widgets
- Standard Python libraries and UI frameworks

DO NOT import from logic/: The logic/ directory is deprecated and being phased out.

Main Components:
----------------
- ui.py: Main application window and controller
- gpr_panel.py: Model training interface
- acquisition_panel.py: Acquisition strategy interface (uses Session API)
- visualizations.py: Plotting and diagnostic visualizations
- notifications.py: Result display windows
- experiment_logger.py: Experiment log file generation (UI-layer utility)
- pool_viz.py: DEPRECATED pool visualization (will be removed in v0.3.0)

Version: 0.2.0-dev (Core-UI Split)
"""

__version__ = "0.2.0-dev"
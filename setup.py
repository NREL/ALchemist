"""
Setup.py for ALchemist - bridges pyproject.toml with custom build hooks.

This file is needed to use custom build commands (build_hooks.py) because
pyproject.toml doesn't support cmdclass directly in modern setuptools.

For editable installs, the custom build hook is skipped.
"""

from setuptools import setup

# Try to import custom build hook, but don't fail if it's not available
# (e.g., during editable installs)
try:
    from build_hooks import BuildWithFrontend
    cmdclass = {'build_py': BuildWithFrontend}
except (ImportError, ModuleNotFoundError):
    # No custom build hook for editable installs
    cmdclass = {}

# All configuration is in pyproject.toml
# This file only exists to register the custom build command
setup(cmdclass=cmdclass)

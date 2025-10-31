"""
Logging configuration for ALchemist Core.

Provides centralized logging setup with configurable verbosity levels.
"""

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for alchemist_core.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string for log messages
    
    Example:
        >>> from alchemist_core.config import configure_logging
        >>> import logging
        >>> 
        >>> # Quiet mode (warnings and errors only)
        >>> configure_logging(level=logging.WARNING)
        >>> 
        >>> # Verbose mode with file logging
        >>> configure_logging(level=logging.DEBUG, log_file="alchemist.log")
        >>> 
        >>> # Custom format
        >>> configure_logging(format_string="%(levelname)s: %(message)s")
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger for alchemist_core
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set level for alchemist_core loggers specifically
    alchemist_logger = logging.getLogger('alchemist_core')
    alchemist_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module (usually __name__)
    
    Returns:
        Logger instance
    
    Example:
        >>> from alchemist_core.config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting computation...")
    
    Note:
        If the name doesn't start with 'alchemist_core', it will be prefixed
        with 'alchemist_core.' to maintain namespace consistency. This ensures
        all alchemist_core loggers are grouped together.
    """
    # Only prepend alchemist_core if the name doesn't already have a package prefix
    # This prevents creating names like "alchemist_core.logic.logging"
    if not name.startswith('alchemist_core') and '.' not in name:
        name = f'alchemist_core.{name}'
    
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers in Jupyter notebook environments
    # where modules may be reloaded
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()
    
    return logger


def set_verbosity(verbose: bool = True) -> None:
    """
    Quick helper to set verbosity level.
    
    Args:
        verbose: If True, set to INFO level. If False, set to WARNING level.
    
    Example:
        >>> from alchemist_core.config import set_verbosity
        >>> set_verbosity(False)  # Quiet mode
        >>> set_verbosity(True)   # Verbose mode
    """
    level = logging.INFO if verbose else logging.WARNING
    configure_logging(level=level)


# Default configuration
configure_logging(level=logging.INFO)

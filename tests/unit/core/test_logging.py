"""
Test logging configuration and usage.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alchemist_core.config import configure_logging, get_logger, set_verbosity


def test_basic_logging():
    """Test that logging works with default configuration."""
    configure_logging()  # Use default INFO level
    logger = get_logger(__name__)
    
    logger.info("This is an INFO message - should appear")
    logger.warning("This is a WARNING message - should appear")
    logger.error("This is an ERROR message - should appear")
    logger.debug("This is a DEBUG message - should NOT appear with default config")
    print("✓ Basic logging test passed")


def test_verbosity_levels():
    """Test changing verbosity levels."""
    # Test DEBUG level
    set_verbosity("DEBUG")
    logger = get_logger(__name__)
    logger.debug("This DEBUG message should now appear")
    
    # Test WARNING level
    set_verbosity("WARNING")
    logger.info("This INFO message should NOT appear")
    logger.warning("This WARNING message should appear")
    
    # Reset to INFO
    set_verbosity("INFO")
    print("✓ Verbosity levels test passed")


def test_module_loggers():
    """Test that different modules get different logger instances."""
    configure_logging()
    
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")
    
    logger1.info("Message from module1")
    logger2.info("Message from module2")
    
    # Loggers are prefixed with "alchemist_core." 
    assert logger1.name == "alchemist_core.module1"
    assert logger2.name == "alchemist_core.module2"
    assert logger1 is not logger2
    print("✓ Module loggers test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing ALchemist Core Logging System")
    print("="*60 + "\n")
    
    test_basic_logging()
    print()
    test_verbosity_levels()
    print()
    test_module_loggers()
    
    print("\n" + "="*60)
    print("All logging tests passed!")
    print("="*60)

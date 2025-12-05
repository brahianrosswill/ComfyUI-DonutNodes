"""
WIDEN Merge Logging Configuration

This module provides centralized logging configuration for WIDEN merge operations.
It includes specialized loggers for different aspects of the merge process:

- widen_logger: General WIDEN merge information and progress
- performance_logger: Performance metrics and timing information
- memory_logger: Memory usage and optimization details
- diagnostic_logger: Detailed diagnostic information for debugging

It also provides progress bar functionality for clean output during merge operations.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Optional


# Configure basic logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Create specialized loggers for different aspects of WIDEN merging
widen_logger = logging.getLogger('widen_merge')
performance_logger = logging.getLogger('widen_performance') 
memory_logger = logging.getLogger('widen_memory')
diagnostic_logger = logging.getLogger('widen_diagnostic')

# Set initial log levels
widen_logger.setLevel(logging.INFO)
performance_logger.setLevel(logging.INFO)
memory_logger.setLevel(logging.WARNING)  # Memory logs are verbose by default
diagnostic_logger.setLevel(logging.WARNING)  # Diagnostic logs are verbose by default


def configure_widen_logging(level="INFO", enable_memory_logs=False, enable_diagnostic_debug=False):
    """
    Configure WIDEN merge logging levels.
    
    Args:
        level: Overall log level ("DEBUG", "INFO", "WARNING", "ERROR")
        enable_memory_logs: Show verbose memory debugging (default: False)  
        enable_diagnostic_debug: Show detailed diagnostic info (default: False)
    
    Examples:
        configure_widen_logging("DEBUG")  # Show everything
        configure_widen_logging("WARNING")  # Only warnings and errors
        configure_widen_logging("INFO", enable_diagnostic_debug=True)  # Info + diagnostics
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO, 
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    widen_logger.setLevel(log_level)
    performance_logger.setLevel(log_level)
    
    # Memory logs are verbose - only enable if requested
    memory_logger.setLevel(logging.DEBUG if enable_memory_logs else logging.WARNING)
    
    # Diagnostic logs default to warnings unless debug enabled
    diagnostic_logger.setLevel(logging.DEBUG if enable_diagnostic_debug else logging.WARNING)
    
    widen_logger.info(f"WIDEN logging configured: level={level}, memory_logs={enable_memory_logs}, diagnostics={enable_diagnostic_debug}")


# Usage examples for logging configuration:
# configure_widen_logging("DEBUG")  # Show all logs including detailed diagnostics
# configure_widen_logging("WARNING")  # Only show warnings and errors (quiet mode)
# configure_widen_logging("INFO", enable_diagnostic_debug=True)  # Show compatibility warnings
# configure_widen_logging("INFO", enable_memory_logs=True)  # Show memory debugging


def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', 
                      decimals: int = 1, length: int = 50, fill: str = '█', print_end: str = "\r"):
    """
    Print a progress bar to the console.
    
    Args:
        iteration: Current iteration (0 to total)
        total: Total iterations
        prefix: Prefix string before the progress bar
        suffix: Suffix string after the progress bar  
        decimals: Number of decimals for percentage display
        length: Character length of the progress bar
        fill: Fill character for the progress bar
        print_end: End character (e.g. "\r", "\r\n")
    """
    if total == 0:
        return
        
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    
    # Print new line on complete
    if iteration == total: 
        print()


@contextmanager
def ProgressBarContext():
    """
    Context manager to suppress verbose logging during progress bar updates.
    
    This temporarily suppresses profiler and other verbose logs to keep
    progress bar output clean and readable.
    """
    # Store original log levels
    original_levels = {}
    
    # List of loggers that might interfere with progress bar output
    interfering_loggers = [
        'widen_memory',
        'widen_diagnostic', 
        'widen_performance'
    ]
    
    try:
        # Temporarily raise log levels to reduce noise during progress updates
        for logger_name in interfering_loggers:
            logger = logging.getLogger(logger_name)
            original_levels[logger_name] = logger.level
            if logger.level < logging.WARNING:
                logger.setLevel(logging.WARNING)
        
        yield
        
    finally:
        # Restore original log levels
        for logger_name, original_level in original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)


# Initialize with default configuration
configure_widen_logging()
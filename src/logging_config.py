"""
Centralized logging configuration for the Garmin Analysis package.
"""
import logging
import sys
from pathlib import Path

def setup_logging(
    level=logging.INFO,
    log_file=None,
    console_output=True,
    file_output=True
):
    """
    Configure logging for the entire package.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: garmin_analysis.log in project root)
        console_output: Whether to output to console (default: True)
        file_output: Whether to output to file (default: True)
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set root logger level
    root_logger.setLevel(level)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            log_file = Path("garmin_analysis.log")
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)

# Default setup
setup_logging()

"""
Garmin Analysis Package

A comprehensive toolkit for analyzing Garmin health and fitness data.
Provides data ingestion, cleaning, analysis, visualization, and reporting capabilities.
"""

__version__ = "0.1.0"
__author__ = "Chang-Hyun Mungai"
__email__ = "changhyunmungai@gmail.com"

# Import logging configuration
from .logging_config import setup_logging, get_logger

# Set up default logging
setup_logging()

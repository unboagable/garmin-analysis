"""
Centralized configuration for Garmin Analysis.

This module provides canonical paths and configuration constants
used throughout the application.
"""
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Project root (assuming config.py is in src/garmin_analysis/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Database directory
DB_DIR = PROJECT_ROOT / "db"

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Output directories
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELING_RESULTS_DIR = PROJECT_ROOT / "modeling_results"
DATA_QUALITY_REPORTS_DIR = PROJECT_ROOT / "data_quality_reports"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# ============================================================================
# Database Paths
# ============================================================================

DB_PATHS = {
    "garmin": DB_DIR / "garmin.db",
    "activities": DB_DIR / "garmin_activities.db",
    "monitoring": DB_DIR / "garmin_monitoring.db",
    "summary": DB_DIR / "garmin_summary.db",
    "summary2": DB_DIR / "summary.db",
}

# ============================================================================
# Data File Paths
# ============================================================================

MASTER_CSV = DATA_DIR / "master_daily_summary.csv"
MODELING_CSV = DATA_DIR / "modeling_ready_dataset.csv"

# ============================================================================
# Configuration File Paths
# ============================================================================

ACTIVITY_MAPPINGS_CONFIG = CONFIG_DIR / "activity_type_mappings.json"

# ============================================================================
# Output Settings
# ============================================================================

# Default figure size for plots
DEFAULT_FIGSIZE = (12, 8)

# Default DPI for saved plots
DEFAULT_DPI = 100

# Plot file format
PLOT_FORMAT = "png"

# ============================================================================
# Analysis Settings
# ============================================================================

# Default imputation strategy for missing values
DEFAULT_IMPUTATION_STRATEGY = "median"

# 24-hour coverage filtering defaults
DEFAULT_MAX_GAP_MINUTES = 2
DEFAULT_DAY_EDGE_TOLERANCE_MINUTES = 2
DEFAULT_COVERAGE_ALLOWANCE_MINUTES = 0

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories_exist():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
        MODELING_RESULTS_DIR,
        DATA_QUALITY_REPORTS_DIR,
        CONFIG_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_db_path(db_name: str) -> Path:
    """
    Get path to a specific database.
    
    Args:
        db_name: One of 'garmin', 'activities', 'monitoring', 'summary', 'summary2'
    
    Returns:
        Path object to the database
    
    Raises:
        KeyError: If db_name is not recognized
    """
    return DB_PATHS[db_name]


# Initialize directories on module import
ensure_directories_exist()


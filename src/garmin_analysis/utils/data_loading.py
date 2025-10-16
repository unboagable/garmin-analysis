import os
import sqlite3
import pandas as pd
import logging

from garmin_analysis.config import MASTER_CSV
from garmin_analysis.utils.error_handling import (
    handle_data_loading_errors,
    handle_database_errors,
    DataLoadingError,
    validate_file_path
)

logger = logging.getLogger(__name__)


@handle_database_errors(default_return={})
def load_garmin_tables(db_path="db/garmin.db"):
    """
    Load Garmin tables from SQLite database.
    
    Args:
        db_path: Path to Garmin database file
    
    Returns:
        Dictionary of DataFrames containing daily, sleep, stress, and rest_hr tables
        Returns empty dict if database not found or loading fails
    
    Raises:
        DatabaseError: If reraise=True in decorator (currently returns {} on error)
    """
    if not os.path.exists(db_path):
        logger.error("Database file '%s' not found. Please run garmindb_cli.py or place it in the root directory.", db_path)
        return {}

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            "daily": pd.read_sql_query("SELECT * FROM daily_summary", conn, parse_dates=["day"]),
            "sleep": pd.read_sql_query("SELECT * FROM sleep", conn, parse_dates=["day"]),
            "stress": pd.read_sql_query("SELECT * FROM stress", conn, parse_dates=["timestamp"]),
            "rest_hr": pd.read_sql_query("SELECT * FROM resting_hr", conn, parse_dates=["day"]),
        }
        logger.info("Successfully loaded Garmin tables from '%s'", db_path)
        return tables
    finally:
        conn.close()


@handle_data_loading_errors(reraise=True)
def load_master_dataframe():
    """
    Load master daily summary DataFrame from CSV.
    
    If CSV doesn't exist, attempts to auto-build it via summarize_and_merge().
    
    Returns:
        DataFrame with master daily summary data
    
    Raises:
        DataLoadingError: If file not found and auto-build fails
        FileNotFoundError: Wrapped in DataLoadingError
    """
    df_path = str(MASTER_CSV)
    if not os.path.exists(df_path):
        logger.warning("%s not found. Attempting to build it via summarize_and_merge().", df_path)
        # Lazy import to avoid circulars at module import time
        from garmin_analysis.data_ingestion.load_all_garmin_dbs import summarize_and_merge, USING_SYNTHETIC_DATA
        
        df = summarize_and_merge(return_df=True)
        
        # Ensure the file is saved for subsequent calls
        os.makedirs(os.path.dirname(df_path), exist_ok=True)
        df.to_csv(df_path, index=False)
        
        if 'USING_SYNTHETIC_DATA' in globals() and USING_SYNTHETIC_DATA:
            logger.warning("Master dataset built from SYNTHETIC data due to missing DBs. File: %s", df_path)
            logger.warning("Do not rely on this dataset for real analysis. Place real DBs under `db/` and rebuild.")
        
        logger.info("Built and saved master dataset to %s (%d rows, %d cols)", df_path, len(df), df.shape[1])
        return df
    
    df = pd.read_csv(df_path, parse_dates=["day"])
    logger.info("Loaded master dataset with %d rows and %d columns", len(df), df.shape[1])
    return df

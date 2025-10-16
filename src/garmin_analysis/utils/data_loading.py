import os
import sqlite3
import pandas as pd
import logging

from garmin_analysis.config import MASTER_CSV


def load_garmin_tables(db_path="db/garmin.db"):
    if not os.path.exists(db_path):
        logging.error("Database file '%s' not found. Please run garmindb_cli.py or place it in the root directory.", db_path)
        return {}

    try:
        conn = sqlite3.connect(db_path)
        tables = {
            "daily": pd.read_sql_query("SELECT * FROM daily_summary", conn, parse_dates=["day"]),
            "sleep": pd.read_sql_query("SELECT * FROM sleep", conn, parse_dates=["day"]),
            "stress": pd.read_sql_query("SELECT * FROM stress", conn, parse_dates=["timestamp"]),
            "rest_hr": pd.read_sql_query("SELECT * FROM resting_hr", conn, parse_dates=["day"]),
        }
        conn.close()
        logging.info("Successfully loaded Garmin tables from '%s'", db_path)
        return tables
    except Exception as e:
        logging.exception("Failed to load Garmin tables: %s", e)
        return {}


def load_master_dataframe():
    df_path = str(MASTER_CSV)
    if not os.path.exists(df_path):
        logging.warning("%s not found. Attempting to build it via summarize_and_merge().", df_path)
        try:
            # Lazy import to avoid circulars at module import time
            from garmin_analysis.data_ingestion.load_all_garmin_dbs import summarize_and_merge, USING_SYNTHETIC_DATA
            df = summarize_and_merge(return_df=True)
            # Ensure the file is saved for subsequent calls
            os.makedirs(os.path.dirname(df_path), exist_ok=True)
            df.to_csv(df_path, index=False)
            if 'USING_SYNTHETIC_DATA' in globals() and USING_SYNTHETIC_DATA:
                logging.warning("Master dataset built from SYNTHETIC data due to missing DBs. File: %s", df_path)
                logging.warning("Do not rely on this dataset for real analysis. Place real DBs under `db/` and rebuild.")
            logging.info("Built and saved master dataset to %s (%d rows, %d cols)", df_path, len(df), df.shape[1])
            return df
        except Exception as e:
            logging.error("Failed to build master dataset automatically: %s", e)
            raise FileNotFoundError(f"{df_path} not found and auto-build failed: {e}")
    df = pd.read_csv(df_path, parse_dates=["day"])
    logging.info("Loaded master dataset with %d rows and %d columns", len(df), df.shape[1])
    return df

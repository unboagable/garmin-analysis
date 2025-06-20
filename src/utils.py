import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_garmin_tables(db_path="garmin.db"):
    """Load key Garmin tables from SQLite into DataFrames."""
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

def normalize_dates(df, col="day"):
    """Normalize datetime column to remove time component."""
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df

def filter_by_date(df, date_col="day", from_date=None, to_date=None, days_back=None, weeks_back=None, months_back=None):
    """Filter DataFrame by relative or absolute date range."""
    if df.empty:
        logging.warning("Received empty DataFrame to filter on column '%s'.", date_col)
        return df

    df[date_col] = pd.to_datetime(df[date_col])
    now = datetime.now()

    if days_back:
        from_date = now - timedelta(days=days_back)
    elif weeks_back:
        from_date = now - timedelta(weeks=weeks_back)
    elif months_back:
        from_date = now - DateOffset(months=months_back)

    if from_date:
        df = df[df[date_col] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df[date_col] <= pd.to_datetime(to_date)]

    logging.info("Filtered data on column '%s' between %s and %s — %d rows returned.",
                 date_col, from_date, to_date, len(df))
    return df


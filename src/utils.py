import os
import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Iterable, Optional
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def normalize_dates(df, col="day"):
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df

def convert_time_to_minutes(time_str):
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 60 + m + s / 60
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m + s / 60
        else:
            return float(time_str)
    except Exception:
        return None

def normalize_day_column(df: pd.DataFrame, source_name: str = "unknown") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "day" in df.columns:
        df = df.assign(day=pd.to_datetime(df["day"]))
    elif "calendarDate" in df.columns:
        df = df.assign(day=pd.to_datetime(df["calendarDate"]))
    elif "timestamp" in df.columns:
        df = df.assign(day=pd.to_datetime(df["timestamp"]).dt.normalize())
    else:
        logging.warning(f"[{source_name}] could not normalize 'day' column (missing day/calendarDate/timestamp)")
    return df

def filter_by_date(df, date_col="day", from_date=None, to_date=None, days_back=None, weeks_back=None, months_back=None):
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

def convert_time_columns(df, columns):
    def time_to_minutes(val):
        try:
            h, m, s = map(int, val.split(":"))
            return h * 60 + m + s / 60
        except:
            return np.nan

    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(time_to_minutes)
    return df

def ensure_datetime_sorted(
    df: pd.DataFrame,
    date_candidates: Iterable[str] = ("date", "day", "start_time", "timestamp"),
    tz: Optional[str] = None,
    drop_dupes: bool = True,
) -> pd.DataFrame:
    """
    - Finds the first present date-like column and converts to pandas datetime.
    - Normalizes to date (YYYY-MM-DD) if column looks daily (name 'date' or 'day').
    - Sorts ascending by that column.
    - Optionally drops duplicate dates, keeping first (stable).
    """
    if df is None or df.empty:
        return df

    # Pick the first date column that exists
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        return df  # nothing to do

    # Parse to datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)

    # Optionally localize/convert timezone if provided
    if tz:
        df[date_col] = df[date_col].dt.tz_convert(tz)

    # If it's a daily table, normalize to date
    if date_col in ("date", "day"):
        df[date_col] = df[date_col].dt.date
        # back to datetime64[ns] (naive) to keep Pandas happy and sortable
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows with unparseable dates
    df = df[df[date_col].notna()]

    # Sort ascending
    df = df.sort_values(by=date_col, kind="mergesort")  # stable

    # Deduplicate on daily date if requested
    if drop_dupes and date_col in ("date", "day"):
        df = df.drop_duplicates(subset=[date_col], keep="first")

    return df

def aggregate_stress(stress_df):
    stress_df["timestamp"] = pd.to_datetime(stress_df["timestamp"])
    stress_df["day"] = stress_df["timestamp"].dt.date
    daily = stress_df.groupby("day").agg(
        stress_avg=("stress", "mean"),
        stress_max=("stress", "max"),
        stress_duration=("stress", lambda x: (x > 0).sum())
    ).reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    return daily

def load_master_dataframe():
    df_path = "data/master_daily_summary.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"{df_path} not found. Please run the ingestion script first.")
    df = pd.read_csv(df_path, parse_dates=["day"])
    logging.info("Loaded master dataset with %d rows and %d columns", len(df), df.shape[1])
    return df

def standardize_features(df, columns):
    df = df.copy()
    df = df.dropna(subset=columns)
    if df.empty:
        logging.warning("No data left after dropping NaNs in columns: %s", columns)
        return np.array([])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    logging.info("Standardized features: %s", columns)
    return scaled

def filter_required_columns(df, required_cols):
    missing_req = [col for col in required_cols if col not in df.columns]
    if missing_req:
        logging.warning(f"Missing required columns for analysis: {missing_req}")
        return df
    before = len(df)
    df = df.dropna(subset=required_cols)
    logging.info(f"Filtered rows missing {required_cols}. Kept {len(df)} of {before} rows.")
    return df

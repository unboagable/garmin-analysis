import os
import sqlite3
import pandas as pd
import logging
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

def convert_time_to_minutes(time_str):
    try:
        h, m, s = map(int, time_str.split(":"))
        return h * 60 + m + s / 60
    except:
        return 0

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
    tables = load_garmin_tables()
    if "daily" not in tables:
        raise ValueError("Missing 'daily_summary' table in Garmin DB")
    return tables["daily"]

def standardize_features(df, columns):
    df = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    logging.info("Standardized features: %s", columns)
    return scaled




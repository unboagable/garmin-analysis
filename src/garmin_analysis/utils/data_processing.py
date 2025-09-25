import pandas as pd
import numpy as np
import logging
from typing import Iterable, Optional


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

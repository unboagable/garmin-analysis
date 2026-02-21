import pandas as pd
import numpy as np
import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


def convert_time_to_minutes(time_str):
    """Convert a time string (HH:MM:SS, MM:SS, or numeric) to minutes. Returns np.nan on invalid input."""
    if time_str is None or (isinstance(time_str, (str, bytes)) and not str(time_str).strip()):
        return np.nan
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 60 + m + s / 60
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m + s / 60
        else:
            val = float(time_str)
            if not np.isfinite(val):
                return np.nan
            return val
    except Exception:
        return np.nan


def normalize_day_column(df: pd.DataFrame, source_name: str = "unknown") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "day" in df.columns:
        try:
            df = df.assign(day=pd.to_datetime(df["day"], errors="coerce"))
        except ValueError:
            df = df.assign(day=pd.to_datetime(df["day"], errors="coerce", utc=True).dt.tz_localize(None))
    elif "calendarDate" in df.columns:
        try:
            df = df.assign(day=pd.to_datetime(df["calendarDate"], errors="coerce"))
        except ValueError:
            df = df.assign(day=pd.to_datetime(df["calendarDate"], errors="coerce", utc=True).dt.tz_localize(None))
    elif "timestamp" in df.columns:
        try:
            df = df.assign(day=pd.to_datetime(df["timestamp"], errors="coerce").dt.normalize())
        except ValueError:
            df = df.assign(day=pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize())
    else:
        logger.warning(f"[{source_name}] could not normalize 'day' column (missing day/calendarDate/timestamp)")
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

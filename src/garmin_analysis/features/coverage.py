import logging
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from garmin_analysis.utils.error_handling import (
    handle_data_loading_errors,
    validate_dataframe,
    DataValidationError
)

logger = logging.getLogger(__name__)


def _to_naive_timestamp_series(
    ts: pd.Series,
    *,
    errors: str = "coerce",
) -> pd.Series:
    """
    Ensure a pandas Series is datetime64[ns] and tz-naive.

    - Parses with errors="coerce" by default
    - Drops timezone information if present
    """
    s = pd.to_datetime(ts, errors=errors)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        # Already tz-naive or not datetime-like; ignore
        pass
    return s


def days_with_continuous_coverage(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
    day_edge_tolerance: pd.Timedelta = pd.Timedelta(minutes=2),
    total_missing_allowance: pd.Timedelta = pd.Timedelta(minutes=0),
) -> List[pd.Timestamp]:
    """
    Return list of days (normalized to midnight) where the timeseries has continuous
    24-hour coverage with no gap larger than `max_gap` and cumulative missing time
    across the day not exceeding `total_missing_allowance`.

    A day qualifies if:
    - First sample is no later than (day_start + day_edge_tolerance)
    - Last sample is no earlier than (day_end - day_edge_tolerance)
    - Cumulative missing time (gaps beyond `max_gap` plus unmet edges beyond
      `day_edge_tolerance`) is <= `total_missing_allowance`

    Args:
        df: DataFrame containing a timestamp column
        timestamp_col: Name of the timestamp column
        max_gap: Maximum allowed gap between consecutive samples
        day_edge_tolerance: Allowed tolerance at the day's edges
        total_missing_allowance: Total allowed missing time within the day (default 0)

    Returns:
        List of tz-naive pandas Timestamps (midnight) representing qualifying days
    """
    if df is None or df.empty or timestamp_col not in df.columns:
        logger.warning("days_with_continuous_coverage received empty df or missing column '%s'", timestamp_col)
        return []

    ts = _to_naive_timestamp_series(df[timestamp_col])
    ts = ts.dropna().sort_values()
    if ts.empty:
        return []

    # Assign day bucket
    day = pd.to_datetime(ts.dt.date)
    grouped = ts.groupby(day)

    qualifying_days: List[pd.Timestamp] = []
    one_day = pd.Timedelta(days=1)

    for current_day, stamps in grouped:
        if stamps.empty:
            continue
        # Ensure sorted
        stamps = stamps.sort_values()
        start_ts = stamps.iloc[0]
        end_ts = stamps.iloc[-1]
        day_start = current_day
        day_end = day_start + one_day

        # Compute cumulative missing time:
        # - Internal missing: sum of (gap - max_gap) for gaps larger than max_gap
        # - Edge missing: time outside tolerated edges at start/end
        if len(stamps) == 1:
            diffs = pd.Series([], dtype="timedelta64[ns]")
            max_consecutive_gap = pd.NaT
        else:
            diffs = stamps.diff().iloc[1:]
            max_consecutive_gap = diffs.max()

        internal_missing = diffs.apply(lambda d: max(pd.Timedelta(0), d - max_gap)).sum() if len(diffs) > 0 else pd.Timedelta(0)

        start_deficit = max(pd.Timedelta(0), (start_ts - (day_start + day_edge_tolerance)))
        end_deficit = max(pd.Timedelta(0), ((day_end - day_edge_tolerance) - end_ts))
        edge_missing = start_deficit + end_deficit

        total_missing = internal_missing + edge_missing

        allowance_ok = (total_missing <= total_missing_allowance)

        # Edge tolerance must still be respected within the allowance budget
        # i.e., we do not require exact edges as long as total_missing stays within allowance
        if allowance_ok:
            qualifying_days.append(day_start)

    return qualifying_days


def filter_master_by_days(
    master_df: pd.DataFrame,
    qualifying_days: Sequence[pd.Timestamp],
    *,
    day_col: str = "day",
) -> pd.DataFrame:
    """
    Filter a master daily dataframe to only rows whose `day_col` is in qualifying_days.

    Args:
        master_df: Daily dataframe containing a date column
        qualifying_days: Sequence of tz-naive midnight Timestamps
        day_col: Name of the day column in master_df

    Returns:
        Filtered dataframe (copy)
    """
    if master_df is None or master_df.empty or day_col not in master_df.columns:
        logger.warning("filter_master_by_days received invalid inputs (empty df or missing '%s')", day_col)
        return master_df

    days_set: Set[pd.Timestamp] = set(pd.to_datetime(qualifying_days))
    df = master_df.copy()
    df[day_col] = pd.to_datetime(df[day_col])
    return df[df[day_col].isin(days_set)].copy()


@handle_data_loading_errors(reraise=False)
def filter_by_24h_coverage(
    master_df: pd.DataFrame,
    *,
    day_col: str = "day",
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
    day_edge_tolerance: pd.Timedelta = pd.Timedelta(minutes=2),
    total_missing_allowance: pd.Timedelta = pd.Timedelta(minutes=0),
    stress_df: Optional[pd.DataFrame] = None,
    db_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter a master daily dataframe to only include days with 24-hour continuous coverage.
    
    This function uses stress timeseries data to determine which days have
    complete 24-hour coverage, then filters the master dataframe accordingly.

    Args:
        master_df: Daily dataframe containing a date column
        day_col: Name of the day column in master_df
        max_gap: Maximum allowed gap between consecutive samples in stress data
        day_edge_tolerance: Allowed tolerance at the day's edges
        total_missing_allowance: Total allowed missing time within day (default 0)
        stress_df: Optional pre-loaded stress DataFrame. If None, will load from database.
        db_path: Optional custom database path. If None, uses default DB_PATHS["garmin"].

    Returns:
        Filtered dataframe containing only days with 24-hour coverage
    
    Note:
        On error, returns original dataframe rather than raising exception.
    """
    if master_df is None or master_df.empty:
        logger.warning("filter_by_24h_coverage received empty dataframe")
        return master_df
    
    # Get stress timeseries data for coverage analysis
    if stress_df is not None:
        # Use pre-loaded DataFrame
        stress = stress_df
        logger.info("Using pre-loaded stress DataFrame for coverage analysis")
    else:
        # Load from database
        from garmin_analysis.data_ingestion.load_all_garmin_dbs import load_table, DB_PATHS
        
        if db_path is not None:
            # Use custom database path
            stress = load_table(db_path, "stress", parse_dates=["timestamp"])
            logger.info(f"Loading stress data from custom path: {db_path}")
        else:
            # Use default database path
            stress = load_table(DB_PATHS["garmin"], "stress", parse_dates=["timestamp"])
            logger.info("Loading stress data from default database path")
    
    if stress is None or stress.empty:
        logger.warning("No stress timeseries data available for coverage analysis")
        return master_df
    
    # Get days with continuous coverage
    qualifying_days = days_with_continuous_coverage(
        stress, 
        timestamp_col="timestamp",
        max_gap=max_gap,
        day_edge_tolerance=day_edge_tolerance,
        total_missing_allowance=total_missing_allowance,
    )
    
    if not qualifying_days:
        logger.warning("No days found with 24-hour continuous coverage")
        return master_df
    
    logger.info(f"Found {len(qualifying_days)} days with 24-hour continuous coverage")
    
    # Filter master dataframe
    filtered_df = filter_master_by_days(master_df, qualifying_days, day_col=day_col)
    
    logger.info(f"Filtered dataset from {len(master_df)} to {len(filtered_df)} days")
    return filtered_df



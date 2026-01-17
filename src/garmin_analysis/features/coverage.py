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


def calculate_daily_coverage_metrics(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    day_col: str = "day",
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
    day_edge_tolerance: pd.Timedelta = pd.Timedelta(minutes=2),
) -> pd.DataFrame:
    """
    Calculate daily coverage metrics for a timeseries DataFrame.
    
    For each day, calculates:
    - coverage_hours: Number of hours with data coverage
    - coverage_pct: Percentage of 24 hours covered
    - has_24h_coverage: Boolean indicating if day has 24-hour coverage
    - first_sample: First timestamp in the day
    - last_sample: Last timestamp in the day
    - gap_count: Number of gaps larger than max_gap
    - total_missing_minutes: Total minutes missing
    
    Args:
        df: DataFrame containing timestamp column
        timestamp_col: Name of the timestamp column
        day_col: Name of the day column (will be created if not present)
        max_gap: Maximum allowed gap between consecutive samples
        day_edge_tolerance: Allowed tolerance at the day's edges
        
    Returns:
        DataFrame with one row per day containing coverage metrics
    """
    if df is None or df.empty or timestamp_col not in df.columns:
        logger.warning("calculate_daily_coverage_metrics received empty df or missing column '%s'", timestamp_col)
        return pd.DataFrame()
    
    ts = _to_naive_timestamp_series(df[timestamp_col])
    ts = ts.dropna().sort_values()
    if ts.empty:
        return pd.DataFrame()
    
    # Assign day bucket
    day = pd.to_datetime(ts.dt.date)
    grouped = ts.groupby(day)
    
    results = []
    one_day = pd.Timedelta(days=1)
    
    for current_day, stamps in grouped:
        if stamps.empty:
            continue
        
        stamps = stamps.sort_values()
        start_ts = stamps.iloc[0]
        end_ts = stamps.iloc[-1]
        day_start = current_day
        day_end = day_start + one_day
        
        # Calculate coverage hours
        coverage_duration = end_ts - start_ts
        coverage_hours = coverage_duration.total_seconds() / 3600.0
        
        # Calculate percentage of 24 hours covered
        coverage_pct = min(100.0, (coverage_hours / 24.0) * 100.0)
        
        # Check for gaps
        if len(stamps) == 1:
            diffs = pd.Series([], dtype="timedelta64[ns]")
            gap_count = 0
            total_missing_minutes = 0
        else:
            diffs = stamps.diff().iloc[1:]
            gap_count = (diffs > max_gap).sum()
            total_missing_minutes = diffs.apply(
                lambda d: max(0, (d - max_gap).total_seconds() / 60.0)
            ).sum()
        
        # Calculate edge deficits
        start_deficit = max(pd.Timedelta(0), (start_ts - (day_start + day_edge_tolerance)))
        end_deficit = max(pd.Timedelta(0), ((day_end - day_edge_tolerance) - end_ts))
        edge_missing_minutes = (start_deficit.total_seconds() + end_deficit.total_seconds()) / 60.0
        
        total_missing_minutes += edge_missing_minutes
        
        # Determine if has 24-hour coverage (strict)
        has_24h_coverage = (
            start_deficit == pd.Timedelta(0) and
            end_deficit == pd.Timedelta(0) and
            gap_count == 0
        )
        
        results.append({
            day_col: current_day,
            "coverage_hours": coverage_hours,
            "coverage_pct": coverage_pct,
            "has_24h_coverage": has_24h_coverage,
            "first_sample": start_ts,
            "last_sample": end_ts,
            "gap_count": gap_count,
            "total_missing_minutes": total_missing_minutes,
        })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


@handle_data_loading_errors(reraise=False)
def filter_by_24h_coverage(
    master_df: pd.DataFrame,
    *,
    day_col: str = "day",
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
    day_edge_tolerance: pd.Timedelta = pd.Timedelta(minutes=2),
    total_missing_allowance: pd.Timedelta = pd.Timedelta(minutes=0),
    hr_df: Optional[pd.DataFrame] = None,
    stress_df: Optional[pd.DataFrame] = None,
    db_path: Optional[str] = None,
    use_monitoring_hr: bool = True,
) -> pd.DataFrame:
    """
    Filter a master daily dataframe to only include days with 24-hour continuous coverage.
    
    This function uses monitoring_hr (heart rate) timeseries data EXCLUSIVELY to determine which days have
    complete 24-hour coverage. Heart rate data requires continuous skin contact, making it the definitive
    indicator that the watch is being worn and powered on. Only valid HR readings (20-250 bpm, non-null)
    are used to determine watch wear time.

    Args:
        master_df: Daily dataframe containing a date column
        day_col: Name of the day column in master_df
        max_gap: Maximum allowed gap between consecutive samples
        day_edge_tolerance: Allowed tolerance at the day's edges
        total_missing_allowance: Total allowed missing time within day (default 0)
        hr_df: Optional pre-loaded monitoring_hr DataFrame. If None, will load from database.
        stress_df: DEPRECATED - no longer used. Heart rate data is the exclusive indicator of watch wear.
        db_path: Optional custom database path. If None, uses default DB_PATHS.
        use_monitoring_hr: If True (default), use monitoring_hr for coverage analysis. If False, returns unfiltered data.

    Returns:
        Filtered dataframe containing only days with 24-hour coverage based on HR data
    
    Note:
        On error or if HR data unavailable, returns original dataframe rather than raising exception.
        Without HR data, we cannot reliably determine watch wear time.
    """
    if master_df is None or master_df.empty:
        logger.warning("filter_by_24h_coverage received empty dataframe")
        return master_df
    
    # Use monitoring_hr (heart rate) EXCLUSIVELY as the indicator of watch wear
    # HR data requires continuous skin contact, making it the definitive indicator that watch is worn/on
    timeseries_df = None
    data_source = None
    
    if use_monitoring_hr:
        if hr_df is not None:
            timeseries_df = hr_df
            data_source = "monitoring_hr (pre-loaded)"
        else:
            from garmin_analysis.data_ingestion.load_all_garmin_dbs import load_table
            from garmin_analysis.config import DB_PATHS
            
            try:
                if db_path is not None:
                    # If db_path is provided, assume it's for monitoring DB or construct path
                    from pathlib import Path
                    db_path_obj = Path(db_path)
                    if "monitoring" in str(db_path_obj):
                        monitoring_path = db_path_obj
                    else:
                        # Construct monitoring DB path from garmin DB path
                        monitoring_path = db_path_obj.parent / "garmin_monitoring.db"
                    timeseries_df = load_table(monitoring_path, "monitoring_hr", parse_dates=["timestamp"])
                    data_source = f"monitoring_hr from {monitoring_path}"
                else:
                    timeseries_df = load_table(DB_PATHS["monitoring"], "monitoring_hr", parse_dates=["timestamp"])
                    data_source = "monitoring_hr from default database"
                
                # Filter for valid heart rate readings (20-250 bpm, non-null)
                if timeseries_df is not None and not timeseries_df.empty and "heart_rate" in timeseries_df.columns:
                    original_rows = len(timeseries_df)
                    timeseries_df = timeseries_df[
                        (timeseries_df["heart_rate"].notna()) &
                        (timeseries_df["heart_rate"] > 0) &
                        (timeseries_df["heart_rate"] >= 20) &
                        (timeseries_df["heart_rate"] <= 250)
                    ].copy()
                    filtered_rows = len(timeseries_df)
                    if filtered_rows < original_rows:
                        logger.info(f"Filtered monitoring_hr: {original_rows} → {filtered_rows} rows with valid HR values (20-250 bpm)")
                
                if timeseries_df is not None and not timeseries_df.empty:
                    logger.info(f"Using {data_source} for coverage analysis")
            except Exception as e:
                logger.warning(f"Could not load monitoring_hr: {e}")
                timeseries_df = None
    
    # HR data is required - without it we cannot determine watch wear
    if timeseries_df is None or timeseries_df.empty:
        logger.warning("No monitoring_hr (heart rate) data available - cannot determine watch wear without HR data")
        logger.warning("Heart rate data is required to determine if watch is being worn (requires skin contact)")
        return master_df
    
    # Get days with continuous coverage
    qualifying_days = days_with_continuous_coverage(
        timeseries_df, 
        timestamp_col="timestamp",
        max_gap=max_gap,
        day_edge_tolerance=day_edge_tolerance,
        total_missing_allowance=total_missing_allowance,
    )
    
    if not qualifying_days:
        logger.warning("No days found with 24-hour continuous coverage")
        return master_df
    
    logger.info(f"Found {len(qualifying_days)} days with 24-hour continuous coverage (using {data_source})")
    
    # Filter master dataframe
    filtered_df = filter_master_by_days(master_df, qualifying_days, day_col=day_col)
    
    logger.info(f"Filtered dataset from {len(master_df)} to {len(filtered_df)} days")
    return filtered_df



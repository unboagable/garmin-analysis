import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler



logger = logging.getLogger(__name__)
def strip_time_from_dates(df, col="day"):
    """
    Strip time component from datetime column, keeping only the date.
    
    This function normalizes datetime values to midnight (00:00:00),
    effectively removing the time component while preserving the date.
    Useful for ensuring date-only comparisons and grouping.
    
    Args:
        df (pd.DataFrame): DataFrame containing datetime column
        col (str): Name of the datetime column to normalize (default: "day")
    
    Returns:
        pd.DataFrame: DataFrame with normalized datetime column
    
    Example:
        >>> df['day'] = ['2024-01-01 14:30:00', '2024-01-02 08:15:00']
        >>> df = strip_time_from_dates(df, col='day')
        >>> # Result: ['2024-01-01 00:00:00', '2024-01-02 00:00:00']
    
    Note:
        This is different from normalize_day_column() in data_processing.py:
        - normalize_day_column(): Detects and converts day/calendarDate/timestamp
          columns during initial data loading (more flexible, multiple column names)
        - strip_time_from_dates(): Strips time from a specific known column
          for consistent date-only operations (simpler, targeted use)
    
    See Also:
        - garmin_analysis.utils.data_processing.normalize_day_column(): For initial data loading
    """
    # Handle empty DataFrame
    if df.empty or col not in df.columns:
        return df
    
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df


# Backward compatibility alias (deprecated)
def normalize_dates(df, col="day"):
    """
    Deprecated: Use strip_time_from_dates() instead.
    
    This function is maintained for backward compatibility but will be
    removed in a future version. Please update your code to use
    strip_time_from_dates() which has the same functionality but a
    clearer name.
    """
    import warnings
    warnings.warn(
        "normalize_dates() is deprecated. Use strip_time_from_dates() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return strip_time_from_dates(df, col)


def filter_by_date(df, date_col="day", from_date=None, to_date=None, days_back=None, weeks_back=None, months_back=None):
    if df.empty:
        logger.warning("Received empty DataFrame to filter on column '%s'.", date_col)
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

    logger.info("Filtered data on column '%s' between %s and %s — %d rows returned.",
                 date_col, from_date, to_date, len(df))
    return df


def convert_time_columns(df, columns):
    """
    Convert time columns from string format to minutes.
    
    Uses the centralized convert_time_to_minutes function which handles:
    - "HH:MM:SS" format (e.g., "1:30:00" -> 90 minutes)
    - "MM:SS" format (e.g., "45:30" -> 45.5 minutes)
    - Direct numeric strings
    
    Args:
        df (pd.DataFrame): DataFrame with time columns
        columns (list): List of column names to convert
    
    Returns:
        pd.DataFrame: DataFrame with converted columns
    
    Example:
        >>> df = convert_time_columns(df, ['total_sleep', 'deep_sleep'])
    """
    from garmin_analysis.utils.data_processing import convert_time_to_minutes

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        # Accept both object and string dtypes (pandas 2.x uses StringDtype for strings)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            converted = df[col].apply(convert_time_to_minutes)
            df[col] = pd.to_numeric(converted, errors="coerce")
    return df


def standardize_features(df, columns):
    df = df.copy()
    df = df.dropna(subset=columns)
    if df.empty:
        logger.warning("No data left after dropping NaNs in columns: %s", columns)
        return np.array([])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    logger.info("Standardized features: %s", columns)
    return scaled


def filter_required_columns(df, required_cols):
    missing_req = [col for col in required_cols if col not in df.columns]
    if missing_req:
        logger.warning(f"Missing required columns for analysis: {missing_req}")
        return df
    before = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"Filtered rows missing {required_cols}. Kept {len(df)} of {before} rows.")
    return df

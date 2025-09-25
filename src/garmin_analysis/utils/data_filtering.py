import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler


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


def convert_time_columns(df, columns):
    def time_to_minutes(val):
        try:
            h, m, s = map(int, val.split(":"))
            return h * 60 + m + s / 60
        except (ValueError, AttributeError, TypeError):
            return np.nan

    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].apply(time_to_minutes)
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

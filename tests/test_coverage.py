import pandas as pd
from garmin_analysis.features.coverage import days_with_continuous_coverage


def test_days_with_continuous_coverage_full_day_minutely():
    # Generate 1-minute samples for a full day
    day = pd.Timestamp("2024-01-01")
    ts = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
    df = pd.DataFrame({"timestamp": ts, "value": range(len(ts))})

    days = days_with_continuous_coverage(df, timestamp_col="timestamp", max_gap=pd.Timedelta(minutes=2))
    assert pd.Timestamp("2024-01-01") in days


def test_days_with_continuous_coverage_with_gap():
    day = pd.Timestamp("2024-01-02")
    # Create 1-minute series but remove a 10-minute chunk in the middle
    ts = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
    mask = ~((ts >= day + pd.Timedelta(hours=12)) & (ts < day + pd.Timedelta(hours=12, minutes=10)))
    ts_gap = ts[mask]
    df = pd.DataFrame({"timestamp": ts_gap})

    days = days_with_continuous_coverage(df, timestamp_col="timestamp", max_gap=pd.Timedelta(minutes=2))
    assert pd.Timestamp("2024-01-02") not in days



import pandas as pd
from garmin_analysis.features.coverage import days_with_continuous_coverage


class TestContinuousCoverage:

    def test_full_day_minutely(self):
        day = pd.Timestamp("2024-01-01")
        ts = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
        df = pd.DataFrame({"timestamp": ts, "value": range(len(ts))})

        days = days_with_continuous_coverage(df, timestamp_col="timestamp", max_gap=pd.Timedelta(minutes=2))
        assert pd.Timestamp("2024-01-01") in days

    def test_with_gap(self):
        day = pd.Timestamp("2024-01-02")
        ts = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
        mask = ~((ts >= day + pd.Timedelta(hours=12)) & (ts < day + pd.Timedelta(hours=12, minutes=10)))
        ts_gap = ts[mask]
        df = pd.DataFrame({"timestamp": ts_gap})

        days = days_with_continuous_coverage(df, timestamp_col="timestamp", max_gap=pd.Timedelta(minutes=2))
        assert pd.Timestamp("2024-01-02") not in days

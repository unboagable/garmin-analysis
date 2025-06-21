import pandas as pd
import pytest
from unittest.mock import patch
from src.features.merge_daily_data import merge_garmin_data

@pytest.fixture
def mock_garmin_tables():
    return {
        "daily": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "steps": [1000, 2000],
            "calories_total": [2100, 2200],
        }),
        "sleep": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "total_sleep": ["7:00:00", "6:30:00"],
            "deep_sleep": ["1:30:00", "1:00:00"],
            "rem_sleep": ["2:00:00", "1:45:00"],
            "score": [80, 70],
        }),
        "stress": pd.DataFrame({
            "timestamp": ["2024-01-01 08:00:00", "2024-01-01 08:01:00"],
            "stress": [20, 30]
        }),
        "rest_hr": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "resting_heart_rate": [60, 58],
        })
    }

def test_merge_garmin_data_output_columns(mock_garmin_tables):
    with patch("src.features.merge_daily_data.load_garmin_tables", return_value=mock_garmin_tables):
        df = merge_garmin_data()
        assert isinstance(df, pd.DataFrame)
        assert "steps" in df.columns
        assert "calories_total" in df.columns
        assert "total_sleep_min" in df.columns
        assert "stress_avg" in df.columns
        assert "resting_heart_rate" in df.columns

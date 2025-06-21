import pandas as pd
import pytest
from unittest.mock import patch
from src.data_ingestion import merge_all_daily_data

@pytest.fixture
def mock_tables():
    activities = pd.DataFrame({
        "activity_id": ["a1", "a2"],
        "start_time": ["2024-01-01 07:00:00", "2024-01-02 08:00:00"],
        "training_effect": [2.5, 2.8],
        "anaerobic_training_effect": [0.5, 0.7],
        "distance": [5.0, 6.2],
        "calories": [300, 350]
    })

    steps_activities = pd.DataFrame({
        "activity_id": ["a1", "a2"],
        "avg_pace": ["6:00", "5:45"],
        "vo2_max": [42, 44]
    })

    return {
        "daily_summary": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "steps": [5000, 6000],
            "calories_total": [2000, 2100]
        }),
        "sleep": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "total_sleep": ["7:00:00", "6:30:00"],
            "deep_sleep": ["1:30:00", "1:00:00"],
            "rem_sleep": ["2:00:00", "1:45:00"],
            "score": [80, 75]
        }),
        "stress": pd.DataFrame({
            "timestamp": ["2024-01-01 08:00:00", "2024-01-02 08:01:00"],
            "stress": [25, 35]
        }),
        "resting_hr": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "resting_heart_rate": [60, 62]
        }),
        "days_summary": pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "calories_bmr_avg": [1400, 1380]
        }),
        "activities": activities,
        "steps_activities": steps_activities
    }

def test_merge_outputs_expected_columns(mock_tables):
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "steps", "calories_total", "score", "resting_heart_rate",
        "training_effect", "anaerobic_training_effect", "calories_bmr_avg", "vo2_max"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"
    assert len(df) == 2

def test_merge_is_stable_with_partial_data(mock_tables):
    del mock_tables["steps_activities"]  # simulate missing optional source
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    assert len(df) == 2
    assert "training_effect" in df.columns

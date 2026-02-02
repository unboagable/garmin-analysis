import pandas as pd
import numpy as np
import pytest
from garmin_analysis.data_ingestion.load_all_garmin_dbs import (
    summarize_and_merge,
    preprocess_sleep,
    convert_time_to_minutes,
    aggregate_stress,
)

# --------------------------
# High-Level Functional (Integration) Tests
# --------------------------

@pytest.mark.integration
def test_summarize_and_merge_output_non_empty(tmp_db):
    df = summarize_and_merge(return_df=True)
    assert not df.empty, "The merged DataFrame should not be empty"


@pytest.mark.integration
def test_summarize_and_merge_key_columns_present(tmp_db):
    df = summarize_and_merge(return_df=True)
    expected_columns = [
        "yesterday_activity_minutes",
        "stress_avg",
    ]
    for col in expected_columns:
        assert col in df.columns, f"{col} should be in the output"


@pytest.mark.integration
def test_summarize_and_merge_lag_feature_shift(tmp_db):
    df = summarize_and_merge(return_df=True)
    if "activity_minutes" in df.columns and "yesterday_activity_minutes" in df.columns:
        actual = df["yesterday_activity_minutes"]
        expected = df["activity_minutes"].shift(1)
        mask = ~(actual.isna() | expected.isna())
        assert np.allclose(actual[mask], expected[mask], atol=1e-2), \
            "Lag feature for activity_minutes is not correctly shifted"


# --------------------------
# Unit Tests with Mock Data
# --------------------------


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


def test_preprocess_sleep_converts_time():
    sleep = pd.DataFrame({
        "day": ["2024-01-01"],
        "total_sleep": ["7:00:00"],
        "deep_sleep": ["1:00:00"],
        "rem_sleep": ["2:00:00"],
        "score": [85]
    })
    result = preprocess_sleep(sleep.copy())
    assert "total_sleep_min" in result.columns
    assert "deep_sleep_min" in result.columns
    assert result["total_sleep_min"].iloc[0] == 420


def test_convert_time_to_minutes():
    assert convert_time_to_minutes("1:30:00") == 90
    assert convert_time_to_minutes("45:00") == 45
    assert pd.isna(convert_time_to_minutes("bad"))


def test_aggregate_stress_output():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 12:00:00"],
        "stress": [10, 20]
    })
    result = aggregate_stress(df)
    assert "stress_avg" in result.columns
    assert result["stress_avg"].iloc[0] == 15


@pytest.mark.integration
def test_summarize_and_merge_column_dtypes(tmp_db):
    df = summarize_and_merge(return_df=True)
    assert pd.api.types.is_datetime64_any_dtype(df["day"])
    assert pd.api.types.is_numeric_dtype(df["steps"])
    if "score" in df.columns:
        assert pd.api.types.is_numeric_dtype(df["score"])
    if "missing_score" in df.columns:
        assert pd.api.types.is_bool_dtype(df["missing_score"])


@pytest.mark.integration
def test_summarize_and_merge_dates_valid(tmp_db):
    df = summarize_and_merge(return_df=True)
    assert df["day"].between("2010-01-01", "2100-01-01").all(), "Dates should be valid and normalized"


@pytest.mark.integration
def test_summarize_and_merge_columns_exist(tmp_db):
    df = summarize_and_merge(return_df=True)
    expected_cols = {
        "day", "steps", "calories_total", "score", "resting_heart_rate",
        "training_effect", "anaerobic_te", "steps_avg_7d",
        "missing_score", "missing_training_effect"
    }
    missing = expected_cols.difference(df.columns)
    assert not missing, f"Missing expected columns: {missing}"
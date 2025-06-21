import pandas as pd
import pytest
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

def test_merge_handles_empty_sleep_data(mock_tables):
    mock_tables["sleep"] = pd.DataFrame()
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    assert "score" not in df.columns or df["score"].isna().all()

def test_preprocess_sleep_works():
    sleep = pd.DataFrame({
        "day": ["2024-01-01"],
        "total_sleep": ["7:00:00"],
        "deep_sleep": ["1:00:00"],
        "rem_sleep": ["2:00:00"],
        "score": [85]
    })
    result = merge_all_daily_data.preprocess_sleep(sleep.copy())
    assert "total_sleep_min" in result.columns
    assert "deep_sleep_min" in result.columns
    assert result["total_sleep_min"].iloc[0] == 420

def test_merge_activity_stats_merges_by_day(mock_tables):
    base_df = mock_tables["daily_summary"].copy()
    result = merge_all_daily_data.merge_activity_stats(base_df, mock_tables["activities"])
    assert "training_effect" in result.columns
    assert len(result) == 2

def test_merge_step_stats_merges_by_day(mock_tables):
    base_df = mock_tables["daily_summary"].copy()
    result = merge_all_daily_data.merge_step_stats(base_df, mock_tables["steps_activities"], mock_tables["activities"])
    assert "vo2_max" in result.columns

def test_no_keyerror_on_optional_fields(mock_tables):
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    try:
        _ = df["training_effect"]
        _ = df["score"]
        _ = df["steps_avg_7d"]
    except KeyError as e:
        pytest.fail(f"Unexpected KeyError on optional column: {e}")

def test_all_merged_dates_are_valid(mock_tables):
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    assert pd.api.types.is_datetime64_any_dtype(df["day"]), "'day' column must be datetime"
    assert df["day"].between("2024-01-01", "2024-12-31").all(), "dates must fall within 2024"

def test_merged_shape_and_columns(mock_tables):
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    expected_cols = {
        "day", "steps", "calories_total", "score", "resting_heart_rate",
        "training_effect", "anaerobic_training_effect", "calories_bmr_avg",
        "steps_avg_7d", "missing_score", "missing_training_effect", "vo2_max"
    }
    assert set(expected_cols).issubset(df.columns), "Missing expected columns"
    assert df.shape[0] == 2, "Expected 2 rows based on mock input"

def test_column_dtypes_are_correct(mock_tables):
    df = merge_all_daily_data.clean_and_merge(mock_tables)
    assert pd.api.types.is_datetime64_any_dtype(df["day"])
    assert pd.api.types.is_numeric_dtype(df["steps"])
    assert pd.api.types.is_numeric_dtype(df["score"])
    assert pd.api.types.is_bool_dtype(df["missing_score"])

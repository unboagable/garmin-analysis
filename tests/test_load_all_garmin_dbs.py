import pandas as pd
from src.data_ingestion.load_all_garmin_dbs import summarize_and_merge

def test_merge_output_non_empty():
    df = summarize_and_merge(return_df=True)
    assert not df.empty, "The merged DataFrame should not be empty"

def test_key_columns_present():
    df = summarize_and_merge(return_df=True)
    expected_columns = [
        "yesterday_activity_minutes",
        "stress_avg",
        "pulse_ox_avg",
    ]
    for col in expected_columns:
        assert col in df.columns, f"{col} should be in the output"

def test_lag_feature_shift_logic():
    df = summarize_and_merge(return_df=True)
    if "activity_minutes" in df.columns and "yesterday_activity_minutes" in df.columns:
        shifted = df["activity_minutes"].shift(1)
        match = df["yesterday_activity_minutes"].fillna(0).round(3).equals(shifted.fillna(0).round(3))
        assert match, "Lag feature for activity_minutes is not correctly shifted"

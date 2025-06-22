import pandas as pd
import os
from pathlib import Path
from src.data_ingestion.prepare_modeling_dataset import prepare_modeling_dataset

TEST_INPUT_PATH = "tests/mock_master_daily_summary.csv"
TEST_OUTPUT_PATH = "tests/mock_modeling_ready_dataset.csv"

def setup_module(module):
    """Create mock input data before tests."""
    df = pd.DataFrame({
        "day": pd.date_range(start="2024-01-01", periods=5),
        "score": [85, None, 90, 75, 88],
        "stress_avg": [30, 40, None, 50, 45],
        "yesterday_activity_minutes": [60, 45, 30, None, 50],
        "some_useless_flag": [True, False, True, False, True],
        "missing_score": [False, True, False, False, False],
        "Unnamed: 0": [0, 1, 2, 3, 4],
        "mostly_nan": [1, None, None, None, None],
    })
    df.to_csv(TEST_INPUT_PATH, index=False)

def teardown_module(module):
    """Clean up test files after tests."""
    Path(TEST_INPUT_PATH).unlink(missing_ok=True)
    Path(TEST_OUTPUT_PATH).unlink(missing_ok=True)

def test_prepare_modeling_dataset_creates_output_file():
    prepare_modeling_dataset(
        input_path=TEST_INPUT_PATH,
        output_path=TEST_OUTPUT_PATH,
        missing_threshold=0.5
    )
    assert Path(TEST_OUTPUT_PATH).exists()

def test_required_feature_filtering():
    prepare_modeling_dataset(
        input_path=TEST_INPUT_PATH,
        output_path=TEST_OUTPUT_PATH,
        required_features=["score", "stress_avg", "yesterday_activity_minutes"]
    )
    df = pd.read_csv(TEST_OUTPUT_PATH)
    assert not df[["score", "stress_avg", "yesterday_activity_minutes"]].isnull().any().any()

def test_metadata_columns_dropped():
    prepare_modeling_dataset(input_path=TEST_INPUT_PATH, output_path=TEST_OUTPUT_PATH)
    df = pd.read_csv(TEST_OUTPUT_PATH)
    assert not any("missing_" in col or "Unnamed" in col for col in df.columns)

def test_columns_with_high_missingness_dropped():
    prepare_modeling_dataset(
        input_path=TEST_INPUT_PATH,
        output_path=TEST_OUTPUT_PATH,
        missing_threshold=0.5
    )
    df = pd.read_csv(TEST_OUTPUT_PATH)
    assert "mostly_nan" not in df.columns

"""Tests for daily data quality score computation."""

import pytest
import pandas as pd
from pathlib import Path

from garmin_analysis.features.daily_data_quality import (
    compute_daily_data_quality_score,
    persist_daily_data_quality,
    load_daily_data_quality,
    compute_and_persist_daily_data_quality,
    KEY_METRIC_COLUMNS,
)


class TestComputeDailyDataQualityScore:
    """Tests for compute_daily_data_quality_score."""

    def test_empty_df(self):
        result = compute_daily_data_quality_score(pd.DataFrame())
        assert result.empty

    def test_none_df(self):
        result = compute_daily_data_quality_score(None)
        assert result.empty

    def test_df_without_day_column(self):
        df = pd.DataFrame({"steps": [1000], "score": [70]})
        result = compute_daily_data_quality_score(df)
        assert result.empty

    def test_minimal_df_with_coverage_pct(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=5, freq="D"),
            "steps": [1000, 2000, 3000, 4000, 5000],
            "score": [70, 75, 80, 85, 90],
            "coverage_pct": [95, 98, 100, 99, 97],
        })
        result = compute_daily_data_quality_score(df)
        assert len(result) == 5
        assert "data_quality_score" in result.columns
        assert "coverage_score" in result.columns
        assert "completeness_score" in result.columns
        assert "key_metrics_count" in result.columns
        assert "key_metrics_total" in result.columns
        assert result["data_quality_score"].min() >= 0
        assert result["data_quality_score"].max() <= 100
        assert (result["coverage_score"] == df["coverage_pct"]).all()

    def test_has_24h_coverage_boolean(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, 2000, 3000],
            "has_24h_coverage": [True, False, True],
        })
        result = compute_daily_data_quality_score(df)
        assert len(result) == 3
        assert result["coverage_score"].iloc[0] == 100.0
        assert result["coverage_score"].iloc[1] == 0.0
        assert result["coverage_score"].iloc[2] == 100.0

    def test_no_coverage_columns_uses_neutral_default(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=2, freq="D"),
            "steps": [1000, 2000],
        })
        result = compute_daily_data_quality_score(df)
        assert len(result) == 2
        assert (result["coverage_score"] == 50.0).all()

    def test_custom_weights(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=2, freq="D"),
            "steps": [1000, 2000],
            "coverage_pct": [100, 0],
        })
        result = compute_daily_data_quality_score(
            df, coverage_weight=0.8, completeness_weight=0.2
        )
        assert len(result) == 2
        assert result["data_quality_score"].iloc[0] > result["data_quality_score"].iloc[1]

    def test_mixed_nulls_in_key_metrics(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, pd.NA, 3000],
            "score": [70, 75, pd.NA],
            "coverage_pct": [100, 100, 100],
        })
        result = compute_daily_data_quality_score(df)
        assert len(result) == 3
        assert result["completeness_score"].iloc[0] == 100.0
        assert result["completeness_score"].iloc[1] == 50.0
        assert result["completeness_score"].iloc[2] == 50.0

    def test_single_row(self):
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-01")],
            "steps": [1000],
            "score": [80],
            "coverage_pct": [95],
        })
        result = compute_daily_data_quality_score(df)
        assert len(result) == 1
        assert result["data_quality_score"].iloc[0] > 0

    def test_all_key_metrics_null(self):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=2, freq="D"),
            "steps": [pd.NA, pd.NA],
            "score": [pd.NA, pd.NA],
            "coverage_pct": [80, 90],
        })
        result = compute_daily_data_quality_score(df)
        assert (result["completeness_score"] == 0).all()
        assert (result["key_metrics_count"] == 0).all()


class TestPersistAndLoad:
    """Tests for persist and load."""

    def test_persist_and_load(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "data_quality_score": [80, 85, 90],
            "coverage_score": [95, 98, 100],
            "completeness_score": [65, 72, 80],
            "key_metrics_count": [5, 6, 7],
            "key_metrics_total": [8, 8, 8],
        })
        path = persist_daily_data_quality(df, output_path=tmp_path / "dq.csv")
        assert path.exists()
        loaded = load_daily_data_quality(path)
        assert len(loaded) == 3
        assert "data_quality_score" in loaded.columns
        assert loaded["data_quality_score"].tolist() == [80.0, 85.0, 90.0]

    def test_load_nonexistent_returns_empty(self, tmp_path):
        result = load_daily_data_quality(tmp_path / "nonexistent.csv")
        assert result.empty

    def test_persist_creates_parent_dir(self, tmp_path):
        out_path = tmp_path / "sub" / "dir" / "dq.csv"
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-01")],
            "data_quality_score": [80],
            "coverage_score": [95],
            "completeness_score": [50],
            "key_metrics_count": [4],
            "key_metrics_total": [8],
        })
        path = persist_daily_data_quality(df, output_path=out_path)
        assert path.parent.exists()
        assert path.exists()


class TestComputeAndPersist:
    """Tests for compute_and_persist_daily_data_quality."""

    def test_with_provided_master_df(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, 2000, 3000],
            "score": [70, 75, 80],
            "coverage_pct": [95, 98, 100],
        })
        result = compute_and_persist_daily_data_quality(
            master_df=df, output_path=tmp_path / "dq.csv"
        )
        assert len(result) == 3
        assert (tmp_path / "dq.csv").exists()
        assert result["data_quality_score"].min() >= 0

    def test_with_empty_master_does_not_persist(self, tmp_path):
        result = compute_and_persist_daily_data_quality(
            master_df=pd.DataFrame(), output_path=tmp_path / "dq.csv"
        )
        assert result.empty
        assert not (tmp_path / "dq.csv").exists()


class TestKeyMetricColumns:
    """Tests for KEY_METRIC_COLUMNS constant."""

    def test_contains_expected_metrics(self):
        expected = {"steps", "score", "stress_avg", "calories_total"}
        assert expected.issubset(set(KEY_METRIC_COLUMNS))

    def test_no_duplicates(self):
        assert len(KEY_METRIC_COLUMNS) == len(set(KEY_METRIC_COLUMNS))

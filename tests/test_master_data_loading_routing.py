"""Tests that modules route master reads through load_master_dataframe()."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from garmin_analysis.data_ingestion.export_master import export_to_parquet
from garmin_analysis.data_ingestion.prepare_modeling_dataset import prepare_modeling_dataset
from garmin_analysis.utils.data_loading import load_master_dataframe


@pytest.fixture
def mock_master_df():
    return pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=5, freq="D"),
            "steps": [1000, 2000, 3000, 4000, 5000],
            "score": [85, 90, 88, None, 92],
            "stress_avg": [30, 35, 40, 45, 50],
            "yesterday_activity_minutes": [60, 45, 30, 50, 55],
            "has_24h_coverage": [True, False, None, True, False],
        }
    )


class TestExportToParquetLoaderRouting:
    """Unit tests: export_to_parquet delegates to load_master_dataframe when df is None."""

    @patch("garmin_analysis.data_ingestion.export_master.load_master_dataframe")
    def test_calls_load_master_dataframe_when_df_none(self, mock_load, mock_master_df, tmp_path):
        mock_load.return_value = mock_master_df
        out = tmp_path / "master.parquet"

        result = export_to_parquet(df=None, output_path=out, include_data_quality=False)

        mock_load.assert_called_once_with()
        assert result == out
        loaded = pd.read_parquet(out)
        assert len(loaded) == 5
        assert "steps" in loaded.columns

    @patch("garmin_analysis.data_ingestion.export_master.load_master_dataframe")
    def test_skips_loader_when_df_provided(self, mock_load, mock_master_df, tmp_path):
        out = tmp_path / "master.parquet"

        export_to_parquet(df=mock_master_df, output_path=out, include_data_quality=False)

        mock_load.assert_not_called()


class TestPrepareModelingDatasetLoaderRouting:
    """Unit tests: prepare_modeling_dataset delegates to load_master_dataframe by default."""

    @patch("garmin_analysis.data_ingestion.prepare_modeling_dataset.load_master_dataframe")
    def test_calls_load_master_dataframe_when_input_path_none(
        self, mock_load, mock_master_df, tmp_path
    ):
        mock_load.return_value = mock_master_df.copy()
        output_path = tmp_path / "modeling.csv"

        prepare_modeling_dataset(output_path=str(output_path))

        mock_load.assert_called_once_with()
        assert output_path.exists()

    @patch("garmin_analysis.data_ingestion.prepare_modeling_dataset.load_master_dataframe")
    def test_explicit_input_path_skips_loader(self, mock_load, tmp_path):
        input_path = tmp_path / "custom_master.csv"
        output_path = tmp_path / "modeling.csv"
        pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "score": [80, 85, 90],
                "stress_avg": [30, 35, 40],
                "yesterday_activity_minutes": [60, 45, 30],
            }
        ).to_csv(input_path, index=False)

        prepare_modeling_dataset(input_path=str(input_path), output_path=str(output_path))

        mock_load.assert_not_called()
        assert output_path.exists()


@pytest.mark.integration
class TestMasterLoaderRoutingIntegration:
    """Integration tests: mock master CSV on disk loaded via load_master_dataframe()."""

    def test_export_to_parquet_reads_mock_master_file(self, tmp_path, monkeypatch):
        master_df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=4, freq="D"),
                "steps": [1000, 2000, 3000, 4000],
                "score": [70, 75, 80, 85],
            }
        )
        master_path = tmp_path / "master_daily_summary.csv"
        master_df.to_csv(master_path, index=False)
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        out = tmp_path / "export" / "master.parquet"
        result = export_to_parquet(df=None, output_path=out, include_data_quality=False)

        assert result == out
        loaded = pd.read_parquet(out)
        assert len(loaded) == 4
        assert loaded["steps"].tolist() == [1000, 2000, 3000, 4000]

    def test_prepare_modeling_dataset_reads_mock_master_file(self, tmp_path, monkeypatch):
        master_df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=4, freq="D"),
                "score": [85, 90, 88, 92],
                "stress_avg": [30, 35, 40, 45],
                "yesterday_activity_minutes": [60, 45, 30, 50],
                "mostly_nan": [1, None, None, None],
            }
        )
        master_path = tmp_path / "master_daily_summary.csv"
        output_path = tmp_path / "modeling_ready.csv"
        master_df.to_csv(master_path, index=False)
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        prepare_modeling_dataset(output_path=str(output_path), missing_threshold=0.5)

        assert output_path.exists()
        loaded = pd.read_csv(output_path, parse_dates=["day"])
        assert len(loaded) == 4
        assert "mostly_nan" not in loaded.columns

    def test_load_master_dataframe_normalizes_boolean_column(self, tmp_path, monkeypatch):
        master_path = tmp_path / "master_daily_summary.csv"
        pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "steps": [1000, 2000, 3000],
                "has_24h_coverage": [True, False, None],
            }
        ).to_csv(master_path, index=False)
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        df = load_master_dataframe()

        assert str(df["has_24h_coverage"].dtype) == "boolean"
        assert df["has_24h_coverage"].tolist() == [True, False, pd.NA]

        out = tmp_path / "master.parquet"
        export_to_parquet(df=None, output_path=out, include_data_quality=False)
        loaded = pd.read_parquet(out)
        assert "has_24h_coverage" in loaded.columns

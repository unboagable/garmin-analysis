"""Tests for export master to Parquet and DuckDB."""

import json
from pathlib import Path

import pandas as pd
import pytest

from garmin_analysis.data_ingestion.export_master import (
    SUPPORTED_EXPORT_FORMATS,
    build_export_output_path,
    export_master,
    export_master_date_range,
    export_to_csv,
    export_to_duckdb,
    export_to_json,
    export_to_parquet,
    filter_master_by_date_range,
    normalize_export_format,
)


@pytest.fixture
def sample_master_df():
    return pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=10, freq="D"),
            "steps": range(1000, 11000, 1000),
            "score": range(70, 80),
        }
    )


class TestExportFormatHelpers:
    def test_supported_formats(self):
        assert "csv" in SUPPORTED_EXPORT_FORMATS
        assert "parquet" in SUPPORTED_EXPORT_FORMATS
        assert "json" in SUPPORTED_EXPORT_FORMATS
        assert "duckdb" in SUPPORTED_EXPORT_FORMATS

    def test_normalize_export_format_accepts_valid_values(self):
        assert normalize_export_format("CSV") == "csv"
        assert normalize_export_format(" parquet ") == "parquet"

    def test_normalize_export_format_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unsupported export format"):
            normalize_export_format("xlsx")

    def test_filter_master_by_date_range_inclusive(self, sample_master_df):
        filtered = filter_master_by_date_range(
            sample_master_df,
            "2024-01-03",
            "2024-01-05",
        )
        assert len(filtered) == 3
        assert filtered["day"].min() == pd.Timestamp("2024-01-03")
        assert filtered["day"].max() == pd.Timestamp("2024-01-05")

    def test_filter_master_by_date_range_rejects_inverted_dates(self, sample_master_df):
        with pytest.raises(ValueError, match="must be on or before"):
            filter_master_by_date_range(sample_master_df, "2024-02-01", "2024-01-01")

    def test_build_export_output_path_includes_date_range(self, tmp_path):
        path = build_export_output_path(
            "csv",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=tmp_path,
        )
        assert path == tmp_path / "master_daily_summary_2024-01-01_2024-01-31.csv"


class TestExportToParquet:
    """Tests for export_to_parquet."""

    def test_export_from_df(self, tmp_path):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=5, freq="D"),
                "steps": [1000, 2000, 3000, 4000, 5000],
                "score": [70, 75, 80, 85, 90],
            }
        )
        out = tmp_path / "master.parquet"
        result = export_to_parquet(df=df, output_path=out, include_data_quality=False)
        assert result == out
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 5
        assert "steps" in loaded.columns
        assert "score" in loaded.columns

    def test_round_trip_preserves_data(self, tmp_path):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "steps": [1000, 2000, 3000],
                "score": [70, 75, 80],
            }
        )
        out = tmp_path / "master.parquet"
        export_to_parquet(df=df, output_path=out, include_data_quality=False)
        loaded = pd.read_parquet(out)
        pd.testing.assert_frame_equal(
            df.astype({"day": "datetime64[ns]"}),
            loaded[["day", "steps", "score"]],
            check_dtype=False,
        )

    def test_include_data_quality_merge_when_file_exists(self, tmp_path, monkeypatch):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "steps": [1000, 2000, 3000],
            }
        )
        dq = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "data_quality_score": [80, 85, 90],
                "coverage_score": [95, 98, 100],
                "completeness_score": [50, 60, 70],
                "key_metrics_count": [4, 5, 6],
                "key_metrics_total": [8, 8, 8],
            }
        )
        dq_path = tmp_path / "daily_data_quality.csv"
        dq.to_csv(dq_path, index=False)
        out = tmp_path / "master.parquet"
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.DAILY_DATA_QUALITY_CSV",
            dq_path,
        )
        export_to_parquet(df=df, output_path=out, include_data_quality=True)
        loaded = pd.read_parquet(out)
        assert "data_quality_score" in loaded.columns
        assert loaded["data_quality_score"].tolist() == [80.0, 85.0, 90.0]

    def test_include_data_quality_false_skips_merge(self, tmp_path):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=2, freq="D"),
                "steps": [1000, 2000],
            }
        )
        out = tmp_path / "master.parquet"
        export_to_parquet(df=df, output_path=out, include_data_quality=False)
        loaded = pd.read_parquet(out)
        assert "data_quality_score" not in loaded.columns

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "export" / "sub" / "master.parquet"
        df = pd.DataFrame({"day": [pd.Timestamp("2024-01-01")], "x": [1]})
        export_to_parquet(df=df, output_path=out, include_data_quality=False)
        assert out.parent.exists()
        assert out.exists()


class TestExportMaster:
    """Tests for export_master."""

    def test_parquet_only(self, tmp_path, monkeypatch):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "steps": [1000, 2000, 3000],
            }
        )
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.EXPORT_DIR",
            tmp_path / "export",
        )
        result = export_master(parquet=True, duckdb=False, df=df)
        assert "parquet_path" in result
        assert Path(result["parquet_path"]).exists()
        assert "duckdb_path" not in result

    def test_parquet_and_duckdb_when_duckdb_installed(self, tmp_path, monkeypatch):
        df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=2, freq="D"),
                "steps": [1000, 2000],
            }
        )
        export_dir = tmp_path / "export"
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.EXPORT_DIR",
            export_dir,
        )
        result = export_master(parquet=True, duckdb=True, df=df)
        assert "parquet_path" in result
        if "duckdb_path" in result:
            assert Path(result["duckdb_path"]).exists()

    def test_parquet_false_duckdb_true_still_exports_parquet_first(self, tmp_path, monkeypatch):
        df = pd.DataFrame({"day": [pd.Timestamp("2024-01-01")], "x": [1]})
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.EXPORT_DIR",
            tmp_path / "export",
        )
        result = export_master(parquet=False, duckdb=True, df=df)
        if "duckdb_path" in result:
            assert (tmp_path / "export" / "master_daily_summary.parquet").exists()


class TestExportToDuckDB:
    """Tests for export_to_duckdb."""

    def test_with_parquet_file(self, tmp_path):
        parquet_path = tmp_path / "master.parquet"
        pd.DataFrame({"day": [1, 2], "x": [10, 20]}).to_parquet(parquet_path)
        result = export_to_duckdb(
            parquet_path=parquet_path,
            duckdb_path=tmp_path / "out.duckdb",
        )
        if result is not None:
            assert result.exists()
            import duckdb

            conn = duckdb.connect(str(result))
            rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()
            assert rows[0] == 2
            conn.close()


class TestExportAdditionalFormats:
    def test_export_to_csv(self, sample_master_df, tmp_path):
        out = tmp_path / "export.csv"
        export_to_csv(sample_master_df, out)
        loaded = pd.read_csv(out, parse_dates=["day"])
        assert len(loaded) == len(sample_master_df)

    def test_export_to_json(self, sample_master_df, tmp_path):
        out = tmp_path / "export.json"
        export_to_json(sample_master_df, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload) == len(sample_master_df)
        assert payload[0]["steps"] == 1000


class TestExportMasterDateRange:
    def test_export_csv_date_range(self, sample_master_df, tmp_path):
        result = export_master_date_range(
            "2024-01-02",
            "2024-01-04",
            "csv",
            df=sample_master_df,
            output_path=tmp_path / "range.csv",
            include_data_quality=False,
            summary_only=True,
        )
        assert result["row_count"] == 3
        assert result["format"] == "csv"
        loaded = pd.read_csv(result["output_path"], parse_dates=["day"])
        assert len(loaded) == 3

    def test_export_parquet_date_range(self, sample_master_df, tmp_path):
        result = export_master_date_range(
            "2024-01-01",
            "2024-01-03",
            "parquet",
            df=sample_master_df,
            output_path=tmp_path / "range.parquet",
            include_data_quality=False,
            summary_only=True,
        )
        loaded = pd.read_parquet(result["output_path"])
        assert len(loaded) == 3

    def test_export_json_date_range(self, sample_master_df, tmp_path):
        result = export_master_date_range(
            "2024-01-08",
            "2024-01-10",
            "json",
            df=sample_master_df,
            output_path=tmp_path / "range.json",
            include_data_quality=False,
            summary_only=True,
        )
        payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
        assert len(payload) == 3

    def test_export_master_wrapper_with_date_range(self, sample_master_df, tmp_path):
        result = export_master(
            start_date="2024-01-01",
            end_date="2024-01-02",
            export_format="csv",
            df=sample_master_df,
            output_path=tmp_path / "wrapper.csv",
            include_data_quality=False,
            summary_only=True,
        )
        assert result["row_count"] == 2


@pytest.mark.integration
class TestExportMasterDateRangeIntegration:
    def test_export_from_master_csv_file(self, tmp_path, monkeypatch):
        master_df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=7, freq="D"),
                "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
            }
        )
        master_path = tmp_path / "master_daily_summary.csv"
        master_df.to_csv(master_path, index=False)
        export_dir = tmp_path / "export"
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.MASTER_CSV",
            master_path,
        )
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        result = export_master_date_range(
            "2024-01-02",
            "2024-01-05",
            "csv",
            output_path=export_dir / "subset.csv",
            include_data_quality=False,
            summary_only=True,
        )

        assert result["row_count"] == 4
        loaded = pd.read_csv(result["output_path"], parse_dates=["day"])
        assert loaded["day"].min() == pd.Timestamp("2024-01-02")
        assert loaded["day"].max() == pd.Timestamp("2024-01-05")

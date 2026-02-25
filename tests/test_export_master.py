"""Tests for export master to Parquet and DuckDB."""

import pytest
import pandas as pd
from pathlib import Path

from garmin_analysis.data_ingestion.export_master import (
    export_to_parquet,
    export_to_duckdb,
    export_master,
)


class TestExportToParquet:
    """Tests for export_to_parquet."""

    def test_export_from_df(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=5, freq="D"),
            "steps": [1000, 2000, 3000, 4000, 5000],
            "score": [70, 75, 80, 85, 90],
        })
        out = tmp_path / "master.parquet"
        result = export_to_parquet(df=df, output_path=out, include_data_quality=False)
        assert result == out
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 5
        assert "steps" in loaded.columns
        assert "score" in loaded.columns

    def test_round_trip_preserves_data(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, 2000, 3000],
            "score": [70, 75, 80],
        })
        out = tmp_path / "master.parquet"
        export_to_parquet(df=df, output_path=out, include_data_quality=False)
        loaded = pd.read_parquet(out)
        pd.testing.assert_frame_equal(
            df.astype({"day": "datetime64[ns]"}),
            loaded[["day", "steps", "score"]],
            check_dtype=False,
        )

    def test_include_data_quality_merge_when_file_exists(self, tmp_path, monkeypatch):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, 2000, 3000],
        })
        dq = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "data_quality_score": [80, 85, 90],
            "coverage_score": [95, 98, 100],
            "completeness_score": [50, 60, 70],
            "key_metrics_count": [4, 5, 6],
            "key_metrics_total": [8, 8, 8],
        })
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
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=2, freq="D"),
            "steps": [1000, 2000],
        })
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
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=3, freq="D"),
            "steps": [1000, 2000, 3000],
        })
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_master.EXPORT_DIR",
            tmp_path / "export",
        )
        result = export_master(parquet=True, duckdb=False, df=df)
        assert "parquet_path" in result
        assert Path(result["parquet_path"]).exists()
        assert "duckdb_path" not in result

    def test_parquet_and_duckdb_when_duckdb_installed(self, tmp_path, monkeypatch):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=2, freq="D"),
            "steps": [1000, 2000],
        })
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

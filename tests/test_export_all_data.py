"""Tests for exporting all Garmin source tables in a date range."""

import json

import pandas as pd
import pytest

from garmin_analysis.data_ingestion.export_all_data import (
    ExportTableSpec,
    discover_export_tables,
    export_all_data_date_range,
    filter_table_by_date_range,
    resolve_date_column,
)


@pytest.fixture
def sample_stress_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-02 08:00:00",
                    "2024-01-05 08:00:00",
                ]
            ),
            "stress": [20, 30, 40],
        }
    )


class TestExportAllDataHelpers:
    def test_resolve_date_column_prefers_registry(self):
        assert resolve_date_column("garmin", "stress", ["timestamp", "stress"]) == "timestamp"
        assert resolve_date_column("garmin", "sleep", ["day", "score"]) == "day"

    def test_filter_table_by_timestamp_range(self, sample_stress_df):
        filtered = filter_table_by_date_range(
            sample_stress_df,
            "2024-01-01",
            "2024-01-02",
            date_column="timestamp",
        )
        assert len(filtered) == 2

    def test_discover_export_tables_from_fixture_db(self, tmp_db):
        result = discover_export_tables(db_paths=tmp_db)
        table_names = {(spec.db_key, spec.table_name) for spec in result}
        assert ("garmin", "daily_summary") in table_names
        assert ("garmin", "stress") in table_names
        assert ("activities", "activities") in table_names


class TestExportAllDataDateRange:
    def test_exports_bundle_with_source_tables_and_master(
        self,
        tmp_path,
        tmp_db,
        monkeypatch,
    ):
        master_df = pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=5, freq="D"),
                "steps": [1000, 2000, 3000, 4000, 5000],
            }
        )
        master_path = tmp_path / "master_daily_summary.csv"
        master_df.to_csv(master_path, index=False)
        output_dir = tmp_path / "bundle"

        monkeypatch.setattr("garmin_analysis.config.DB_PATHS", tmp_db)
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_all_data.DB_PATHS",
            tmp_db,
        )
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        result = export_all_data_date_range(
            "2024-01-02",
            "2024-01-04",
            "csv",
            output_dir=output_dir,
            include_data_quality=False,
            include_daily_data_quality_file=False,
        )

        assert result["scope"] == "all"
        assert result["table_count"] >= 4
        assert (output_dir / "master_daily_summary.csv").exists()
        assert (output_dir / "garmin" / "daily_summary.csv").exists()
        assert (output_dir / "garmin" / "stress.csv").exists()

        stress = pd.read_csv(output_dir / "garmin" / "stress.csv", parse_dates=["timestamp"])
        assert len(stress) >= 1
        master = pd.read_csv(output_dir / "master_daily_summary.csv", parse_dates=["day"])
        assert len(master) == 3

    def test_exports_json_bundle(self, tmp_path, tmp_db, monkeypatch):
        master_path = tmp_path / "master_daily_summary.csv"
        pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=3, freq="D"),
                "steps": [1, 2, 3],
            }
        ).to_csv(master_path, index=False)
        output_dir = tmp_path / "bundle"

        monkeypatch.setattr("garmin_analysis.config.DB_PATHS", tmp_db)
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_all_data.DB_PATHS",
            tmp_db,
        )
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        specs = [
            ExportTableSpec(
                db_key="garmin",
                db_path=tmp_db["garmin"],
                table_name="daily_summary",
                date_column="day",
                parse_dates=("day",),
            )
        ]
        result = export_all_data_date_range(
            "2024-01-01",
            "2024-01-02",
            "json",
            output_dir=output_dir,
            include_master=False,
            include_daily_data_quality_file=False,
            table_specs=specs,
        )

        payload = json.loads((output_dir / "garmin" / "daily_summary.json").read_text())
        assert len(payload) == 2
        assert result["tables"]["garmin__daily_summary"] == 2


@pytest.mark.integration
class TestExportAllDataIntegration:
    def test_cli_exports_all_data_bundle(self, tmp_path, tmp_db, monkeypatch):
        import sys

        from garmin_analysis.cli_export import main

        master_path = tmp_path / "master_daily_summary.csv"
        pd.DataFrame(
            {
                "day": pd.date_range("2024-01-01", periods=5, freq="D"),
                "steps": [1000, 2000, 3000, 4000, 5000],
            }
        ).to_csv(master_path, index=False)
        output_dir = tmp_path / "export_bundle"

        monkeypatch.setattr("garmin_analysis.config.DB_PATHS", tmp_db)
        monkeypatch.setattr(
            "garmin_analysis.data_ingestion.export_all_data.DB_PATHS",
            tmp_db,
        )
        monkeypatch.setattr(
            "garmin_analysis.utils.data_loading.MASTER_CSV",
            master_path,
        )

        sys.argv = [
            "cli_export",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-04",
            "--format",
            "csv",
            "--output",
            str(output_dir),
            "--no-data-quality",
        ]
        assert main() == 0
        assert (output_dir / "master_daily_summary.csv").exists()
        assert (output_dir / "garmin" / "sleep.csv").exists()

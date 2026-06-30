"""Tests for cli_export CLI."""

import sys
from pathlib import Path

import pandas as pd
import pytest


def test_cli_export_returns_zero_on_success(tmp_path, monkeypatch):
    from garmin_analysis.cli_export import main

    monkeypatch.setattr(
        "garmin_analysis.cli_export.export_master",
        lambda **kw: {"parquet_path": str(tmp_path / "out.parquet")},
    )
    sys.argv = ["cli_export"]
    assert main() == 0


def test_cli_export_returns_one_on_failure(monkeypatch):
    from garmin_analysis.cli_export import main

    def fail(**kw):
        raise ValueError("Export failed")

    monkeypatch.setattr("garmin_analysis.cli_export.export_master", fail)
    sys.argv = ["cli_export"]
    assert main() == 1


def test_cli_export_parquet_only_flag(monkeypatch):
    from garmin_analysis.cli_export import main

    export_calls = []

    def capture(**kw):
        export_calls.append(kw)
        return {"parquet_path": "/tmp/out.parquet"}

    monkeypatch.setattr("garmin_analysis.cli_export.export_master", capture)
    sys.argv = ["cli_export", "--parquet-only"]
    main()
    assert export_calls[0]["duckdb"] is False
    assert export_calls[0]["parquet"] is True


def test_cli_export_duckdb_flag(monkeypatch):
    from garmin_analysis.cli_export import main

    export_calls = []

    def capture(**kw):
        export_calls.append(kw)
        return {"parquet_path": "/tmp/out.parquet"}

    monkeypatch.setattr("garmin_analysis.cli_export.export_master", capture)
    sys.argv = ["cli_export", "--duckdb"]
    main()
    assert export_calls[0]["duckdb"] is True


def test_cli_export_date_range_requires_both_dates(monkeypatch):
    from garmin_analysis.cli_export import main

    sys.argv = ["cli_export", "--start-date", "2024-01-01"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_cli_export_date_range_requires_format(monkeypatch):
    from garmin_analysis.cli_export import main

    sys.argv = [
        "cli_export",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-31",
    ]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_cli_export_date_range_passes_arguments(monkeypatch):
    from garmin_analysis.cli_export import main

    export_calls = []

    def capture(**kw):
        export_calls.append(kw)
        return {
            "format": "csv",
            "output_path": "/tmp/range.csv",
            "row_count": 10,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

    monkeypatch.setattr("garmin_analysis.cli_export.export_master", capture)
    sys.argv = [
        "cli_export",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-31",
        "--format",
        "csv",
        "--output",
        "/tmp/range.csv",
    ]
    assert main() == 0
    assert export_calls[0]["start_date"] == "2024-01-01"
    assert export_calls[0]["end_date"] == "2024-01-31"
    assert export_calls[0]["export_format"] == "csv"
    assert export_calls[0]["output_path"] == Path("/tmp/range.csv")


@pytest.mark.integration
def test_cli_export_summary_only_end_to_end(tmp_path, monkeypatch):
    from garmin_analysis.cli_export import main

    master_df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=5, freq="D"),
            "steps": [1000, 2000, 3000, 4000, 5000],
        }
    )
    master_path = tmp_path / "master_daily_summary.csv"
    master_df.to_csv(master_path, index=False)
    output_path = tmp_path / "export" / "subset.csv"

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
        str(output_path),
        "--no-data-quality",
        "--summary-only",
    ]
    assert main() == 0
    loaded = pd.read_csv(output_path, parse_dates=["day"])
    assert len(loaded) == 3

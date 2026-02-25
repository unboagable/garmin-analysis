"""Tests for cli_export CLI."""

import pytest
import sys
from pathlib import Path


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

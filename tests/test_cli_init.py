"""Tests for garmin init bootstrap CLI."""

import sqlite3
import pytest
from pathlib import Path

from garmin_analysis.cli_init import (
    check_databases,
    validate_schema,
    run_init,
)


class TestCheckDatabases:
    """Tests for check_databases."""

    def test_returns_dict(self):
        result = check_databases()
        assert isinstance(result, dict)
        assert "garmin" in result
        assert "exists" in result["garmin"]
        assert "path" in result["garmin"]

    def test_all_db_keys_present(self):
        result = check_databases()
        expected_keys = {"garmin", "activities", "monitoring", "summary", "summary2"}
        assert expected_keys.issubset(result.keys())

    def test_each_entry_has_required_fields(self):
        result = check_databases()
        for name, info in result.items():
            assert "exists" in info
            assert "path" in info
            assert "size_mb" in info
            assert isinstance(info["exists"], bool)
            assert isinstance(info["size_mb"], (int, float))


class TestValidateSchema:
    """Tests for validate_schema."""

    def test_nonexistent_db(self):
        ok, msg = validate_schema(Path("/nonexistent/db.db"))
        assert ok is False
        assert "not found" in msg.lower() or "no" in msg.lower()

    def test_empty_db_no_tables(self, tmp_path):
        db = tmp_path / "empty.db"
        sqlite3.connect(db).close()
        ok, msg = validate_schema(db)
        assert ok is False
        assert "no tables" in msg.lower() or "not found" in msg.lower()

    def test_garmin_db_with_required_tables(self, tmp_path):
        db = tmp_path / "garmin.db"
        with sqlite3.connect(db) as conn:
            for t in ["daily_summary", "sleep", "stress", "resting_hr"]:
                conn.execute(f"CREATE TABLE {t} (id INTEGER)")
        ok, msg = validate_schema(db)
        assert ok is True
        assert "OK" in msg

    def test_garmin_db_missing_tables(self, tmp_path):
        db = tmp_path / "garmin.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE daily_summary (id INTEGER)")
            conn.execute("CREATE TABLE sleep (id INTEGER)")
            # Missing: stress, resting_hr
        ok, msg = validate_schema(db)
        assert ok is False
        assert "missing" in msg.lower() or "stress" in msg.lower()

    def test_activities_db_valid(self, tmp_path):
        db = tmp_path / "garmin_activities.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE activities (id INTEGER)")
        ok, msg = validate_schema(db)
        assert ok is True

    def test_monitoring_db_valid(self, tmp_path):
        db = tmp_path / "garmin_monitoring.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE monitoring_hr (id INTEGER)")
        ok, msg = validate_schema(db)
        assert ok is True

    def test_unknown_db_name_returns_ok(self, tmp_path):
        db = tmp_path / "other.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE foo (id INTEGER)")
        ok, msg = validate_schema(db)
        assert ok is True
        assert "no schema spec" in msg or "Found" in msg


class TestRunInit:
    """Tests for run_init."""

    def test_exits_zero(self):
        assert run_init(verbose=False) == 0

    def test_verbose_mode(self):
        assert run_init(verbose=True) == 0

    def test_output_contains_folders_section(self, capsys):
        run_init(verbose=False)
        out, _ = capsys.readouterr()
        assert "Folders" in out
        assert "Databases" in out
        assert "Next commands" in out

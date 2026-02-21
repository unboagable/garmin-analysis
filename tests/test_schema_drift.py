import os
import sqlite3
import tempfile
import logging

import pytest

from garmin_analysis.data_ingestion.inspect_sqlite_schema import extract_schema, detect_schema_drift


@pytest.fixture
def temp_db_with_schema():
    """Create a temporary SQLite DB that mirrors key Garmin tables/types."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE daily_summary (
            day DATE,
            hr_min INTEGER,
            hr_max INTEGER,
            stress_avg INTEGER,
            steps INTEGER,
            calories_total INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE sleep (
            day DATE,
            total_sleep TIME,
            rem_sleep TIME,
            score INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE stress (
            timestamp DATETIME,
            stress INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE resting_hr (
            day DATE,
            resting_heart_rate FLOAT
        )
        """
    )

    conn.commit()
    conn.close()
    try:
        yield path
    finally:
        os.remove(path)


class TestSchemaDetection:

    def test_returns_table_info(self, temp_db_with_schema):
        schema = extract_schema(temp_db_with_schema)
        assert "daily_summary" in schema
        for pair in [("day", "DATE"), ("hr_min", "INTEGER"), ("hr_max", "INTEGER"), ("stress_avg", "INTEGER"), ("steps", "INTEGER"), ("calories_total", "INTEGER")]:
            assert pair in schema["daily_summary"]

        assert "sleep" in schema
        for pair in [("day", "DATE"), ("total_sleep", "TIME"), ("rem_sleep", "TIME"), ("score", "INTEGER")]:
            assert pair in schema["sleep"]

        assert ("timestamp", "DATETIME") in schema["stress"]
        assert ("stress", "INTEGER") in schema["stress"]

        assert ("day", "DATE") in schema["resting_hr"]
        assert ("resting_heart_rate", "FLOAT") in schema["resting_hr"]

    def test_no_changes(self, temp_db_with_schema):
        expected = extract_schema(temp_db_with_schema)
        actual = extract_schema(temp_db_with_schema)
        report = detect_schema_drift(expected, actual)
        assert report == {}

    def test_with_changes(self, temp_db_with_schema):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE daily_summary (
                day DATE,
                hr_min INTEGER,
                hr_max INTEGER,
                stress_avg INTEGER
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE sleep (
                day DATE,
                total_sleep TIME,
                rem_sleep TIME,
                score INTEGER,
                qualifier VARCHAR
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE stress (
                timestamp DATETIME,
                stress INTEGER
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE resting_hr (
                day DATE,
                resting_heart_rate INTEGER
            )
            """
        )

        conn.commit()
        conn.close()

        expected = extract_schema(temp_db_with_schema)
        actual = extract_schema(path)
        try:
            report = detect_schema_drift(expected, actual)

            assert "daily_summary" in report
            missing = report["daily_summary"]["missing_columns"]
            assert "steps" in missing and "calories_total" in missing

            assert "sleep" in report
            assert "qualifier" in report["sleep"]["extra_columns"]

            assert "resting_hr" in report
            mismatches = dict((c, (e_t, a_t)) for c, e_t, a_t in report["resting_hr"]["type_mismatches"])
            assert mismatches.get("resting_heart_rate") == ("FLOAT", "INTEGER")
        finally:
            os.remove(path)

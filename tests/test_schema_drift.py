import os
import sqlite3
import tempfile
import logging

import pytest

from src.data_ingestion.inspect_sqlite_schema import extract_schema, detect_schema_drift


@pytest.fixture
def temp_db_with_schema():
    """Create a temporary SQLite DB that mirrors key Garmin tables/types."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Subset of daily_summary
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

    # Subset of sleep
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

    # Stress
    cur.execute(
        """
        CREATE TABLE stress (
            timestamp DATETIME,
            stress INTEGER
        )
        """
    )

    # Resting HR
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


def test_extract_schema_basic(temp_db_with_schema):
    schema = extract_schema(temp_db_with_schema)
    # daily_summary
    assert "daily_summary" in schema
    for pair in [("day", "DATE"), ("hr_min", "INTEGER"), ("hr_max", "INTEGER"), ("stress_avg", "INTEGER"), ("steps", "INTEGER"), ("calories_total", "INTEGER")]:
        assert pair in schema["daily_summary"]

    # sleep
    assert "sleep" in schema
    for pair in [("day", "DATE"), ("total_sleep", "TIME"), ("rem_sleep", "TIME"), ("score", "INTEGER")]:
        assert pair in schema["sleep"]

    # stress
    assert ("timestamp", "DATETIME") in schema["stress"]
    assert ("stress", "INTEGER") in schema["stress"]

    # resting_hr
    assert ("day", "DATE") in schema["resting_hr"]
    assert ("resting_heart_rate", "FLOAT") in schema["resting_hr"]


def test_detect_schema_drift_no_drift(temp_db_with_schema):
    expected = extract_schema(temp_db_with_schema)
    actual = extract_schema(temp_db_with_schema)
    report = detect_schema_drift(expected, actual)
    assert report == {}


def test_detect_schema_drift_with_changes(temp_db_with_schema):
    # Build an actual schema with missing columns, extra columns, and a type change
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # daily_summary missing steps and calories_total
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

    # sleep with extra column 'qualifier'
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

    # stress unchanged
    cur.execute(
        """
        CREATE TABLE stress (
            timestamp DATETIME,
            stress INTEGER
        )
        """
    )

    # resting_hr with type drift on resting_heart_rate
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

        # daily_summary: missing steps and calories_total
        assert "daily_summary" in report
        missing = report["daily_summary"]["missing_columns"]
        assert "steps" in missing and "calories_total" in missing

        # sleep: extra qualifier column
        assert "sleep" in report
        assert "qualifier" in report["sleep"]["extra_columns"]

        # resting_hr: type mismatch
        assert "resting_hr" in report
        mismatches = dict((c, (e_t, a_t)) for c, e_t, a_t in report["resting_hr"]["type_mismatches"])
        assert mismatches.get("resting_heart_rate") == ("FLOAT", "INTEGER")
    finally:
        os.remove(path)

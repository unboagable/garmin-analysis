"""
Tests for check_missing_data module.

These tests verify:
- Missing data identification across tables
- Pattern analysis for different data scenarios
- Report generation and export functionality
"""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path
import os

from garmin_analysis.features.check_missing_data import audit_table_health, main


class TestAuditTableHealth:

    def test_with_valid_tables(self, mem_db):
        """Test identification of missing data in database tables."""
        # Test basic table health audit
        tables_to_check = ["daily_summary", "sleep", "stress", "resting_hr"]
        
        report = audit_table_health(mem_db, tables_to_check)
        
        # Verify report structure
        assert isinstance(report, pd.DataFrame)
        assert len(report) == len(tables_to_check)
        assert list(report.columns) == ["table", "rows", "status"]
        
        # Verify each table is found and has data
        for _, row in report.iterrows():
            assert row["table"] in tables_to_check
            assert row["rows"] > 0
            assert row["status"] in ["OK", "all null/empty", "not found"]
        
        # Verify specific tables have correct status
        daily_summary_row = report[report["table"] == "daily_summary"].iloc[0]
        assert daily_summary_row["status"] == "OK"
        assert daily_summary_row["rows"] == 5

    def test_with_various_patterns(self, mem_db):
        """Test analysis of different missing data patterns."""
        # Create additional test tables with different patterns
        cur = mem_db.cursor()
        
        # Pattern 1: Table with all NULL values
        cur.execute("""
            CREATE TABLE IF NOT EXISTS null_table (
                day TEXT,
                value1 REAL,
                value2 REAL
            )
        """)
        cur.executemany(
            "INSERT INTO null_table (day, value1, value2) VALUES (?, ?, ?)",
            [("2024-01-01", None, None), ("2024-01-02", None, None)]
        )
        
        # Pattern 2: Empty table (no rows)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS empty_table (
                day TEXT,
                value REAL
            )
        """)
        
        # Pattern 3: Table with mixed data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mixed_table (
                day TEXT,
                value1 REAL,
                value2 REAL
            )
        """)
        cur.executemany(
            "INSERT INTO mixed_table (day, value1, value2) VALUES (?, ?, ?)",
            [("2024-01-01", 100.0, None), ("2024-01-02", None, 200.0), ("2024-01-03", 150.0, 250.0)]
        )
        
        mem_db.commit()
        
        # Test audit on tables with different patterns
        tables_to_check = ["null_table", "empty_table", "mixed_table", "nonexistent_table"]
        report = audit_table_health(mem_db, tables_to_check)
        
        # Verify all tables are reported
        assert len(report) == len(tables_to_check)
        
        # Verify null table detection
        # Note: null_table has the 'day' column with values, so not ALL values are null
        # Only the value1 and value2 columns are null, so status should be "OK"
        null_row = report[report["table"] == "null_table"].iloc[0]
        assert null_row["status"] == "OK"  # Has day column with data
        assert null_row["rows"] == 2
        
        # Verify empty table detection (has 0 rows but table exists)
        empty_row = report[report["table"] == "empty_table"].iloc[0]
        assert empty_row["rows"] == 0
        # Empty table (0 rows) results in empty DataFrame, which isnull().all() returns True
        assert empty_row["status"] == "all null/empty"
        
        # Verify mixed data table (has some data, some nulls)
        mixed_row = report[report["table"] == "mixed_table"].iloc[0]
        assert mixed_row["status"] == "OK"  # Not all columns are null
        assert mixed_row["rows"] == 3
        
        # Verify nonexistent table detection
        nonexistent_row = report[report["table"] == "nonexistent_table"].iloc[0]
        assert nonexistent_row["status"] == "not found"
        assert nonexistent_row["rows"] == 0


class TestCheckMissingDataMain:

    @pytest.mark.integration
    def test_end_to_end(self, tmp_path):
        """Test CSV report generation from missing data analysis."""
        # Create a temporary database
        db_path = tmp_path / "test_garmin.db"
        conn = sqlite3.connect(db_path)
        
        # Create and populate test tables
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE daily_summary (
                day TEXT,
                steps REAL,
                calories_total REAL
            );
            CREATE TABLE sleep (
                day TEXT,
                total_sleep TEXT,
                score REAL
            );
            CREATE TABLE weight (
                day TEXT,
                weight REAL
            );
        """)
        
        # Add some data
        cur.executemany(
            "INSERT INTO daily_summary (day, steps, calories_total) VALUES (?, ?, ?)",
            [("2024-01-01", 5000.0, 2000.0), ("2024-01-02", 6000.0, 2100.0)]
        )
        cur.executemany(
            "INSERT INTO sleep (day, total_sleep, score) VALUES (?, ?, ?)",
            [("2024-01-01", "07:30:00", 75.0), ("2024-01-02", "08:00:00", 80.0)]
        )
        # Weight table left empty to test missing data
        
        conn.commit()
        conn.close()
        
        # Create a data directory in the temp path
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Temporarily change working directory to tmp_path for CSV export
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Run main with export enabled
            main(db_path=str(db_path), export_csv=True)
            
            # Verify CSV was created
            report_path = data_dir / "missing_report.csv"
            assert report_path.exists(), "Report CSV should be created"
            
            # Read and verify CSV contents
            report_df = pd.read_csv(report_path)
            
            # Verify structure
            assert list(report_df.columns) == ["table", "rows", "status"]
            assert len(report_df) > 0
            
            # Verify specific tables are in report
            table_names = report_df["table"].tolist()
            assert "daily_summary" in table_names
            assert "sleep" in table_names
            assert "weight" in table_names
            
            # Verify data accuracy
            daily_summary_row = report_df[report_df["table"] == "daily_summary"]
            assert len(daily_summary_row) == 1
            assert daily_summary_row.iloc[0]["rows"] == 2
            assert daily_summary_row.iloc[0]["status"] == "OK"
            
            sleep_row = report_df[report_df["table"] == "sleep"]
            assert len(sleep_row) == 1
            assert sleep_row.iloc[0]["rows"] == 2
            assert sleep_row.iloc[0]["status"] == "OK"
            
            weight_row = report_df[report_df["table"] == "weight"]
            assert len(weight_row) == 1
            assert weight_row.iloc[0]["rows"] == 0
            # Empty table (0 rows) shows as "all null/empty"
            assert weight_row.iloc[0]["status"] == "all null/empty"
            
        finally:
            os.chdir(original_cwd)

    def test_no_export(self, tmp_path):
        """Test main function without CSV export."""
        # Create a temporary database
        db_path = tmp_path / "test_garmin.db"
        conn = sqlite3.connect(db_path)
        
        # Create simple test table
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE daily_summary (
                day TEXT,
                steps REAL
            )
        """)
        cur.execute("INSERT INTO daily_summary (day, steps) VALUES (?, ?)", ("2024-01-01", 5000.0))
        conn.commit()
        conn.close()
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Run main without export
            main(db_path=str(db_path), export_csv=False)
            
            # Verify CSV was NOT created
            data_dir = tmp_path / "data"
            if data_dir.exists():
                report_path = data_dir / "missing_report.csv"
                assert not report_path.exists(), "Report CSV should not be created when export_csv=False"
        
        finally:
            os.chdir(original_cwd)

    def test_nonexistent_database(self):
        """Test handling of nonexistent database file."""
        # This should log an error and return without crashing
        nonexistent_path = "/path/to/nonexistent/database.db"
        
        # Should not raise an exception
        result = main(db_path=nonexistent_path, export_csv=False)
        
        # Function returns None when DB not found
        assert result is None

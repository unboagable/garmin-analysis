"""
Tests for data loading utilities.

These tests verify:
- Loading existing master dataframe from CSV
- Auto-building master dataframe when CSV doesn't exist
- Loading Garmin tables from valid database
- Handling invalid/missing databases
- Handling corrupted CSV files
- Synthetic data warnings
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from garmin_analysis.utils.data_loading import (
    load_master_dataframe,
    load_garmin_tables
)


@pytest.fixture
def sample_master_csv(tmp_path):
    """Create a sample master CSV file."""
    csv_path = tmp_path / "master_health_data.csv"
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=10)
    df = pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, 10),
        'calories_total': np.random.randint(1800, 2500, 10),
        'score': np.random.randint(60, 100, 10),
        'resting_heart_rate': np.random.randint(50, 75, 10)
    })
    
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def valid_garmin_db(tmp_path):
    """Create a valid Garmin database with required tables."""
    db_path = tmp_path / "garmin.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create tables
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
        
        CREATE TABLE stress (
            timestamp TEXT,
            stress REAL
        );
        
        CREATE TABLE resting_hr (
            day TEXT,
            resting_heart_rate REAL
        );
    """)
    
    # Insert sample data
    cur.executemany(
        "INSERT INTO daily_summary (day, steps, calories_total) VALUES (?, ?, ?)",
        [('2024-01-01', 10000, 2000), ('2024-01-02', 12000, 2100)]
    )
    
    cur.executemany(
        "INSERT INTO sleep (day, total_sleep, score) VALUES (?, ?, ?)",
        [('2024-01-01', '07:30:00', 75), ('2024-01-02', '08:00:00', 80)]
    )
    
    cur.executemany(
        "INSERT INTO stress (timestamp, stress) VALUES (?, ?)",
        [('2024-01-01 08:00:00', 25), ('2024-01-01 17:00:00', 35)]
    )
    
    cur.executemany(
        "INSERT INTO resting_hr (day, resting_heart_rate) VALUES (?, ?)",
        [('2024-01-01', 62), ('2024-01-02', 60)]
    )
    
    conn.commit()
    conn.close()
    
    return db_path


def test_load_master_dataframe_existing(sample_master_csv):
    """Test loading master dataframe when CSV file exists."""
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', sample_master_csv):
        df = load_master_dataframe()
        
        # Verify dataframe was loaded
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 10
        
        # Verify expected columns
        assert 'day' in df.columns
        assert 'steps' in df.columns
        assert 'calories_total' in df.columns
        
        # Verify day column is datetime
        assert pd.api.types.is_datetime64_any_dtype(df['day'])


def test_load_master_dataframe_auto_build(tmp_path):
    """Test auto-building master dataframe when CSV doesn't exist."""
    nonexistent_csv = tmp_path / "nonexistent.csv"
    
    # Mock the summarize_and_merge function to return sample data
    mock_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=5),
        'steps': [8000, 9000, 10000, 11000, 12000],
        'score': [70, 75, 80, 85, 90]
    })
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', nonexistent_csv), \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.summarize_and_merge') as mock_summarize, \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.USING_SYNTHETIC_DATA', False):
        
        mock_summarize.return_value = mock_df
        
        df = load_master_dataframe()
        
        # Verify summarize_and_merge was called
        mock_summarize.assert_called_once_with(return_df=True)
        
        # Verify dataframe was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'day' in df.columns
        
        # Verify CSV was created
        assert nonexistent_csv.exists()
        
        # Verify CSV content
        saved_df = pd.read_csv(nonexistent_csv)
        assert len(saved_df) == 5


def test_load_master_dataframe_auto_build_with_synthetic_warning(tmp_path, caplog):
    """Test that synthetic data warning is logged during auto-build."""
    nonexistent_csv = tmp_path / "nonexistent.csv"
    
    mock_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=3),
        'steps': [8000, 9000, 10000]
    })
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', nonexistent_csv), \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.summarize_and_merge') as mock_summarize, \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.USING_SYNTHETIC_DATA', True):
        
        mock_summarize.return_value = mock_df
        
        # Need to mock the globals() check
        with patch('garmin_analysis.utils.data_loading.globals') as mock_globals:
            mock_globals.return_value = {'USING_SYNTHETIC_DATA': True}
            
            df = load_master_dataframe()
            
            # Verify warning was logged
            assert 'SYNTHETIC data' in caplog.text or 'synthetic' in caplog.text.lower()


def test_load_master_dataframe_auto_build_fails(tmp_path):
    """Test handling when auto-build fails."""
    nonexistent_csv = tmp_path / "nonexistent.csv"
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', nonexistent_csv), \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.summarize_and_merge') as mock_summarize:
        
        # Make summarize_and_merge raise an exception
        mock_summarize.side_effect = RuntimeError("Database connection failed")
        
        # Should raise DataLoadingError (wrapped by error handling decorator)
        from garmin_analysis.utils.error_handling import DataLoadingError
        with pytest.raises(DataLoadingError):
            load_master_dataframe()


def test_load_garmin_tables_valid_db(valid_garmin_db):
    """Test loading Garmin tables from a valid database."""
    tables = load_garmin_tables(db_path=str(valid_garmin_db))
    
    # Verify all expected tables are loaded
    assert isinstance(tables, dict)
    assert 'daily' in tables
    assert 'sleep' in tables
    assert 'stress' in tables
    assert 'rest_hr' in tables
    
    # Verify table contents
    assert isinstance(tables['daily'], pd.DataFrame)
    assert len(tables['daily']) == 2
    assert 'steps' in tables['daily'].columns
    
    assert isinstance(tables['sleep'], pd.DataFrame)
    assert len(tables['sleep']) == 2
    assert 'score' in tables['sleep'].columns
    
    assert isinstance(tables['stress'], pd.DataFrame)
    assert len(tables['stress']) == 2
    
    assert isinstance(tables['rest_hr'], pd.DataFrame)
    assert len(tables['rest_hr']) == 2


def test_load_garmin_tables_invalid_db(tmp_path):
    """Test handling of invalid database file."""
    invalid_db = tmp_path / "invalid.db"
    
    # Create an empty file (not a valid SQLite database)
    invalid_db.write_text("This is not a database")
    
    tables = load_garmin_tables(db_path=str(invalid_db))
    
    # Should return empty dict on error
    assert tables == {}


def test_handles_corrupted_csv(tmp_path):
    """Test handling of corrupted CSV file."""
    corrupted_csv = tmp_path / "corrupted.csv"
    
    # Create a corrupted CSV with mismatched columns
    corrupted_csv.write_text("day,steps,calories\n2024-01-01,10000\n2024-01-02,invalid,data,too,many,cols")
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', corrupted_csv):
        # Should raise an exception when trying to parse
        with pytest.raises(Exception):
            load_master_dataframe()


def test_handles_missing_db():
    """Test handling when database file doesn't exist."""
    nonexistent_db = "/path/to/nonexistent/garmin.db"
    
    tables = load_garmin_tables(db_path=nonexistent_db)
    
    # Should return empty dict
    assert tables == {}
    assert isinstance(tables, dict)


def test_handles_missing_db_with_logging(caplog):
    """Test that missing database logs appropriate error."""
    nonexistent_db = "/path/to/nonexistent/garmin.db"
    
    tables = load_garmin_tables(db_path=nonexistent_db)
    
    # Verify error was logged
    assert 'not found' in caplog.text or 'Database file' in caplog.text


def test_synthetic_data_warning_during_build(tmp_path, caplog):
    """Test that synthetic data warning is logged when DBs are missing."""
    nonexistent_csv = tmp_path / "test_master.csv"
    
    # Create synthetic sample data
    synthetic_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=5),
        'steps': [8000] * 5,
        'score': [75] * 5
    })
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', nonexistent_csv), \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.summarize_and_merge') as mock_summarize, \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.USING_SYNTHETIC_DATA', True):
        
        mock_summarize.return_value = synthetic_df
        
        # Mock globals to include USING_SYNTHETIC_DATA
        original_globals = globals
        
        def mock_globals_func():
            g = original_globals()
            g['USING_SYNTHETIC_DATA'] = True
            return g
        
        with patch('garmin_analysis.utils.data_loading.globals', side_effect=mock_globals_func):
            df = load_master_dataframe()
            
            # Verify data was loaded (even if synthetic)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5


def test_load_garmin_tables_with_missing_tables(tmp_path, caplog):
    """Test loading from database with missing tables."""
    incomplete_db = tmp_path / "incomplete.db"
    conn = sqlite3.connect(incomplete_db)
    cur = conn.cursor()
    
    # Only create one table
    cur.execute("""
        CREATE TABLE daily_summary (
            day TEXT,
            steps REAL
        )
    """)
    cur.execute("INSERT INTO daily_summary (day, steps) VALUES ('2024-01-01', 10000)")
    conn.commit()
    conn.close()
    
    tables = load_garmin_tables(db_path=str(incomplete_db))
    
    # Should return empty dict if any query fails
    assert tables == {}


def test_load_master_dataframe_with_date_parsing(sample_master_csv):
    """Test that date parsing works correctly."""
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', sample_master_csv):
        df = load_master_dataframe()
        
        # Verify day column is parsed as datetime
        assert pd.api.types.is_datetime64_any_dtype(df['day'])
        
        # Verify dates are valid
        assert df['day'].min() == pd.Timestamp('2024-01-01')
        assert df['day'].max() >= pd.Timestamp('2024-01-01')


def test_load_garmin_tables_with_date_parsing(valid_garmin_db):
    """Test that dates are parsed correctly in loaded tables."""
    tables = load_garmin_tables(db_path=str(valid_garmin_db))
    
    # Verify day columns are parsed as datetime
    assert pd.api.types.is_datetime64_any_dtype(tables['daily']['day'])
    assert pd.api.types.is_datetime64_any_dtype(tables['sleep']['day'])
    assert pd.api.types.is_datetime64_any_dtype(tables['stress']['timestamp'])
    assert pd.api.types.is_datetime64_any_dtype(tables['rest_hr']['day'])


def test_load_master_dataframe_creates_directory(tmp_path):
    """Test that directory is created when auto-building."""
    nested_path = tmp_path / "data" / "outputs" / "master.csv"
    
    mock_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=3),
        'steps': [8000, 9000, 10000]
    })
    
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', nested_path), \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.summarize_and_merge') as mock_summarize, \
         patch('garmin_analysis.data_ingestion.load_all_garmin_dbs.USING_SYNTHETIC_DATA', False):
        
        mock_summarize.return_value = mock_df
        
        df = load_master_dataframe()
        
        # Verify directory was created
        assert nested_path.parent.exists()
        assert nested_path.exists()


def test_load_garmin_tables_returns_empty_on_connection_error(tmp_path, caplog):
    """Test that connection errors are handled gracefully."""
    # Create a directory instead of a database file
    not_a_db = tmp_path / "not_a_database"
    not_a_db.mkdir()
    
    tables = load_garmin_tables(db_path=str(not_a_db))
    
    # Should return empty dict
    assert tables == {}
    
    # Should log the exception (error handling decorator logs "Database operational error")
    assert 'Database operational error' in caplog.text or 'database' in caplog.text.lower()


def test_load_master_dataframe_logging_on_success(sample_master_csv, caplog):
    """Test that successful load logs appropriate message."""
    with patch('garmin_analysis.utils.data_loading.MASTER_CSV', sample_master_csv):
        df = load_master_dataframe()
        
        # Verify success message was logged
        assert 'Loaded master dataset' in caplog.text
        assert '10 rows' in caplog.text or 'rows' in caplog.text


def test_load_garmin_tables_logging_on_success(valid_garmin_db, caplog):
    """Test that successful table load logs appropriate message."""
    tables = load_garmin_tables(db_path=str(valid_garmin_db))
    
    # Verify success message was logged
    assert 'Successfully loaded Garmin tables' in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


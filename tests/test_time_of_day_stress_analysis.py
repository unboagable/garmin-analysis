"""
Tests for time-of-day stress analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import tempfile
import os
from pathlib import Path

from garmin_analysis.features.time_of_day_stress_analysis import (
    load_stress_data,
    calculate_hourly_stress_averages,
    calculate_hourly_stress_by_weekday,
    plot_hourly_stress_pattern,
    plot_stress_heatmap_by_weekday,
    print_stress_summary
)


@pytest.fixture
def temp_stress_db():
    """Create a temporary database with stress data for testing."""
    # Create temporary database
    temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(temp_fd)
    
    conn = sqlite3.connect(temp_path)
    cursor = conn.cursor()
    
    # Create stress table
    cursor.execute("""
        CREATE TABLE stress (
            timestamp DATETIME PRIMARY KEY,
            stress INTEGER NOT NULL
        )
    """)
    
    # Generate sample stress data for 7 days, every 3 minutes
    base_date = datetime(2024, 1, 1, 0, 0, 0)
    stress_data = []
    
    for day in range(7):
        for hour in range(24):
            for minute in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57]:
                timestamp = base_date + timedelta(days=day, hours=hour, minutes=minute)
                # Create a pattern: lower stress at night, higher during day
                base_stress = 30 if hour < 6 or hour > 22 else 50
                # Add some variation by hour
                hour_variation = (hour - 12) ** 2 / 10 if 9 <= hour <= 17 else 0
                # Add random noise
                noise = np.random.randint(-10, 10)
                stress_value = int(max(0, min(100, base_stress + hour_variation + noise)))
                
                stress_data.append((timestamp.strftime("%Y-%m-%d %H:%M:%S"), stress_value))
    
    cursor.executemany("INSERT INTO stress (timestamp, stress) VALUES (?, ?)", stress_data)
    conn.commit()
    conn.close()
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_stress_df():
    """Create a sample stress DataFrame for testing."""
    # Generate 24 hours of data
    base_date = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_date + timedelta(hours=h, minutes=m) 
                 for h in range(24) 
                 for m in range(0, 60, 3)]
    
    # Create stress pattern: low at night, higher during day
    stress_values = []
    for ts in timestamps:
        hour = ts.hour
        if hour < 6 or hour > 22:
            stress = np.random.randint(20, 40)
        elif 9 <= hour <= 17:
            stress = np.random.randint(40, 80)
        else:
            stress = np.random.randint(30, 60)
        stress_values.append(stress)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'stress': stress_values
    })
    
    return df


def test_load_stress_data_success(temp_stress_db):
    """Test successful loading of stress data."""
    df = load_stress_data(temp_stress_db)
    
    assert not df.empty
    assert 'timestamp' in df.columns
    assert 'stress' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
    assert len(df) > 0


def test_load_stress_data_missing_file():
    """Test loading stress data from non-existent file."""
    df = load_stress_data("/nonexistent/path/to/garmin.db")
    
    assert df.empty


def test_calculate_hourly_stress_averages(sample_stress_df):
    """Test calculation of hourly stress averages."""
    hourly_stats = calculate_hourly_stress_averages(sample_stress_df)
    
    assert not hourly_stats.empty
    assert len(hourly_stats) == 24  # One entry per hour
    assert all(col in hourly_stats.columns for col in 
              ['hour', 'mean', 'median', 'std', 'min', 'max', 'count', 'ci_lower', 'ci_upper'])
    assert hourly_stats['hour'].min() == 0
    assert hourly_stats['hour'].max() == 23


def test_calculate_hourly_stress_averages_empty_df():
    """Test hourly averages with empty DataFrame."""
    empty_df = pd.DataFrame()
    hourly_stats = calculate_hourly_stress_averages(empty_df)
    
    assert hourly_stats.empty


def test_calculate_hourly_stress_by_weekday(sample_stress_df):
    """Test calculation of hourly stress by weekday."""
    # Add more days to sample data for better weekday coverage
    multi_day_data = []
    for day in range(7):
        day_df = sample_stress_df.copy()
        day_df['timestamp'] = day_df['timestamp'] + timedelta(days=day)
        multi_day_data.append(day_df)
    
    multi_day_df = pd.concat(multi_day_data, ignore_index=True)
    
    hourly_weekday_stats = calculate_hourly_stress_by_weekday(multi_day_df)
    
    assert not hourly_weekday_stats.empty
    assert all(col in hourly_weekday_stats.columns for col in 
              ['hour', 'day_of_week', 'day_of_week_num', 'mean', 'std', 'count'])
    assert hourly_weekday_stats['hour'].min() == 0
    assert hourly_weekday_stats['hour'].max() == 23
    
    # Check that day_of_week is categorical with proper ordering
    assert pd.api.types.is_categorical_dtype(hourly_weekday_stats['day_of_week'])


def test_calculate_hourly_stress_by_weekday_empty_df():
    """Test weekday analysis with empty DataFrame."""
    empty_df = pd.DataFrame()
    hourly_weekday_stats = calculate_hourly_stress_by_weekday(empty_df)
    
    assert hourly_weekday_stats.empty


def test_plot_hourly_stress_pattern(sample_stress_df, tmp_path):
    """Test plotting hourly stress patterns."""
    hourly_stats = calculate_hourly_stress_averages(sample_stress_df)
    
    # Mock PLOTS_DIR to use temp directory
    import garmin_analysis.features.time_of_day_stress_analysis as analysis_module
    original_plots_dir = analysis_module.PLOTS_DIR
    analysis_module.PLOTS_DIR = tmp_path
    
    try:
        plot_files = plot_hourly_stress_pattern(
            hourly_stats, 
            save_plots=True, 
            show_plots=False
        )
        
        assert len(plot_files) == 2  # hourly_stress and hourly_stress_bars
        assert 'hourly_stress' in plot_files
        assert 'hourly_stress_bars' in plot_files
        
        # Check that files were created
        for filepath in plot_files.values():
            assert Path(filepath).exists()
            assert Path(filepath).suffix == '.png'
    
    finally:
        # Restore original PLOTS_DIR
        analysis_module.PLOTS_DIR = original_plots_dir


def test_plot_hourly_stress_pattern_empty_df():
    """Test plotting with empty DataFrame."""
    empty_stats = pd.DataFrame()
    plot_files = plot_hourly_stress_pattern(empty_stats, save_plots=False, show_plots=False)
    
    assert plot_files == {}


def test_plot_stress_heatmap_by_weekday(sample_stress_df, tmp_path):
    """Test plotting stress heatmap by weekday."""
    # Add more days for weekday coverage
    multi_day_data = []
    for day in range(7):
        day_df = sample_stress_df.copy()
        day_df['timestamp'] = day_df['timestamp'] + timedelta(days=day)
        multi_day_data.append(day_df)
    
    multi_day_df = pd.concat(multi_day_data, ignore_index=True)
    hourly_weekday_stats = calculate_hourly_stress_by_weekday(multi_day_df)
    
    # Mock PLOTS_DIR to use temp directory
    import garmin_analysis.features.time_of_day_stress_analysis as analysis_module
    original_plots_dir = analysis_module.PLOTS_DIR
    analysis_module.PLOTS_DIR = tmp_path
    
    try:
        plot_files = plot_stress_heatmap_by_weekday(
            hourly_weekday_stats,
            save_plots=True,
            show_plots=False
        )
        
        assert len(plot_files) == 2  # stress_heatmap and stress_weekday_lines
        assert 'stress_heatmap' in plot_files
        assert 'stress_weekday_lines' in plot_files
        
        # Check that files were created
        for filepath in plot_files.values():
            assert Path(filepath).exists()
            assert Path(filepath).suffix == '.png'
    
    finally:
        # Restore original PLOTS_DIR
        analysis_module.PLOTS_DIR = original_plots_dir


def test_plot_stress_heatmap_empty_df():
    """Test heatmap plotting with empty DataFrame."""
    empty_stats = pd.DataFrame()
    plot_files = plot_stress_heatmap_by_weekday(empty_stats, save_plots=False, show_plots=False)
    
    assert plot_files == {}


def test_print_stress_summary(sample_stress_df, caplog):
    """Test printing stress summary."""
    hourly_stats = calculate_hourly_stress_averages(sample_stress_df)
    
    # Add more days for weekday coverage
    multi_day_data = []
    for day in range(7):
        day_df = sample_stress_df.copy()
        day_df['timestamp'] = day_df['timestamp'] + timedelta(days=day)
        multi_day_data.append(day_df)
    
    multi_day_df = pd.concat(multi_day_data, ignore_index=True)
    hourly_weekday_stats = calculate_hourly_stress_by_weekday(multi_day_df)
    
    # Capture log output
    import logging
    with caplog.at_level(logging.INFO):
        print_stress_summary(hourly_stats, hourly_weekday_stats)
    
    # Check that summary was printed
    log_text = caplog.text
    assert "STRESS ANALYSIS BY TIME OF DAY" in log_text
    assert "Overall Stress Statistics" in log_text
    assert "Peak Stress Times" in log_text
    assert "Low Stress Times" in log_text
    assert "Hourly Stress Breakdown" in log_text
    assert "Time Period Analysis" in log_text
    assert "Average Stress by Day of Week" in log_text


def test_print_stress_summary_empty_df(caplog):
    """Test printing summary with empty DataFrame."""
    empty_stats = pd.DataFrame()
    
    import logging
    with caplog.at_level(logging.WARNING):
        print_stress_summary(empty_stats, None)
    
    assert "No data to summarize" in caplog.text


def test_full_analysis_integration(temp_stress_db, tmp_path):
    """Integration test for full analysis workflow."""
    # Load data
    df = load_stress_data(temp_stress_db)
    assert not df.empty
    
    # Calculate hourly averages
    hourly_stats = calculate_hourly_stress_averages(df)
    assert not hourly_stats.empty
    assert len(hourly_stats) == 24
    
    # Calculate weekday patterns
    hourly_weekday_stats = calculate_hourly_stress_by_weekday(df)
    assert not hourly_weekday_stats.empty
    
    # Mock PLOTS_DIR for plotting
    import garmin_analysis.features.time_of_day_stress_analysis as analysis_module
    original_plots_dir = analysis_module.PLOTS_DIR
    analysis_module.PLOTS_DIR = tmp_path
    
    try:
        # Create plots
        plot_files = plot_hourly_stress_pattern(hourly_stats, save_plots=True, show_plots=False)
        weekday_plots = plot_stress_heatmap_by_weekday(hourly_weekday_stats, save_plots=True, show_plots=False)
        
        all_plots = {**plot_files, **weekday_plots}
        assert len(all_plots) == 4  # 2 hourly plots + 2 weekday plots
        
        # Verify all plot files exist
        for filepath in all_plots.values():
            assert Path(filepath).exists()
    
    finally:
        # Restore original PLOTS_DIR
        analysis_module.PLOTS_DIR = original_plots_dir


def test_stress_pattern_validation(sample_stress_df):
    """Test that stress patterns follow expected trends."""
    hourly_stats = calculate_hourly_stress_averages(sample_stress_df)
    
    # Night hours (0-5, 23) should have lower average stress
    night_hours = hourly_stats[hourly_stats['hour'].isin([0, 1, 2, 3, 4, 5, 23])]
    night_avg = night_hours['mean'].mean()
    
    # Day hours (9-17) should have higher average stress
    day_hours = hourly_stats[hourly_stats['hour'].between(9, 17)]
    day_avg = day_hours['mean'].mean()
    
    # Day stress should be higher than night stress
    assert day_avg > night_avg, "Day stress should be higher than night stress"


@pytest.mark.integration
def test_real_database_integration():
    """Integration test with real database (if available)."""
    from garmin_analysis.config import DB_DIR
    
    db_path = DB_DIR / "garmin.db"
    
    if not db_path.exists():
        pytest.skip("Real database not available")
    
    # Load real data
    df = load_stress_data(str(db_path))
    
    if df.empty:
        pytest.skip("No stress data in database")
    
    # Calculate statistics
    hourly_stats = calculate_hourly_stress_averages(df)
    
    # Basic validation
    assert not hourly_stats.empty
    assert len(hourly_stats) <= 24
    assert all(hourly_stats['mean'] >= 0)
    assert all(hourly_stats['mean'] <= 100)


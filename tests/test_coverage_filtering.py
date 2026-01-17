"""
Test for 24-hour coverage filtering functionality
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from garmin_analysis.features.coverage import filter_by_24h_coverage, days_with_continuous_coverage


def test_filter_by_24h_coverage_with_empty_dataframe():
    """Test that empty dataframe is handled gracefully"""
    empty_df = pd.DataFrame()
    result = filter_by_24h_coverage(empty_df)
    assert result.empty


def test_filter_by_24h_coverage_with_no_stress_data():
    """Test that missing HR data is handled gracefully"""
    # Create a simple dataframe with day column
    df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=5),
        'steps': [1000, 2000, 3000, 4000, 5000]
    })
    
    # Test with empty HR DataFrame
    empty_hr_df = pd.DataFrame()
    result = filter_by_24h_coverage(df, hr_df=empty_hr_df)
    # Should return original dataframe when no HR data (cannot determine watch wear)
    assert len(result) == len(df)
    assert result.equals(df)


def test_filter_by_24h_coverage_with_stress_data():
    """Test filtering with pre-loaded HR data (heart rate indicates watch wear)"""
    # Create master dataframe
    master_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=3),
        'steps': [1000, 2000, 3000]
    })
    
    # Create HR data with continuous coverage for first day only (heart rate indicates watch wear)
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]  # 24 hours of 1-minute intervals
    
    # HR data requires heart_rate column with valid values (20-250 bpm)
    hr_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': [70] * len(timestamps)  # Valid HR values (20-250 bpm)
    })
    
    result = filter_by_24h_coverage(master_df, hr_df=hr_df)
    
    # Should only include the first day (2024-01-01) which has continuous coverage
    assert len(result) == 1
    assert result['day'].iloc[0] == pd.Timestamp('2024-01-01')
    assert result['steps'].iloc[0] == 1000


def test_days_with_continuous_coverage_with_full_day():
    """Test basic functionality of days_with_continuous_coverage"""
    # Create a simple stress dataframe with continuous coverage for one day
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]  # 24 hours of 1-minute intervals
    
    stress_df = pd.DataFrame({
        'timestamp': timestamps,
        'stress_level': [50] * len(timestamps)
    })
    
    qualifying_days = days_with_continuous_coverage(stress_df, timestamp_col='timestamp')
    
    # Should find one qualifying day
    assert len(qualifying_days) == 1
    assert qualifying_days[0] == pd.Timestamp('2024-01-01')


def test_days_with_continuous_coverage_with_gap():
    """Test that days with gaps are not included"""
    # Create stress data with a gap
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = []
    
    # First 12 hours
    for i in range(720):  # 12 hours
        timestamps.append(start_time + timedelta(minutes=i))
    
    # Gap of 1 hour
    # Then next 12 hours
    for i in range(720, 1440):  # remaining 12 hours
        timestamps.append(start_time + timedelta(minutes=i + 60))  # +60 minute gap
    
    stress_df = pd.DataFrame({
        'timestamp': timestamps,
        'stress_level': [50] * len(timestamps)
    })
    
    qualifying_days = days_with_continuous_coverage(stress_df, timestamp_col='timestamp')
    
    # Should not find any qualifying days due to the gap
    assert len(qualifying_days) == 0


def test_days_with_continuous_coverage_with_allowance_internal_gap():
    """A 30-minute internal gap should be allowed when allowance >= 30."""
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = []
    # First block: 8 hours, minutely
    for i in range(8 * 60):
        timestamps.append(start_time + pd.Timedelta(minutes=i))
    # Gap: 30 minutes
    # Second block resumes after 30 minute gap
    resume = start_time + pd.Timedelta(hours=8, minutes=30)
    for i in range(16 * 60 - 30):  # remaining minutes to fill until 24h
        timestamps.append(resume + pd.Timedelta(minutes=i))

    stress_df = pd.DataFrame({'timestamp': timestamps})

    days = days_with_continuous_coverage(
        stress_df,
        timestamp_col='timestamp',
        max_gap=pd.Timedelta(minutes=2),
        day_edge_tolerance=pd.Timedelta(minutes=2),
        total_missing_allowance=pd.Timedelta(minutes=30),
    )

    assert pd.Timestamp('2024-01-01') in days


def test_days_with_continuous_coverage_with_allowance_edge_deficit():
    """A late start and early end totalling 20 minutes should be allowed when allowance >= 20."""
    day = pd.Timestamp('2024-01-01')
    # Start 10 minutes late, end 10 minutes early, otherwise minutely
    ts = pd.date_range(day + pd.Timedelta(minutes=10), day + pd.Timedelta(days=1) - pd.Timedelta(minutes=10), freq="1min")
    stress_df = pd.DataFrame({'timestamp': ts})

    days = days_with_continuous_coverage(
        stress_df,
        timestamp_col='timestamp',
        max_gap=pd.Timedelta(minutes=2),
        day_edge_tolerance=pd.Timedelta(minutes=2),
        total_missing_allowance=pd.Timedelta(minutes=20),
    )

    assert pd.Timestamp('2024-01-01') in days


def test_filter_by_24h_coverage_with_total_allowance():
    """Master df is filtered when HR coverage meets total allowance but violates max_gap."""
    master_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=2),
        'steps': [1000, 2000]
    })

    # Build HR data with one 15-minute internal gap on the first day
    # HR data requires heart_rate column with valid values (20-250 bpm)
    day = pd.Timestamp('2024-01-01')
    ts1a = pd.date_range(day, day + pd.Timedelta(hours=6), freq='1min', inclusive='left')
    ts1b = pd.date_range(day + pd.Timedelta(hours=6, minutes=15), day + pd.Timedelta(days=1), freq='1min', inclusive='left')
    hr_timestamps = ts1a.append(ts1b)

    # Second day has no data; should not pass
    hr_df = pd.DataFrame({
        'timestamp': hr_timestamps,
        'heart_rate': [70] * len(hr_timestamps)  # Valid HR values (20-250 bpm)
    })

    filtered = filter_by_24h_coverage(
        master_df,
        hr_df=hr_df,
        max_gap=pd.Timedelta(minutes=2),
        day_edge_tolerance=pd.Timedelta(minutes=2),
        total_missing_allowance=pd.Timedelta(minutes=15),
    )

    assert len(filtered) == 1
    assert filtered['day'].iloc[0] == pd.Timestamp('2024-01-01')


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test for 24-hour coverage filtering functionality
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from garmin_analysis.features.coverage import filter_by_24h_coverage, days_with_continuous_coverage


def test_filter_by_24h_coverage_empty_dataframe():
    """Test that empty dataframe is handled gracefully"""
    empty_df = pd.DataFrame()
    result = filter_by_24h_coverage(empty_df)
    assert result.empty


def test_filter_by_24h_coverage_no_stress_data():
    """Test that missing stress data is handled gracefully"""
    # Create a simple dataframe with day column
    df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=5),
        'steps': [1000, 2000, 3000, 4000, 5000]
    })
    
    # Test with empty stress DataFrame
    empty_stress_df = pd.DataFrame()
    result = filter_by_24h_coverage(df, stress_df=empty_stress_df)
    # Should return original dataframe when no stress data
    assert len(result) == len(df)
    assert result.equals(df)


def test_filter_by_24h_coverage_with_stress_data():
    """Test filtering with pre-loaded stress data"""
    # Create master dataframe
    master_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=3),
        'steps': [1000, 2000, 3000]
    })
    
    # Create stress data with continuous coverage for first day only
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]  # 24 hours of 1-minute intervals
    
    stress_df = pd.DataFrame({
        'timestamp': timestamps,
        'stress_level': [50] * len(timestamps)
    })
    
    result = filter_by_24h_coverage(master_df, stress_df=stress_df)
    
    # Should only include the first day (2024-01-01) which has continuous coverage
    assert len(result) == 1
    assert result['day'].iloc[0] == pd.Timestamp('2024-01-01')
    assert result['steps'].iloc[0] == 1000


def test_days_with_continuous_coverage_basic():
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


if __name__ == "__main__":
    pytest.main([__file__])

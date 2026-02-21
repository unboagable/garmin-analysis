"""
Test for 24-hour coverage filtering functionality
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from garmin_analysis.features.coverage import filter_by_24h_coverage, days_with_continuous_coverage


class TestFilterBy24hCoverage:

    def test_empty_dataframe(self):
        """Test that empty dataframe is handled gracefully"""
        empty_df = pd.DataFrame()
        result = filter_by_24h_coverage(empty_df)
        assert result.empty

    def test_no_stress_data(self):
        """Test that missing HR data is handled gracefully"""
        df = pd.DataFrame({
            'day': pd.date_range('2024-01-01', periods=5),
            'steps': [1000, 2000, 3000, 4000, 5000]
        })
        
        empty_hr_df = pd.DataFrame()
        result = filter_by_24h_coverage(df, hr_df=empty_hr_df)
        assert len(result) == len(df)
        assert result.equals(df)

    def test_stress_data(self):
        """Test filtering with pre-loaded HR data (heart rate indicates watch wear)"""
        master_df = pd.DataFrame({
            'day': pd.date_range('2024-01-01', periods=3),
            'steps': [1000, 2000, 3000]
        })
        
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]
        
        hr_df = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': [70] * len(timestamps)
        })
        
        result = filter_by_24h_coverage(master_df, hr_df=hr_df)
        
        assert len(result) == 1
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01')
        assert result['steps'].iloc[0] == 1000

    def test_total_allowance(self):
        """Master df is filtered when HR coverage meets total allowance but violates max_gap."""
        master_df = pd.DataFrame({
            'day': pd.date_range('2024-01-01', periods=2),
            'steps': [1000, 2000]
        })

        day = pd.Timestamp('2024-01-01')
        ts1a = pd.date_range(day, day + pd.Timedelta(hours=6), freq='1min', inclusive='left')
        ts1b = pd.date_range(day + pd.Timedelta(hours=6, minutes=15), day + pd.Timedelta(days=1), freq='1min', inclusive='left')
        hr_timestamps = ts1a.append(ts1b)

        hr_df = pd.DataFrame({
            'timestamp': hr_timestamps,
            'heart_rate': [70] * len(hr_timestamps)
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


class TestDaysWithContinuousCoverage:

    def test_full_day(self):
        """Test basic functionality of days_with_continuous_coverage"""
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = [start_time + timedelta(minutes=i) for i in range(1440)]
        
        stress_df = pd.DataFrame({
            'timestamp': timestamps,
            'stress_level': [50] * len(timestamps)
        })
        
        qualifying_days = days_with_continuous_coverage(stress_df, timestamp_col='timestamp')
        
        assert len(qualifying_days) == 1
        assert qualifying_days[0] == pd.Timestamp('2024-01-01')

    def test_gap(self):
        """Test that days with gaps are not included"""
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = []
        
        for i in range(720):
            timestamps.append(start_time + timedelta(minutes=i))
        
        for i in range(720, 1440):
            timestamps.append(start_time + timedelta(minutes=i + 60))
        
        stress_df = pd.DataFrame({
            'timestamp': timestamps,
            'stress_level': [50] * len(timestamps)
        })
        
        qualifying_days = days_with_continuous_coverage(stress_df, timestamp_col='timestamp')
        
        assert len(qualifying_days) == 0

    def test_allowance_internal_gap(self):
        """A 30-minute internal gap should be allowed when allowance >= 30."""
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = []
        for i in range(8 * 60):
            timestamps.append(start_time + pd.Timedelta(minutes=i))
        resume = start_time + pd.Timedelta(hours=8, minutes=30)
        for i in range(16 * 60 - 30):
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

    def test_allowance_edge_deficit(self):
        """A late start and early end totalling 20 minutes should be allowed when allowance >= 20."""
        day = pd.Timestamp('2024-01-01')
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


if __name__ == "__main__":
    pytest.main([__file__])

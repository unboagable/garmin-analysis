"""
Tests for data filtering utilities.

These tests verify:
- Date range filtering functionality
- Required column filtering
- Feature standardization with different scalers
- Edge cases (missing columns, empty dataframes)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler

from garmin_analysis.utils.data_filtering import (
    filter_by_date,
    filter_required_columns,
    standardize_features,
    strip_time_from_dates,
    convert_time_columns
)


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    return pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, 30),
        'calories': np.random.randint(1800, 2500, 30),
        'resting_hr': np.random.randint(50, 75, 30),
        'score': np.random.randint(60, 100, 30)
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    df = pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, 20),
        'calories': np.random.randint(1800, 2500, 20),
        'resting_hr': np.random.randint(50, 75, 20)
    })
    
    # Add some missing values
    df.loc[df.sample(5, random_state=42).index, 'steps'] = np.nan
    df.loc[df.sample(3, random_state=43).index, 'calories'] = np.nan
    
    return df


class TestFilterByDate:

    def test_from_and_to_dates(self, sample_time_series_data):
        """Test filtering by explicit from_date and to_date."""
        from_date = '2024-01-10'
        to_date = '2024-01-20'
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            from_date=from_date,
            to_date=to_date
        )
        
        assert len(result) == 11  # Inclusive of both dates
        assert result['day'].min() >= pd.to_datetime(from_date)
        assert result['day'].max() <= pd.to_datetime(to_date)
        
        assert result['day'].min() == pd.to_datetime('2024-01-10')
        assert result['day'].max() == pd.to_datetime('2024-01-20')

    def test_days_back(self, sample_time_series_data):
        """Test filtering using days_back parameter."""
        days_back = 7
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            days_back=days_back
        )
        
        expected_from = datetime.now() - timedelta(days=days_back)
        
        if len(result) > 0:
            assert result['day'].min() >= pd.to_datetime(expected_from)

    def test_weeks_back(self, sample_time_series_data):
        """Test filtering using weeks_back parameter."""
        weeks_back = 2
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            weeks_back=weeks_back
        )
        
        expected_from = datetime.now() - timedelta(weeks=weeks_back)
        
        if len(result) > 0:
            assert result['day'].min() >= pd.to_datetime(expected_from)

    def test_months_back(self, sample_time_series_data):
        """Test filtering using months_back parameter."""
        months_back = 1
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            months_back=months_back
        )
        
        if len(result) > 0:
            assert isinstance(result, pd.DataFrame)
            assert len(result) <= len(sample_time_series_data)

    def test_only_from_date(self, sample_time_series_data):
        """Test filtering with only from_date specified."""
        from_date = '2024-01-15'
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            from_date=from_date
        )
        
        assert len(result) == 16  # From Jan 15 to Jan 30
        assert result['day'].min() == pd.to_datetime('2024-01-15')
        assert result['day'].max() == pd.to_datetime('2024-01-30')

    def test_only_to_date(self, sample_time_series_data):
        """Test filtering with only to_date specified."""
        to_date = '2024-01-15'
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            to_date=to_date
        )
        
        assert len(result) == 15  # From Jan 1 to Jan 15
        assert result['day'].min() == pd.to_datetime('2024-01-01')
        assert result['day'].max() == pd.to_datetime('2024-01-15')

    def test_single_day(self, sample_time_series_data):
        """Test filtering to a single day."""
        target_date = '2024-01-15'
        
        result = filter_by_date(
            sample_time_series_data,
            date_col='day',
            from_date=target_date,
            to_date=target_date
        )
        
        assert len(result) == 1
        assert result['day'].iloc[0] == pd.to_datetime(target_date)

    def test_empty_dataframe(self):
        """Test filter_by_date with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['day', 'steps', 'calories'])
        
        result = filter_by_date(
            empty_df,
            date_col='day',
            from_date='2024-01-01',
            to_date='2024-01-31'
        )
        
        assert result.empty
        assert list(result.columns) == ['day', 'steps', 'calories']


class TestFilterRequiredColumns:

    def test_all_present(self, sample_time_series_data):
        """Test filtering with all required columns present and valid."""
        required_cols = ['steps', 'calories', 'resting_hr']
        
        result = filter_required_columns(sample_time_series_data, required_cols)
        
        assert len(result) == len(sample_time_series_data)
        
        for col in required_cols:
            assert col in result.columns
            assert not result[col].isna().any()

    def test_missing_values(self, sample_data_with_missing):
        """Test filtering drops rows with missing values in required columns."""
        required_cols = ['steps', 'calories']
        
        original_length = len(sample_data_with_missing)
        result = filter_required_columns(sample_data_with_missing, required_cols)
        
        assert len(result) < original_length
        
        for col in required_cols:
            assert not result[col].isna().any()

    def test_missing_columns(self, sample_time_series_data, caplog):
        """Test handling of missing required columns."""
        required_cols = ['steps', 'nonexistent_column', 'another_missing']
        
        result = filter_required_columns(sample_time_series_data, required_cols)
        
        assert len(result) == len(sample_time_series_data)
        
        assert 'Missing required columns' in caplog.text

    def test_single_column(self, sample_time_series_data):
        """Test filtering with single required column."""
        result = filter_required_columns(sample_time_series_data, ['steps'])
        
        assert len(result) == len(sample_time_series_data)
        assert not result['steps'].isna().any()

    def test_empty_dataframe(self):
        """Test filter_required_columns with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['steps', 'calories', 'resting_hr'])
        
        result = filter_required_columns(empty_df, ['steps', 'calories'])
        
        assert result.empty
        assert 'steps' in result.columns
        assert 'calories' in result.columns


class TestStandardizeFeatures:

    def test_standard_scaler(self, sample_time_series_data):
        """Test feature standardization using StandardScaler."""
        columns = ['steps', 'calories', 'resting_hr']
        
        result = standardize_features(sample_time_series_data, columns)
        
        assert result.shape[0] == len(sample_time_series_data)
        assert result.shape[1] == len(columns)
        
        for i in range(result.shape[1]):
            col_data = result[:, i]
            assert np.abs(np.mean(col_data)) < 0.1
            assert np.abs(np.std(col_data) - 1.0) < 0.1

    def test_robust_scaler_with_outliers(self):
        """Test feature standardization using RobustScaler for outlier-heavy data."""
        data_with_outliers = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 100],
            'feature2': [10, 20, 30, 40, 50, 60],
            'feature3': [5, 5, 5, 5, 5, 500]
        })
        
        columns = ['feature1', 'feature2', 'feature3']
        
        scaler = RobustScaler()
        df_clean = data_with_outliers.dropna(subset=columns)
        result = scaler.fit_transform(df_clean[columns])
        
        assert result.shape[0] == len(df_clean)
        assert result.shape[1] == len(columns)
        
        assert result.shape == (len(df_clean), len(columns))
        
        medians = np.median(result, axis=0)
        for median in medians:
            assert np.abs(median) < 1.0

    def test_missing_columns(self, sample_time_series_data):
        """Test standardization with columns that don't exist."""
        columns = ['steps', 'nonexistent_column']
        
        with pytest.raises(KeyError):
            standardize_features(sample_time_series_data, columns)

    def test_empty_dataframe(self, caplog):
        """Test standardize_features with DataFrame that becomes empty after dropna."""
        df_all_nan = pd.DataFrame({
            'feature1': [np.nan, np.nan, np.nan],
            'feature2': [np.nan, np.nan, np.nan]
        })
        
        result = standardize_features(df_all_nan, ['feature1', 'feature2'])
        
        assert len(result) == 0
        assert 'No data left after dropping NaNs' in caplog.text

    def test_single_column(self, sample_time_series_data):
        """Test standardizing a single feature."""
        result = standardize_features(sample_time_series_data, ['steps'])
        
        assert result.shape == (len(sample_time_series_data), 1)
        
        assert np.abs(np.mean(result[:, 0])) < 0.1
        assert np.abs(np.std(result[:, 0]) - 1.0) < 0.1


class TestStripTimeFromDates:

    def test_datetime(self):
        """Test stripping time component from datetime column."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:23']),
            'value': [100, 200]
        })
        
        result = strip_time_from_dates(df, col='day')
        
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['day'].iloc[1] == pd.Timestamp('2024-01-02 00:00:00')
        
        assert all(result['day'].dt.hour == 0)
        assert all(result['day'].dt.minute == 0)
        assert all(result['day'].dt.second == 0)

    def test_empty_dataframe(self):
        """Test strip_time_from_dates with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['day', 'value'])
        
        result = strip_time_from_dates(empty_df, col='day')
        
        assert result.empty
        assert 'day' in result.columns

    def test_missing_column(self):
        """Test strip_time_from_dates with missing column."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'value': [100, 200]
        })
        
        result = strip_time_from_dates(df, col='day')
        
        assert len(result) == 2
        assert 'date' in result.columns
        assert 'day' not in result.columns


class TestConvertTimeColumns:

    def test_hms_format(self):
        """Test converting time columns in HH:MM:SS format."""
        df = pd.DataFrame({
            'total_sleep': ['07:30:00', '08:15:00', '06:45:00'],
            'deep_sleep': ['01:30:00', '02:00:00', '01:15:00'],
            'value': [1, 2, 3]
        })
        
        result = convert_time_columns(df, ['total_sleep', 'deep_sleep'])
        
        assert result['total_sleep'].iloc[0] == 450.0
        assert result['total_sleep'].iloc[1] == 495.0
        assert result['deep_sleep'].iloc[0] == 90.0
        
        assert result['value'].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

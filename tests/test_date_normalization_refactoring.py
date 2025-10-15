"""
Test date normalization function refactoring.

Tests verify:
1. New function strip_time_from_dates() works correctly
2. Old function normalize_dates() still works (with deprecation warning)
3. Both produce identical results
4. Documentation clarity improvements
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from garmin_analysis.utils.data_filtering import strip_time_from_dates, normalize_dates


class TestStripTimeFromDates:
    """Test the new strip_time_from_dates function."""
    
    def test_strips_time_component(self):
        """Test that time component is removed from datetime."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:00', '2024-01-03 23:59:59'])
        })
        
        result = strip_time_from_dates(df, col='day')
        
        # All times should be midnight (00:00:00)
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['day'].iloc[1] == pd.Timestamp('2024-01-02 00:00:00')
        assert result['day'].iloc[2] == pd.Timestamp('2024-01-03 00:00:00')
    
    def test_default_column_name(self):
        """Test that default column name is 'day'."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:00'])
        })
        
        # Should work without specifying col parameter
        result = strip_time_from_dates(df)
        
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
    
    def test_custom_column_name(self):
        """Test that custom column names work."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:00'])
        })
        
        result = strip_time_from_dates(df, col='timestamp')
        
        assert result['timestamp'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
    
    def test_already_normalized_dates(self):
        """Test that dates already at midnight remain unchanged."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-02 00:00:00'])
        })
        
        result = strip_time_from_dates(df)
        
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['day'].iloc[1] == pd.Timestamp('2024-01-02 00:00:00')
    
    def test_handles_string_dates(self):
        """Test conversion of string dates to datetime with stripped time."""
        df = pd.DataFrame({
            'day': ['2024-01-01 14:30:00', '2024-01-02 08:15:00']
        })
        
        result = strip_time_from_dates(df)
        
        # Should convert to datetime and strip time
        assert isinstance(result['day'].iloc[0], pd.Timestamp)
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
    
    def test_preserves_date_ordering(self):
        """Test that date ordering is preserved."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-03 10:00:00', '2024-01-01 20:00:00', '2024-01-02 15:00:00'])
        })
        
        result = strip_time_from_dates(df)
        
        # Order should be preserved (not sorted)
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-03 00:00:00')
        assert result['day'].iloc[1] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['day'].iloc[2] == pd.Timestamp('2024-01-02 00:00:00')
    
    def test_handles_nat_values(self):
        """Test that NaT (Not a Time) values are preserved."""
        df = pd.DataFrame({
            'day': [pd.Timestamp('2024-01-01 14:30:00'), pd.NaT, pd.Timestamp('2024-01-03 10:00:00')]
        })
        
        result = strip_time_from_dates(df)
        
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert pd.isna(result['day'].iloc[1])
        assert result['day'].iloc[2] == pd.Timestamp('2024-01-03 00:00:00')


class TestBackwardCompatibility:
    """Test that normalize_dates still works for backward compatibility."""
    
    def test_normalize_dates_deprecated_warning(self):
        """Test that normalize_dates issues deprecation warning."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00'])
        })
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_dates(df)
            
            # Should issue DeprecationWarning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "strip_time_from_dates" in str(w[0].message)
    
    def test_normalize_dates_still_works(self):
        """Test that normalize_dates still produces correct results."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:00'])
        })
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = normalize_dates(df)
        
        # Should produce same results as strip_time_from_dates
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
        assert result['day'].iloc[1] == pd.Timestamp('2024-01-02 00:00:00')
    
    def test_both_functions_produce_identical_results(self):
        """Test that both functions produce identical results."""
        df1 = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00', '2024-01-02 08:15:00', '2024-01-03 23:59:59'])
        })
        df2 = df1.copy()
        
        result1 = strip_time_from_dates(df1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result2 = normalize_dates(df2)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestFunctionClarityVsNormalizeDayColumn:
    """Test that clarifies difference from normalize_day_column."""
    
    def test_strip_time_is_simpler_single_purpose(self):
        """Demonstrate strip_time_from_dates is simpler and single-purpose."""
        # strip_time_from_dates: You specify the column, it strips time
        df = pd.DataFrame({
            'my_date_col': pd.to_datetime(['2024-01-01 14:30:00'])
        })
        
        result = strip_time_from_dates(df, col='my_date_col')
        assert result['my_date_col'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
    
    def test_normalize_day_column_is_more_flexible(self):
        """Show that normalize_day_column from data_processing is more flexible."""
        from garmin_analysis.utils.data_processing import normalize_day_column
        
        # normalize_day_column: Auto-detects different column names
        # Test with 'calendarDate' instead of 'day'
        df = pd.DataFrame({
            'calendarDate': ['2024-01-01', '2024-01-02']
        })
        
        result = normalize_day_column(df, source_name='test')
        
        # Should auto-detect and convert calendarDate -> day
        assert 'day' in result.columns
        assert isinstance(result['day'].iloc[0], pd.Timestamp)
    
    def test_use_cases_are_different(self):
        """Document the different use cases for each function."""
        from garmin_analysis.utils.data_processing import normalize_day_column
        
        # Use Case 1: Initial data loading (unknown column names)
        # Use normalize_day_column() - it detects day/calendarDate/timestamp
        df_loading = pd.DataFrame({
            'calendarDate': ['2024-01-01 10:00:00']
        })
        result_loading = normalize_day_column(df_loading, 'source1')
        assert 'day' in result_loading.columns
        
        # Use Case 2: Post-processing (known column, strip time)
        # Use strip_time_from_dates() - simpler, targeted
        df_processing = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00'])
        })
        result_processing = strip_time_from_dates(df_processing)
        assert result_processing['day'].iloc[0].hour == 0  # Time stripped


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = strip_time_from_dates(df)
        assert result.empty
    
    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({
            'day': pd.to_datetime(['2024-01-01 14:30:00'])
        })
        result = strip_time_from_dates(df)
        assert len(result) == 1
        assert result['day'].iloc[0] == pd.Timestamp('2024-01-01 00:00:00')
    
    def test_large_dataframe(self):
        """Test with large DataFrame."""
        dates = pd.date_range('2020-01-01 12:00:00', periods=10000, freq='H')
        df = pd.DataFrame({'day': dates})
        
        result = strip_time_from_dates(df)
        
        # All should be normalized to midnight
        assert all(result['day'].dt.hour == 0)
        assert all(result['day'].dt.minute == 0)
        assert all(result['day'].dt.second == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


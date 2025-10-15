"""
Test to verify time conversion refactoring works correctly.

This test verifies that convert_time_columns now properly uses
convert_time_to_minutes from data_processing.py
"""

import pytest
import pandas as pd
import numpy as np
from garmin_analysis.utils.data_filtering import convert_time_columns
from garmin_analysis.utils.data_processing import convert_time_to_minutes


class TestTimeConversionRefactoring:
    """Test that convert_time_columns uses convert_time_to_minutes correctly."""
    
    def test_convert_time_columns_hms_format(self):
        """Test conversion of HH:MM:SS format."""
        df = pd.DataFrame({
            'sleep_time': ['1:30:00', '2:15:00', '0:45:30'],
            'awake_time': ['0:30:00', '0:15:00', '1:00:00']
        })
        
        result = convert_time_columns(df, ['sleep_time', 'awake_time'])
        
        # 1:30:00 = 90 minutes
        assert result['sleep_time'].iloc[0] == 90.0
        # 2:15:00 = 135 minutes
        assert result['sleep_time'].iloc[1] == 135.0
        # 0:45:30 = 45.5 minutes
        assert result['sleep_time'].iloc[2] == 45.5
    
    def test_convert_time_columns_ms_format(self):
        """Test conversion of MM:SS format (handled by convert_time_to_minutes)."""
        df = pd.DataFrame({
            'duration': ['45:30', '30:00', '15:45']
        })
        
        result = convert_time_columns(df, ['duration'])
        
        # 45:30 = 45.5 minutes
        assert result['duration'].iloc[0] == 45.5
        # 30:00 = 30 minutes
        assert result['duration'].iloc[1] == 30.0
        # 15:45 = 15.75 minutes
        assert result['duration'].iloc[2] == 15.75
    
    def test_convert_time_columns_handles_invalid(self):
        """Test that invalid values are handled gracefully."""
        df = pd.DataFrame({
            'time': ['1:30:00', 'invalid', '', None, '2:00:00']
        })
        
        result = convert_time_columns(df, ['time'])
        
        assert result['time'].iloc[0] == 90.0
        assert pd.isna(result['time'].iloc[1])  # 'invalid' -> NaN
        assert pd.isna(result['time'].iloc[2])  # '' -> NaN
        assert pd.isna(result['time'].iloc[3])  # None -> NaN
        assert result['time'].iloc[4] == 120.0
    
    def test_convert_time_columns_only_object_dtype(self):
        """Test that only object dtype columns are converted."""
        df = pd.DataFrame({
            'time_str': ['1:30:00', '2:00:00'],
            'numeric': [90, 120],
            'other': [1.5, 2.0]
        })
        
        result = convert_time_columns(df, ['time_str', 'numeric', 'other'])
        
        # Only time_str should be converted (it's object dtype)
        assert result['time_str'].iloc[0] == 90.0
        # numeric and other should remain unchanged
        assert result['numeric'].iloc[0] == 90
        assert result['other'].iloc[0] == 1.5
    
    def test_convert_time_columns_nonexistent_columns(self):
        """Test that nonexistent columns are handled gracefully."""
        df = pd.DataFrame({
            'time': ['1:30:00', '2:00:00']
        })
        
        # Should not raise error for nonexistent column
        result = convert_time_columns(df, ['time', 'nonexistent'])
        
        assert result['time'].iloc[0] == 90.0
        assert 'nonexistent' not in result.columns
    
    def test_convert_time_columns_multiple_columns(self):
        """Test converting multiple columns at once."""
        df = pd.DataFrame({
            'sleep': ['8:30:00', '7:45:00'],
            'deep_sleep': ['2:15:00', '1:30:00'],
            'rem_sleep': ['1:45:00', '2:00:00']
        })
        
        result = convert_time_columns(df, ['sleep', 'deep_sleep', 'rem_sleep'])
        
        assert result['sleep'].iloc[0] == 510.0  # 8:30:00
        assert result['deep_sleep'].iloc[0] == 135.0  # 2:15:00
        assert result['rem_sleep'].iloc[0] == 105.0  # 1:45:00
    
    def test_uses_same_logic_as_convert_time_to_minutes(self):
        """Verify that convert_time_columns uses convert_time_to_minutes."""
        test_values = ['1:30:00', '45:30', 'invalid', '2:15:00']
        
        # Test with convert_time_to_minutes directly
        direct_results = [convert_time_to_minutes(val) for val in test_values]
        
        # Test with convert_time_columns
        df = pd.DataFrame({'time': test_values})
        df_result = convert_time_columns(df, ['time'])
        
        # Results should be identical
        for i, direct_result in enumerate(direct_results):
            if pd.isna(direct_result):
                assert pd.isna(df_result['time'].iloc[i])
            else:
                assert df_result['time'].iloc[i] == direct_result
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame is handled."""
        df = pd.DataFrame()
        result = convert_time_columns(df, ['time'])
        assert result.empty
    
    def test_empty_columns_list(self):
        """Test with empty columns list."""
        df = pd.DataFrame({'time': ['1:30:00']})
        result = convert_time_columns(df, [])
        # Should return unchanged
        assert result['time'].iloc[0] == '1:30:00'


class TestRefactoringBenefits:
    """Test the benefits of using the centralized function."""
    
    def test_handles_all_formats_from_convert_time_to_minutes(self):
        """Test that all formats supported by convert_time_to_minutes work."""
        df = pd.DataFrame({
            'col': [
                '1:30:00',  # HH:MM:SS
                '45:30',    # MM:SS
                '90',       # Direct number as string
            ]
        })
        
        result = convert_time_columns(df, ['col'])
        
        assert result['col'].iloc[0] == 90.0
        assert result['col'].iloc[1] == 45.5
        assert result['col'].iloc[2] == 90.0
    
    def test_consistent_error_handling(self):
        """Test that error handling is consistent with convert_time_to_minutes."""
        # Both should handle the same errors the same way
        error_values = ['', 'invalid', 'abc:def:ghi', '::']
        
        df = pd.DataFrame({'time': error_values})
        result = convert_time_columns(df, ['time'])
        
        # All should be NaN
        assert result['time'].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


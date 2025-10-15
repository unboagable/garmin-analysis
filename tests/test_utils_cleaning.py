"""
Unit tests for utils_cleaning.py

Tests the clean_data() function with various scenarios including:
- Placeholder replacement
- Data type conversion
- Outlier removal (optional)
- Column name normalization
"""

import pytest
import pandas as pd
import numpy as np
from garmin_analysis.utils_cleaning import clean_data


class TestCleanDataBasics:
    """Test basic cleaning functionality."""
    
    def test_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        df = pd.DataFrame()
        result = clean_data(df)
        assert result.empty
        assert result.shape == (0, 0)
    
    def test_preserves_original_dataframe(self):
        """Test that original DataFrame is not modified (copy is made)."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        original_id = id(df)
        result = clean_data(df)
        assert id(result) != original_id
        assert df.equals(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
    
    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = clean_data(df)
        assert isinstance(result, pd.DataFrame)


class TestPlaceholderReplacement:
    """Test placeholder value replacement with NaN."""
    
    def test_replaces_empty_strings(self):
        """Test that empty strings are replaced with NaN."""
        df = pd.DataFrame({'col': ['', 'value', '']})
        result = clean_data(df)
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 'value'
        assert pd.isna(result['col'].iloc[2])
    
    def test_replaces_na_strings(self):
        """Test that 'NA' strings are replaced with NaN."""
        df = pd.DataFrame({'col': ['NA', 'value', 'NA']})
        result = clean_data(df)
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 'value'
        assert pd.isna(result['col'].iloc[2])
    
    def test_replaces_null_strings(self):
        """Test that 'null' strings are replaced with NaN."""
        df = pd.DataFrame({'col': ['null', 'value', 'null']})
        result = clean_data(df)
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 'value'
    
    def test_replaces_none_strings(self):
        """Test that 'None' strings are replaced with NaN."""
        df = pd.DataFrame({'col': ['None', 'value', 'None']})
        result = clean_data(df)
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 'value'
    
    def test_replaces_negative_one(self):
        """Test that -1 is replaced with NaN in numeric columns."""
        df = pd.DataFrame({'col': [-1, 5, -1, 10]})
        result = clean_data(df)
        assert pd.isna(result['col'].iloc[0])
        assert result['col'].iloc[1] == 5
        assert pd.isna(result['col'].iloc[2])
        assert result['col'].iloc[3] == 10
    
    def test_mixed_placeholders(self):
        """Test DataFrame with multiple placeholder types."""
        df = pd.DataFrame({
            'a': ['', 'NA', 'null', 'value'],
            'b': [-1, 5, -1, 10],
            'c': ['None', '', 'test', 'NA']
        })
        result = clean_data(df)
        assert pd.isna(result['a'].iloc[0])
        assert pd.isna(result['a'].iloc[1])
        assert pd.isna(result['a'].iloc[2])
        assert result['a'].iloc[3] == 'value'
        assert pd.isna(result['b'].iloc[0])
        assert pd.isna(result['b'].iloc[2])


class TestDataTypeConversion:
    """Test numeric data type standardization."""
    
    def test_converts_integers_to_int64(self):
        """Test that integer columns are converted to nullable Int64."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        result = clean_data(df)
        assert result['col'].dtype == 'Int64'
    
    def test_converts_floats_to_float32(self):
        """Test that float columns are converted to float32."""
        df = pd.DataFrame({'col': [1.5, 2.7, 3.2, 4.8]})
        result = clean_data(df)
        assert result['col'].dtype == 'float32'
    
    def test_preserves_string_columns(self):
        """Test that string columns remain unchanged."""
        df = pd.DataFrame({'col': ['a', 'b', 'c']})
        result = clean_data(df)
        assert result['col'].dtype == object
    
    def test_handles_mixed_types(self):
        """Test DataFrame with multiple data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        result = clean_data(df)
        assert result['int_col'].dtype == 'Int64'
        assert result['float_col'].dtype == 'float32'
        assert result['str_col'].dtype == object
    
    def test_handles_integer_like_floats(self):
        """Test that floats with only integer values are converted to Int64."""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})
        result = clean_data(df)
        assert result['col'].dtype == 'Int64'
    
    def test_handles_empty_numeric_columns(self):
        """Test that empty numeric columns don't cause errors."""
        df = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        result = clean_data(df)
        # Should handle gracefully without errors
        assert len(result) == 3


class TestOutlierRemoval:
    """Test optional outlier removal using IQR method."""
    
    def test_preserves_data_by_default(self):
        """Test that outliers are preserved by default (remove_outliers=False)."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5, 100]})  # 100 is an outlier
        result = clean_data(df)
        assert len(result) == 6  # All rows preserved
        assert 100 in result['col'].values
    
    def test_removes_outliers_when_enabled(self):
        """Test that outliers are removed when remove_outliers=True."""
        # Create data with clear outlier
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5, 100]})  # 100 is an outlier
        result = clean_data(df, remove_outliers=True)
        assert len(result) < 6  # Some rows removed
        assert 100 not in result['col'].values
    
    def test_outlier_removal_with_iqr(self):
        """Test IQR-based outlier detection (1.5 * IQR rule)."""
        # Data: [10, 12, 14, 16, 18, 50]
        # Q1 = 12, Q3 = 18, IQR = 6
        # Lower bound = 12 - 1.5*6 = 3
        # Upper bound = 18 + 1.5*6 = 27
        # 50 should be removed
        df = pd.DataFrame({'col': [10, 12, 14, 16, 18, 50]})
        result = clean_data(df, remove_outliers=True)
        assert 50 not in result['col'].values
        assert 10 in result['col'].values
    
    def test_preserves_nan_during_outlier_removal(self):
        """Test that NaN values are preserved during outlier removal."""
        df = pd.DataFrame({'col': [1, 2, np.nan, 3, 4, 100]})
        result = clean_data(df, remove_outliers=True)
        assert result['col'].isna().any()  # NaN still present
    
    def test_outlier_removal_multiple_columns(self):
        """Test outlier removal across multiple columns."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'b': [10, 20, 30, 40, 50, 60]  # No outliers
        })
        result = clean_data(df, remove_outliers=True)
        assert 100 not in result['a'].values
        # b column should still have all values
        assert len(result['b'].dropna()) > 0
    
    def test_no_outliers_detected(self):
        """Test when no outliers are present."""
        df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})  # Normal distribution
        result = clean_data(df, remove_outliers=True)
        assert len(result) == 5  # All rows preserved


class TestColumnNameNormalization:
    """Test column name cleaning and standardization."""
    
    def test_lowercases_column_names(self):
        """Test that column names are converted to lowercase."""
        df = pd.DataFrame({'ColA': [1, 2], 'ColB': [3, 4]})
        result = clean_data(df)
        assert 'cola' in result.columns
        assert 'colb' in result.columns
        assert 'ColA' not in result.columns
    
    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        df = pd.DataFrame({' col1 ': [1, 2], '  col2': [3, 4]})
        result = clean_data(df)
        assert 'col1' in result.columns
        assert 'col2' in result.columns
    
    def test_replaces_spaces_with_underscores(self):
        """Test that spaces in column names are replaced with underscores."""
        df = pd.DataFrame({'col a': [1, 2], 'col b': [3, 4]})
        result = clean_data(df)
        assert 'col_a' in result.columns
        assert 'col_b' in result.columns
    
    def test_combined_normalization(self):
        """Test combined lowercasing, stripping, and space replacement."""
        df = pd.DataFrame({' Col Name ': [1, 2], 'Another Col': [3, 4]})
        result = clean_data(df)
        assert 'col_name' in result.columns
        assert 'another_col' in result.columns
    
    def test_already_normalized_columns(self):
        """Test that already-normalized columns remain unchanged."""
        df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})
        result = clean_data(df)
        assert 'col_a' in result.columns
        assert 'col_b' in result.columns


class TestIntegration:
    """Integration tests combining multiple cleaning operations."""
    
    def test_full_cleaning_pipeline(self):
        """Test complete cleaning pipeline with all operations."""
        df = pd.DataFrame({
            ' Col A ': [1, '', 3, -1, 5],
            'Col B': [1.5, 2.5, 'NA', 4.5, 5.5],
            'Col_C': [10, 20, 30, 40, 500]  # 500 is outlier
        })
        
        result = clean_data(df, remove_outliers=False)
        
        # Check column names normalized
        assert 'col_a' in result.columns
        assert 'col_b' in result.columns
        assert 'col_c' in result.columns
        
        # Check placeholders replaced
        assert pd.isna(result['col_a'].iloc[1])
        assert pd.isna(result['col_a'].iloc[3])
        
        # Check data types
        assert result['col_a'].dtype == 'Int64'
        assert result['col_c'].dtype == 'Int64'
        
        # Check outliers preserved (default)
        assert 500 in result['col_c'].values
    
    def test_health_data_scenario(self):
        """Test realistic health data cleaning scenario."""
        df = pd.DataFrame({
            'Day': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Steps': [8000, -1, 12000],  # -1 is placeholder
            'HR Max': [165, 180, 200],  # 200 might be extreme but valid
            'Sleep Score': [75.5, '', 88.2],  # Empty string placeholder
        })
        
        result = clean_data(df)
        
        # Steps placeholder replaced
        assert pd.isna(result['steps'].iloc[1])
        
        # HR Max preserved (important for anomaly detection)
        assert 200 in result['hr_max'].values
        
        # Sleep score placeholder replaced
        assert pd.isna(result['sleep_score'].iloc[1])
        
        # Column names normalized
        assert 'day' in result.columns
        assert 'steps' in result.columns
        assert 'hr_max' in result.columns
    
    def test_preserves_data_for_modeling(self):
        """Test that data is preserved for ML modeling use cases."""
        # Simulate fitness data with legitimate extremes
        df = pd.DataFrame({
            'hr_min': [45, 48, 42, 50, 47],  # Normal resting HR
            'hr_max': [155, 165, 185, 160, 162],  # 185 during sprint is valid
            'steps': [8000, 12000, 3000, 9500, 11000],  # 3000 on recovery day
        })
        
        result = clean_data(df)
        
        # All data preserved
        assert len(result) == 5
        assert 185 in result['hr_max'].values  # Extreme but valid
        assert 3000 in result['steps'].values  # Recovery day preserved


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_nan_column(self):
        """Test column with all NaN values."""
        df = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        result = clean_data(df)
        assert result['col'].isna().all()
    
    def test_single_row_dataframe(self):
        """Test DataFrame with only one row."""
        df = pd.DataFrame({'col': [42]})
        result = clean_data(df)
        assert len(result) == 1
        assert result['col'].iloc[0] == 42
    
    def test_single_column_dataframe(self):
        """Test DataFrame with only one column."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = clean_data(df)
        assert len(result.columns) == 1
        assert 'col' in result.columns
    
    def test_very_large_dataframe(self):
        """Test that function handles larger DataFrames efficiently."""
        df = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.randn(10000),
            'col3': ['value'] * 10000
        })
        result = clean_data(df)
        assert len(result) == 10000
    
    def test_special_characters_in_column_names(self):
        """Test column names with special characters."""
        df = pd.DataFrame({'col@1': [1, 2], 'col#2': [3, 4]})
        result = clean_data(df)
        # Should handle without errors
        assert len(result.columns) == 2
    
    def test_unicode_in_data(self):
        """Test that Unicode characters are preserved."""
        df = pd.DataFrame({'col': ['测试', 'тест', '試験']})
        result = clean_data(df)
        assert '测试' in result['col'].values
        assert 'тест' in result['col'].values


class TestLogging:
    """Test that function logs appropriate information."""
    
    def test_logs_cleaning_operations(self, caplog):
        """Test that cleaning operations are logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        df = pd.DataFrame({'Col A': [1, '', 3]})
        result = clean_data(df)
        
        # Check that logging occurred
        assert len(caplog.records) > 0
        
        # Should log start, placeholders, types, names, completion
        log_messages = [record.message for record in caplog.records]
        assert any('Starting data cleaning' in msg for msg in log_messages)
        assert any('Data cleaning complete' in msg for msg in log_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


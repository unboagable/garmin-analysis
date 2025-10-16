"""
Unit tests for shared imputation utility.

Tests cover:
- All imputation strategies (median, mean, drop, forward_fill, backward_fill, none)
- Error handling
- Missing value summary
- Strategy recommendations
"""

import pytest
import pandas as pd
import numpy as np
from garmin_analysis.utils.imputation import (
    impute_missing_values,
    get_missing_value_summary,
    recommend_imputation_strategy,
    impute_median,
    impute_mean,
    drop_missing
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame with missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        'a': [1.0, np.nan, 3.0, 4.0, 5.0],
        'b': [10.0, 20.0, np.nan, 40.0, 50.0],
        'c': [100.0, 200.0, 300.0, 400.0, 500.0],  # No missing
    })


@pytest.fixture
def df_with_outliers():
    """Create DataFrame with outliers for testing strategy recommendations."""
    return pd.DataFrame({
        'normal': [10, 12, 11, 13, 12, 10, 11],
        'with_outlier': [10, 12, 11, 100, 12, 10, 11],  # 100 is outlier
    })


class TestImputeMissingValues:
    """Test impute_missing_values function."""
    
    def test_median_imputation(self, sample_df):
        """Test median imputation strategy."""
        result = impute_missing_values(sample_df, ['a', 'b'], strategy='median')
        
        # Should have no NaN values
        assert not result['a'].isna().any()
        assert not result['b'].isna().any()
        
        # Should fill with median
        assert result.loc[1, 'a'] == 3.5  # median of [1, 3, 4, 5]
        assert result.loc[2, 'b'] == 30.0  # median of [10, 20, 40, 50]
    
    def test_mean_imputation(self, sample_df):
        """Test mean imputation strategy."""
        result = impute_missing_values(sample_df, ['a', 'b'], strategy='mean')
        
        # Should have no NaN values
        assert not result['a'].isna().any()
        assert not result['b'].isna().any()
        
        # Should fill with mean
        assert result.loc[1, 'a'] == 3.25  # mean of [1, 3, 4, 5]
        assert result.loc[2, 'b'] == 30.0  # mean of [10, 20, 40, 50]
    
    def test_drop_strategy(self, sample_df):
        """Test drop rows strategy."""
        result = impute_missing_values(sample_df, ['a', 'b'], strategy='drop')
        
        # Should have dropped rows with NaN
        assert len(result) == 3  # Started with 5, dropped 2
        assert not result['a'].isna().any()
        assert not result['b'].isna().any()
    
    def test_forward_fill_strategy(self, sample_df):
        """Test forward fill strategy."""
        result = impute_missing_values(sample_df, ['a'], strategy='forward_fill')
        
        # Should forward fill
        assert result.loc[1, 'a'] == 1.0  # Filled from previous value
    
    def test_backward_fill_strategy(self, sample_df):
        """Test backward fill strategy."""
        result = impute_missing_values(sample_df, ['a'], strategy='backward_fill')
        
        # Should backward fill
        assert result.loc[1, 'a'] == 3.0  # Filled from next value
    
    def test_none_strategy(self, sample_df):
        """Test no imputation strategy."""
        result = impute_missing_values(sample_df, ['a', 'b'], strategy='none')
        
        # Should keep NaN values
        assert result['a'].isna().sum() == 1
        assert result['b'].isna().sum() == 1
    
    def test_copy_parameter(self, sample_df):
        """Test that copy parameter works correctly."""
        # With copy=True (default)
        result = impute_missing_values(sample_df, ['a'], strategy='median', copy=True)
        assert sample_df['a'].isna().sum() == 1  # Original unchanged
        
        # With copy=False
        df_copy = sample_df.copy()
        result = impute_missing_values(df_copy, ['a'], strategy='median', copy=False)
        assert result is df_copy  # Same object
    
    def test_invalid_strategy(self, sample_df):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid imputation strategy"):
            impute_missing_values(sample_df, ['a'], strategy='invalid')
    
    def test_missing_columns(self, sample_df):
        """Test that missing columns raise ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            impute_missing_values(sample_df, ['nonexistent'], strategy='median')
    
    def test_empty_columns_list(self, sample_df):
        """Test that empty columns list raises ValueError."""
        with pytest.raises(ValueError, match="columns list cannot be empty"):
            impute_missing_values(sample_df, [], strategy='median')
    
    def test_invalid_input_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            impute_missing_values([1, 2, 3], ['a'], strategy='median')


class TestGetMissingValueSummary:
    """Test get_missing_value_summary function."""
    
    def test_summary_generation(self, sample_df):
        """Test that summary is generated correctly."""
        summary = get_missing_value_summary(sample_df, ['a', 'b', 'c'])
        
        assert len(summary) == 3
        assert 'column' in summary.columns
        assert 'missing_count' in summary.columns
        assert 'missing_pct' in summary.columns
        assert 'has_missing' in summary.columns
    
    def test_summary_values(self, sample_df):
        """Test that summary values are correct."""
        summary = get_missing_value_summary(sample_df, ['a', 'b', 'c'])
        
        # Check column a
        a_row = summary[summary['column'] == 'a'].iloc[0]
        assert a_row['missing_count'] == 1
        assert a_row['missing_pct'] == 20.0  # 1/5 * 100
        assert a_row['has_missing'] == True
        
        # Check column c (no missing)
        c_row = summary[summary['column'] == 'c'].iloc[0]
        assert c_row['missing_count'] == 0
        assert c_row['missing_pct'] == 0.0
        assert c_row['has_missing'] == False
    
    def test_summary_sorted_by_missing_pct(self, sample_df):
        """Test that summary is sorted by missing percentage."""
        summary = get_missing_value_summary(sample_df, ['a', 'b', 'c'])
        
        # Should be sorted descending by missing_pct
        assert summary['missing_pct'].is_monotonic_decreasing
    
    def test_summary_default_columns(self, sample_df):
        """Test that default columns are numeric columns."""
        summary = get_missing_value_summary(sample_df)
        
        # Should include all numeric columns
        assert len(summary) == 3


class TestRecommendImputationStrategy:
    """Test recommend_imputation_strategy function."""
    
    def test_recommend_for_normal_data(self):
        """Test recommendation for normally distributed data."""
        df = pd.DataFrame({
            'normal': [10, 12, 11, 13, 12, 10, 11]
        })
        
        recommendations = recommend_imputation_strategy(df, ['normal'])
        
        assert 'normal' in recommendations
        assert recommendations['normal']['strategy'] in ['mean', 'median']
    
    def test_recommend_for_outliers(self, df_with_outliers):
        """Test recommendation for data with outliers."""
        recommendations = recommend_imputation_strategy(df_with_outliers, ['with_outlier'])
        
        assert 'with_outlier' in recommendations
        assert recommendations['with_outlier']['strategy'] == 'median'
        assert 'outliers' in recommendations['with_outlier']['reason'].lower()
    
    def test_recommend_for_empty_column(self):
        """Test recommendation for column with no non-null values."""
        df = pd.DataFrame({
            'empty': [np.nan, np.nan, np.nan]
        })
        
        recommendations = recommend_imputation_strategy(df, ['empty'])
        
        assert 'empty' in recommendations
        assert recommendations['empty']['strategy'] == 'drop'


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_impute_median(self, sample_df):
        """Test impute_median convenience function."""
        result = impute_median(sample_df, ['a', 'b'])
        
        assert not result['a'].isna().any()
        assert not result['b'].isna().any()
    
    def test_impute_mean(self, sample_df):
        """Test impute_mean convenience function."""
        result = impute_mean(sample_df, ['a', 'b'])
        
        assert not result['a'].isna().any()
        assert not result['b'].isna().any()
    
    def test_drop_missing(self, sample_df):
        """Test drop_missing convenience function."""
        result = drop_missing(sample_df, ['a', 'b'])
        
        assert len(result) == 3  # Dropped 2 rows


class TestEdgeCases:
    """Test edge cases."""
    
    def test_all_missing(self):
        """Test with column that's all missing."""
        df = pd.DataFrame({
            'all_nan': [np.nan, np.nan, np.nan]
        })
        
        # Median of all NaN should be NaN (suppress expected RuntimeWarning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = impute_missing_values(df, ['all_nan'], strategy='median')
        assert result['all_nan'].isna().all()
    
    def test_no_missing(self, sample_df):
        """Test with column that has no missing values."""
        result = impute_missing_values(sample_df, ['c'], strategy='median')
        
        # Should be unchanged
        assert result['c'].equals(sample_df['c'])
    
    def test_single_value(self):
        """Test with single non-NaN value."""
        df = pd.DataFrame({
            'single': [1.0, np.nan, np.nan]
        })
        
        result = impute_missing_values(df, ['single'], strategy='median')
        
        # Median of single value is that value
        assert result['single'].iloc[1] == 1.0
        assert result['single'].iloc[2] == 1.0
    
    def test_large_dataset(self):
        """Test with larger dataset for performance."""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'col1': np.random.randn(n),
            'col2': np.random.randn(n),
        })
        
        # Introduce 10% missing values
        df.loc[np.random.choice(n, size=int(n*0.1), replace=False), 'col1'] = np.nan
        df.loc[np.random.choice(n, size=int(n*0.1), replace=False), 'col2'] = np.nan
        
        result = impute_missing_values(df, ['col1', 'col2'], strategy='median')
        
        # Should have no missing values
        assert not result.isna().any().any()


class TestIntegrationWithRealData:
    """Test with realistic health data patterns."""
    
    def test_heart_rate_data(self):
        """Test with heart rate-like data (outliers common)."""
        np.random.seed(42)
        df = pd.DataFrame({
            'rhr': [55, 58, np.nan, 54, 200, 56, 57],  # 200 is outlier
        })
        
        # Median should handle outlier better than mean
        median_result = impute_missing_values(df, ['rhr'], strategy='median')
        mean_result = impute_missing_values(df, ['rhr'], strategy='mean')
        
        # Median should be closer to typical values
        median_val = median_result.loc[2, 'rhr']
        mean_val = mean_result.loc[2, 'rhr']
        
        assert abs(median_val - 56) < abs(mean_val - 56)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


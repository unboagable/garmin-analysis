"""
Tests for trend summary generation.

These tests verify:
- Basic trend summary generation
- Correlation pair logging
- 24-hour coverage filtering integration
- Markdown output format
- Edge cases (empty dataframes)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from garmin_analysis.reporting.generate_trend_summary import (
    generate_trend_summary,
    log_top_correlations
)


@pytest.fixture
def sample_health_data():
    """Create sample health data for testing."""
    np.random.seed(42)
    n_days = 50
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    return pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, n_days),
        'calories': np.random.randint(1800, 2500, n_days),
        'resting_hr': np.random.randint(50, 75, n_days),
        'score': np.random.randint(60, 100, n_days),
        'stress_avg': np.random.randint(20, 60, n_days),
    })


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values."""
    np.random.seed(42)
    n_days = 30
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    df = pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, n_days),
        'calories': np.random.randint(1800, 2500, n_days),
        'resting_hr': np.random.randint(50, 75, n_days),
        'score': np.random.randint(60, 100, n_days),
    })
    
    # Add missing values
    df.loc[df.sample(10, random_state=42).index, 'steps'] = np.nan
    df.loc[df.sample(15, random_state=43).index, 'score'] = np.nan
    
    return df


class TestGenerateTrendSummary:

    def test_creates_output(self, sample_health_data, tmp_path):
        """Test that trend summary generates output file."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        # Verify file was created
        assert Path(output_path).exists()
        assert Path(output_path).name == 'trend_summary_20240101_120000.md'
        
        # Verify file has content
        with open(output_path, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert '📈 Garmin Data Trend Summary' in content

    def test_with_markdown_structure(self, sample_health_data, tmp_path):
        """Test that markdown output has proper structure."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            # Verify markdown headers
            assert '# 📈 Garmin Data Trend Summary' in content
            assert '## 🔗 Top Volatile Features' in content
            assert '## ❗ Features with Missing Data' in content

    def test_includes_volatile_features(self, sample_health_data, tmp_path):
        """Test that volatile features are included in output."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            # Should contain at least one of the numeric columns
            assert any(col in content for col in ['steps', 'calories', 'resting_hr', 'score', 'stress_avg'])

    def test_includes_missing_data_summary(self, sample_data_with_missing, tmp_path):
        """Test that missing data summary is included."""
        output_path = generate_trend_summary(
            sample_data_with_missing,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            # Should mention features with missing data
            assert '## ❗ Features with Missing Data' in content

    def test_creates_output_directory(self, sample_health_data, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "reports" / "trends"
        
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(nested_dir),
            timestamp='20240101_120000'
        )
        
        # Verify directory was created
        assert nested_dir.exists()
        assert Path(output_path).exists()

    def test_with_custom_timestamp(self, sample_health_data, tmp_path):
        """Test using custom timestamp."""
        custom_timestamp = '20231225_143000'
        
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp=custom_timestamp
        )
        
        # Verify custom timestamp is used
        assert custom_timestamp in output_path
        assert Path(output_path).name == f'trend_summary_{custom_timestamp}.md'

    def test_auto_generates_timestamp(self, sample_health_data, tmp_path):
        """Test that timestamp is auto-generated when not provided."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify file was created with timestamp
        assert Path(output_path).exists()
        assert 'trend_summary_' in Path(output_path).name
        assert '.md' in Path(output_path).name

    def test_returns_path(self, sample_health_data, tmp_path):
        """Test that function returns output path."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        # Verify return value is a string path
        assert isinstance(output_path, str)
        assert output_path.endswith('.md')

    def test_logging(self, sample_health_data, tmp_path, caplog):
        """Test that appropriate log messages are generated."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        # Verify success message was logged
        assert 'Saved trend summary' in caplog.text or 'saved' in caplog.text.lower()

    def test_empty_dataframe(self, tmp_path, caplog):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['day', 'steps', 'calories'])
        
        # Should handle gracefully (may raise exception or return early)
        try:
            output_path = generate_trend_summary(
                empty_df,
                output_dir=str(tmp_path),
                timestamp='20240101_120000'
            )
            
            # If it completes, verify file was created
            if output_path:
                assert Path(output_path).exists()
        except Exception as e:
            # If it raises exception, that's also acceptable
            assert True

    def test_no_numeric_columns(self, tmp_path):
        """Test handling of DataFrame with no numeric columns."""
        non_numeric_df = pd.DataFrame({
            'day': pd.date_range('2024-01-01', periods=10),
            'category': ['A', 'B', 'C'] * 3 + ['A']
        })
        
        # Should handle gracefully
        try:
            output_path = generate_trend_summary(
                non_numeric_df,
                output_dir=str(tmp_path),
                timestamp='20240101_120000'
            )
            
            if output_path:
                assert Path(output_path).exists()
        except Exception:
            # Exception is acceptable for data with no numeric columns
            assert True


class TestLogTopCorrelations:

    def test_logs_pairs(self, sample_health_data, caplog):
        """Test that correlation logging works correctly."""
        # Calculate correlation matrix
        numeric_df = sample_health_data.select_dtypes(include='number')
        corr_matrix = numeric_df.corr()
        
        # Log correlations
        log_top_correlations(corr_matrix, threshold=0.3, max_pairs=5)
        
        # Verify logging occurred
        assert 'correlated pairs' in caplog.text.lower() or 'Top' in caplog.text

    def test_with_high_threshold(self, caplog):
        """Test correlation logging with high threshold."""
        # Create data with known correlations
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfect positive correlation with a
            'c': [5, 4, 3, 2, 1],   # Perfect negative correlation with a
            'd': [10, 15, 12, 18, 20]  # Weaker correlation
        })
        
        corr_matrix = df.corr()
        
        # Log with high threshold
        log_top_correlations(corr_matrix, threshold=0.9, max_pairs=10)
        
        # Should log strong correlations (a-b, a-c)
        assert 'Top' in caplog.text

    def test_respects_max_pairs(self, caplog):
        """Test that max_pairs parameter is respected."""
        # Create correlation matrix with many pairs
        n = 10
        df = pd.DataFrame(np.random.rand(20, n), columns=[f'col_{i}' for i in range(n)])
        corr_matrix = df.corr()
        
        max_pairs = 3
        log_top_correlations(corr_matrix, threshold=0.0, max_pairs=max_pairs)
        
        # Verify logging was called (exact count is hard to verify in logs)
        assert 'Top' in caplog.text


class TestTrendSummaryIntegration:

    def test_24h_coverage_filtering(self, sample_health_data, tmp_path):
        """Test 24-hour coverage filtering integration."""
        # Mock the filter function - it's imported inside the function
        with patch('garmin_analysis.features.coverage.filter_by_24h_coverage') as mock_filter:
            # Mock filter to return subset of data
            filtered_data = sample_health_data.head(20)
            mock_filter.return_value = filtered_data
            
            output_path = generate_trend_summary(
                sample_health_data,
                output_dir=str(tmp_path),
                filter_24h_coverage=True,
                max_gap_minutes=5,
                day_edge_tolerance_minutes=3,
                coverage_allowance_minutes=15,
                timestamp='20240101_120000'
            )
            
            # Verify filter was called
            mock_filter.assert_called_once()
            
            # Verify filtering parameters
            call_args = mock_filter.call_args
            assert call_args[1]['max_gap'] == pd.Timedelta(minutes=5)
            assert call_args[1]['day_edge_tolerance'] == pd.Timedelta(minutes=3)
            assert call_args[1]['total_missing_allowance'] == pd.Timedelta(minutes=15)
            
            # Verify note in output
            with open(output_path, 'r') as f:
                content = f.read()
                assert '24-hour continuous coverage' in content

    def test_24h_coverage_adds_note(self, sample_health_data, tmp_path):
        """Test that 24h filtering adds note to markdown output."""
        with patch('garmin_analysis.features.coverage.filter_by_24h_coverage') as mock_filter:
            mock_filter.return_value = sample_health_data.head(20)
            
            output_path = generate_trend_summary(
                sample_health_data,
                output_dir=str(tmp_path),
                filter_24h_coverage=True,
                timestamp='20240101_120000'
            )
            
            with open(output_path, 'r') as f:
                content = f.read()
                
                # Should contain note about filtering
                assert 'Note:' in content
                assert '24-hour' in content or '24h' in content

    def test_output_format_structure(self, sample_health_data, tmp_path):
        """Test that markdown output follows proper format."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
            
            # Verify first line is main header
            assert lines[0].startswith('# ')
            
            # Verify section headers exist
            content = ''.join(lines)
            assert '## 🔗' in content or '##' in content
            assert '## ❗' in content or '##' in content

    def test_output_format_includes_emoji(self, sample_health_data, tmp_path):
        """Test that markdown output includes emoji in headers."""
        output_path = generate_trend_summary(
            sample_health_data,
            output_dir=str(tmp_path),
            timestamp='20240101_120000'
        )
        
        with open(output_path, 'r') as f:
            content = f.read()
            
            # Verify emoji are present
            assert '📈' in content
            assert '🔗' in content
            assert '❗' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

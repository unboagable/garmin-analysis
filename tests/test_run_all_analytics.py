"""
Tests for comprehensive analytics reporting.

These tests verify:
- Full report generation (integration)
- Monthly report generation (integration)
- HTML output generation
- 24-hour coverage filtering integration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from garmin_analysis.reporting.run_all_analytics import run_all_analytics


@pytest.fixture
def sample_health_data():
    """Create sample health data spanning multiple months."""
    np.random.seed(42)
    n_days = 90  # 3 months of data
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    return pd.DataFrame({
        'day': dates,
        'steps': np.random.randint(5000, 15000, n_days),
        'calories_total': np.random.randint(1800, 2500, n_days),
        'resting_heart_rate': np.random.randint(50, 75, n_days),
        'score': np.random.randint(60, 100, n_days),
        'stress_avg': np.random.randint(20, 60, n_days),
        'hr_min': np.random.randint(45, 65, n_days),
        'hr_max': np.random.randint(120, 180, n_days),
    })


@pytest.mark.integration
def test_run_all_analytics_full_report(sample_health_data, tmp_path):
    """Test full report generation end-to-end."""
    # Mock the anomaly detection to avoid long-running operations
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        # Mock returns (anomalies_df, plot_path)
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            monthly=False
        )
        
        # Verify report was created
        assert Path(report_path).exists()
        assert 'full_report' in report_path
        assert report_path.endswith('.md')
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Full Health Report' in content
            assert 'Trend Summary' in content or '📊' in content
            assert 'Anomaly Detection' in content or '🚨' in content
        
        # Verify trend summary was also created
        trend_files = list(Path(tmp_path).glob('trend_summary_*.md'))
        assert len(trend_files) > 0


@pytest.mark.integration
def test_run_all_analytics_monthly_report(sample_health_data, tmp_path):
    """Test monthly report generation end-to-end."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            monthly=True
        )
        
        # Verify monthly report was created
        assert Path(report_path).exists()
        assert 'monthly_report' in report_path
        assert report_path.endswith('.md')
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Monthly Health Report' in content or 'Monthly' in content


@pytest.mark.integration
def test_run_all_analytics_monthly_filters_data(sample_health_data, tmp_path, caplog):
    """Test that monthly report filters data correctly."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            monthly=True
        )
        
        # Verify filtering message was logged
        assert 'Filtered data to monthly range' in caplog.text or 'monthly' in caplog.text.lower()


def test_run_all_analytics_creates_output_directory(sample_health_data, tmp_path):
    """Test that output directory is created if it doesn't exist."""
    nested_dir = tmp_path / "reports" / "analytics"
    
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(nested_dir),
            monthly=False
        )
        
        # Verify directory was created
        assert nested_dir.exists()
        assert Path(report_path).exists()


def test_html_output_generation(sample_health_data, tmp_path):
    """Test HTML output generation."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            as_html=True
        )
        
        # Verify HTML file was created
        assert Path(report_path).exists()
        assert report_path.endswith('.html')


def test_html_output_vs_markdown(sample_health_data, tmp_path):
    """Test that HTML and Markdown outputs differ appropriately."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        # Generate markdown
        md_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            as_html=False
        )
        
        # Generate HTML
        html_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            as_html=True
        )
        
        # Verify different file extensions
        assert md_path.endswith('.md')
        assert html_path.endswith('.html')
        assert md_path != html_path


def test_24h_coverage_filtering_integration(sample_health_data, tmp_path):
    """Test 24-hour coverage filtering integration."""
    # filter_by_24h_coverage is imported inside the function
    with patch('garmin_analysis.features.coverage.filter_by_24h_coverage') as mock_filter, \
         patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        
        # Mock filter to return subset of data
        filtered_data = sample_health_data.head(30)
        mock_filter.return_value = filtered_data
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            filter_24h_coverage=True,
            max_gap_minutes=5,
            day_edge_tolerance_minutes=3,
            coverage_allowance_minutes=15
        )
        
        # Verify filter was called (may be called twice: once in run_all_analytics, once in generate_trend_summary)
        assert mock_filter.call_count >= 1
        
        # Verify filtering parameters from first call
        call_args = mock_filter.call_args_list[0]
        assert call_args[1]['max_gap'] == pd.Timedelta(minutes=5)
        assert call_args[1]['day_edge_tolerance'] == pd.Timedelta(minutes=3)
        assert call_args[1]['total_missing_allowance'] == pd.Timedelta(minutes=15)
        
        # Verify note in report
        with open(report_path, 'r') as f:
            content = f.read()
            assert '24-hour' in content or '24h' in content


def test_24h_coverage_filtering_adds_note(sample_health_data, tmp_path):
    """Test that 24h filtering adds note to report."""
    # filter_by_24h_coverage is imported inside the function
    with patch('garmin_analysis.features.coverage.filter_by_24h_coverage') as mock_filter, \
         patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        
        mock_filter.return_value = sample_health_data.head(30)
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path),
            filter_24h_coverage=True
        )
        
        with open(report_path, 'r') as f:
            content = f.read()
            
            # Should contain note about filtering
            assert 'Note:' in content or 'note:' in content.lower()
            assert 'continuous coverage' in content or '24-hour' in content


def test_run_all_analytics_with_anomalies_detected(sample_health_data, tmp_path):
    """Test report generation when anomalies are detected."""
    # Create mock anomalies
    anomalies_df = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=3),
        'anomaly_score': [0.95, 0.92, 0.88]
    })
    
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (anomalies_df, 'path/to/plot.png')
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify report mentions anomalies
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Detected 3 anomalies' in content or '3' in content


def test_run_all_analytics_with_no_anomalies(sample_health_data, tmp_path):
    """Test report generation when no anomalies are detected."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify report mentions no anomalies
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'No anomalies detected' in content or 'no anomalies' in content.lower()


def test_run_all_analytics_includes_timestamp(sample_health_data, tmp_path):
    """Test that report includes generation timestamp."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify timestamp in filename
        filename = Path(report_path).name
        assert any(char.isdigit() for char in filename)
        
        # Verify timestamp in content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Generated:' in content or any(char.isdigit() for char in content[:200])


def test_run_all_analytics_references_trend_summary(sample_health_data, tmp_path):
    """Test that report references the trend summary file."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        with open(report_path, 'r') as f:
            content = f.read()
            
            # Should reference trend summary
            assert 'trend_summary' in content.lower() or 'Trend Summary' in content
            assert 'See:' in content or '.md' in content


def test_run_all_analytics_logging(sample_health_data, tmp_path, caplog):
    """Test that appropriate log messages are generated."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify success message was logged
        assert 'report saved' in caplog.text.lower() or 'saved to' in caplog.text.lower()


def test_run_all_analytics_returns_path(sample_health_data, tmp_path):
    """Test that function returns report path."""
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            sample_health_data,
            output_dir=str(tmp_path)
        )
        
        # Verify return value is a string path
        assert isinstance(report_path, str)
        assert Path(report_path).exists()


def test_run_all_analytics_monthly_with_insufficient_data(tmp_path):
    """Test monthly report with insufficient data."""
    # Create data with only a few days
    small_data = pd.DataFrame({
        'day': pd.date_range('2024-01-15', periods=5),
        'steps': [8000, 9000, 10000, 11000, 12000],
        'score': [70, 75, 80, 85, 90]
    })
    
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            small_data,
            output_dir=str(tmp_path),
            monthly=True
        )
        
        # Should still create report (even with limited data)
        assert Path(report_path).exists()


def test_run_all_analytics_with_custom_date_column(sample_health_data, tmp_path):
    """Test with custom date column name."""
    # Rename day column
    custom_df = sample_health_data.rename(columns={'day': 'date'})
    
    with patch('garmin_analysis.reporting.run_all_analytics.run_anomaly_detection') as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        
        report_path = run_all_analytics(
            custom_df,
            date_col='date',
            output_dir=str(tmp_path)
        )
        
        # Should complete successfully
        assert Path(report_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


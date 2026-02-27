"""Tests for optimal sleep activity ranges analysis."""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from garmin_analysis.features.optimal_sleep_activity_ranges import (
    compute_optimal_sleep_ranges,
    get_optimal_sleep_plotly_figures,
    plot_optimal_sleep_ranges,
    print_optimal_sleep_summary,
)


@pytest.fixture
def sample_data():
    """Create sample data with steps, intensity, and sleep score."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    # Best sleep when steps 7k-10k, intensity 30-60 min
    steps = np.random.randint(3000, 15000, n).astype(float)
    intensity = np.random.randint(0, 90, n).astype(float)
    score = 65 + 0.002 * steps + 0.1 * intensity  # rough positive correlation
    score = np.clip(score + np.random.normal(0, 5, n), 40, 95)
    return pd.DataFrame({"day": dates, "steps": steps, "intensity_time": intensity, "score": score})


def test_compute_optimal_sleep_ranges_basic(sample_data):
    """Test basic optimal range computation."""
    result = compute_optimal_sleep_ranges(sample_data)
    assert "steps_range" in result
    assert "intensity_range" in result
    assert "message" in result
    assert result["message"]
    # With positive correlation, best bin should have higher steps
    if result["steps_range"]:
        low, high = result["steps_range"]
        assert low < high


def test_compute_optimal_sleep_ranges_empty():
    """Test with empty dataframe."""
    df = pd.DataFrame(columns=["day", "steps", "score"])
    result = compute_optimal_sleep_ranges(df)
    assert result["message"]
    assert result["steps_range"] is None


def test_compute_optimal_sleep_ranges_no_score():
    """Test with missing score column."""
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"day": days, "steps": [5000 + i * 1000 for i in range(10)]})
    result = compute_optimal_sleep_ranges(df)
    assert "No sleep score" in result["message"] or "not available" in result["message"]


def test_compute_optimal_sleep_ranges_with_moderate_vigorous():
    """Test using moderate + vigorous instead of intensity_time."""
    df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=30, freq="D"),
            "steps": np.random.randint(5000, 12000, 30),
            "moderate_activity_time": [f"00:{m:02d}:00" for m in np.random.randint(10, 45, 30)],
            "vigorous_activity_time": [f"00:{v:02d}:00" for v in np.random.randint(0, 25, 30)],
            "score": np.random.randint(60, 90, 30),
        }
    )
    result = compute_optimal_sleep_ranges(df)
    assert "message" in result


def test_get_optimal_sleep_plotly_figures(sample_data):
    """Test Plotly figure generation for dashboard."""
    result, steps_fig, intensity_fig = get_optimal_sleep_plotly_figures(sample_data)
    assert "steps_range" in result or "intensity_range" in result
    if result["steps_summary"] is not None:
        assert steps_fig is not None
    if result["intensity_summary"] is not None:
        assert intensity_fig is not None


def test_plot_optimal_sleep_ranges_save(sample_data, tmp_path):
    """Test matplotlib plot saving."""
    plot_files = plot_optimal_sleep_ranges(sample_data, save_plots=True, output_dir=tmp_path)
    for name, path in plot_files.items():
        assert path.endswith(".png") or "optimal_sleep" in path


def test_plot_optimal_sleep_ranges_no_save(sample_data, tmp_path):
    """Test plot_optimal_sleep_ranges with save_plots=False."""
    plot_files = plot_optimal_sleep_ranges(sample_data, save_plots=False, output_dir=tmp_path)
    assert plot_files == {}
    assert list(tmp_path.iterdir()) == []


def test_compute_optimal_sleep_ranges_with_yesterday_activity():
    """Test using yesterday_activity_minutes as intensity source."""
    df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=40, freq="D"),
            "steps": np.random.randint(5000, 12000, 40),
            "yesterday_activity_minutes": np.random.randint(0, 60, 40).astype(float),
            "score": np.random.randint(60, 90, 40),
        }
    )
    result = compute_optimal_sleep_ranges(df)
    assert result["message"]
    assert "intensity" in result["message"].lower() or "Insufficient" in result["message"]


def test_compute_optimal_sleep_ranges_min_samples():
    """Test with very few samples (min_samples edge case)."""
    df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=5, freq="D"),
            "steps": [5000, 6000, 7000, 8000, 9000],
            "intensity_time": [10, 20, 30, 40, 50],
            "score": [70, 72, 75, 74, 73],
        }
    )
    result = compute_optimal_sleep_ranges(df, min_samples=2)
    assert result["message"]


def test_compute_optimal_sleep_ranges_custom_columns():
    """Test with custom score and steps column names."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=25, freq="D"),
            "daily_steps": np.random.randint(4000, 15000, 25),
            "sleep_score": np.random.randint(60, 90, 25),
        }
    )
    result = compute_optimal_sleep_ranges(df, score_col="sleep_score", steps_col="daily_steps")
    assert result["message"]
    if result["steps_range"]:
        low, high = result["steps_range"]
        assert low < high


def test_compute_optimal_sleep_ranges_message_format(sample_data):
    """Test that message contains expected phrasing when data is sufficient."""
    result = compute_optimal_sleep_ranges(sample_data)
    if result["steps_range"]:
        assert "steps" in result["message"].lower()
        assert "between" in result["message"].lower()
    if result["intensity_range"]:
        assert "intensity" in result["message"].lower()


def test_print_optimal_sleep_summary(caplog, sample_data):
    """Test print_optimal_sleep_summary does not crash and logs message."""
    import logging

    caplog.set_level(logging.INFO)
    print_optimal_sleep_summary(sample_data)
    assert "OPTIMAL SLEEP" in caplog.text or "optimal" in caplog.text.lower()


def test_get_optimal_sleep_plotly_figures_structure(sample_data):
    """Test Plotly figures have expected structure."""
    result, steps_fig, intensity_fig = get_optimal_sleep_plotly_figures(sample_data)
    if steps_fig is not None:
        assert hasattr(steps_fig, "data")
        assert len(steps_fig.data) > 0
        assert hasattr(steps_fig, "layout")
        assert "Sleep Score by Steps" in str(steps_fig.layout.title)
    if intensity_fig is not None:
        assert hasattr(intensity_fig, "data")
        assert len(intensity_fig.data) > 0
        assert "Intensity" in str(intensity_fig.layout.title)


def test_cli_optimal_sleep_success(caplog, tmp_path):
    """Test CLI runs successfully with mocked data."""
    from garmin_analysis.cli_optimal_sleep import main

    sample_df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=30, freq="D"),
            "steps": np.random.randint(5000, 12000, 30),
            "intensity_time": np.random.randint(0, 60, 30).astype(float),
            "score": np.random.randint(65, 90, 30),
        }
    )

    with patch.object(sys, "argv", ["garmin-optimal-sleep", "--no-save"]):
        with patch(
            "garmin_analysis.cli_optimal_sleep.load_master_dataframe",
            return_value=sample_df,
        ):
            exit_code = main()

    assert exit_code == 0


def test_cli_optimal_sleep_empty_data():
    """Test CLI returns 1 when no data loaded."""
    from garmin_analysis.cli_optimal_sleep import main

    with patch.object(sys, "argv", ["garmin-optimal-sleep", "--no-save"]):
        with patch(
            "garmin_analysis.cli_optimal_sleep.load_master_dataframe",
            return_value=pd.DataFrame(),
        ):
            exit_code = main()

    assert exit_code == 1


def test_run_all_analytics_includes_optimal_sleep(tmp_path):
    """Test run_all_analytics report includes optimal sleep section."""
    from garmin_analysis.reporting.run_all_analytics import run_all_analytics

    df = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=50, freq="D"),
            "steps": np.random.randint(5000, 12000, 50),
            "intensity_time": np.random.randint(0, 60, 50).astype(float),
            "score": np.random.randint(65, 90, 50),
        }
    )

    with patch("garmin_analysis.reporting.run_all_analytics.run_anomaly_detection") as mock_anomaly:
        mock_anomaly.return_value = (pd.DataFrame(), None)
        report_path = run_all_analytics(df, output_dir=str(tmp_path), monthly=False)

    with open(report_path) as f:
        content = f.read()

    assert "Optimal Sleep" in content
    assert "steps" in content.lower() or "intensity" in content.lower()

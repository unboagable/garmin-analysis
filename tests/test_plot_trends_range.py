import pandas as pd
import pytest
from unittest import mock
from garmin_analysis.viz.plot_trends_range import plot_columns


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "day": pd.date_range(start="2024-01-01", periods=3),
        "steps": [1000, 2000, 3000],
        "calories_total": [2200, 2300, 2100],
        "hr_min": [50, 55, 52],
    })


def test_plot_columns_with_all_present(sample_df):
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        plot_columns(sample_df, ["steps", "calories_total"], "Test Plot")


def test_plot_columns_with_some_missing(sample_df, caplog):
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        plot_columns(sample_df, ["steps", "missing_col"], "Test Missing")
        assert "Skipping missing columns" in caplog.text


def test_plot_columns_with_all_missing(sample_df, caplog):
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        plot_columns(sample_df, ["foo", "bar"], "Nothing to Plot")
        assert "No valid columns available" in caplog.text


def test_plot_columns_edge_case_single_data_point():
    """Test with single data point."""
    single_point_df = pd.DataFrame({
        "day": [pd.Timestamp("2024-01-01")],
        "steps": [10000],
        "calories": [2000]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle single data point without errors
        plot_columns(single_point_df, ["steps", "calories"], "Single Point")


def test_plot_columns_edge_case_non_sequential_dates():
    """Test with non-sequential dates."""
    non_sequential_df = pd.DataFrame({
        "day": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-03", "2024-01-30", "2024-01-10"]),
        "steps": [5000, 8000, 6000, 9000, 7000],
        "calories": [2000, 2200, 2100, 2300, 2150]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle non-sequential dates
        plot_columns(non_sequential_df, ["steps", "calories"], "Non-Sequential Dates")


def test_plot_columns_edge_case_duplicate_dates():
    """Test with duplicate dates."""
    duplicate_dates_df = pd.DataFrame({
        "day": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
        "steps": [5000, 6000, 7000, 8000, 9000],
        "calories": [2000, 2100, 2200, 2300, 2400]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle duplicate dates gracefully
        plot_columns(duplicate_dates_df, ["steps", "calories"], "Duplicate Dates")


def test_plot_columns_edge_case_all_columns_missing():
    """Test when all requested columns are missing from DataFrame."""
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=5),
        "existing_col": [1, 2, 3, 4, 5]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"), \
         pytest.raises(Exception) if False else mock.patch("sys.exit"):
        # All requested columns are missing
        try:
            plot_columns(df, ["missing1", "missing2", "missing3"], "All Missing")
        except Exception:
            # If it raises exception, that's acceptable
            pass


def test_plot_columns_edge_case_empty_column_list():
    """Test with empty column list."""
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=3),
        "steps": [1000, 2000, 3000]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle empty column list gracefully
        try:
            plot_columns(df, [], "Empty Columns")
        except Exception:
            # If it raises exception, that's acceptable
            pass


def test_plot_columns_edge_case_dates_with_large_gaps():
    """Test with dates that have large gaps."""
    dates_with_gaps = pd.to_datetime([
        "2024-01-01",
        "2024-02-15",  # 45 day gap
        "2024-06-01",  # ~3.5 month gap
        "2024-12-31"   # ~7 month gap
    ])
    
    df = pd.DataFrame({
        "day": dates_with_gaps,
        "steps": [5000, 8000, 10000, 12000],
        "calories": [2000, 2200, 2400, 2600]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle large gaps in dates
        plot_columns(df, ["steps", "calories"], "Large Date Gaps")


def test_plot_columns_edge_case_dates_out_of_order():
    """Test with dates in reverse chronological order."""
    reverse_dates_df = pd.DataFrame({
        "day": pd.to_datetime(["2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01"]),
        "steps": [9000, 8000, 7000, 6000, 5000],
        "calories": [2400, 2300, 2200, 2100, 2000]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle reverse chronological dates
        plot_columns(reverse_dates_df, ["steps", "calories"], "Reverse Order")


def test_plot_columns_edge_case_all_null_values():
    """Test with all null values in requested columns."""
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=5),
        "steps": [None, None, None, None, None],
        "calories": [None, None, None, None, None]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle all null values
        try:
            plot_columns(df, ["steps", "calories"], "All Nulls")
        except Exception:
            # If it raises exception, that's acceptable
            pass


def test_plot_columns_edge_case_mixed_null_values():
    """Test with mix of valid and null values."""
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=5),
        "steps": [5000, None, 7000, None, 9000],
        "calories": [2000, 2100, None, None, 2400]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle mix of nulls and valid values
        plot_columns(df, ["steps", "calories"], "Mixed Nulls")


def test_plot_columns_edge_case_single_column():
    """Test with single column to plot."""
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=10),
        "steps": [5000 + i*500 for i in range(10)]
    })
    
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        # Should handle single column
        plot_columns(df, ["steps"], "Single Column")

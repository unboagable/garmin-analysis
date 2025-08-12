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


def test_plot_columns_all_missing(sample_df, caplog):
    with mock.patch("pandas.DataFrame.plot"), \
         mock.patch("matplotlib.pyplot.tight_layout"), \
         mock.patch("matplotlib.pyplot.show"):
        plot_columns(sample_df, ["foo", "bar"], "Nothing to Plot")
        assert "No valid columns available" in caplog.text

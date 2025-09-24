import pytest
import pandas as pd
import numpy as np
from garmin_analysis.utils import load_master_dataframe
from dash import Dash
from garmin_analysis.dashboard import app as dashboard_app


@pytest.mark.integration
def test_load_master_dataframe(tmp_db):
    df = load_master_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "day" in df.columns


@pytest.mark.integration
def test_correlation_matrix_valid(tmp_db):
    df = load_master_dataframe()
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    assert np.all(corr.columns == corr.index)


@pytest.mark.integration
def test_correlation_matrix_edge_cases(tmp_db):
    """Test correlation matrix edge cases"""
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    numeric_df = empty_df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    if numeric_df.shape[1] == 0:
        corr = pd.DataFrame([[0]], columns=['No numeric data'], index=['No numeric data'])
    else:
        corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    
    # Test with non-numeric DataFrame
    non_numeric_df = pd.DataFrame({
        'day': ['2024-01-01', '2024-01-02'],
        'category': ['A', 'B']
    })
    numeric_df = non_numeric_df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    if numeric_df.shape[1] == 0:
        corr = pd.DataFrame([[0]], columns=['No numeric data'], index=['No numeric data'])
    else:
        corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    
    # Test with all-NA numeric columns
    all_na_df = pd.DataFrame({
        'day': ['2024-01-01', '2024-01-02'],
        'numeric_col': [np.nan, np.nan]
    })
    numeric_df = all_na_df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    if numeric_df.shape[1] == 0:
        corr = pd.DataFrame([[0]], columns=['No numeric data'], index=['No numeric data'])
    else:
        corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]


def test_dash_app_layout_smoke():
    # Ensure the app can be created and has the expected structure
    assert dashboard_app.app is not None
    assert hasattr(dashboard_app.app, 'layout')
    assert hasattr(dashboard_app.app, 'title')
    assert dashboard_app.app.title == "Garmin Health Dashboard"

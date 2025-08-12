import pytest
import pandas as pd
import numpy as np
from src.utils import load_master_dataframe
from dash import Dash
from src.dashboard import app as dashboard_app

def test_load_master_dataframe():
    df = load_master_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "day" in df.columns

def test_correlation_matrix_valid():
    df = load_master_dataframe()
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    assert np.all(corr.columns == corr.index)

def test_dash_app_layout_smoke():
    # Ensure the app can be created and has the expected structure
    assert dashboard_app.app is not None
    assert hasattr(dashboard_app.app, 'layout')
    assert hasattr(dashboard_app.app, 'title')
    assert dashboard_app.app.title == "Garmin Health Dashboard"

import pytest
import pandas as pd
import numpy as np
from src.utils import load_master_dataframe
from src.modeling.sleep_predictor import run_sleep_model
from dash import Dash
from src.dashboard import app as dashboard_app

def test_load_master_dataframe():
    df = load_master_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "day" in df.columns

def test_run_sleep_model_keys():
    result = run_sleep_model()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"r2", "mse", "plot_path"}

def test_run_sleep_model_outputs():
    result = run_sleep_model()
    assert isinstance(result["r2"], (float, type(None)))
    assert isinstance(result["mse"], (float, type(None)))
    assert isinstance(result["plot_path"], (str, type(None)))

def test_correlation_matrix_valid():
    df = load_master_dataframe()
    numeric_df = df.select_dtypes(include=np.number).dropna(axis=1, how='any')
    corr = numeric_df.corr()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape[0] == corr.shape[1]
    assert np.all(corr.columns == corr.index)

def test_dash_app_layout_smoke():
    # Ensure the app layout compiles
    app = Dash(__name__)
    app.layout = dashboard_app.app.layout
    assert app.layout is not None

import pytest
import pandas as pd
from src.utils import convert_time_to_minutes, normalize_day_column

def test_convert_time_to_minutes_hms():
    assert convert_time_to_minutes("01:30:00") == 90
    assert convert_time_to_minutes("00:45:30") == 45.5

def test_convert_time_to_minutes_ms():
    assert convert_time_to_minutes("05:30") == 5.5

def test_convert_time_to_minutes_invalid():
    assert convert_time_to_minutes("nonsense") is None
    assert convert_time_to_minutes("") is None

def test_normalize_day_column_day():
    df = pd.DataFrame({"day": ["2023-01-01", "2023-01-02"]})
    result = normalize_day_column(df.copy(), "test")
    assert pd.api.types.is_datetime64_any_dtype(result["day"])

def test_normalize_day_column_calendarDate():
    df = pd.DataFrame({"calendarDate": ["2023-01-01", "2023-01-02"]})
    result = normalize_day_column(df.copy(), "test")
    assert "day" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["day"])

def test_normalize_day_column_timestamp():
    df = pd.DataFrame({"timestamp": ["2023-01-01T14:23:00", "2023-01-02T00:01:00"]})
    result = normalize_day_column(df.copy(), "test")
    assert "day" in result.columns
    assert all(result["day"].dt.hour == 0)

def test_normalize_day_column_missing_column(caplog):
    df = pd.DataFrame({"value": [1, 2]})
    result = normalize_day_column(df.copy(), "test_missing")
    assert "day" not in result.columns
    assert any("could not normalize" in message for message in caplog.text.splitlines())

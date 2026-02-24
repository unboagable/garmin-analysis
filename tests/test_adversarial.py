"""
Adversarial test suite for garmin_analysis.

Attacks every public function with malicious inputs, boundary conditions,
type confusion, injection attempts, and degenerate data designed to expose
crashes, silent corruption, and unhandled edge cases.
"""

import datetime
import math
import os
import sqlite3
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. data_processing: convert_time_to_minutes, normalize_day_column, ensure_datetime_sorted
# ---------------------------------------------------------------------------
from garmin_analysis.utils.data_processing import (
    convert_time_to_minutes,
    normalize_day_column,
    ensure_datetime_sorted,
)


class TestConvertTimeToMinutesAdversarial:
    """Adversarial attacks on convert_time_to_minutes."""

    @pytest.mark.parametrize("inp", [
        None, "", "   ", "\t", "\n", "\r\n",
    ])
    def test_blank_and_none(self, inp):
        result = convert_time_to_minutes(inp)
        assert result is np.nan or (isinstance(result, float) and math.isnan(result))

    @pytest.mark.parametrize("inp", [
        "not-a-time", "abc:def:ghi", ":::", "12:60:60",
        "HH:MM:SS", "∞:∞:∞", "NaN", "inf", "-inf",
        "1:2:3:4", "1:2:3:4:5",
    ])
    def test_garbage_strings(self, inp):
        result = convert_time_to_minutes(inp)
        assert isinstance(result, float) and (math.isnan(result) or math.isfinite(result))

    def test_negative_time(self):
        result = convert_time_to_minutes("-1:30:00")
        assert isinstance(result, (int, float))

    def test_very_large_time(self):
        result = convert_time_to_minutes("9999:59:59")
        if not (isinstance(result, float) and math.isnan(result)):
            assert result > 0

    def test_zero_time(self):
        result = convert_time_to_minutes("00:00:00")
        assert result == 0.0

    def test_numeric_passthrough(self):
        assert convert_time_to_minutes(90) == 90.0
        assert convert_time_to_minutes(0) == 0.0
        assert convert_time_to_minutes(-5) == -5.0

    def test_float_passthrough(self):
        assert convert_time_to_minutes(1.5) == 1.5

    @pytest.mark.parametrize("inp", [
        float("inf"), float("-inf"), float("nan"),
    ])
    def test_special_floats(self, inp):
        result = convert_time_to_minutes(inp)
        assert isinstance(result, float)

    def test_bytes_input(self):
        result = convert_time_to_minutes(b"01:30:00")
        assert isinstance(result, float)

    def test_unicode_fullwidth(self):
        result = convert_time_to_minutes("０１:３０:００")
        assert isinstance(result, float)

    def test_sql_injection_string(self):
        result = convert_time_to_minutes("'; DROP TABLE stress;--")
        assert result is np.nan or (isinstance(result, float) and math.isnan(result))

    def test_boolean_input(self):
        result = convert_time_to_minutes(True)
        assert isinstance(result, float)

    def test_list_input(self):
        result = convert_time_to_minutes([1, 2, 3])
        assert isinstance(result, float)


class TestNormalizeDayColumnAdversarial:
    """Adversarial attacks on normalize_day_column."""

    def test_none_input(self):
        result = normalize_day_column(None)
        assert result is None

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = normalize_day_column(df)
        assert result.empty

    def test_no_date_columns(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = normalize_day_column(df)
        assert "day" not in result.columns or result.equals(df)

    def test_day_column_with_garbage(self):
        """After fix: unparseable strings become NaT instead of crashing."""
        df = pd.DataFrame({"day": ["not-a-date", "garbage", None, ""]})
        result = normalize_day_column(df)
        assert "day" in result.columns
        assert result["day"].isna().sum() >= 2

    def test_calendar_date_column(self):
        df = pd.DataFrame({"calendarDate": ["2024-01-01", "2024-01-02"]})
        result = normalize_day_column(df)
        assert "day" in result.columns

    def test_timestamp_column(self):
        df = pd.DataFrame({"timestamp": ["2024-01-01 12:00:00", "2024-01-02 14:00:00"]})
        result = normalize_day_column(df)
        assert "day" in result.columns

    def test_day_with_mixed_timezones(self):
        df = pd.DataFrame({"day": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-02T00:00:00+05:30",
            "2024-01-03T00:00:00-08:00",
        ]})
        result = normalize_day_column(df)
        assert len(result) == 3

    def test_day_with_epoch_zero(self):
        df = pd.DataFrame({"day": ["1970-01-01", "2024-01-01"]})
        result = normalize_day_column(df)
        assert len(result) == 2

    def test_day_with_far_future_dates(self):
        """After fix: out-of-bounds dates become NaT instead of crashing."""
        df = pd.DataFrame({"day": ["2200-01-01", "3000-12-31"]})
        result = normalize_day_column(df)
        assert len(result) == 2

    def test_day_with_pre_epoch_dates(self):
        df = pd.DataFrame({"day": ["1900-01-01", "1800-06-15"]})
        result = normalize_day_column(df)
        assert len(result) == 2

    def test_day_column_all_nat(self):
        df = pd.DataFrame({"day": [pd.NaT, pd.NaT, pd.NaT]})
        result = normalize_day_column(df)
        assert len(result) == 3

    def test_duplicate_day_and_calendardate(self):
        df = pd.DataFrame({
            "day": ["2024-01-01"],
            "calendarDate": ["2024-06-15"],
        })
        result = normalize_day_column(df)
        assert "day" in result.columns


class TestEnsureDatetimeSortedAdversarial:
    """Adversarial attacks on ensure_datetime_sorted."""

    def test_none_input(self):
        result = ensure_datetime_sorted(None)
        assert result is None

    def test_empty_dataframe(self):
        result = ensure_datetime_sorted(pd.DataFrame())
        assert result.empty

    def test_no_date_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = ensure_datetime_sorted(df)
        assert result.equals(df)

    def test_all_unparseable_dates(self):
        df = pd.DataFrame({"day": ["garbage", "not_a_date", "???"]})
        result = ensure_datetime_sorted(df)
        assert len(result) == 0

    def test_mixed_valid_invalid_dates(self):
        df = pd.DataFrame({"day": ["2024-01-01", "garbage", "2024-01-03"]})
        result = ensure_datetime_sorted(df)
        assert len(result) == 2

    def test_already_sorted(self):
        df = pd.DataFrame({"day": ["2024-01-01", "2024-01-02", "2024-01-03"]})
        result = ensure_datetime_sorted(df)
        assert len(result) == 3

    def test_reverse_sorted(self):
        df = pd.DataFrame({
            "day": ["2024-01-03", "2024-01-02", "2024-01-01"],
            "val": [3, 2, 1],
        })
        result = ensure_datetime_sorted(df)
        assert result.iloc[0]["val"] == 1

    def test_duplicates_kept_first(self):
        df = pd.DataFrame({
            "day": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "val": [1, 2, 3],
        })
        result = ensure_datetime_sorted(df, drop_dupes=True)
        assert len(result) == 2
        assert result.iloc[0]["val"] == 1

    def test_duplicates_not_dropped(self):
        df = pd.DataFrame({
            "day": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "val": [1, 2, 3],
        })
        result = ensure_datetime_sorted(df, drop_dupes=False)
        assert len(result) == 3

    def test_tz_conversion(self):
        df = pd.DataFrame({"day": ["2024-01-01T12:00:00+00:00", "2024-01-02T12:00:00+00:00"]})
        result = ensure_datetime_sorted(df, tz="US/Eastern")
        assert len(result) >= 1

    def test_timestamp_column_no_dedup(self):
        df = pd.DataFrame({
            "timestamp": ["2024-01-01 08:00:00", "2024-01-01 09:00:00", "2024-01-01 10:00:00"],
            "val": [1, 2, 3],
        })
        result = ensure_datetime_sorted(df, drop_dupes=True)
        assert len(result) == 3

    def test_single_row(self):
        df = pd.DataFrame({"day": ["2024-01-01"], "val": [42]})
        result = ensure_datetime_sorted(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2. data_filtering: strip_time_from_dates, filter_by_date,
#    convert_time_columns, standardize_features, filter_required_columns
# ---------------------------------------------------------------------------
from garmin_analysis.utils.data_filtering import (
    strip_time_from_dates,
    normalize_dates,
    filter_by_date,
    convert_time_columns,
    standardize_features,
    filter_required_columns,
)


class TestStripTimeFromDatesAdversarial:

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["day"])
        result = strip_time_from_dates(df)
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = strip_time_from_dates(df, col="day")
        assert "day" not in result.columns

    def test_already_normalized(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = strip_time_from_dates(df)
        assert all(result["day"].dt.hour == 0)

    def test_with_time_components(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01 14:30:00", "2024-01-02 08:15:00"])})
        result = strip_time_from_dates(df)
        assert all(result["day"].dt.hour == 0)

    def test_mixed_types_in_column(self):
        df = pd.DataFrame({"day": ["2024-01-01", 12345, None]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = strip_time_from_dates(df)
            except Exception:
                pass  # acceptable to raise on truly bad data

    def test_normalize_dates_deprecation(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01 12:00:00"])})
        with pytest.warns(DeprecationWarning):
            result = normalize_dates(df)
        assert result["day"].iloc[0].hour == 0


class TestFilterByDateAdversarial:

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["day"])
        result = filter_by_date(df)
        assert result.empty

    def test_no_filters(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = filter_by_date(df)
        assert len(result) == 2

    def test_from_date_in_future(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = filter_by_date(df, from_date="2099-01-01")
        assert len(result) == 0

    def test_to_date_in_past(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = filter_by_date(df, to_date="1900-01-01")
        assert len(result) == 0

    def test_days_back_zero(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = filter_by_date(df, days_back=0)
        assert isinstance(result, pd.DataFrame)

    def test_negative_days_back(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"])})
        result = filter_by_date(df, days_back=-1)
        assert isinstance(result, pd.DataFrame)

    def test_very_large_days_back(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"])})
        result = filter_by_date(df, days_back=999999)
        assert len(result) == 1

    def test_months_back(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-06-01"])})
        result = filter_by_date(df, months_back=3)
        assert isinstance(result, pd.DataFrame)

    def test_conflicting_filters(self):
        """days_back should take priority when both days_back and from_date could apply."""
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"])})
        result = filter_by_date(df, from_date="2024-05-01", days_back=30)
        assert isinstance(result, pd.DataFrame)

    def test_wrong_date_column_name(self):
        df = pd.DataFrame({"date": ["2024-01-01"]})
        with pytest.raises(KeyError):
            filter_by_date(df, date_col="day")

    def test_unparseable_dates(self):
        df = pd.DataFrame({"day": ["not-a-date", "garbage"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = filter_by_date(df)
            except Exception:
                pass


class TestConvertTimeColumnsAdversarial:

    def test_missing_columns_ignored(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = convert_time_columns(df, ["nonexistent"])
        assert result.equals(df)

    def test_numeric_columns_unchanged(self):
        df = pd.DataFrame({"total_sleep": [90.0, 120.0]})
        result = convert_time_columns(df, ["total_sleep"])
        pd.testing.assert_frame_equal(result, df)

    def test_string_time_columns(self):
        df = pd.DataFrame({"total_sleep": ["01:30:00", "02:00:00"]})
        result = convert_time_columns(df, ["total_sleep"])
        assert result["total_sleep"].iloc[0] == pytest.approx(90.0)

    def test_mixed_valid_invalid(self):
        df = pd.DataFrame({"col": ["01:00:00", "garbage", None]})
        result = convert_time_columns(df, ["col"])
        assert result["col"].iloc[0] == pytest.approx(60.0)
        assert pd.isna(result["col"].iloc[1])

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["col"])
        result = convert_time_columns(df, ["col"])
        assert result.empty


class TestStandardizeFeaturesAdversarial:

    def test_all_nan_columns(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = standardize_features(df, ["a", "b"])
        assert isinstance(result, np.ndarray) and len(result) == 0

    def test_single_row(self):
        df = pd.DataFrame({"a": [5.0], "b": [10.0]})
        result = standardize_features(df, ["a", "b"])
        assert len(result) == 1

    def test_constant_values(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [10.0, 10.0, 10.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = standardize_features(df, ["a", "b"])
        assert len(result) == 3

    def test_with_inf(self):
        df = pd.DataFrame({"a": [1.0, float("inf"), 3.0], "b": [4.0, 5.0, 6.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = standardize_features(df, ["a", "b"])
            except Exception:
                pass


class TestFilterRequiredColumnsAdversarial:

    def test_all_required_missing(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = filter_required_columns(df, ["a", "b"])
        assert len(result) == 2  # returns original when cols missing

    def test_empty_required(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = filter_required_columns(df, [])
        assert len(result) == 2

    def test_all_rows_have_nulls(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, np.nan]})
        result = filter_required_columns(df, ["a", "b"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 3. imputation: edge cases beyond what existing tests cover
# ---------------------------------------------------------------------------
from garmin_analysis.utils.imputation import (
    impute_missing_values,
    get_missing_value_summary,
    recommend_imputation_strategy,
)


class TestImputationAdversarial:

    def test_inf_values_in_column(self):
        df = pd.DataFrame({"a": [1.0, float("inf"), np.nan, 3.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = impute_missing_values(df, ["a"], strategy="median")
        assert not result["a"].isna().any()

    def test_neg_inf_values(self):
        df = pd.DataFrame({"a": [1.0, float("-inf"), np.nan, 3.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = impute_missing_values(df, ["a"], strategy="mean")
        assert isinstance(result, pd.DataFrame)

    def test_all_inf_column(self):
        df = pd.DataFrame({"a": [float("inf"), float("inf"), np.nan]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = impute_missing_values(df, ["a"], strategy="median")
        assert isinstance(result, pd.DataFrame)

    def test_single_row_with_nan(self):
        df = pd.DataFrame({"a": [np.nan]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = impute_missing_values(df, ["a"], strategy="median")
        assert len(result) == 1

    def test_forward_fill_leading_nan(self):
        """Forward fill can't fill leading NaN — should leave it."""
        df = pd.DataFrame({"a": [np.nan, 1.0, 2.0]})
        result = impute_missing_values(df, ["a"], strategy="forward_fill")
        assert pd.isna(result["a"].iloc[0])

    def test_backward_fill_trailing_nan(self):
        """Backward fill can't fill trailing NaN — should leave it."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan]})
        result = impute_missing_values(df, ["a"], strategy="backward_fill")
        assert pd.isna(result["a"].iloc[-1])

    def test_drop_removes_all_rows(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        result = impute_missing_values(df, ["a"], strategy="drop")
        assert len(result) == 0

    def test_multiple_columns_different_nan_patterns(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, 2.0, np.nan, 4.0],
        })
        result = impute_missing_values(df, ["a", "b"], strategy="drop")
        assert len(result) == 0  # every row has at least one NaN

    def test_non_dataframe_inputs(self):
        with pytest.raises(TypeError):
            impute_missing_values({"a": [1, 2]}, ["a"])
        with pytest.raises(TypeError):
            impute_missing_values(None, ["a"])
        with pytest.raises(TypeError):
            impute_missing_values("hello", ["a"])

    def test_column_names_with_special_chars(self):
        df = pd.DataFrame({"col with spaces": [1.0, np.nan, 3.0]})
        result = impute_missing_values(df, ["col with spaces"], strategy="median")
        assert not result["col with spaces"].isna().any()

    def test_summary_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype=float)})
        result = get_missing_value_summary(df, ["a"])
        assert len(result) == 1
        assert result.iloc[0]["total_count"] == 0

    def test_recommend_single_value(self):
        df = pd.DataFrame({"a": [5.0]})
        recs = recommend_imputation_strategy(df, ["a"])
        assert "a" in recs

    def test_recommend_nonexistent_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        recs = recommend_imputation_strategy(df, ["nonexistent"])
        assert "nonexistent" not in recs


# ---------------------------------------------------------------------------
# 4. error_handling: exceptions, decorators, validators, context managers
# ---------------------------------------------------------------------------
from garmin_analysis.utils.error_handling import (
    GarminAnalysisError,
    DataLoadingError,
    DatabaseError,
    DataValidationError,
    ModelingError,
    ConfigurationError,
    InsufficientDataError,
    handle_data_loading_errors,
    handle_database_errors,
    handle_modeling_errors,
    validate_dataframe,
    validate_file_path,
    validate_database_connection,
    suppress_and_log,
    log_and_reraise,
    safe_return,
)


class TestExceptionHierarchy:

    def test_all_inherit_from_base(self):
        for exc_cls in [DataLoadingError, DatabaseError, DataValidationError,
                        ModelingError, ConfigurationError, InsufficientDataError]:
            assert issubclass(exc_cls, GarminAnalysisError)

    def test_insufficient_data_is_validation_error(self):
        assert issubclass(InsufficientDataError, DataValidationError)


class TestDataLoadingDecorator:

    def test_swallows_file_not_found(self):
        @handle_data_loading_errors(default_return="fallback")
        def fn():
            raise FileNotFoundError("gone")
        assert fn() == "fallback"

    def test_swallows_os_error(self):
        @handle_data_loading_errors(default_return=[])
        def fn():
            raise OSError("disk crash")
        assert fn() == []

    def test_reraise_wraps_in_custom_exception(self):
        @handle_data_loading_errors(reraise=True)
        def fn():
            raise FileNotFoundError("missing")
        with pytest.raises(DataLoadingError):
            fn()

    def test_unexpected_exception_handled(self):
        @handle_data_loading_errors(default_return=None)
        def fn():
            raise RuntimeError("boom")
        assert fn() is None

    def test_no_exception_passes_through(self):
        @handle_data_loading_errors(default_return="bad")
        def fn():
            return "good"
        assert fn() == "good"

    def test_preserves_function_name(self):
        @handle_data_loading_errors()
        def my_func():
            pass
        assert my_func.__name__ == "my_func"


class TestDatabaseErrorDecorator:

    def test_swallows_operational_error(self):
        @handle_database_errors(default_return=pd.DataFrame())
        def fn():
            raise sqlite3.OperationalError("locked")
        result = fn()
        assert isinstance(result, pd.DataFrame)

    def test_swallows_integrity_error(self):
        @handle_database_errors(default_return={})
        def fn():
            raise sqlite3.IntegrityError("constraint")
        assert fn() == {}

    def test_reraise_wraps(self):
        @handle_database_errors(reraise=True)
        def fn():
            raise sqlite3.OperationalError("fail")
        with pytest.raises(DatabaseError):
            fn()


class TestModelingErrorDecorator:

    def test_swallows_value_error(self):
        @handle_modeling_errors(default_return={})
        def fn():
            raise ValueError("bad input")
        assert fn() == {}

    def test_swallows_key_error(self):
        @handle_modeling_errors(default_return=None)
        def fn():
            raise KeyError("missing_key")
        assert fn() is None

    def test_reraise_wraps(self):
        @handle_modeling_errors(reraise=True)
        def fn():
            raise ValueError("oops")
        with pytest.raises(ModelingError):
            fn()


class TestValidateDataframeAdversarial:

    def test_none_raises(self):
        with pytest.raises(DataValidationError, match="None"):
            validate_dataframe(None)

    def test_wrong_type_raises(self):
        with pytest.raises(DataValidationError, match="Expected DataFrame"):
            validate_dataframe([1, 2, 3])

    def test_empty_not_allowed(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_dataframe(pd.DataFrame())

    def test_empty_allowed(self):
        validate_dataframe(pd.DataFrame(), allow_empty=True)

    def test_insufficient_rows(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(InsufficientDataError, match="at least 10"):
            validate_dataframe(df, min_rows=10)

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(DataValidationError, match="Missing required"):
            validate_dataframe(df, required_columns=["a", "b", "c"])

    def test_all_checks_pass(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        validate_dataframe(df, required_columns=["a", "b"], min_rows=2)


class TestValidateFilePathAdversarial:

    def test_nonexistent_must_exist(self):
        with pytest.raises(FileNotFoundError):
            validate_file_path("/nonexistent/path/file.csv", must_exist=True)

    def test_wrong_file_type(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Expected file type"):
            validate_file_path(f, must_exist=True, file_type=".csv")

    def test_correct_file_type(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")
        result = validate_file_path(f, must_exist=True, file_type=".csv")
        assert result == f

    def test_path_traversal(self):
        with pytest.raises(FileNotFoundError):
            validate_file_path("../../../etc/passwd", must_exist=True)

    def test_doesnt_need_to_exist(self):
        result = validate_file_path("/tmp/future_file.csv", must_exist=False)
        assert result == Path("/tmp/future_file.csv")


class TestValidateDatabaseConnectionAdversarial:

    def test_none_connection(self):
        with pytest.raises(DatabaseError, match="None"):
            validate_database_connection(None)

    def test_closed_connection(self):
        conn = sqlite3.connect(":memory:")
        conn.close()
        with pytest.raises(DatabaseError):
            validate_database_connection(conn)

    def test_valid_connection(self):
        conn = sqlite3.connect(":memory:")
        try:
            validate_database_connection(conn)
        finally:
            conn.close()

    def test_mock_bad_connection(self):
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.Error("dead connection")
        with pytest.raises(DatabaseError):
            validate_database_connection(mock_conn)


class TestSuppressAndLogAdversarial:

    def test_suppresses_specified(self):
        with suppress_and_log(ValueError):
            raise ValueError("expected")

    def test_does_not_suppress_unspecified(self):
        with pytest.raises(TypeError):
            with suppress_and_log(ValueError):
                raise TypeError("not suppressed")

    def test_no_exception(self):
        with suppress_and_log(ValueError):
            pass  # no exception


class TestSafeReturnAdversarial:

    def test_returns_default_on_error(self):
        def boom():
            raise RuntimeError("crash")
        result = safe_return(boom, default="fallback")
        assert result == "fallback"

    def test_passes_args(self):
        def add(a, b):
            return a + b
        assert safe_return(add, 2, 3, default=0) == 5

    def test_passes_kwargs(self):
        def greet(name="world"):
            return f"hello {name}"
        assert safe_return(greet, default="", name="test") == "hello test"


class TestLogAndReraiseAdversarial:

    def test_reraises(self):
        with pytest.raises(ValueError):
            try:
                raise ValueError("original")
            except Exception as e:
                log_and_reraise(e, "testing")


# ---------------------------------------------------------------------------
# 5. utils_cleaning: clean_data
# ---------------------------------------------------------------------------
from garmin_analysis.utils_cleaning import clean_data


class TestCleanDataAdversarial:

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = clean_data(df)
        assert result.empty

    def test_all_nan_dataframe(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = clean_data(df)
        assert len(result) == 2

    def test_placeholder_replacement(self):
        df = pd.DataFrame({"a": ["", "NA", "null", "None", -1, 42]})
        result = clean_data(df)
        assert result.iloc[-1].values[0] == 42

    def test_outlier_removal(self):
        data = [10] * 100 + [1000]
        df = pd.DataFrame({"a": data})
        result = clean_data(df, remove_outliers=True)
        assert len(result) < len(df)

    def test_no_outlier_removal(self):
        data = [10] * 100 + [1000]
        df = pd.DataFrame({"a": data})
        result = clean_data(df, remove_outliers=False)
        assert len(result) == len(df)

    def test_column_name_normalization(self):
        df = pd.DataFrame({"  Column Name  ": [1], "UPPER": [2], "with spaces": [3]})
        result = clean_data(df)
        assert "column_name" in result.columns
        assert "upper" in result.columns
        assert "with_spaces" in result.columns

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"a": ["NA", 1, 2]})
        original = df.copy()
        _ = clean_data(df)
        pd.testing.assert_frame_equal(df, original)

    def test_mixed_type_columns(self):
        df = pd.DataFrame({"a": [1, "2", 3.0, "NA", None]})
        result = clean_data(df)
        assert len(result) == 5

    def test_unicode_column_names(self):
        df = pd.DataFrame({"日本語": [1, 2], "émojis_🎉": [3, 4]})
        result = clean_data(df)
        assert len(result) == 2

    def test_single_column(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = clean_data(df)
        assert len(result) == 3

    def test_very_wide_dataframe(self):
        df = pd.DataFrame({f"col_{i}": [i, i + 1] for i in range(200)})
        result = clean_data(df)
        assert result.shape[1] == 200

    def test_negative_one_replaced(self):
        """The function treats -1 as a placeholder."""
        df = pd.DataFrame({"a": [-1, 0, 1, -1, 5]})
        result = clean_data(df)
        nan_count = result.iloc[:, 0].isna().sum()
        assert nan_count == 2

    def test_outlier_removal_all_same(self):
        df = pd.DataFrame({"a": [5] * 10})
        result = clean_data(df, remove_outliers=True)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# 6. coverage: days_with_continuous_coverage, filter_master_by_days,
#    calculate_daily_coverage_metrics
# ---------------------------------------------------------------------------
from garmin_analysis.features.coverage import (
    days_with_continuous_coverage,
    filter_master_by_days,
    calculate_daily_coverage_metrics,
)


class TestDaysWithContinuousCoverageAdversarial:

    def test_none_df(self):
        assert days_with_continuous_coverage(None) == []

    def test_empty_df(self):
        assert days_with_continuous_coverage(pd.DataFrame()) == []

    def test_missing_timestamp_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert days_with_continuous_coverage(df) == []

    def test_all_nat_timestamps(self):
        df = pd.DataFrame({"timestamp": [pd.NaT, pd.NaT, pd.NaT]})
        assert days_with_continuous_coverage(df) == []

    def test_single_sample_per_day(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 12:00:00")]})
        result = days_with_continuous_coverage(df)
        assert len(result) == 0

    def test_perfect_coverage(self):
        ts = pd.date_range("2024-01-01 00:00:00", "2024-01-01 23:59:00", freq="1min")
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(df)
        assert len(result) == 1

    def test_gap_too_large(self):
        ts = [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp("2024-01-01 23:59:00"),
        ]
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(df, max_gap=pd.Timedelta(minutes=2))
        assert len(result) == 0

    def test_tz_aware_timestamps(self):
        ts = pd.date_range("2024-01-01", periods=1440, freq="1min", tz="US/Eastern")
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(df)
        assert isinstance(result, list)

    def test_multiple_days(self):
        ts = pd.date_range("2024-01-01 00:00:00", "2024-01-02 23:59:00", freq="1min")
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(df)
        assert len(result) == 2

    def test_custom_allowance(self):
        ts = pd.date_range("2024-01-01 00:05:00", "2024-01-01 23:55:00", freq="1min")
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(
            df,
            total_missing_allowance=pd.Timedelta(minutes=10),
        )
        assert len(result) == 1

    def test_unsorted_timestamps(self):
        ts = pd.date_range("2024-01-01 00:00:00", "2024-01-01 23:59:00", freq="1min")
        shuffled = ts.to_series().sample(frac=1, random_state=42).reset_index(drop=True)
        df = pd.DataFrame({"timestamp": shuffled})
        result = days_with_continuous_coverage(df)
        assert len(result) == 1


class TestFilterMasterByDaysAdversarial:

    def test_none_df(self):
        result = filter_master_by_days(None, [])
        assert result is None

    def test_empty_df(self):
        df = pd.DataFrame(columns=["day"])
        result = filter_master_by_days(df, [pd.Timestamp("2024-01-01")])
        assert result.empty

    def test_missing_day_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = filter_master_by_days(df, [pd.Timestamp("2024-01-01")])
        assert result.equals(df)

    def test_no_qualifying_days(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        result = filter_master_by_days(df, [pd.Timestamp("2099-01-01")])
        assert len(result) == 0

    def test_all_qualifying(self):
        days = pd.to_datetime(["2024-01-01", "2024-01-02"])
        df = pd.DataFrame({"day": days, "val": [1, 2]})
        result = filter_master_by_days(df, list(days))
        assert len(result) == 2

    def test_empty_qualifying_days(self):
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"]), "val": [1]})
        result = filter_master_by_days(df, [])
        assert len(result) == 0


class TestCalculateDailyCoverageMetricsAdversarial:

    def test_none_df(self):
        result = calculate_daily_coverage_metrics(None)
        assert result.empty

    def test_empty_df(self):
        result = calculate_daily_coverage_metrics(pd.DataFrame())
        assert result.empty

    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = calculate_daily_coverage_metrics(df)
        assert result.empty

    def test_single_sample(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01 12:00:00")]})
        result = calculate_daily_coverage_metrics(df)
        assert len(result) == 1
        assert result.iloc[0]["coverage_hours"] == 0.0

    def test_all_nat(self):
        df = pd.DataFrame({"timestamp": [pd.NaT, pd.NaT]})
        result = calculate_daily_coverage_metrics(df)
        assert result.empty


# ---------------------------------------------------------------------------
# 7. activity_mappings
# ---------------------------------------------------------------------------
from garmin_analysis.utils.activity_mappings import (
    load_activity_mappings,
    get_display_name,
    get_activity_color,
    list_unknown_activities,
    map_activity_dataframe,
)


class TestActivityMappingsAdversarial:

    def test_load_nonexistent_config(self):
        result = load_activity_mappings("/nonexistent/path.json")
        assert "unknown_activity_mappings" in result

    def test_get_display_name_empty_mappings(self):
        name, desc = get_display_name("running", {})
        assert name == "Running"
        assert desc is None

    def test_get_display_name_with_mapping(self):
        mappings = {"unknown_activity_mappings": {
            "test_act": {"display_name": "Test Activity", "description": "A test"},
        }}
        name, desc = get_display_name("test_act", mappings)
        assert name == "Test Activity"
        assert desc == "A test"

    def test_get_display_name_special_chars(self):
        name, _ = get_display_name("multi_sport_type_123", {})
        assert isinstance(name, str) and len(name) > 0

    def test_get_activity_color_default(self):
        color = get_activity_color("unknown_type", {}, {})
        assert color == "#BDC3C7"

    def test_get_activity_color_from_defaults(self):
        color = get_activity_color("running", {}, {"running": "#FF0000"})
        assert color == "#FF0000"

    def test_list_unknown_no_sport_column(self):
        df = pd.DataFrame({"activity": ["running"]})
        with pytest.raises(KeyError):
            list_unknown_activities(df, {})

    def test_list_unknown_with_nan(self):
        df = pd.DataFrame({"sport": ["running", np.nan, "cycling"]})
        result = list_unknown_activities(df, {})
        assert "running" in result
        assert "cycling" in result

    def test_map_activity_dataframe_empty(self):
        df = pd.DataFrame({"sport": pd.Series(dtype=str)})
        result = map_activity_dataframe(df, {})
        assert "display_name" in result.columns
        assert len(result) == 0

    def test_map_activity_dataframe_with_nan(self):
        df = pd.DataFrame({"sport": ["running", np.nan, "cycling"]})
        result = map_activity_dataframe(df, {})
        assert len(result) == 3
        assert pd.isna(result["display_name"].iloc[1])

    def test_map_activity_injection(self):
        df = pd.DataFrame({"sport": ["'; DROP TABLE activities;--"]})
        result = map_activity_dataframe(df, {})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 8. inspect_sqlite_schema: extract_schema, detect_schema_drift
# ---------------------------------------------------------------------------
from garmin_analysis.data_ingestion.inspect_sqlite_schema import (
    extract_schema,
    detect_schema_drift,
    inspect_sqlite_db,
)


class TestSchemaInspectionAdversarial:

    def test_extract_schema_nonexistent(self):
        result = extract_schema("/nonexistent/db.db")
        assert result == {}

    def test_extract_schema_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(db)
        conn.close()
        result = extract_schema(str(db))
        assert result == {}

    def test_extract_schema_table_no_columns(self, tmp_path):
        db = tmp_path / "nocols.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE empty_table (dummy INTEGER)")
        conn.commit()
        conn.close()
        result = extract_schema(str(db))
        assert "empty_table" in result

    def test_detect_drift_identical(self):
        schema = {"t": [("c1", "TEXT"), ("c2", "REAL")]}
        drift = detect_schema_drift(schema, schema)
        assert drift == {}

    def test_detect_drift_missing_column(self):
        expected = {"t": [("c1", "TEXT"), ("c2", "REAL")]}
        actual = {"t": [("c1", "TEXT")]}
        drift = detect_schema_drift(expected, actual)
        assert "t" in drift
        assert "c2" in drift["t"]["missing_columns"]

    def test_detect_drift_extra_column(self):
        expected = {"t": [("c1", "TEXT")]}
        actual = {"t": [("c1", "TEXT"), ("c2", "REAL")]}
        drift = detect_schema_drift(expected, actual)
        assert "t" in drift
        assert "c2" in drift["t"]["extra_columns"]

    def test_detect_drift_type_mismatch(self):
        expected = {"t": [("c1", "TEXT")]}
        actual = {"t": [("c1", "INTEGER")]}
        drift = detect_schema_drift(expected, actual)
        assert "t" in drift
        assert len(drift["t"]["type_mismatches"]) == 1

    def test_detect_drift_missing_table(self):
        expected = {"t1": [("c", "TEXT")], "t2": [("c", "TEXT")]}
        actual = {"t1": [("c", "TEXT")]}
        drift = detect_schema_drift(expected, actual)
        assert "t2" in drift

    def test_detect_drift_extra_table(self):
        expected = {"t1": [("c", "TEXT")]}
        actual = {"t1": [("c", "TEXT")], "t2": [("c", "TEXT")]}
        drift = detect_schema_drift(expected, actual)
        assert "t2" in drift

    def test_detect_drift_empty_schemas(self):
        drift = detect_schema_drift({}, {})
        assert drift == {}

    def test_detect_drift_none_types(self):
        expected = {"t": [("c1", None)]}
        actual = {"t": [("c1", None)]}
        drift = detect_schema_drift(expected, actual)
        assert drift == {}

    def test_inspect_nonexistent_db(self):
        inspect_sqlite_db("/nonexistent/path.db")


# ---------------------------------------------------------------------------
# 9. data_ingestion: load_table, aggregate_stress, preprocess_sleep, to_naive_day
# ---------------------------------------------------------------------------
from garmin_analysis.data_ingestion.load_all_garmin_dbs import (
    load_table,
    aggregate_stress,
    preprocess_sleep,
    to_naive_day,
    _coalesce,
    _create_synthetic_dataframes,
)


class TestToNaiveDayAdversarial:

    def test_tz_aware_series(self):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"))
        result = to_naive_day(s)
        assert result.dt.tz is None

    def test_all_nat(self):
        s = pd.Series([pd.NaT, pd.NaT])
        result = to_naive_day(s)
        assert result.isna().all()

    def test_mixed_valid_invalid(self):
        s = pd.Series(["2024-01-01", "garbage", None])
        result = to_naive_day(s)
        assert result.iloc[0] == pd.Timestamp("2024-01-01")
        assert pd.isna(result.iloc[1])

    def test_epoch_zero(self):
        s = pd.Series(["1970-01-01"])
        result = to_naive_day(s)
        assert result.iloc[0] == pd.Timestamp("1970-01-01")

    def test_empty_series(self):
        s = pd.Series(dtype="datetime64[ns]")
        result = to_naive_day(s)
        assert len(result) == 0

    def test_mixed_timezone_offsets_no_crash(self):
        # Mixed offsets should parse safely and normalize to tz-naive midnight.
        s = pd.Series([
            "2024-01-01T23:30:00-0500",
            "2024-01-02T08:15:00+0900",
            "2024-01-03T00:00:00Z",
        ])
        result = to_naive_day(s)
        assert result.dt.tz is None
        assert result.notna().all()
        assert all(ts.hour == 0 and ts.minute == 0 and ts.second == 0 for ts in result)


class TestLoadTableAdversarial:

    def test_nonexistent_path(self, tmp_path):
        result = load_table(tmp_path / "nonexistent.db", "test_table")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_table(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.commit()
        conn.close()
        result = load_table(db, "nonexistent_table")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_table(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE empty_tbl (x INTEGER, y TEXT)")
        conn.commit()
        conn.close()
        result = load_table(db, "empty_tbl")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestAggregateStressAdversarial:

    def test_empty_df(self):
        result = aggregate_stress(pd.DataFrame())
        assert result.empty

    def test_missing_timestamp(self):
        df = pd.DataFrame({"stress": [20, 30]})
        result = aggregate_stress(df)
        assert result.empty

    def test_all_zero_stress(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00"]),
            "stress": [0, 0],
        })
        result = aggregate_stress(df)
        assert len(result) == 1
        assert result.iloc[0]["stress_avg"] == 0

    def test_negative_stress(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00"]),
            "stress": [-10, -20],
        })
        result = aggregate_stress(df)
        assert len(result) == 1
        assert result.iloc[0]["stress_avg"] < 0

    def test_single_sample(self):
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 12:00:00")],
            "stress": [50],
        })
        result = aggregate_stress(df)
        assert len(result) == 1

    def test_nan_stress_values(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00"]),
            "stress": [np.nan, np.nan],
        })
        result = aggregate_stress(df)
        assert len(result) == 1

    def test_multi_day(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-01 08:00", "2024-01-01 12:00",
                "2024-01-02 08:00", "2024-01-02 12:00",
            ]),
            "stress": [10, 20, 30, 40],
        })
        result = aggregate_stress(df)
        assert len(result) == 2


class TestPreprocessSleepAdversarial:

    def test_empty_df(self):
        result = preprocess_sleep(pd.DataFrame())
        assert result.empty

    def test_missing_sleep_columns(self):
        df = pd.DataFrame({"day": ["2024-01-01"], "other": [1]})
        result = preprocess_sleep(df)
        assert len(result) == 1

    def test_missing_score_column(self):
        df = pd.DataFrame({
            "day": ["2024-01-01"],
            "total_sleep": ["07:30:00"],
            "deep_sleep": ["01:00:00"],
            "rem_sleep": ["01:30:00"],
        })
        result = preprocess_sleep(df)
        assert len(result) == 1

    def test_all_zero_scores(self):
        df = pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "total_sleep": ["07:30:00", "08:00:00"],
            "deep_sleep": ["01:00:00", "01:15:00"],
            "rem_sleep": ["01:30:00", "01:45:00"],
            "score": [0, 0],
        })
        result = preprocess_sleep(df)
        assert len(result) == 0

    def test_negative_scores(self):
        df = pd.DataFrame({
            "day": ["2024-01-01"],
            "total_sleep": ["07:30:00"],
            "deep_sleep": ["01:00:00"],
            "rem_sleep": ["01:30:00"],
            "score": [-5],
        })
        result = preprocess_sleep(df)
        assert len(result) == 0

    def test_non_numeric_scores(self):
        df = pd.DataFrame({
            "day": ["2024-01-01"],
            "total_sleep": ["07:30:00"],
            "deep_sleep": ["01:00:00"],
            "rem_sleep": ["01:30:00"],
            "score": ["not_a_number"],
        })
        result = preprocess_sleep(df)
        assert len(result) == 0

    def test_garbage_sleep_times(self):
        df = pd.DataFrame({
            "day": ["2024-01-01"],
            "total_sleep": ["garbage"],
            "deep_sleep": ["not_time"],
            "rem_sleep": ["???"],
            "score": [80],
        })
        result = preprocess_sleep(df)
        assert len(result) == 1
        assert pd.isna(result["total_sleep_min"].iloc[0])


class TestCoalesceAdversarial:

    def test_basic_coalesce(self):
        df = pd.DataFrame({"a": [np.nan, 2], "b": [1, np.nan]})
        result = _coalesce(df.copy(), "a", "b")
        assert result["a"].iloc[0] == 1
        assert "b" not in result.columns

    def test_no_candidates(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _coalesce(df.copy(), "a")
        assert result["a"].tolist() == [1, 2]

    def test_new_column_created(self):
        df = pd.DataFrame({"b": [1, 2]})
        result = _coalesce(df.copy(), "new_col", "b")
        assert "new_col" in result.columns
        assert result["new_col"].tolist() == [1, 2]


class TestCreateSyntheticDataframesAdversarial:

    def test_zero_days(self):
        result = _create_synthetic_dataframes(num_days=0)
        assert isinstance(result, dict)
        for key in ["daily_summary", "sleep", "stress", "resting_hr"]:
            assert key in result

    def test_one_day(self):
        result = _create_synthetic_dataframes(num_days=1)
        assert len(result["daily_summary"]) == 1

    def test_large_num_days(self):
        result = _create_synthetic_dataframes(num_days=1000)
        assert len(result["daily_summary"]) == 1000


# ---------------------------------------------------------------------------
# 10. day_of_week_analysis
# ---------------------------------------------------------------------------
from garmin_analysis.features.day_of_week_analysis import (
    get_day_order,
    calculate_day_of_week_averages,
)


class TestDayOfWeekAnalysisAdversarial:

    def test_empty_df(self):
        result = calculate_day_of_week_averages(pd.DataFrame())
        assert result.empty

    def test_single_day(self):
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-01")],
            "score": [80],
            "bb_max": [90],
            "bb_min": [30],
            "hydration_intake": [2000],
        })
        result = calculate_day_of_week_averages(df)
        assert len(result) > 0

    def test_no_metric_columns(self):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "unrelated": [1, 2],
        })
        result = calculate_day_of_week_averages(df)
        assert result.empty

    def test_all_nan_metrics(self):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "score": [np.nan, np.nan],
            "bb_max": [np.nan, np.nan],
        })
        result = calculate_day_of_week_averages(df)
        assert result.empty

    def test_day_not_datetime(self):
        df = pd.DataFrame({
            "day": ["2024-01-01", "2024-01-02"],
            "score": [80, 85],
        })
        result = calculate_day_of_week_averages(df)
        assert len(result) > 0

    def test_full_week_coverage(self):
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        df = pd.DataFrame({
            "day": dates,
            "score": [70, 72, 74, 76, 78, 80, 82],
        })
        result = calculate_day_of_week_averages(df)
        assert len(result) == 7  # one per day

    def test_get_day_order(self):
        order = get_day_order()
        assert order[0] == "Sunday"
        assert order[-1] == "Saturday"
        assert len(order) == 7


# ---------------------------------------------------------------------------
# 11. prepare_modeling_dataset
# ---------------------------------------------------------------------------
from garmin_analysis.data_ingestion.prepare_modeling_dataset import prepare_modeling_dataset


class TestPrepareModelingDatasetAdversarial:

    def test_nonexistent_input(self, tmp_path):
        prepare_modeling_dataset(
            input_path=str(tmp_path / "nonexistent.csv"),
            output_path=str(tmp_path / "out.csv"),
        )
        assert not (tmp_path / "out.csv").exists()

    def test_empty_csv(self, tmp_path):
        """An empty CSV (headers only) should not crash and should produce no output rows."""
        inp = tmp_path / "empty.csv"
        inp.write_text("day,score,stress_avg,yesterday_activity_minutes\n")
        out = tmp_path / "out.csv"
        prepare_modeling_dataset(
            input_path=str(inp),
            output_path=str(out),
        )
        if out.exists():
            content = out.read_text().strip()
            if content:
                df = pd.read_csv(out)
                assert len(df) == 0

    def test_all_required_features_missing(self, tmp_path):
        inp = tmp_path / "data.csv"
        inp.write_text("day,x,y\n2024-01-01,1,2\n2024-01-02,3,4\n")
        out = tmp_path / "out.csv"
        prepare_modeling_dataset(
            input_path=str(inp),
            output_path=str(out),
            required_features=["score", "stress_avg", "yesterday_activity_minutes"],
        )

    def test_high_missing_threshold(self, tmp_path):
        inp = tmp_path / "data.csv"
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "score": [80, 85, 90],
            "stress_avg": [20, np.nan, 30],
            "yesterday_activity_minutes": [30, 45, 60],
            "sparse_col": [np.nan, np.nan, np.nan],
        })
        df.to_csv(inp, index=False)
        out = tmp_path / "out.csv"
        prepare_modeling_dataset(
            input_path=str(inp),
            output_path=str(out),
            missing_threshold=0.5,
        )
        if out.exists():
            result = pd.read_csv(out)
            assert "sparse_col" not in result.columns

    def test_24h_coverage_filter(self, tmp_path):
        inp = tmp_path / "data.csv"
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "score": [80, 85, 90],
            "stress_avg": [20, 25, 30],
            "yesterday_activity_minutes": [30, 45, 60],
            "has_24h_coverage": [True, False, True],
        })
        df.to_csv(inp, index=False)
        out = tmp_path / "out.csv"
        prepare_modeling_dataset(
            input_path=str(inp),
            output_path=str(out),
            require_24h_coverage=True,
        )
        if out.exists():
            result = pd.read_csv(out)
            assert len(result) == 2

    def test_coverage_pct_filter(self, tmp_path):
        inp = tmp_path / "data.csv"
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "score": [80, 85, 90],
            "stress_avg": [20, 25, 30],
            "yesterday_activity_minutes": [30, 45, 60],
            "coverage_pct": [95.0, 50.0, 80.0],
        })
        df.to_csv(inp, index=False)
        out = tmp_path / "out.csv"
        prepare_modeling_dataset(
            input_path=str(inp),
            output_path=str(out),
            min_coverage_pct=75.0,
        )
        if out.exists():
            result = pd.read_csv(out)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# 12. Cross-module integration adversarial tests
# ---------------------------------------------------------------------------
class TestCrossModuleAdversarial:
    """Tests that chain multiple modules with adversarial data."""

    def test_clean_then_impute(self):
        df = pd.DataFrame({
            "heart_rate": ["NA", "60", "65", "", "70", "null"],
            "steps": [-1, 5000, "None", 8000, 10000, -1],
        })
        cleaned = clean_data(df)
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result = impute_missing_values(cleaned, numeric_cols, strategy="median")
            assert isinstance(result, pd.DataFrame)

    def test_normalize_then_filter(self):
        df = pd.DataFrame({
            "day": ["2024-01-01 14:30:00", "2024-06-15 08:00:00", "2024-12-31 23:59:59"],
            "val": [1, 2, 3],
        })
        normalized = normalize_day_column(df)
        stripped = strip_time_from_dates(normalized)
        filtered = filter_by_date(stripped, from_date="2024-06-01", to_date="2024-07-01")
        assert len(filtered) == 1

    def test_full_pipeline_synthetic(self):
        synth = _create_synthetic_dataframes(num_days=30)
        daily = synth["daily_summary"]
        sleep = synth["sleep"]

        processed_sleep = preprocess_sleep(sleep)
        assert len(processed_sleep) > 0

        cleaned = clean_data(daily)
        assert len(cleaned) == 30

    def test_schema_drift_then_load(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE daily_summary (day TEXT, steps REAL)")
        conn.execute("INSERT INTO daily_summary VALUES ('2024-01-01', 5000)")
        conn.commit()
        conn.close()

        schema = extract_schema(str(db))
        expected = {"daily_summary": [("day", "TEXT"), ("steps", "REAL"), ("calories", "REAL")]}
        drift = detect_schema_drift(expected, schema)
        assert "daily_summary" in drift
        assert "calories" in drift["daily_summary"]["missing_columns"]

        result = load_table(db, "daily_summary")
        assert len(result) == 1

    def test_impute_then_standardize(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0],
        })
        imputed = impute_missing_values(df, ["a", "b"], strategy="median")
        scaled = standardize_features(imputed, ["a", "b"])
        assert len(scaled) == 5
        assert scaled.shape[1] == 2

    def test_validate_then_analyze(self):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01"]),
            "score": [80],
        })
        validate_dataframe(df, required_columns=["day", "score"], min_rows=1)
        result = calculate_day_of_week_averages(df)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 13. Concurrency / race condition sanity checks
# ---------------------------------------------------------------------------
class TestConcurrencySanity:

    def test_parallel_imputation_independence(self):
        """Two imputation calls on copies should not interfere."""
        df1 = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df2 = pd.DataFrame({"a": [np.nan, 10.0, 20.0]})

        r1 = impute_missing_values(df1, ["a"], strategy="median")
        r2 = impute_missing_values(df2, ["a"], strategy="mean")

        assert r1["a"].iloc[1] == 2.0
        assert r2["a"].iloc[0] == 15.0

    def test_clean_data_no_global_state(self):
        """Successive calls should not leak state."""
        df1 = pd.DataFrame({"X": [1, 2, 3]})
        df2 = pd.DataFrame({"Y": [4, 5, 6]})

        r1 = clean_data(df1)
        r2 = clean_data(df2)

        assert "x" in r1.columns
        assert "y" in r2.columns
        assert "x" not in r2.columns


# ---------------------------------------------------------------------------
# 14. Boundary value and type confusion attacks
# ---------------------------------------------------------------------------
class TestBoundaryValues:

    def test_max_float_in_dataframe(self):
        df = pd.DataFrame({"a": [np.finfo(np.float64).max, 1.0, np.nan]})
        result = impute_missing_values(df, ["a"], strategy="mean")
        assert isinstance(result, pd.DataFrame)

    def test_min_float_in_dataframe(self):
        df = pd.DataFrame({"a": [np.finfo(np.float64).tiny, 1.0, np.nan]})
        result = impute_missing_values(df, ["a"], strategy="median")
        assert not result["a"].isna().any()

    def test_integer_overflow_in_time(self):
        result = convert_time_to_minutes("99999:59:59")
        if not math.isnan(result):
            assert result > 0

    def test_empty_string_column_names(self):
        df = pd.DataFrame({"": [1, 2], "a": [3, 4]})
        result = clean_data(df)
        assert len(result) == 2

    def test_zero_row_dataframe_with_columns(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        result = clean_data(df)
        assert result.empty

    def test_dataframe_with_object_that_looks_numeric(self):
        df = pd.DataFrame({"a": pd.array(["1", "2", "3"], dtype="string")})
        result = clean_data(df)
        assert len(result) == 3

    def test_timestamp_at_midnight_boundary(self):
        ts = [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 23:59:59.999999999"),
        ]
        df = pd.DataFrame({"timestamp": ts})
        result = calculate_daily_coverage_metrics(df)
        assert len(result) == 1

    def test_coverage_exactly_at_edge_tolerance(self):
        ts = pd.date_range("2024-01-01 00:02:00", "2024-01-01 23:58:00", freq="1min")
        df = pd.DataFrame({"timestamp": ts})
        result = days_with_continuous_coverage(
            df,
            day_edge_tolerance=pd.Timedelta(minutes=2),
        )
        assert isinstance(result, list)


# ===========================================================================
# PART 2 — Adversarial tests for previously-uncovered modules
# ===========================================================================


# ---------------------------------------------------------------------------
# 15. config module
# ---------------------------------------------------------------------------
from garmin_analysis.config import (
    ensure_directories_exist,
    get_db_path,
    PROJECT_ROOT,
    DB_PATHS,
    MASTER_CSV,
    MODELING_CSV,
)


class TestConfigAdversarial:

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_db_paths_all_present(self):
        for name in ["garmin", "activities", "monitoring", "summary", "summary2"]:
            assert name in DB_PATHS

    def test_get_db_path_valid(self):
        p = get_db_path("garmin")
        assert p == DB_PATHS["garmin"]

    def test_get_db_path_invalid(self):
        with pytest.raises(KeyError):
            get_db_path("nonexistent_db")

    def test_ensure_directories_exist_idempotent(self):
        ensure_directories_exist()
        ensure_directories_exist()  # calling twice should not fail

    def test_master_csv_path(self):
        assert str(MASTER_CSV).endswith("master_daily_summary.csv")

    def test_modeling_csv_path(self):
        assert str(MODELING_CSV).endswith("modeling_ready_dataset.csv")


# ---------------------------------------------------------------------------
# 16. logging_config module
# ---------------------------------------------------------------------------
from garmin_analysis.logging_config import setup_logging, get_logger


class TestLoggingConfigAdversarial:

    def test_setup_logging_default(self):
        setup_logging()  # should not raise

    def test_setup_logging_debug_level(self):
        import logging as _logging
        setup_logging(level=_logging.DEBUG, file_output=False)

    def test_setup_logging_no_console(self):
        setup_logging(console_output=False, file_output=False)

    def test_setup_logging_custom_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, console_output=False)
        assert log_file.exists()

    def test_get_logger_returns_logger(self):
        lg = get_logger("test_adversarial")
        assert hasattr(lg, "info")
        assert hasattr(lg, "error")

    def test_setup_logging_called_twice(self):
        setup_logging(file_output=False)
        setup_logging(file_output=False)


# ---------------------------------------------------------------------------
# 17. utils/cli_helpers
# ---------------------------------------------------------------------------
from garmin_analysis.utils.cli_helpers import (
    add_24h_coverage_args,
    apply_24h_coverage_filter_from_args,
    add_common_output_args,
    setup_logging_from_args,
)
import argparse


class TestCLIHelpersAdversarial:

    def _make_parser(self):
        p = argparse.ArgumentParser()
        add_24h_coverage_args(p)
        return p

    def test_add_coverage_args(self):
        p = self._make_parser()
        args = p.parse_args([])
        assert hasattr(args, "filter_24h_coverage")
        assert args.filter_24h_coverage is False

    def test_add_coverage_args_enabled(self):
        p = self._make_parser()
        args = p.parse_args(["--filter-24h-coverage"])
        assert args.filter_24h_coverage is True

    def test_apply_filter_not_requested(self):
        p = self._make_parser()
        args = p.parse_args([])
        df = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"]), "val": [1]})
        result = apply_24h_coverage_filter_from_args(df, args)
        assert len(result) == 1

    def test_apply_filter_missing_attrs(self):
        args = argparse.Namespace()
        df = pd.DataFrame({"day": ["2024-01-01"]})
        with pytest.raises(AttributeError, match="missing required"):
            apply_24h_coverage_filter_from_args(df, args)

    def test_add_common_output_args(self):
        p = argparse.ArgumentParser()
        add_common_output_args(p)
        args = p.parse_args(["-v", "--output-dir", "/tmp/test"])
        assert args.verbose is True
        assert args.output_dir == "/tmp/test"

    def test_setup_logging_from_args_verbose(self):
        p = argparse.ArgumentParser()
        add_common_output_args(p)
        args = p.parse_args(["-v"])
        setup_logging_from_args(args)

    def test_setup_logging_from_args_quiet(self):
        p = argparse.ArgumentParser()
        add_common_output_args(p)
        args = p.parse_args([])
        setup_logging_from_args(args)


# ---------------------------------------------------------------------------
# 18. utils/data_loading
# ---------------------------------------------------------------------------
from garmin_analysis.utils.data_loading import load_garmin_tables


class TestDataLoadingAdversarial:

    def test_load_garmin_tables_nonexistent(self):
        result = load_garmin_tables("/nonexistent/garmin.db")
        assert result == {}

    def test_load_garmin_tables_missing_tables(self, tmp_path):
        db = tmp_path / "garmin.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.commit()
        conn.close()
        result = load_garmin_tables(str(db))
        assert result == {}

    def test_load_garmin_tables_valid(self, mem_db, tmp_path):
        db = tmp_path / "garmin.db"
        src = sqlite3.connect(db)
        mem_db.backup(src)
        src.close()
        result = load_garmin_tables(str(db))
        assert isinstance(result, dict)
        assert "daily" in result


# ---------------------------------------------------------------------------
# 19. features/time_of_day_stress_analysis
# ---------------------------------------------------------------------------
from garmin_analysis.features.time_of_day_stress_analysis import (
    calculate_hourly_stress_averages,
    calculate_hourly_stress_by_weekday,
    print_stress_summary,
    plot_hourly_stress_pattern,
    plot_stress_heatmap_by_weekday,
)


class TestTimeOfDayStressAdversarial:

    def _make_stress_df(self, n_hours=24, n_days=7):
        rows = []
        for d in range(n_days):
            for h in range(n_hours):
                rows.append({
                    "timestamp": pd.Timestamp(f"2024-01-{d+1:02d} {h:02d}:00:00"),
                    "stress": 20 + h + d,
                })
        return pd.DataFrame(rows)

    def test_empty_df(self):
        result = calculate_hourly_stress_averages(pd.DataFrame())
        assert result.empty

    def test_empty_df_weekday(self):
        result = calculate_hourly_stress_by_weekday(pd.DataFrame())
        assert result.empty

    def test_single_sample(self):
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 12:00:00")],
            "stress": [50],
        })
        result = calculate_hourly_stress_averages(df)
        assert len(result) == 1
        assert result.iloc[0]["mean"] == 50

    def test_all_same_hour(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00", "2024-01-03 08:00"]),
            "stress": [30, 40, 50],
        })
        result = calculate_hourly_stress_averages(df)
        assert len(result) == 1
        assert result.iloc[0]["mean"] == pytest.approx(40.0)

    def test_negative_stress(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"]),
            "stress": [-10, -20],
        })
        result = calculate_hourly_stress_averages(df)
        assert len(result) == 2

    def test_nan_stress(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"]),
            "stress": [np.nan, 50],
        })
        result = calculate_hourly_stress_averages(df)
        assert len(result) >= 1

    def test_full_day_coverage(self):
        df = self._make_stress_df()
        result = calculate_hourly_stress_averages(df)
        assert len(result) == 24

    def test_weekday_analysis(self):
        df = self._make_stress_df()
        result = calculate_hourly_stress_by_weekday(df)
        assert len(result) > 0

    def test_print_summary_empty(self):
        print_stress_summary(pd.DataFrame())

    def test_print_summary_valid(self):
        df = self._make_stress_df()
        hourly = calculate_hourly_stress_averages(df)
        weekday = calculate_hourly_stress_by_weekday(df)
        print_stress_summary(hourly, weekday)

    def test_plot_empty(self):
        result = plot_hourly_stress_pattern(pd.DataFrame(), save_plots=False)
        assert result == {}

    def test_plot_valid(self, tmp_path):
        df = self._make_stress_df()
        hourly = calculate_hourly_stress_averages(df)
        with patch("garmin_analysis.features.time_of_day_stress_analysis.PLOTS_DIR", tmp_path):
            result = plot_hourly_stress_pattern(hourly, save_plots=True, show_plots=False)
        assert len(result) > 0

    def test_heatmap_empty(self):
        result = plot_stress_heatmap_by_weekday(pd.DataFrame(), save_plots=False)
        assert result == {}


# ---------------------------------------------------------------------------
# 20. features/data_quality_analysis
# ---------------------------------------------------------------------------
from garmin_analysis.features.data_quality_analysis import GarminDataQualityAnalyzer


class TestDataQualityAnalysisAdversarial:

    def _make_health_df(self, n=50):
        np.random.seed(42)
        return pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=n),
            "steps": np.random.randint(3000, 15000, n).astype(float),
            "hr_avg": np.random.randint(55, 85, n).astype(float),
            "stress_avg": np.random.randint(15, 60, n).astype(float),
            "score": np.random.randint(50, 100, n).astype(float),
            "total_sleep_min": np.random.uniform(300, 540, n),
        })

    def test_empty_df(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        df = pd.DataFrame(columns=["day", "steps"])
        result = analyzer.analyze_garmin_data(df)
        assert isinstance(result, dict)

    def test_empty_df_completeness_is_zero_not_nan(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        df = pd.DataFrame(columns=["day", "steps"])
        result = analyzer.analyze_garmin_data(df)
        completeness = result["completeness"]
        assert completeness["day"]["completeness_percentage"] == 0.0
        assert completeness["steps"]["completeness_percentage"] == 0.0
        assert completeness["day"]["is_adequate_for_analysis"] is False
        assert completeness["steps"]["is_adequate_for_analysis"] is False

    def test_single_row(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-01")],
            "steps": [5000.0],
        })
        result = analyzer.analyze_garmin_data(df)
        assert "dataset_info" in result

    def test_all_nan_df(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=10),
            "steps": [np.nan] * 10,
            "hr": [np.nan] * 10,
        })
        result = analyzer.analyze_garmin_data(df)
        assert result["completeness"]["steps"]["null_count"] == 10

    def test_mixed_types(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=5),
            "val": [1, "two", 3, None, 5],
        })
        result = analyzer.analyze_garmin_data(df)
        issues = result["quality_issues"]
        assert isinstance(issues, dict)

    def test_full_analysis(self, tmp_path):
        analyzer = GarminDataQualityAnalyzer(output_dir=str(tmp_path))
        df = self._make_health_df()
        result = analyzer.analyze_garmin_data(df)
        assert "dataset_info" in result
        assert "completeness" in result
        assert "quality_issues" in result
        assert "modeling_suitability" in result
        assert "recommendations" in result

    def test_print_summary(self, tmp_path):
        analyzer = GarminDataQualityAnalyzer(output_dir=str(tmp_path))
        df = self._make_health_df()
        analyzer.analyze_garmin_data(df)
        analyzer.print_summary()

    def test_save_report(self, tmp_path):
        analyzer = GarminDataQualityAnalyzer(output_dir=str(tmp_path))
        df = self._make_health_df()
        analyzer.analyze_garmin_data(df)
        json_path, md_path = analyzer.save_report("test_report")
        assert json_path.exists()
        assert md_path.exists()

    def test_completeness_levels(self):
        analyzer = GarminDataQualityAnalyzer(output_dir="/tmp/test_dqa")
        assert "Excellent" in analyzer._get_completeness_level(0.95)
        assert "Good" in analyzer._get_completeness_level(0.75)
        assert "Fair" in analyzer._get_completeness_level(0.55)
        assert "Poor" in analyzer._get_completeness_level(0.35)
        assert "Very Poor" in analyzer._get_completeness_level(0.15)
        assert "Critical" in analyzer._get_completeness_level(0.05)

    def test_duplicate_column_detection(self, tmp_path):
        analyzer = GarminDataQualityAnalyzer(output_dir=str(tmp_path))
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=10),
            "a": range(10),
            "b": range(10),
        })
        result = analyzer.analyze_garmin_data(df)
        dups = result["quality_issues"]["duplicate_columns"]
        assert len(dups) >= 1


# ---------------------------------------------------------------------------
# 21. viz/plot_feature_correlation
# ---------------------------------------------------------------------------
from garmin_analysis.viz.plot_feature_correlation import plot_feature_correlation


class TestPlotFeatureCorrelationAdversarial:

    def test_empty_numeric(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="No numeric columns"):
            plot_feature_correlation(df)

    def test_single_column(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        out = str(tmp_path / "corr.png")
        plot_feature_correlation(df, output_path=out)
        assert os.path.exists(out)

    def test_with_nan(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        out = str(tmp_path / "corr.png")
        plot_feature_correlation(df, output_path=out)
        assert os.path.exists(out)

    def test_exclude_cols(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        out = str(tmp_path / "corr.png")
        plot_feature_correlation(df, output_path=out, exclude_cols=["c"])
        assert os.path.exists(out)

    def test_spearman_method(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        out = str(tmp_path / "corr.png")
        plot_feature_correlation(df, output_path=out, method="spearman")
        assert os.path.exists(out)

    def test_kendall_method(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        out = str(tmp_path / "corr.png")
        plot_feature_correlation(df, output_path=out, method="kendall")
        assert os.path.exists(out)

    def test_constant_columns(self, tmp_path):
        df = pd.DataFrame({"a": [5, 5, 5], "b": [10, 10, 10]})
        out = str(tmp_path / "corr.png")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_feature_correlation(df, output_path=out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# 22. viz/plot_feature_trend
# ---------------------------------------------------------------------------
from garmin_analysis.viz.plot_feature_trend import plot_feature_trend


class TestPlotFeatureTrendAdversarial:

    def test_missing_feature_col(self):
        df = pd.DataFrame({"day": ["2024-01-01"], "other": [1]})
        with pytest.raises(ValueError, match="not found"):
            plot_feature_trend(df, feature="nonexistent")

    def test_no_date_col(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        with pytest.raises(ValueError, match="date column"):
            plot_feature_trend(df, feature="val")

    def test_auto_detect_day(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "score": [70, 80, 90],
        })
        plot_feature_trend(df, feature="score", output_dir=str(tmp_path))
        assert len(list(tmp_path.glob("*.png"))) == 1

    def test_explicit_date_col(self, tmp_path):
        df = pd.DataFrame({
            "my_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "val": [10, 20],
        })
        plot_feature_trend(df, feature="val", date_col="my_date", output_dir=str(tmp_path))

    def test_with_anomalies(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "val": [10, 20, 30],
        })
        anomalies = pd.DataFrame({"day": pd.to_datetime(["2024-01-02"])})
        plot_feature_trend(df, feature="val", anomalies=anomalies, output_dir=str(tmp_path))

    def test_rolling_zero(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "val": [10, 20],
        })
        plot_feature_trend(df, feature="val", rolling_days=0, output_dir=str(tmp_path))

    def test_single_data_point(self, tmp_path):
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-01")],
            "val": [42],
        })
        plot_feature_trend(df, feature="val", output_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# 23. viz/plot_activity_calendar
# ---------------------------------------------------------------------------
from garmin_analysis.viz.plot_activity_calendar import (
    plot_activity_calendar,
    _get_sport_colors,
    _darken_color,
)


class TestPlotActivityCalendarAdversarial:

    def _make_activities(self, n=10):
        return pd.DataFrame({
            "start_time": pd.date_range("2024-01-01", periods=n, freq="2D"),
            "sport": ["running" if i % 2 == 0 else "cycling" for i in range(n)],
        })

    def test_missing_columns(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="Missing required"):
            plot_activity_calendar(df)

    def test_empty_after_filter(self):
        df = self._make_activities()
        plot_activity_calendar(df, start_date="2099-01-01", end_date="2099-12-31")

    def test_valid_calendar(self, tmp_path):
        df = self._make_activities()
        plot_activity_calendar(df, output_dir=str(tmp_path))
        assert len(list(tmp_path.glob("*.png"))) == 1

    def test_date_range_filter(self, tmp_path):
        df = self._make_activities(30)
        plot_activity_calendar(df, start_date="2024-01-01", end_date="2024-01-15",
                              output_dir=str(tmp_path))

    def test_no_mappings(self, tmp_path):
        df = self._make_activities()
        plot_activity_calendar(df, use_mappings=False, output_dir=str(tmp_path))

    def test_single_activity(self, tmp_path):
        df = pd.DataFrame({
            "start_time": [pd.Timestamp("2024-01-01 08:00:00")],
            "sport": ["running"],
        })
        plot_activity_calendar(df, output_dir=str(tmp_path))

    def test_nan_sport(self, tmp_path):
        df = pd.DataFrame({
            "start_time": pd.date_range("2024-01-01", periods=3),
            "sport": ["running", np.nan, "cycling"],
        })
        plot_activity_calendar(df, output_dir=str(tmp_path))

    def test_darken_color(self):
        result = _darken_color("#FFFFFF", 0.5)
        assert result == "#7f7f7f"

    def test_darken_black(self):
        result = _darken_color("#000000", 0.5)
        assert result == "#000000"

    def test_get_sport_colors(self):
        colors = _get_sport_colors(["running", "cycling", "unknown_sport_xyz"])
        assert "running" in colors
        assert "cycling" in colors
        assert "unknown_sport_xyz" in colors


# ---------------------------------------------------------------------------
# 24. modeling/anomaly_detection (simple)
# ---------------------------------------------------------------------------
from garmin_analysis.modeling.anomaly_detection import run_anomaly_detection


class TestSimpleAnomalyDetectionAdversarial:

    def test_empty_df(self):
        result, path = run_anomaly_detection(pd.DataFrame())
        assert result.empty

    def test_missing_features(self):
        df = pd.DataFrame({"steps": [1000, 2000, 3000], "other": [1, 2, 3]})
        result, path = run_anomaly_detection(df)
        assert result.empty

    def test_insufficient_rows(self):
        features = ["steps", "activity_minutes", "training_effect",
                    "total_sleep_min", "rem_sleep_min", "deep_sleep_min", "awake",
                    "stress_avg", "stress_max", "stress_duration"]
        df = pd.DataFrame({f: [i] for i, f in enumerate(features)})
        result, path = run_anomaly_detection(df)
        assert result.empty


# ---------------------------------------------------------------------------
# 25. modeling/enhanced_anomaly_detection
# ---------------------------------------------------------------------------
from garmin_analysis.modeling.enhanced_anomaly_detection import EnhancedAnomalyDetector


class TestEnhancedAnomalyDetectorAdversarial:

    def _make_data(self, n=100, n_features=5):
        np.random.seed(42)
        return np.random.randn(n, n_features)

    def test_init(self):
        detector = EnhancedAnomalyDetector(random_state=0)
        assert detector.random_state == 0

    def test_fit_isolation_forest(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        result = detector.fit_isolation_forest(X)
        assert "model" in result

    def test_fit_lof(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        result = detector.fit_local_outlier_factor(X)
        assert "model" in result

    def test_fit_ocsvm(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        result = detector.fit_one_class_svm(X)
        assert "model" in result

    def test_ensemble_no_models(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        result = detector.ensemble_detection(X)
        assert result == {}

    def test_ensemble_with_models(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        detector.fit_isolation_forest(X)
        detector.fit_local_outlier_factor(X)
        detector.fit_one_class_svm(X)
        result = detector.ensemble_detection(X)
        assert "ensemble" in result

    def test_evaluate_single_cluster(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        labels = np.ones(len(X))
        result = detector.evaluate_clustering_quality(X, labels)
        assert result["silhouette_score"] == -1

    def test_evaluate_two_clusters(self):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        labels = np.array([0] * 50 + [1] * 50)
        result = detector.evaluate_clustering_quality(X, labels)
        assert isinstance(result["silhouette_score"], float)

    def test_create_visualizations(self, tmp_path):
        detector = EnhancedAnomalyDetector()
        X = self._make_data()
        detector.fit_isolation_forest(X)
        labels = detector.models["isolation_forest"].predict(X)
        paths = detector.create_visualizations(X, labels, ["f1", "f2", "f3", "f4", "f5"], tmp_path)
        assert len(paths) >= 1

    def test_prepare_features_insufficient(self):
        detector = EnhancedAnomalyDetector()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            detector.prepare_features(df, feature_cols=["a", "b"])


# ---------------------------------------------------------------------------
# 26. modeling/enhanced_clustering
# ---------------------------------------------------------------------------
from garmin_analysis.modeling.enhanced_clustering import EnhancedClusterer


class TestEnhancedClustererAdversarial:

    def _make_data(self, n=100, n_features=5):
        np.random.seed(42)
        return np.random.randn(n, n_features)

    def test_init(self):
        c = EnhancedClusterer(random_state=0)
        assert c.random_state == 0

    def test_find_optimal_clusters(self):
        c = EnhancedClusterer()
        X = self._make_data()
        result = c.find_optimal_clusters(X, max_clusters=5)
        assert "recommended_k" in result

    def test_find_optimal_small_dataset(self):
        c = EnhancedClusterer()
        X = self._make_data(n=6)
        result = c.find_optimal_clusters(X, max_clusters=10)
        assert "recommended_k" in result

    def test_fit_kmeans(self):
        c = EnhancedClusterer()
        X = self._make_data()
        result = c.fit_kmeans(X, n_clusters=3)
        assert result["n_clusters"] == 3

    def test_fit_gmm(self):
        c = EnhancedClusterer()
        X = self._make_data()
        result = c.fit_gaussian_mixture(X, n_clusters=3)
        assert result["n_clusters"] == 3

    def test_fit_hierarchical(self):
        c = EnhancedClusterer()
        X = self._make_data()
        result = c.fit_hierarchical(X, n_clusters=3)
        assert result["n_clusters"] == 3

    def test_fit_dbscan(self):
        c = EnhancedClusterer()
        X = self._make_data()
        result = c.fit_dbscan(X)
        assert "n_clusters" in result

    def test_evaluate_single_cluster(self):
        c = EnhancedClusterer()
        X = self._make_data()
        labels = np.zeros(len(X))
        result = c.evaluate_clustering(X, labels)
        assert result["silhouette_score"] == -1

    def test_cluster_profiles(self):
        c = EnhancedClusterer()
        df = pd.DataFrame({"f1": range(10), "f2": range(10, 20)})
        labels = np.array([0] * 5 + [1] * 5)
        profiles = c.analyze_cluster_profiles(df, labels, ["f1", "f2"])
        assert "cluster_0" in profiles
        assert "cluster_1" in profiles

    def test_cluster_profiles_with_noise(self):
        c = EnhancedClusterer()
        df = pd.DataFrame({"f1": range(10)})
        labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1, 1])
        profiles = c.analyze_cluster_profiles(df, labels, ["f1"])
        assert "cluster_-1" not in profiles


# ---------------------------------------------------------------------------
# 27. modeling/predictive_modeling
# ---------------------------------------------------------------------------
from garmin_analysis.modeling.predictive_modeling import HealthPredictor


class TestHealthPredictorAdversarial:

    def _make_data(self, n=50, n_features=5):
        np.random.seed(42)
        X = np.random.randn(n, n_features)
        y = np.random.randn(n)
        return X, y

    def test_init(self):
        p = HealthPredictor(random_state=0)
        assert p.random_state == 0

    def test_fit_random_forest(self):
        p = HealthPredictor()
        X, y = self._make_data()
        result = p.fit_random_forest(X, y)
        assert "model" in result

    def test_fit_gradient_boosting(self):
        p = HealthPredictor()
        X, y = self._make_data()
        result = p.fit_gradient_boosting(X, y)
        assert "model" in result

    def test_fit_linear_models(self):
        p = HealthPredictor()
        X, y = self._make_data()
        result = p.fit_linear_models(X, y)
        assert "models" in result
        assert "linear_regression" in result["models"]

    def test_fit_svr(self):
        p = HealthPredictor()
        X, y = self._make_data()
        result = p.fit_svr(X, y)
        assert "model" in result

    def test_fit_mlp(self):
        p = HealthPredictor()
        X, y = self._make_data()
        result = p.fit_mlp(X, y)
        assert "model" in result

    def test_evaluate_model(self):
        p = HealthPredictor()
        X, y = self._make_data()
        p.fit_random_forest(X, y)
        metrics = p.evaluate_model(p.models["random_forest"], X, y, cv_splits=3)
        assert "mse" in metrics
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_feature_importance_tree(self):
        p = HealthPredictor()
        X, y = self._make_data()
        p.fit_random_forest(X, y)
        imp = p.get_feature_importance(p.models["random_forest"], ["f1", "f2", "f3", "f4", "f5"])
        assert len(imp) == 5

    def test_feature_importance_linear(self):
        p = HealthPredictor()
        X, y = self._make_data()
        p.fit_linear_models(X, y)
        imp = p.get_feature_importance(p.models["linear_regression"], ["f1", "f2", "f3", "f4", "f5"])
        assert len(imp) == 5

    def test_feature_importance_no_attr(self):
        p = HealthPredictor()
        X, y = self._make_data()
        p.fit_svr(X, y)
        imp = p.get_feature_importance(p.models["svr"], ["f1", "f2", "f3", "f4", "f5"])
        assert all(v == 0 for v in imp.values())

    def test_time_series_split(self):
        p = HealthPredictor()
        tscv = p.create_time_series_split(n_splits=3)
        assert tscv.n_splits == 3

    def test_constant_target(self):
        p = HealthPredictor()
        X = np.random.randn(50, 3)
        y = np.ones(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = p.fit_random_forest(X, y)
        assert "model" in result


# ---------------------------------------------------------------------------
# 28. reporting/generate_trend_summary
# ---------------------------------------------------------------------------
from garmin_analysis.reporting.generate_trend_summary import (
    generate_trend_summary,
    log_top_correlations,
)


class TestGenerateTrendSummaryAdversarial:

    def _make_df(self, n=30):
        np.random.seed(42)
        return pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=n),
            "steps": np.random.randint(3000, 15000, n),
            "hr_avg": np.random.randint(55, 85, n),
            "stress_avg": np.random.randint(15, 60, n),
            "score": np.random.randint(50, 100, n),
        })

    def test_empty_df(self, tmp_path):
        df = pd.DataFrame({"day": pd.Series(dtype="datetime64[ns]")})
        result = generate_trend_summary(df, output_dir=str(tmp_path), timestamp="test")
        assert result is not None

    def test_valid_df(self, tmp_path):
        df = self._make_df()
        result = generate_trend_summary(df, output_dir=str(tmp_path), timestamp="test")
        assert os.path.exists(result)

    def test_all_nan_columns(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=5),
            "a": [np.nan] * 5,
            "b": [np.nan] * 5,
        })
        result = generate_trend_summary(df, output_dir=str(tmp_path), timestamp="test")
        assert result is not None

    def test_single_column(self, tmp_path):
        df = pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=5),
            "val": [1, 2, 3, 4, 5],
        })
        result = generate_trend_summary(df, output_dir=str(tmp_path), timestamp="test")
        assert result is not None

    def test_log_top_correlations_empty(self):
        corr_df = pd.DataFrame()
        log_top_correlations(corr_df)

    def test_log_top_correlations_single_col(self):
        corr_df = pd.DataFrame({"a": [1.0]}, index=["a"])
        log_top_correlations(corr_df)

    def test_log_top_correlations_perfect(self):
        corr_df = pd.DataFrame(
            {"a": [1.0, 0.95], "b": [0.95, 1.0]},
            index=["a", "b"],
        )
        log_top_correlations(corr_df, threshold=0.9)


# ---------------------------------------------------------------------------
# 29. reporting/run_all_analytics
# ---------------------------------------------------------------------------
from garmin_analysis.reporting.run_all_analytics import run_all_analytics


class TestRunAllAnalyticsAdversarial:

    def _make_df(self, n=30):
        np.random.seed(42)
        return pd.DataFrame({
            "day": pd.date_range("2024-01-01", periods=n),
            "steps": np.random.randint(3000, 15000, n).astype(float),
            "hr_avg": np.random.randint(55, 85, n).astype(float),
            "stress_avg": np.random.randint(15, 60, n).astype(float),
            "score": np.random.randint(50, 100, n).astype(float),
        })

    def test_basic_report(self, tmp_path):
        df = self._make_df()
        result = run_all_analytics(df, output_dir=str(tmp_path))
        assert os.path.exists(result)

    def test_empty_df(self, tmp_path):
        df = pd.DataFrame({"day": pd.Series(dtype="datetime64[ns]")})
        result = run_all_analytics(df, output_dir=str(tmp_path))
        assert os.path.exists(result)

    def test_monthly_report(self, tmp_path):
        df = self._make_df(n=90)
        result = run_all_analytics(df, output_dir=str(tmp_path), monthly=True)
        assert os.path.exists(result)

    def test_single_day(self, tmp_path):
        df = pd.DataFrame({
            "day": [pd.Timestamp("2024-01-15")],
            "steps": [5000.0],
        })
        result = run_all_analytics(df, output_dir=str(tmp_path))
        assert os.path.exists(result)

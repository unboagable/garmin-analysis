"""
Tests for error handling utilities.

These tests verify:
- Custom exception classes
- Error handling decorators
- Validation functions
- Context managers
- Helper functions
"""

import pytest
import pandas as pd
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

from garmin_analysis.utils.error_handling import (
    # Exceptions
    GarminAnalysisError,
    DataLoadingError,
    DatabaseError,
    DataValidationError,
    ModelingError,
    ConfigurationError,
    InsufficientDataError,
    # Decorators
    handle_data_loading_errors,
    handle_database_errors,
    handle_modeling_errors,
    # Validation
    validate_dataframe,
    validate_file_path,
    validate_database_connection,
    # Context managers
    suppress_and_log,
    # Helpers
    log_and_reraise,
    safe_return,
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""
    
    def test_garmin_analysis_error_is_base(self):
        """Test that GarminAnalysisError is base exception."""
        error = GarminAnalysisError("test")
        assert isinstance(error, Exception)
        assert str(error) == "test"
    
    def test_data_loading_error_inherits_from_base(self):
        """Test DataLoadingError inheritance."""
        error = DataLoadingError("test")
        assert isinstance(error, GarminAnalysisError)
        assert isinstance(error, Exception)
    
    def test_database_error_inherits_from_base(self):
        """Test DatabaseError inheritance."""
        error = DatabaseError("test")
        assert isinstance(error, GarminAnalysisError)
    
    def test_insufficient_data_error_inherits_from_validation(self):
        """Test InsufficientDataError inheritance."""
        error = InsufficientDataError("test")
        assert isinstance(error, DataValidationError)
        assert isinstance(error, GarminAnalysisError)


class TestHandleDataLoadingErrors:
    """Test handle_data_loading_errors decorator."""
    
    def test_successful_execution(self):
        """Test that successful execution passes through."""
        @handle_data_loading_errors(default_return=None)
        def load_data():
            return pd.DataFrame({'a': [1, 2, 3]})
        
        result = load_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_file_not_found_returns_default(self):
        """Test FileNotFoundError returns default value."""
        @handle_data_loading_errors(default_return=pd.DataFrame())
        def load_missing_file():
            raise FileNotFoundError("file not found")
        
        result = load_missing_file()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_file_not_found_reraises_as_data_loading_error(self):
        """Test FileNotFoundError is wrapped when reraise=True."""
        @handle_data_loading_errors(reraise=True)
        def load_missing_file():
            raise FileNotFoundError("file not found")
        
        with pytest.raises(DataLoadingError, match="File not found"):
            load_missing_file()
    
    def test_parser_error_handled(self):
        """Test pandas ParserError is handled."""
        @handle_data_loading_errors(default_return=pd.DataFrame())
        def parse_bad_csv():
            raise pd.errors.ParserError("bad CSV")
        
        result = parse_bad_csv()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_unexpected_error_logged(self, caplog):
        """Test unexpected errors are logged."""
        @handle_data_loading_errors(default_return=None)
        def raise_unexpected():
            raise ValueError("unexpected")
        
        result = raise_unexpected()
        assert result is None
        assert "Unexpected error" in caplog.text


class TestHandleDatabaseErrors:
    """Test handle_database_errors decorator."""
    
    def test_successful_query(self):
        """Test successful database operation."""
        @handle_database_errors(default_return=pd.DataFrame())
        def query_db():
            return pd.DataFrame({'col': [1, 2, 3]})
        
        result = query_db()
        assert len(result) == 3
    
    def test_operational_error_returns_default(self):
        """Test sqlite3.OperationalError returns default."""
        @handle_database_errors(default_return=pd.DataFrame())
        def locked_db():
            raise sqlite3.OperationalError("database is locked")
        
        result = locked_db()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_integrity_error_wrapped(self):
        """Test IntegrityError is wrapped when reraise=True."""
        @handle_database_errors(reraise=True)
        def violate_constraint():
            raise sqlite3.IntegrityError("constraint violation")
        
        with pytest.raises(DatabaseError, match="integrity error"):
            violate_constraint()
    
    def test_generic_sqlite_error_handled(self):
        """Test generic sqlite3.Error is handled."""
        @handle_database_errors(default_return={})
        def db_error():
            raise sqlite3.Error("database error")
        
        result = db_error()
        assert result == {}


class TestHandleModelingErrors:
    """Test handle_modeling_errors decorator."""
    
    def test_value_error_handled(self):
        """Test ValueError in modeling is handled."""
        @handle_modeling_errors(default_return={})
        def bad_values():
            raise ValueError("invalid values")
        
        result = bad_values()
        assert result == {}
    
    def test_key_error_handled(self):
        """Test KeyError in modeling is handled."""
        @handle_modeling_errors(default_return=None)
        def missing_key():
            raise KeyError("missing_column")
        
        result = missing_key()
        assert result is None
    
    def test_modeling_error_wrapped(self):
        """Test error is wrapped as ModelingError when reraise=True."""
        @handle_modeling_errors(reraise=True)
        def bad_model():
            raise ValueError("bad model parameters")
        
        with pytest.raises(ModelingError, match="Value error"):
            bad_model()


class TestValidateDataFrame:
    """Test validate_dataframe function."""
    
    def test_valid_dataframe_passes(self):
        """Test valid DataFrame passes validation."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        validate_dataframe(df, required_columns=['a', 'b'], min_rows=2)
        # No exception means success
    
    def test_none_dataframe_raises(self):
        """Test None DataFrame raises DataValidationError."""
        with pytest.raises(DataValidationError, match="DataFrame is None"):
            validate_dataframe(None)
    
    def test_empty_dataframe_raises_by_default(self):
        """Test empty DataFrame raises error by default."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_dataframe(df)
    
    def test_empty_dataframe_allowed_when_specified(self):
        """Test empty DataFrame is allowed when allow_empty=True."""
        df = pd.DataFrame()
        validate_dataframe(df, allow_empty=True)
        # No exception
    
    def test_missing_required_columns_raises(self):
        """Test missing required columns raises error."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df, required_columns=['a', 'b', 'c'])
    
    def test_insufficient_rows_raises(self):
        """Test insufficient rows raises InsufficientDataError."""
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            validate_dataframe(df, min_rows=10)
    
    def test_wrong_type_raises(self):
        """Test non-DataFrame raises DataValidationError."""
        with pytest.raises(DataValidationError, match="Expected DataFrame"):
            validate_dataframe([1, 2, 3])


class TestValidateFilePath:
    """Test validate_file_path function."""
    
    def test_existing_file_valid(self, tmp_path):
        """Test existing file passes validation."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        result = validate_file_path(test_file, must_exist=True)
        assert result == test_file
    
    def test_missing_file_raises_when_must_exist(self):
        """Test missing file raises FileNotFoundError when must_exist=True."""
        with pytest.raises(FileNotFoundError):
            validate_file_path("/nonexistent/file.csv", must_exist=True)
    
    def test_missing_file_ok_when_not_required(self):
        """Test missing file is ok when must_exist=False."""
        result = validate_file_path("/nonexistent/file.csv", must_exist=False)
        assert result == Path("/nonexistent/file.csv")
    
    def test_wrong_file_type_raises(self, tmp_path):
        """Test wrong file type raises ValueError."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("data")
        
        with pytest.raises(ValueError, match="Expected file type"):
            validate_file_path(test_file, must_exist=True, file_type='.csv')
    
    def test_correct_file_type_passes(self, tmp_path):
        """Test correct file type passes validation."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        
        result = validate_file_path(test_file, must_exist=True, file_type='.csv')
        assert result == test_file


class TestValidateDatabaseConnection:
    """Test validate_database_connection function."""
    
    def test_valid_connection_passes(self):
        """Test valid database connection passes."""
        conn = sqlite3.connect(":memory:")
        validate_database_connection(conn)
        conn.close()
    
    def test_none_connection_raises(self):
        """Test None connection raises DatabaseError."""
        with pytest.raises(DatabaseError, match="connection is None"):
            validate_database_connection(None)
    
    def test_closed_connection_raises(self):
        """Test closed connection raises DatabaseError."""
        conn = sqlite3.connect(":memory:")
        conn.close()
        
        with pytest.raises(DatabaseError, match="Invalid database connection"):
            validate_database_connection(conn)


class TestSuppressAndLog:
    """Test suppress_and_log context manager."""
    
    def test_suppresses_specified_exception(self, caplog):
        """Test context manager suppresses specified exception."""
        with suppress_and_log(ValueError):
            raise ValueError("test error")
        
        # If we get here, exception was suppressed
        assert "Suppressed ValueError" in caplog.text
    
    def test_does_not_suppress_other_exceptions(self):
        """Test context manager doesn't suppress other exceptions."""
        with pytest.raises(TypeError):
            with suppress_and_log(ValueError):
                raise TypeError("different error")
    
    def test_suppresses_multiple_exception_types(self, caplog):
        """Test suppressing multiple exception types."""
        with suppress_and_log(ValueError, KeyError):
            raise KeyError("test")
        
        assert "Suppressed KeyError" in caplog.text


class TestLogAndReraise:
    """Test log_and_reraise function."""
    
    def test_logs_and_reraises(self, caplog):
        """Test function logs and re-raises exception."""
        with pytest.raises(ValueError, match="test error"):
            try:
                raise ValueError("test error")
            except ValueError as e:
                log_and_reraise(e, "processing data")
        
        assert "Error while processing data" in caplog.text


class TestSafeReturn:
    """Test safe_return function."""
    
    def test_returns_function_result_on_success(self):
        """Test returns function result when successful."""
        def successful_func(x):
            return x * 2
        
        result = safe_return(successful_func, 5, default=0, log_errors=False)
        assert result == 10
    
    def test_returns_default_on_error(self):
        """Test returns default value on error."""
        def failing_func():
            raise ValueError("error")
        
        result = safe_return(failing_func, default=42, log_errors=False)
        assert result == 42
    
    def test_logs_errors_when_requested(self, caplog):
        """Test errors are logged when log_errors=True."""
        def failing_func():
            raise ValueError("error")
        
        result = safe_return(failing_func, default=None, log_errors=True)
        assert result is None
        assert "Error calling" in caplog.text
    
    def test_passes_args_and_kwargs(self):
        """Test function receives args and kwargs."""
        def func_with_args(a, b, c=3):
            return a + b + c
        
        result = safe_return(func_with_args, 1, 2, default=0, log_errors=False, c=4)
        assert result == 7


class TestIntegration:
    """Integration tests for error handling."""
    
    def test_decorator_with_validation(self):
        """Test combining decorator with validation."""
        @handle_data_loading_errors(reraise=True)
        def load_and_validate(path):
            df = pd.read_csv(path)
            validate_dataframe(df, required_columns=['a', 'b'], min_rows=5)
            return df
        
        # Test with missing file
        with pytest.raises(DataLoadingError):
            load_and_validate("/nonexistent.csv")
    
    def test_multiple_decorators(self):
        """Test stacking multiple error handling decorators."""
        @handle_data_loading_errors(default_return=None)
        @handle_modeling_errors(default_return=None)
        def complex_operation():
            # Could raise either data loading or modeling errors
            raise ValueError("test")
        
        result = complex_operation()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


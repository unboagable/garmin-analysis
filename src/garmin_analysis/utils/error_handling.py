"""
Centralized error handling utilities for the Garmin Analysis package.

This module provides:
- Custom exception classes for domain-specific errors
- Decorator functions for common error handling patterns
- Error context managers for resource cleanup
- Utilities for logging and re-raising exceptions with context

Usage:
    from garmin_analysis.utils.error_handling import (
        handle_data_loading_errors,
        handle_database_errors,
        DataLoadingError,
        DatabaseError
    )
    
    @handle_data_loading_errors(default_return=pd.DataFrame())
    def load_data(path):
        return pd.read_csv(path)
"""

import logging
import sqlite3
import functools
from pathlib import Path
from typing import Callable, Any, Optional, TypeVar, Union
import pandas as pd

logger = logging.getLogger(__name__)

# Type variable for decorators
T = TypeVar('T')


# ============================================================================
# Custom Exception Classes
# ============================================================================

class GarminAnalysisError(Exception):
    """Base exception for all Garmin Analysis errors."""
    pass


class DataLoadingError(GarminAnalysisError):
    """Raised when data loading fails (CSV, database, etc.)."""
    pass


class DatabaseError(GarminAnalysisError):
    """Raised when database operations fail."""
    pass


class DataValidationError(GarminAnalysisError):
    """Raised when data validation fails."""
    pass


class ModelingError(GarminAnalysisError):
    """Raised when modeling/prediction fails."""
    pass


class ConfigurationError(GarminAnalysisError):
    """Raised when configuration is invalid."""
    pass


class InsufficientDataError(DataValidationError):
    """Raised when there's not enough data for analysis."""
    pass


# ============================================================================
# Error Handling Decorators
# ============================================================================

def handle_data_loading_errors(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling errors in data loading functions.
    
    Catches common data loading exceptions and handles them gracefully:
    - FileNotFoundError: Missing files
    - pd.errors.EmptyDataError: Empty CSV files
    - pd.errors.ParserError: Malformed CSV files
    - sqlite3.Error: Database errors
    - OSError: File system errors
    
    Args:
        default_return: Value to return on error (default: None)
        log_level: Logging level for error messages (default: ERROR)
        reraise: Whether to re-raise the exception after logging (default: False)
    
    Returns:
        Decorator function
    
    Example:
        @handle_data_loading_errors(default_return=pd.DataFrame())
        def load_csv(path):
            return pd.read_csv(path)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                logger.log(log_level, f"File not found in {func.__name__}: {e}")
                if reraise:
                    raise DataLoadingError(f"File not found: {e}") from e
                return default_return
            except pd.errors.EmptyDataError as e:
                logger.log(log_level, f"Empty data in {func.__name__}: {e}")
                if reraise:
                    raise DataLoadingError(f"Empty data: {e}") from e
                return default_return
            except pd.errors.ParserError as e:
                logger.log(log_level, f"Parse error in {func.__name__}: {e}")
                if reraise:
                    raise DataLoadingError(f"Parse error: {e}") from e
                return default_return
            except (OSError, IOError) as e:
                logger.log(log_level, f"I/O error in {func.__name__}: {e}")
                if reraise:
                    raise DataLoadingError(f"I/O error: {e}") from e
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}")
                if reraise:
                    raise DataLoadingError(f"Unexpected error: {e}") from e
                return default_return
        return wrapper
    return decorator


def handle_database_errors(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling database operation errors.
    
    Catches SQLite and database-related exceptions:
    - sqlite3.Error: Database errors
    - sqlite3.OperationalError: Database locked, etc.
    - sqlite3.IntegrityError: Constraint violations
    
    Args:
        default_return: Value to return on error (default: None)
        log_level: Logging level for error messages (default: ERROR)
        reraise: Whether to re-raise as DatabaseError (default: False)
    
    Returns:
        Decorator function
    
    Example:
        @handle_database_errors(default_return=pd.DataFrame())
        def query_database(conn, query):
            return pd.read_sql(query, conn)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                logger.log(log_level, f"Database operational error in {func.__name__}: {e}")
                if reraise:
                    raise DatabaseError(f"Database operational error: {e}") from e
                return default_return
            except sqlite3.IntegrityError as e:
                logger.log(log_level, f"Database integrity error in {func.__name__}: {e}")
                if reraise:
                    raise DatabaseError(f"Database integrity error: {e}") from e
                return default_return
            except sqlite3.Error as e:
                logger.log(log_level, f"Database error in {func.__name__}: {e}")
                if reraise:
                    raise DatabaseError(f"Database error: {e}") from e
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected error in {func.__name__}")
                if reraise:
                    raise DatabaseError(f"Unexpected error: {e}") from e
                return default_return
        return wrapper
    return decorator


def handle_modeling_errors(
    default_return: Any = None,
    log_level: int = logging.WARNING,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling modeling/ML errors.
    
    Catches scikit-learn and modeling-related exceptions gracefully.
    
    Args:
        default_return: Value to return on error (default: None)
        log_level: Logging level for error messages (default: WARNING)
        reraise: Whether to re-raise as ModelingError (default: False)
    
    Returns:
        Decorator function
    
    Example:
        @handle_modeling_errors(default_return={})
        def train_model(X, y):
            model = RandomForestRegressor()
            model.fit(X, y)
            return {'model': model}
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except ValueError as e:
                logger.log(log_level, f"Value error in {func.__name__}: {e}")
                if reraise:
                    raise ModelingError(f"Value error: {e}") from e
                return default_return
            except KeyError as e:
                logger.log(log_level, f"Key error in {func.__name__}: {e}")
                if reraise:
                    raise ModelingError(f"Key error: {e}") from e
                return default_return
            except Exception as e:
                logger.exception(f"Unexpected modeling error in {func.__name__}")
                if reraise:
                    raise ModelingError(f"Unexpected error: {e}") from e
                return default_return
        return wrapper
    return decorator


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 0,
    allow_empty: bool = False
) -> None:
    """
    Validate DataFrame meets requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        allow_empty: Whether to allow empty DataFrames
    
    Raises:
        DataValidationError: If validation fails
        
    Example:
        validate_dataframe(df, required_columns=['day', 'steps'], min_rows=10)
    """
    if df is None:
        raise DataValidationError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected DataFrame, got {type(df)}")
    
    if df.empty and not allow_empty:
        raise DataValidationError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Insufficient data: need at least {min_rows} rows, got {len(df)}"
        )
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}. "
                f"Available: {list(df.columns)}"
            )


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    file_type: Optional[str] = None
) -> Path:
    """
    Validate file path.
    
    Args:
        path: File path to validate
        must_exist: Whether file must exist
        file_type: Expected file extension (e.g., '.csv', '.db')
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If file must exist but doesn't
        ValueError: If file type doesn't match
        
    Example:
        path = validate_file_path('data/file.csv', must_exist=True, file_type='.csv')
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_type and path.suffix != file_type:
        raise ValueError(
            f"Expected file type {file_type}, got {path.suffix} for {path}"
        )
    
    return path


def validate_database_connection(conn) -> None:
    """
    Validate SQLite database connection.
    
    Args:
        conn: Database connection to validate
    
    Raises:
        DatabaseError: If connection is invalid
    """
    if conn is None:
        raise DatabaseError("Database connection is None")
    
    try:
        # Test connection with a simple query
        conn.execute("SELECT 1")
    except sqlite3.Error as e:
        raise DatabaseError(f"Invalid database connection: {e}") from e


# ============================================================================
# Context Managers
# ============================================================================

class suppress_and_log:
    """
    Context manager to suppress exceptions and log them.
    
    Similar to contextlib.suppress but with logging.
    
    Example:
        with suppress_and_log(ValueError, KeyError, log_level=logging.WARNING):
            # Code that might raise ValueError or KeyError
            risky_operation()
    """
    def __init__(self, *exceptions, log_level: int = logging.WARNING):
        self.exceptions = exceptions
        self.log_level = log_level
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, self.exceptions):
            logger.log(self.log_level, f"Suppressed {exc_type.__name__}: {exc_val}")
            return True  # Suppress the exception
        return False


# ============================================================================
# Helper Functions
# ============================================================================

def log_and_reraise(
    exception: Exception,
    context: str,
    log_level: int = logging.ERROR
) -> None:
    """
    Log an exception with context and re-raise it.
    
    Args:
        exception: Exception to log and re-raise
        context: Context string describing what was being done
        log_level: Logging level (default: ERROR)
    
    Raises:
        The original exception
    
    Example:
        try:
            risky_operation()
        except Exception as e:
            log_and_reraise(e, "loading data from database")
    """
    logger.log(log_level, f"Error while {context}: {exception}")
    raise


def safe_return(
    func: Callable,
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely call a function, returning default value on error.
    
    Args:
        func: Function to call
        *args: Positional arguments for func
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for func
    
    Returns:
        Function result or default value
    
    Example:
        result = safe_return(risky_func, arg1_value, default=[], log_errors=True, kwarg1='value')
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"Error calling {func.__name__}: {e}")
        return default


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example of using the decorators
    
    @handle_data_loading_errors(default_return=pd.DataFrame())
    def load_example_csv(path):
        """Load CSV with error handling."""
        return pd.read_csv(path)
    
    @handle_database_errors(default_return={})
    def query_example_db(db_path):
        """Query database with error handling."""
        conn = sqlite3.connect(db_path)
        result = pd.read_sql("SELECT * FROM table", conn)
        conn.close()
        return result
    
    # Example of validation
    try:
        df = pd.DataFrame({'col1': [1, 2, 3]})
        validate_dataframe(df, required_columns=['col1'], min_rows=2)
        print("Validation passed!")
    except DataValidationError as e:
        print(f"Validation failed: {e}")


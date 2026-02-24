"""
Shared utilities for handling missing values across the codebase.

This module provides consistent, configurable strategies for imputing
or handling missing values in health/fitness data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)


def impute_missing_values(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = 'median',
    copy: bool = True
) -> pd.DataFrame:
    """
    Impute missing values in specified columns using various strategies.
    
    This function provides a consistent interface for handling missing values
    across all modeling and analysis modules. It's particularly designed for
    health and fitness data which often contains outliers.
    
    Args:
        df: Input DataFrame
        columns: List of column names to impute
        strategy: Imputation strategy to use:
            - 'median': Fill with column median (robust to outliers, recommended for health data)
            - 'mean': Fill with column mean (good for normally distributed data)
            - 'drop': Drop rows with any missing values in specified columns
            - 'forward_fill': Forward fill (use previous value, good for time series)
            - 'backward_fill': Backward fill (use next value, good for time series)
            - 'none': No imputation (keep NaN values)
        copy: Whether to copy the DataFrame (default True). Set to False to modify in place.
    
    Returns:
        DataFrame with imputed values
        
    Raises:
        ValueError: If strategy is invalid or columns don't exist
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        
        >>> # Median imputation (robust to outliers)
        >>> impute_missing_values(df, ['a', 'b'], strategy='median')
        
        >>> # Drop rows with missing values
        >>> impute_missing_values(df, ['a', 'b'], strategy='drop')
        
        >>> # Forward fill for time series
        >>> impute_missing_values(df, ['a'], strategy='forward_fill')
    
    Notes:
        - For health/fitness data, 'median' is recommended as it's robust to outliers
        - The 'drop' strategy may significantly reduce your dataset size
        - Always check data loss after imputation with log messages
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if not columns:
        raise ValueError("columns list cannot be empty")
    
    # Check all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Copy if requested
    df_result = df.copy() if copy else df
    
    # Count missing values before imputation
    missing_before = {col: df_result[col].isna().sum() for col in columns}
    total_missing_before = sum(missing_before.values())
    
    # Apply imputation strategy
    if strategy == 'drop':
        # Drop rows with any missing values in specified columns
        rows_before = len(df_result)
        df_result = df_result.dropna(subset=columns)
        rows_after = len(df_result)
        rows_dropped = rows_before - rows_after
        pct_str = f"{rows_after/rows_before*100:.1f}%" if rows_before > 0 else "N/A"
        logger.info(
            f"Dropped {rows_dropped} rows with missing values. "
            f"Remaining: {rows_after}/{rows_before} ({pct_str})"
        )
        
    elif strategy == 'median':
        # Fill with median (robust to outliers)
        for col in columns:
            if df_result[col].isna().any():
                median_val = df_result[col].median()
                if pd.isna(median_val):
                    logger.warning(
                        f"Column '{col}' is all-NaN; cannot compute median. "
                        "Skipping imputation for this column."
                    )
                    continue
                df_result[col] = df_result[col].fillna(median_val)
                logger.info(
                    f"Filled {missing_before[col]} missing values in '{col}' "
                    f"with median: {median_val:.4f}"
                )
                
    elif strategy == 'mean':
        # Fill with mean
        for col in columns:
            if df_result[col].isna().any():
                mean_val = df_result[col].mean()
                if pd.isna(mean_val):
                    logger.warning(
                        f"Column '{col}' is all-NaN; cannot compute mean. "
                        "Skipping imputation for this column."
                    )
                    continue
                df_result[col] = df_result[col].fillna(mean_val)
                logger.info(
                    f"Filled {missing_before[col]} missing values in '{col}' "
                    f"with mean: {mean_val:.4f}"
                )
                
    elif strategy == 'forward_fill':
        # Forward fill (use previous value)
        for col in columns:
            if df_result[col].isna().any():
                df_result[col] = df_result[col].ffill()
                filled = missing_before[col] - df_result[col].isna().sum()
                logger.info(
                    f"Forward filled {filled} missing values in '{col}' "
                    f"({df_result[col].isna().sum()} still missing)"
                )
                
    elif strategy == 'backward_fill':
        # Backward fill (use next value)
        for col in columns:
            if df_result[col].isna().any():
                df_result[col] = df_result[col].bfill()
                filled = missing_before[col] - df_result[col].isna().sum()
                logger.info(
                    f"Backward filled {filled} missing values in '{col}' "
                    f"({df_result[col].isna().sum()} still missing)"
                )
                
    elif strategy == 'none':
        # No imputation
        logger.info(
            f"No imputation applied. Preserving {total_missing_before} "
            f"missing values across {len(columns)} columns."
        )
        
    else:
        raise ValueError(
            f"Invalid imputation strategy: '{strategy}'. "
            f"Choose from: 'median', 'mean', 'drop', 'forward_fill', "
            f"'backward_fill', or 'none'"
        )
    
    return df_result


def get_missing_value_summary(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Generate a summary of missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of columns to analyze (default: all numeric columns)
        
    Returns:
        DataFrame with missing value statistics per column
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, 6]})
        >>> get_missing_value_summary(df)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = []
    for col in columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            total_count = len(df)
            missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
            
            summary.append({
                'column': col,
                'missing_count': missing_count,
                'total_count': total_count,
                'missing_pct': missing_pct,
                'has_missing': missing_count > 0
            })
    
    return pd.DataFrame(summary).sort_values('missing_pct', ascending=False)


def recommend_imputation_strategy(
    df: pd.DataFrame,
    columns: List[str],
    outlier_threshold: float = 3.0
) -> dict:
    """
    Recommend an imputation strategy based on data characteristics.
    
    This function analyzes the data and suggests the most appropriate
    imputation strategy for each column.
    
    Args:
        df: Input DataFrame
        columns: Columns to analyze
        outlier_threshold: Number of std deviations to consider as outlier
        
    Returns:
        Dictionary with recommendations per column
        
    Examples:
        >>> df = pd.DataFrame({'hr': [60, 65, 200, 62, 58]})  # Has outlier
        >>> recommend_imputation_strategy(df, ['hr'])
        {'hr': {'strategy': 'median', 'reason': 'Contains outliers'}}
    """
    recommendations = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        values = df[col].dropna()
        
        if len(values) == 0:
            recommendations[col] = {
                'strategy': 'drop',
                'reason': 'No non-null values'
            }
            continue
        
        # Check for outliers
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val > 0:
            outliers = values[(values < mean_val - outlier_threshold * std_val) | 
                            (values > mean_val + outlier_threshold * std_val)]
            has_outliers = len(outliers) > 0
        else:
            has_outliers = False
        
        # Check skewness
        skewness = values.skew() if len(values) > 2 else 0
        is_skewed = abs(skewness) > 1.0
        
        # Determine recommendation
        if has_outliers or is_skewed:
            recommendations[col] = {
                'strategy': 'median',
                'reason': f'Contains outliers or skewed distribution (skew={skewness:.2f})'
            }
        else:
            recommendations[col] = {
                'strategy': 'mean',
                'reason': 'Normal distribution, no significant outliers'
            }
    
    return recommendations


# Convenience functions for common strategies

def impute_median(df: pd.DataFrame, columns: List[str], copy: bool = True) -> pd.DataFrame:
    """Convenience function for median imputation."""
    return impute_missing_values(df, columns, strategy='median', copy=copy)


def impute_mean(df: pd.DataFrame, columns: List[str], copy: bool = True) -> pd.DataFrame:
    """Convenience function for mean imputation."""
    return impute_missing_values(df, columns, strategy='mean', copy=copy)


def drop_missing(df: pd.DataFrame, columns: List[str], copy: bool = True) -> pd.DataFrame:
    """Convenience function for dropping rows with missing values."""
    return impute_missing_values(df, columns, strategy='drop', copy=copy)


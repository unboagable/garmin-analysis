"""
CLI Helper Utilities for Command-Line Scripts.

This module provides shared CLI argument patterns and helper functions
to eliminate code duplication across command-line tools.
"""
import argparse
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def add_24h_coverage_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard 24-hour coverage filtering arguments to an argument parser.
    
    This function adds four arguments related to 24-hour continuous coverage
    filtering, which are commonly used across multiple CLI tools.
    
    Args:
        parser: The ArgumentParser instance to add arguments to.
    
    Returns:
        The same ArgumentParser instance with added arguments (for chaining).
    
    Example:
        >>> parser = argparse.ArgumentParser(description='My tool')
        >>> parser = add_24h_coverage_args(parser)
        >>> args = parser.parse_args(['--filter-24h-coverage', '--max-gap', '5'])
    """
    parser.add_argument(
        '--filter-24h-coverage',
        action='store_true',
        help='Filter to only days with 24-hour continuous coverage'
    )
    parser.add_argument(
        '--max-gap',
        type=int,
        default=2,
        help='Maximum gap in minutes for continuous coverage (default: 2)'
    )
    parser.add_argument(
        '--day-edge-tolerance',
        type=int,
        default=2,
        help='Day edge tolerance in minutes for continuous coverage (default: 2)'
    )
    parser.add_argument(
        '--coverage-allowance-minutes',
        type=int,
        default=0,
        help='Total allowed missing minutes within a day (0-300, default: 0)'
    )
    return parser


def apply_24h_coverage_filter_from_args(
    df: pd.DataFrame,
    args: argparse.Namespace,
    hr_df: Optional[pd.DataFrame] = None,
    stress_df: Optional[pd.DataFrame] = None,  # Deprecated - no longer used
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply 24-hour coverage filtering based on parsed CLI arguments.
    
    This function checks if filtering was requested via CLI arguments and applies
    the appropriate filtering parameters. It handles the conversion of arguments
    to the expected types and provides logging.
    
    Args:
        df: The DataFrame to filter.
        args: Parsed command-line arguments (must have filter_24h_coverage,
              max_gap, day_edge_tolerance, and coverage_allowance_minutes attributes).
        hr_df: Optional pre-loaded monitoring_hr DataFrame for coverage analysis.
               Heart rate data is required to determine watch wear.
        stress_df: DEPRECATED - no longer used. Heart rate data is required.
        db_path: Optional custom database path for loading HR data.
    
    Returns:
        Filtered DataFrame if filtering was requested, otherwise original DataFrame.
        Returns original DataFrame if HR data is not available (cannot determine watch wear).
    
    Raises:
        AttributeError: If args is missing required attributes.
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_24h_coverage_args(parser)
        >>> args = parser.parse_args(['--filter-24h-coverage'])
        >>> filtered_df = apply_24h_coverage_filter_from_args(df, args)
    """
    # Validate args has required attributes
    required_attrs = ['filter_24h_coverage', 'max_gap', 'day_edge_tolerance', 
                     'coverage_allowance_minutes']
    missing_attrs = [attr for attr in required_attrs if not hasattr(args, attr)]
    if missing_attrs:
        raise AttributeError(
            f"args missing required attributes: {missing_attrs}. "
            f"Did you call add_24h_coverage_args() on the parser?"
        )
    
    # If filtering not requested, return original dataframe
    if not args.filter_24h_coverage:
        return df
    
    # Import here to avoid circular dependencies
    from garmin_analysis.features.coverage import filter_by_24h_coverage
    
    logger.info("Filtering to days with 24-hour continuous coverage...")
    
    # Convert arguments to appropriate types
    max_gap = pd.Timedelta(minutes=args.max_gap)
    day_edge_tolerance = pd.Timedelta(minutes=args.day_edge_tolerance)
    # Clamp coverage allowance to [0, 300] minutes
    allowance_minutes = max(0, min(args.coverage_allowance_minutes, 300))
    total_missing_allowance = pd.Timedelta(minutes=allowance_minutes)
    
    # Apply filtering (use hr_df if provided, otherwise let function load from database)
    df_filtered = filter_by_24h_coverage(
        df,
        max_gap=max_gap,
        day_edge_tolerance=day_edge_tolerance,
        total_missing_allowance=total_missing_allowance,
        hr_df=hr_df,
        db_path=db_path
    )
    
    logger.info(
        f"After 24h coverage filtering: {len(df_filtered)} days remaining "
        f"(from {len(df)} days, filtered {len(df) - len(df_filtered)} days)"
    )
    
    return df_filtered


def add_common_output_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common output-related arguments to an argument parser.
    
    This function adds standard arguments for output directory and verbosity
    that are commonly used across CLI tools.
    
    Args:
        parser: The ArgumentParser instance to add arguments to.
    
    Returns:
        The same ArgumentParser instance with added arguments (for chaining).
    
    Example:
        >>> parser = argparse.ArgumentParser(description='My tool')
        >>> parser = add_common_output_args(parser)
        >>> args = parser.parse_args(['--output-dir', 'results', '-v'])
    """
    parser.add_argument(
        '--output-dir',
        default='plots',
        help='Output directory for generated files (default: plots)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser


def setup_logging_from_args(args: argparse.Namespace, default_level: int = logging.INFO) -> None:
    """
    Configure logging based on CLI arguments.
    
    This is a convenience function to set up logging based on the verbose flag
    that is added by add_common_output_args().
    
    Args:
        args: Parsed command-line arguments (must have verbose attribute).
        default_level: Logging level to use when verbose is False (default: INFO).
    
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_common_output_args(parser)
        >>> args = parser.parse_args(['-v'])
        >>> setup_logging_from_args(args)
    """
    from garmin_analysis.logging_config import setup_logging
    
    if hasattr(args, 'verbose') and args.verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=default_level)


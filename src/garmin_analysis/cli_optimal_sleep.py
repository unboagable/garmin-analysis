#!/usr/bin/env python3
"""
CLI for optimal sleep activity ranges analysis.

Identifies steps and intensity minutes ranges associated with best sleep.
"""

import argparse
import logging

import pandas as pd

from garmin_analysis.features.coverage import filter_by_24h_coverage
from garmin_analysis.features.optimal_sleep_activity_ranges import (
    plot_optimal_sleep_ranges,
    print_optimal_sleep_summary,
)
from garmin_analysis.logging_config import setup_logging
from garmin_analysis.utils.data_loading import load_master_dataframe

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point for optimal sleep activity ranges analysis."""
    parser = argparse.ArgumentParser(
        description="Find steps and intensity minutes ranges associated with best sleep"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively (default: save only)"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save plots to files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--filter-24h-coverage",
        action="store_true",
        help="Filter to only days with 24-hour continuous coverage",
    )
    parser.add_argument(
        "--max-gap", type=int, default=2, help="Maximum allowed gap in minutes (default: 2)"
    )
    parser.add_argument(
        "--day-edge-tolerance",
        type=int,
        default=2,
        help="Day edge tolerance in minutes (default: 2)",
    )
    parser.add_argument(
        "--coverage-allowance-minutes", type=int, default=0, help="Coverage allowance (default: 0)"
    )

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=logging.INFO)
    else:
        setup_logging(level=logging.WARNING)

    try:
        logger.info("Loading master daily summary data...")
        df = load_master_dataframe()
        if args.filter_24h_coverage:
            max_gap = max(1, int(args.max_gap))
            edge_tol = max(0, int(args.day_edge_tolerance))
            allowance = max(0, int(args.coverage_allowance_minutes))
            logger.info(
                "Applying 24h coverage filter (max_gap=%sm, edge_tol=%sm, allowance=%sm)",
                max_gap,
                edge_tol,
                allowance,
            )
            df = filter_by_24h_coverage(
                df,
                max_gap=pd.Timedelta(minutes=max_gap),
                day_edge_tolerance=pd.Timedelta(minutes=edge_tol),
                total_missing_allowance=pd.Timedelta(minutes=allowance),
            )

        if df.empty:
            logger.error("No data loaded")
            return 1

        logger.info("Loaded %d days of data", len(df))
        print_optimal_sleep_summary(df)

        if not args.no_save or args.show_plots:
            logger.info("Generating visualizations...")
            plot_files = plot_optimal_sleep_ranges(
                df, save_plots=not args.no_save, show_plots=args.show_plots
            )
            if plot_files and not args.no_save:
                logger.info("Generated %d plots", len(plot_files))
                for name, path in plot_files.items():
                    logger.info("  %s: %s", name, path)

        return 0
    except Exception as e:
        logger.exception("Error in optimal sleep analysis: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())

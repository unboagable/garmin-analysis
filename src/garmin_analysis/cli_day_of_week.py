#!/usr/bin/env python3
"""
CLI entry point for day-of-week analysis of sleep score, body battery, and water intake.
"""

import argparse
import logging
from pathlib import Path
from .features.day_of_week_analysis import (
    calculate_day_of_week_averages,
    plot_day_of_week_averages,
    print_day_of_week_summary
)
from .utils.data_loading import load_master_dataframe
from .features.coverage import filter_by_24h_coverage

def main():
    """CLI entry point for day-of-week analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze sleep score, body battery, and water intake by day of week"
    )
    parser.add_argument(
        "--show-plots", 
        action="store_true", 
        help="Display plots interactively (default: save only)"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true", 
        help="Don't save plots to files"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--filter-24h-coverage",
        action="store_true",
        help="Filter to only days with 24-hour continuous coverage (uses stress data)"
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum allowed gap between samples in minutes (default: 2)"
    )
    parser.add_argument(
        "--day-edge-tolerance",
        type=int,
        default=2,
        help="Allowed tolerance at day start/end in minutes (default: 2)"
    )
    parser.add_argument(
        "--coverage-allowance-minutes",
        type=int,
        default=0,
        help="Total allowed missing minutes within a day (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    try:
        # Load data
        logging.info("Loading master daily summary data...")
        df = load_master_dataframe()
        # Optional 24h coverage filtering
        if args.filter_24h_coverage:
            max_gap = max(1, int(args.max_gap))
            edge_tol = max(0, int(args.day_edge_tolerance))
            allowance = max(0, int(args.coverage_allowance_minutes))
            logging.info(
                "Applying 24h coverage filter (max_gap=%sm, edge_tol=%sm, allowance=%sm)",
                max_gap, edge_tol, allowance
            )
            df = filter_by_24h_coverage(
                df,
                max_gap=pd.Timedelta(minutes=max_gap),
                day_edge_tolerance=pd.Timedelta(minutes=edge_tol),
                total_missing_allowance=pd.Timedelta(minutes=allowance),
            )
        
        if df.empty:
            logging.error("No data loaded")
            return 1
        
        logging.info(f"Loaded {len(df)} days of data")
        
        # Print summary
        print_day_of_week_summary(df)
        
        # Create visualizations
        if not args.no_save or args.show_plots:
            logging.info("Generating day-of-week visualizations...")
            plot_files = plot_day_of_week_averages(
                df, 
                save_plots=not args.no_save, 
                show_plots=args.show_plots
            )
            
            if plot_files and not args.no_save:
                logging.info(f"Generated {len(plot_files)} plots:")
                for metric, filepath in plot_files.items():
                    logging.info(f"  {metric}: {filepath}")
        
        return 0
        
    except Exception as e:
        logging.exception("Error in day-of-week analysis: %s", e)
        return 1

if __name__ == "__main__":
    exit(main())

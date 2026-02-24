#!/usr/bin/env python3
"""
CLI entry point for time-of-day stress analysis.

Analyzes stress patterns by hour of day and day of week, showing:
- Average stress levels throughout the day
- Peak and low stress periods
- Stress heatmaps by weekday and hour
- Statistical summaries
"""

import argparse
import logging

from garmin_analysis.features.time_of_day_stress_analysis import (
    calculate_hourly_stress_averages,
    calculate_hourly_stress_by_weekday,
    load_stress_data,
    plot_hourly_stress_pattern,
    plot_stress_heatmap_by_weekday,
    print_stress_summary,
)
from garmin_analysis.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """CLI entry point for time-of-day stress analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze stress patterns by time of day and day of week",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis with all visualizations
  python -m garmin_analysis.cli_time_of_day_stress
  
  # Display plots interactively
  python -m garmin_analysis.cli_time_of_day_stress --show-plots
  
  # Use custom database path
  python -m garmin_analysis.cli_time_of_day_stress --db-path /path/to/garmin.db
  
  # Skip weekday analysis (faster)
  python -m garmin_analysis.cli_time_of_day_stress --no-weekday-analysis
  
  # Verbose output
  python -m garmin_analysis.cli_time_of_day_stress -v
        """,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to garmin.db database (default: db/garmin.db)",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively (default: save only)"
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save plots to files")
    parser.add_argument(
        "--no-weekday-analysis",
        action="store_true",
        help="Skip weekday-specific analysis (faster for large datasets)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        setup_logging(level=logging.INFO)
    else:
        setup_logging(level=logging.WARNING)

    try:
        # Load stress data
        logger.info("Loading stress data from database...")
        stress_df = load_stress_data(args.db_path)

        if stress_df.empty:
            logger.error("No stress data loaded. Check database path and data availability.")
            return 1

        logger.info(f"Loaded {len(stress_df):,} stress measurements")

        # Calculate hourly averages
        logger.info("Calculating hourly stress averages...")
        hourly_stats = calculate_hourly_stress_averages(stress_df)

        # Calculate weekday patterns if requested
        hourly_weekday_stats = None
        if not args.no_weekday_analysis:
            logger.info("Calculating hourly stress averages by weekday...")
            hourly_weekday_stats = calculate_hourly_stress_by_weekday(stress_df)

        # Print summary
        print_stress_summary(hourly_stats, hourly_weekday_stats)

        # Create visualizations
        save_plots = not args.no_save

        if save_plots or args.show_plots:
            logger.info("\nGenerating stress pattern visualizations...")
            plot_files = plot_hourly_stress_pattern(
                hourly_stats, save_plots=save_plots, show_plots=args.show_plots
            )

            if not args.no_weekday_analysis:
                logger.info("Generating stress heatmap by weekday and hour...")
                weekday_plot_files = plot_stress_heatmap_by_weekday(
                    hourly_weekday_stats, save_plots=save_plots, show_plots=args.show_plots
                )
                plot_files.update(weekday_plot_files)

            if plot_files and save_plots:
                logger.info(f"\n✅ Generated {len(plot_files)} plots:")
                for plot_name, filepath in plot_files.items():
                    logger.info(f"  {plot_name}: {filepath}")

        logger.info("\n✅ Time-of-day stress analysis complete!")
        return 0

    except Exception as e:
        logger.exception(f"Error in time-of-day stress analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

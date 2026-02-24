#!/usr/bin/env python3
"""
Command-line interface for creating activity calendar plots.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from garmin_analysis.logging_config import setup_logging
from garmin_analysis.viz.plot_activity_calendar import (
    load_activities_data,
    plot_activity_calendar,
    suggest_activity_mappings,
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Create activity calendar plots from Garmin data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create calendar for all available data
  python cli_activity_calendar.py
  
  # Create calendar for last 6 months
  python cli_activity_calendar.py --months 6
  
  # Create calendar for specific date range
  python cli_activity_calendar.py --start-date 2024-01-01 --end-date 2024-12-31
  
  # Create calendar with custom output directory
  python cli_activity_calendar.py --output-dir my_plots
        """,
    )

    parser.add_argument(
        "--db-path",
        default="db/garmin_activities.db",
        help="Path to the activities database (default: db/garmin_activities.db)",
    )

    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")

    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD format")

    parser.add_argument(
        "--months",
        type=int,
        help="Number of months back from today to include (overrides start-date)",
    )

    parser.add_argument(
        "--output-dir", default="plots", help="Output directory for the plot (default: plots)"
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[16, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 16 10)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--no-mappings",
        action="store_true",
        help="Disable activity type mappings (use raw sport names)",
    )

    parser.add_argument(
        "--suggest-mappings",
        action="store_true",
        help="Suggest mappings for unknown activity types and exit",
    )

    parser.add_argument(
        "--mappings-config",
        default="config/activity_type_mappings.json",
        help="Path to activity mappings configuration file",
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    # Validate database path
    if not Path(args.db_path).exists():
        logger.error(f"Database not found at {args.db_path}")
        sys.exit(1)

    try:
        # Load activities data
        logger.info(f"Loading activities data from {args.db_path}")
        activities_df = load_activities_data(args.db_path)

        if activities_df.empty:
            logger.warning("No activities found in the database")
            sys.exit(1)

        logger.info(f"Loaded {len(activities_df)} activities")

        # Handle mapping suggestions
        if args.suggest_mappings:
            logger.info("Analyzing activity types for mapping suggestions...")
            suggest_activity_mappings(activities_df, args.mappings_config)
            return

        # Determine date range
        start_date = args.start_date
        end_date = args.end_date

        if args.months:
            # Calculate start date based on months back
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")
            logger.info(f"Using {args.months} months back: {start_date} to {end_date}")

        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Create the calendar plot
        logger.info("Creating activity calendar...")
        plot_activity_calendar(
            activities_df,
            start_date=start_date,
            end_date=end_date,
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            use_mappings=not args.no_mappings,
            mappings_config_path=args.mappings_config,
        )

        logger.info("Activity calendar created successfully!")

    except Exception as e:
        logger.error(f"Error creating activity calendar: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()

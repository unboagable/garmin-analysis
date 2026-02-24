"""CLI entry-point for the weekly health report."""

import argparse
import logging

from garmin_analysis.reporting.generate_weekly_report import generate_weekly_report
from garmin_analysis.utils.cli_helpers import (
    add_24h_coverage_args,
    add_common_output_args,
    apply_24h_coverage_filter_from_args,
    setup_logging_from_args,
)
from garmin_analysis.utils.data_loading import load_master_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a weekly Markdown health report "
        "(sleep score trend, resting HR delta, stress minutes delta)."
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="Number of recent weeks to include (default: 4)",
    )
    add_common_output_args(parser)
    add_24h_coverage_args(parser)

    args = parser.parse_args()
    setup_logging_from_args(args)

    logger = logging.getLogger(__name__)

    df = load_master_dataframe()
    df = apply_24h_coverage_filter_from_args(df, args)

    report_path = generate_weekly_report(
        df,
        num_weeks=args.weeks,
        output_dir=args.output_dir if args.output_dir != "plots" else None,
    )

    if report_path:
        logger.info("Weekly report: %s", report_path)
    else:
        logger.warning("No report generated (insufficient data).")


if __name__ == "__main__":
    main()

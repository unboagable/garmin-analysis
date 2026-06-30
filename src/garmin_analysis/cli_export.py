#!/usr/bin/env python3
"""
CLI for exporting master dataset to Parquet and optionally DuckDB.
"""

import argparse
import logging
import sys
from pathlib import Path

from garmin_analysis.data_ingestion.export_master import (
    SUPPORTED_EXPORT_FORMATS,
    export_master,
)
from garmin_analysis.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export master dataset to CSV, Parquet, JSON, or DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset (default Parquet export)
  poetry run python -m garmin_analysis.cli_export
  poetry run python -m garmin_analysis.cli_export --duckdb

  # Date-range export (all Garmin tables + merged master)
  poetry run python -m garmin_analysis.cli_export \\
    --start-date 2024-01-01 --end-date 2024-03-31 --format csv

  # Merged master summary only
  poetry run python -m garmin_analysis.cli_export \\
    --start-date 2024-01-01 --end-date 2024-03-31 --format parquet --summary-only
        """,
    )
    parser.add_argument(
        "--start-date",
        metavar="YYYY-MM-DD",
        help="Inclusive start date for export (requires --end-date)",
    )
    parser.add_argument(
        "--end-date",
        metavar="YYYY-MM-DD",
        help="Inclusive end date for export (requires --start-date)",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_EXPORT_FORMATS,
        help=(
            "Output format for date-range export "
            f"({', '.join(SUPPORTED_EXPORT_FORMATS)}). "
            "Required when using --start-date/--end-date."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for all-data export (default: data/export/garmin_export_<start>_<end>/)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Export only the merged master daily summary (not raw Garmin tables)",
    )
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Export only to Parquet (default: Parquet + optionally DuckDB)",
    )
    parser.add_argument(
        "--duckdb",
        action="store_true",
        help="Also export to DuckDB (requires: pip install duckdb)",
    )
    parser.add_argument(
        "--no-data-quality",
        action="store_true",
        help="Skip merging daily data quality columns into the export",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    has_start = args.start_date is not None
    has_end = args.end_date is not None
    if has_start ^ has_end:
        parser.error("--start-date and --end-date must be provided together")
    if (has_start or has_end) and args.format is None:
        parser.error("--format is required when exporting a date range")

    parquet = True
    duckdb = args.duckdb
    if args.parquet_only:
        duckdb = False

    try:
        if has_start and has_end:
            result = export_master(
                parquet=False,
                duckdb=False,
                start_date=args.start_date,
                end_date=args.end_date,
                export_format=args.format,
                output_path=args.output,
                include_data_quality=not args.no_data_quality,
                summary_only=args.summary_only,
            )
        else:
            if args.format is not None:
                parser.error("--format can only be used with --start-date and --end-date")
            result = export_master(
                parquet=parquet,
                duckdb=duckdb,
                include_data_quality=not args.no_data_quality,
            )

        for key, value in result.items():
            logger.info("  %s: %s", key, value)
        return 0
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
CLI for exporting master dataset to Parquet and optionally DuckDB.
"""

import argparse
import logging
import sys

from garmin_analysis.data_ingestion.export_master import export_master
from garmin_analysis.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export master dataset to Parquet and optionally DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python -m garmin_analysis.cli_export
  poetry run python -m garmin_analysis.cli_export --duckdb
  poetry run python -m garmin_analysis.cli_export --parquet-only
        """,
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    parquet = True
    duckdb = args.duckdb
    if args.parquet_only:
        duckdb = False

    try:
        result = export_master(parquet=parquet, duckdb=duckdb)
        for k, v in result.items():
            logger.info(f"  {k}: {v}")
        return 0
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

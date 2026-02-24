#!/usr/bin/env python3
"""
CLI tool for syncing Garmin Connect data using GarminDB.

This provides a convenient command-line interface for downloading and syncing
your Garmin health data without manually using GarminDB.
"""

import argparse
import logging
import sys

from garmin_analysis.data_ingestion.garmin_connect_sync import (
    GARMINDB_CONFIG_FILE,
    backup_garmindb,
    check_garmindb_installed,
    copy_garmindb_databases,
    create_garmindb_config,
    find_garmindb_databases,
    get_garmindb_stats,
    sync_garmin_data,
)
from garmin_analysis.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync Garmin Connect data using GarminDB integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial setup with credentials
  %(prog)s --setup --username your@email.com --password yourpassword
  
  # Download all historical data (first time)
  %(prog)s --sync --all
  
  # Download only latest data (daily update)
  %(prog)s --sync --latest
  
  # Backup your databases
  %(prog)s --backup
  
  # Show statistics about your data
  %(prog)s --stats

For more information, see: https://github.com/tcgoetz/GarminDB
        """,
    )

    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--setup",
        action="store_true",
        help="Set up GarminDB configuration with your Garmin Connect credentials",
    )
    action_group.add_argument("--sync", action="store_true", help="Sync data from Garmin Connect")
    action_group.add_argument("--backup", action="store_true", help="Backup GarminDB databases")
    action_group.add_argument(
        "--stats", action="store_true", help="Show statistics about synced data"
    )
    action_group.add_argument(
        "--copy-dbs",
        action="store_true",
        help="Copy GarminDB databases from ~/.GarminDb/ to project db/ directory",
    )
    action_group.add_argument(
        "--find-dbs", action="store_true", help="Find and list GarminDB databases (without copying)"
    )

    # Setup options
    setup_group = parser.add_argument_group("setup options")
    setup_group.add_argument(
        "--username", help="Garmin Connect username/email (required for --setup)"
    )
    setup_group.add_argument("--password", help="Garmin Connect password (required for --setup)")
    setup_group.add_argument(
        "--start-date",
        help="Start date for data download in MM/DD/YYYY format (default: 1 year ago)",
    )
    setup_group.add_argument(
        "--end-date", help="End date for data download in MM/DD/YYYY format (default: today)"
    )
    setup_group.add_argument(
        "--download-latest-activities",
        type=int,
        default=25,
        help="Number of recent activities to download with --latest (default: 25)",
    )
    setup_group.add_argument(
        "--download-all-activities",
        type=int,
        default=1000,
        help="Total activities to download with --all (default: 1000)",
    )

    # Sync options
    sync_group = parser.add_argument_group("sync options")
    sync_group.add_argument(
        "--latest", action="store_true", help="Only download latest data (for daily updates)"
    )
    sync_group.add_argument(
        "--all",
        dest="all_data",
        action="store_true",
        help="Download all data types (monitoring, sleep, weight, activities)",
    )
    sync_group.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download step (only import/analyze existing data)",
    )
    sync_group.add_argument(
        "--no-import", action="store_true", help="Skip import step (only download)"
    )
    sync_group.add_argument(
        "--no-analyze", action="store_true", help="Skip analyze step (only download and import)"
    )

    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        # Check if GarminDB is installed
        if not check_garmindb_installed():
            logger.error("GarminDB not found!")
            logger.error("Install it with: poetry add garmindb")
            logger.error("Or: pip install garmindb")
            return 1

        # Handle setup
        if args.setup:
            if not args.username or not args.password:
                logger.error("--setup requires --username and --password")
                parser.print_help()
                return 1

            logger.info("Setting up GarminDB configuration...")
            config_path = create_garmindb_config(
                username=args.username,
                password=args.password,
                start_date=args.start_date,
                end_date=args.end_date,
                download_latest_activities=args.download_latest_activities,
                download_all_activities=args.download_all_activities,
            )
            logger.info(f"✅ Configuration saved to {config_path}")
            logger.info("\nNext steps:")
            logger.info(f"  1. Review config: cat {config_path}")
            logger.info(f"  2. Run: {sys.argv[0]} --sync --all")
            return 0

        # Handle sync
        elif args.sync:
            if not GARMINDB_CONFIG_FILE.exists():
                logger.error("GarminDB not configured!")
                logger.error(
                    f"Run setup first: {sys.argv[0]} --setup --username your@email.com --password yourpassword"
                )
                return 1

            logger.info("Starting Garmin Connect data sync...")
            result = sync_garmin_data(
                download=not args.no_download,
                import_data=not args.no_import,
                analyze=not args.no_analyze,
                latest=args.latest,
                all_data=args.all_data,
            )

            if result["success"]:
                logger.info(f"✅ {result['message']}")
                logger.info(f"Database available at: {result['db_path']}")
                logger.info("\nNext steps:")
                logger.info(
                    "  1. Generate unified dataset: poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs"
                )
                logger.info("  2. Launch dashboard: poetry run python run_dashboard.py")
                return 0
            else:
                logger.error(f"❌ Sync failed: {result.get('message', 'Unknown error')}")
                return 1

        # Handle backup
        elif args.backup:
            logger.info("Backing up GarminDB databases...")
            result = backup_garmindb()
            if result["success"]:
                logger.info(f"✅ {result['message']}")
                return 0
            else:
                logger.error(f"❌ {result['message']}")
                return 1

        # Handle stats
        elif args.stats:
            stats = get_garmindb_stats()
            if stats:
                print("\n" + "=" * 70)
                print("GARMIN DATA STATISTICS")
                print("=" * 70)
                print(stats)
                print("=" * 70)
                return 0
            else:
                logger.warning("No statistics available. Run --sync first.")
                return 1

        # Handle find-dbs
        elif args.find_dbs:
            logger.info("Searching for GarminDB databases...")
            dbs = find_garmindb_databases()

            if dbs:
                print("\n" + "=" * 70)
                print("FOUND GARMINDB DATABASES")
                print("=" * 70)
                for name, path in dbs.items():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"  {name:12s} {path} ({size_mb:.1f} MB)")
                print("=" * 70)
                print(f"\n✅ Found {len(dbs)} database(s)")
                print("\nTo copy to project:")
                print(f"  {sys.argv[0]} --copy-dbs")
                return 0
            else:
                logger.error("No GarminDB databases found in ~/.GarminDb/")
                logger.info(
                    "Run GarminDB first: garmindb_cli.py --all --download --import --analyze"
                )
                return 1

        # Handle copy-dbs
        elif args.copy_dbs:
            logger.info("Copying GarminDB databases to project...")
            result = copy_garmindb_databases()

            if result["copied"]:
                print("\n" + "=" * 70)
                print("DATABASES COPIED")
                print("=" * 70)
                for db in result["copied"]:
                    print(f"  ✅ {db['name']}: {db['size_mb']:.1f} MB")
                    print(f"     {db['source']}")
                    print(f"  -> {db['target']}")
                    print()
                print("=" * 70)
                logger.info(f"✅ Copied {len(result['copied'])} database(s)")
                logger.info("\nNext step: Generate unified dataset")
                logger.info(
                    "  poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs"
                )
                return 0
            elif result["errors"]:
                logger.error(f"❌ Errors occurred: {len(result['errors'])}")
                for error in result["errors"]:
                    logger.error(f"  - {error}")
                return 1
            else:
                logger.warning("No databases to copy")
                return 1

    except KeyboardInterrupt:
        logger.info("\nSync interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

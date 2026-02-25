#!/usr/bin/env python3
"""
Single-command bootstrap for Garmin Analysis.

garmin init checks for databases, creates folders, validates schema,
and prints recommended next commands.
"""

import argparse
import logging
import sys
from pathlib import Path

from garmin_analysis.config import (
    CONFIG_DIR,
    DATA_DIR,
    DATA_QUALITY_REPORTS_DIR,
    DB_DIR,
    DB_PATHS,
    EXPORT_DIR,
    MASTER_CSV,
    MODELING_RESULTS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    ensure_directories_exist,
)
from garmin_analysis.data_ingestion.inspect_sqlite_schema import (
    detect_schema_drift,
    extract_schema,
)

logger = logging.getLogger(__name__)


def check_databases() -> dict:
    """Check which databases exist and report status."""
    status = {}
    for name, path in DB_PATHS.items():
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
        status[name] = {"exists": exists, "path": str(path), "size_mb": size_mb}
    return status


def validate_schema(db_path: Path) -> tuple[bool, str]:
    """
    Validate that a database has expected tables.
    Returns (ok, message).
    """
    if not db_path.exists():
        return False, "Database not found"

    schema = extract_schema(str(db_path))
    if not schema:
        return False, "No tables found"

    # Expected tables per GarminDB (minimal check)
    expected_by_db = {
        "garmin.db": ["daily_summary", "sleep", "stress", "resting_hr"],
        "garmin_activities.db": ["activities"],
        "garmin_monitoring.db": ["monitoring_hr"],
        "garmin_summary.db": ["days_summary"],
        "summary.db": ["days_summary"],
    }
    db_name = db_path.name
    expected_tables = expected_by_db.get(db_name, [])

    if not expected_tables:
        return True, f"Found {len(schema)} tables (no schema spec)"

    missing = [t for t in expected_tables if t not in schema]
    if missing:
        return False, f"Missing tables: {missing}"
    return True, f"OK ({len(schema)} tables)"


def run_init(verbose: bool = False) -> int:
    """Run the init bootstrap. Returns 0 on success, 1 on failure."""
    setup_logging = None
    try:
        from garmin_analysis.logging_config import setup_logging
    except ImportError:
        pass
    if setup_logging:
        setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    print("\n" + "=" * 60)
    print("  Garmin Analysis — Bootstrap (garmin init)")
    print("=" * 60)

    # 1. Create folders
    ensure_directories_exist()
    dirs = [DB_DIR, DATA_DIR, EXPORT_DIR, REPORTS_DIR, PLOTS_DIR, MODELING_RESULTS_DIR, DATA_QUALITY_REPORTS_DIR, CONFIG_DIR]
    cwd = Path.cwd()
    print("\n📁 Folders:")
    for d in dirs:
        exists = "✓" if d.exists() else "created"
        try:
            rel = d.relative_to(cwd)
        except ValueError:
            rel = d
        print(f"   {rel} — {exists}")

    # 2. Check databases
    print("\n🗄️  Databases:")
    db_status = check_databases()
    any_db = False
    for name, info in db_status.items():
        path = Path(info["path"])
        if info["exists"]:
            any_db = True
            ok, msg = validate_schema(path)
            icon = "✓" if ok else "⚠"
            size_str = f" ({info['size_mb']:.1f} MB)" if info["size_mb"] > 0 else ""
            print(f"   {icon} {name}{size_str}: {msg}")
        else:
            print(f"   ✗ {name}: not found")

    # 3. Master CSV
    master_exists = MASTER_CSV.exists()
    print("\n📊 Master dataset:")
    if master_exists:
        rows = sum(1 for _ in open(MASTER_CSV)) - 1
        print(f"   ✓ {MASTER_CSV.relative_to(Path.cwd())} ({rows} rows)")
    else:
        print(f"   ✗ {MASTER_CSV.relative_to(Path.cwd())} — not yet generated")

    # 4. Next commands
    print("\n" + "-" * 60)
    print("  Next commands")
    print("-" * 60)

    if not any_db:
        print("\n  No Garmin databases found. Choose one:")
        print("    A) Sync from Garmin Connect:")
        print("       poetry run python -m garmin_analysis.cli_garmin_sync --setup \\")
        print("         --username your@email.com --password yourpassword --start-date 01/01/2024")
        print("       poetry run python -m garmin_analysis.cli_garmin_sync --sync --all")
        print("")
        print("    B) Copy existing garmin.db:")
        print("       mkdir -p db && cp /path/to/GarminDB/garmin.db db/")
        print("")
    else:
        if not master_exists:
            print("\n  Generate unified dataset:")
            print("    poetry run python -m garmin_analysis.data_ingestion.load_all_garmin_dbs")
            print("")
        print("  Launch dashboard:")
        print("    poetry run python run_dashboard.py")
        print("")
        print("  Other useful commands:")
        print("    poetry run python -m garmin_analysis.features.quick_data_check --summary")
        print("    poetry run python -m garmin_analysis.cli_weekly_report")
        print("    poetry run python -m garmin_analysis.cli_export  # Export to Parquet")
        print("")

    print("=" * 60 + "\n")
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Garmin Analysis: check DBs, create folders, validate schema, print next commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python -m garmin_analysis.cli_init
  poetry run garmin-init
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    return run_init(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())

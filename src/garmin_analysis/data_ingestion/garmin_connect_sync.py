"""
Direct integration with GarminDB for automated data download from Garmin Connect.

This module provides a wrapper around the GarminDB library to download and sync
health data directly from Garmin Connect without requiring manual export.

Credits:
    This module integrates with GarminDB by Tom Goetz, an excellent open-source
    project for downloading and parsing Garmin health data.
    
    GarminDB Project: https://github.com/tcgoetz/GarminDB
    Author: Tom Goetz
    License: GPL-2.0
    
    We are grateful to Tom Goetz and all GarminDB contributors for their work
    that makes this integration possible.

Reference: https://github.com/tcgoetz/GarminDB
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

from garmin_analysis.config import DB_DIR, PROJECT_ROOT, CONFIG_DIR
from garmin_analysis.utils.error_handling import (
    handle_data_loading_errors,
    DataLoadingError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

# GarminDB configuration file location
GARMINDB_CONFIG_DIR = Path.home() / ".GarminDb"
GARMINDB_CONFIG_FILE = GARMINDB_CONFIG_DIR / "GarminConnectConfig.json"

# Default configuration template
def _get_default_config(
    username: str,
    password: str,
    start_date: str = None,
    download_latest_activities: int = 25,
    download_all_activities: int = 1000
):
    """
    Generate GarminDB config matching the official format.
    
    Args:
        username: Garmin Connect username/email
        password: Garmin Connect password
        start_date: Start date in MM/DD/YYYY format (default: 1 year ago)
        download_latest_activities: Number of recent activities to download
        download_all_activities: Number of total activities to download
    
    Returns:
        Complete GarminDB configuration dictionary
    """
    from datetime import datetime, timedelta
    
    # Calculate default start date (1 year ago) if not provided
    if start_date is None:
        start_date_obj = datetime.now() - timedelta(days=365)
        start_date = start_date_obj.strftime("%m/%d/%Y")
    
    return {
        "db": {
            "type": "sqlite"
        },
        "garmin": {
            "domain": "garmin.com"
        },
        "credentials": {
            "user": username,
            "secure_password": False,
            "password": password
        },
        "data": {
            "weight_start_date": start_date,
            "sleep_start_date": start_date,
            "rhr_start_date": start_date,
            "monitoring_start_date": start_date,
            "download_latest_activities": download_latest_activities,
            "download_all_activities": download_all_activities
        },
        "directories": {
            "relative_to_home": True,
            "base_dir": "HealthData",
            "mount_dir": "/Volumes/GARMIN"
        },
        "enabled_stats": {
            "monitoring": True,
            "steps": True,
            "itime": True,
            "sleep": True,
            "rhr": True,
            "weight": True,
            "activities": True
        },
        "course_views": {
            "steps": []
        },
        "modes": {},
        "activities": {
            "display": []
        },
        "settings": {
            "metric": False,
            "default_display_activities": ["walking", "running", "cycling"]
        },
        "checkup": {
            "look_back_days": 90
        }
    }


def check_garmindb_installed() -> bool:
    """
    Check if garmindb CLI is available.
    
    Returns:
        True if garmindb_cli.py is available, False otherwise
    """
    return shutil.which("garmindb_cli.py") is not None


@handle_data_loading_errors(reraise=True)
def create_garmindb_config(
    username: str,
    password: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    download_latest_activities: int = 25,
    download_all_activities: int = 1000,
    config_path: Optional[Path] = None
) -> Path:
    """
    Create GarminDB configuration file in the official format.
    
    Args:
        username: Garmin Connect username/email
        password: Garmin Connect password
        start_date: Start date in MM/DD/YYYY format (default: 1 year ago)
        end_date: End date in MM/DD/YYYY format (default: today, currently unused by GarminDB)
        download_latest_activities: Number of recent activities to download (default: 25)
        download_all_activities: Total activities to download on --all (default: 1000)
        config_path: Optional custom path for config file
    
    Returns:
        Path to created config file
    
    Raises:
        ConfigurationError: If config creation fails
        
    Example:
        >>> create_garmindb_config(
        ...     username="your@email.com",
        ...     password="yourpass",
        ...     start_date="01/01/2024"
        ... )
    """
    if config_path is None:
        config_path = GARMINDB_CONFIG_FILE
    
    # Create config directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create configuration with official GarminDB format
    config = _get_default_config(
        username=username,
        password=password,
        start_date=start_date,
        download_latest_activities=download_latest_activities,
        download_all_activities=download_all_activities
    )
    
    # Write config file
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created GarminDB config at {config_path}")
        logger.info(f"Start date: {config['data']['monitoring_start_date']}")
        logger.info(f"Download activities: {download_latest_activities} latest, {download_all_activities} all")
        return config_path
    except Exception as e:
        raise ConfigurationError(f"Failed to create GarminDB config: {e}") from e


@handle_data_loading_errors(reraise=True)
def sync_garmin_data(
    download: bool = True,
    import_data: bool = True,
    analyze: bool = True,
    latest: bool = False,
    all_data: bool = True
) -> Dict[str, Any]:
    """
    Sync data from Garmin Connect using GarminDB.
    
    This function calls the garmindb_cli.py command to download and process
    Garmin Connect data into SQLite databases.
    
    Args:
        download: Download data from Garmin Connect (default: True)
        import_data: Import downloaded data into databases (default: True)
        analyze: Analyze and create summary tables (default: True)
        latest: Only download latest data (default: False, downloads all)
        all_data: Process all data types (monitoring, sleep, weight, activities)
    
    Returns:
        Dictionary with sync results including:
        - success: bool
        - message: str
        - db_path: Path to garmin.db
    
    Raises:
        DataLoadingError: If GarminDB sync fails
        ConfigurationError: If GarminDB is not configured
    
    Example:
        >>> from garmin_analysis.data_ingestion.garmin_connect_sync import sync_garmin_data
        >>> result = sync_garmin_data(latest=True)
        >>> print(result['message'])
    """
    # Check if GarminDB is installed
    if not check_garmindb_installed():
        raise ConfigurationError(
            "GarminDB not found. Install it with: pip install garmindb\n"
            "Or run: poetry add garmindb"
        )
    
    # Check if config exists
    if not GARMINDB_CONFIG_FILE.exists():
        raise ConfigurationError(
            f"GarminDB config not found at {GARMINDB_CONFIG_FILE}\n"
            f"Create config with credentials using create_garmindb_config() or:\n"
            f"garmin-connect-setup --username your_email --password your_password"
        )
    
    # Build command
    cmd = ["garmindb_cli.py"]
    
    # Note: --all is required even with --latest to specify data types
    if all_data or latest:
        cmd.append("--all")
    
    if download:
        cmd.append("--download")
    
    if import_data:
        cmd.append("--import")
    
    if analyze:
        cmd.append("--analyze")
    
    if latest:
        cmd.append("--latest")
    
    logger.info(f"Running GarminDB sync: {' '.join(cmd)}")
    
    try:
        # Run garmindb_cli
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("GarminDB sync completed successfully")
        logger.debug(f"GarminDB output: {result.stdout}")
        
        # GarminDB creates databases based on config (~/HealthData/DBs by default)
        garmindb_dir = Path.home() / "HealthData" / "DBs"
        
        # Fallback to legacy location if needed
        if not garmindb_dir.exists():
            legacy_dir = Path.home() / ".GarminDb"
            if legacy_dir.exists():
                garmindb_dir = legacy_dir
        
        source_db = garmindb_dir / "garmin.db"
        
        # Copy to our project's db directory
        if source_db.exists():
            target_db = DB_DIR / "garmin.db"
            DB_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_db, target_db)
            logger.info(f"Copied garmin.db to {target_db}")
            
            return {
                "success": True,
                "message": "Garmin data synced successfully",
                "db_path": target_db,
                "source_db": source_db
            }
        else:
            raise DataLoadingError(f"Expected database not found at {source_db}")
            
    except subprocess.CalledProcessError as e:
        error_msg = f"GarminDB sync failed: {e.stderr}"
        logger.error(error_msg)
        raise DataLoadingError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error during GarminDB sync: {e}"
        logger.error(error_msg)
        raise DataLoadingError(error_msg) from e


def find_garmindb_databases(garmindb_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Find GarminDB database files in the GarminDB directory.
    
    Args:
        garmindb_dir: Optional custom GarminDB directory (default: ~/HealthData/DBs)
    
    Returns:
        Dictionary mapping database names to their paths
        
    Example:
        >>> dbs = find_garmindb_databases()
        >>> print(dbs)
        {'garmin': Path('/Users/you/HealthData/DBs/garmin.db'), ...}
    """
    if garmindb_dir is None:
        # Try the configured location first (based on our GarminDB config)
        garmindb_dir = Path.home() / "HealthData" / "DBs"
        
        # If it doesn't exist, try the legacy location
        if not garmindb_dir.exists():
            legacy_dir = Path.home() / ".GarminDb"
            if legacy_dir.exists():
                logger.info(f"Using legacy GarminDB directory: {legacy_dir}")
                garmindb_dir = legacy_dir
    
    db_files = {
        "garmin": garmindb_dir / "garmin.db",
        "activities": garmindb_dir / "garmin_activities.db",
        "monitoring": garmindb_dir / "garmin_monitoring.db",
        "summary": garmindb_dir / "garmin_summary.db",
    }
    
    # Return only databases that exist
    found_dbs = {name: path for name, path in db_files.items() if path.exists()}
    
    if found_dbs:
        logger.info(f"Found {len(found_dbs)} GarminDB database(s) in {garmindb_dir}")
        for name, path in found_dbs.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"  - {name}: {path.name} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"No GarminDB databases found in {garmindb_dir}")
        logger.info(f"Searched locations:")
        logger.info(f"  - {Path.home() / 'HealthData' / 'DBs'}")
        logger.info(f"  - {Path.home() / '.GarminDb'}")
    
    return found_dbs


def copy_garmindb_databases(
    target_dir: Optional[Path] = None,
    garmindb_dir: Optional[Path] = None,
    databases: Optional[list] = None
) -> Dict[str, Any]:
    """
    Copy GarminDB databases from GarminDB directory to project directory.
    
    Args:
        target_dir: Target directory for databases (default: project's db/ directory)
        garmindb_dir: Source GarminDB directory (default: ~/.GarminDb)
        databases: List of database names to copy (default: all found databases)
                  Options: 'garmin', 'activities', 'monitoring', 'summary'
    
    Returns:
        Dictionary with copy results including:
        - copied: list of successfully copied databases
        - skipped: list of databases not found
        - errors: list of errors encountered
        
    Example:
        >>> result = copy_garmindb_databases()
        >>> print(f"Copied {len(result['copied'])} databases")
    """
    from garmin_analysis.config import DB_DIR
    
    if target_dir is None:
        target_dir = DB_DIR
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available databases
    found_dbs = find_garmindb_databases(garmindb_dir)
    
    if not found_dbs:
        return {
            "copied": [],
            "skipped": ["No databases found"],
            "errors": []
        }
    
    # Determine which databases to copy
    if databases is None:
        databases_to_copy = found_dbs
    else:
        databases_to_copy = {name: path for name, path in found_dbs.items() if name in databases}
    
    copied = []
    skipped = []
    errors = []
    
    for name, source_path in databases_to_copy.items():
        target_path = target_dir / source_path.name
        
        try:
            shutil.copy2(source_path, target_path)
            size_mb = target_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Copied {name}: {source_path} -> {target_path} ({size_mb:.1f} MB)")
            copied.append({
                "name": name,
                "source": str(source_path),
                "target": str(target_path),
                "size_mb": size_mb
            })
        except Exception as e:
            error_msg = f"Failed to copy {name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    # Check for databases that were requested but not found
    if databases:
        not_found = set(databases) - set(found_dbs.keys())
        for name in not_found:
            skip_msg = f"{name}: not found in {garmindb_dir or Path.home() / '.GarminDb'}"
            logger.warning(skip_msg)
            skipped.append(skip_msg)
    
    logger.info(f"Database copy complete: {len(copied)} copied, {len(skipped)} skipped, {len(errors)} errors")
    
    return {
        "copied": copied,
        "skipped": skipped,
        "errors": errors
    }


def get_garmindb_stats() -> Optional[str]:
    """
    Get statistics about synced GarminDB data.
    
    Returns:
        String with statistics or None if stats file doesn't exist
    """
    # Try configured location first
    garmindb_dir = Path.home() / "HealthData" / "DBs"
    stats_file = garmindb_dir / "stats.txt"
    
    # Fallback to legacy location
    if not stats_file.exists():
        legacy_dir = Path.home() / ".GarminDb"
        stats_file = legacy_dir / "stats.txt"
    
    if stats_file.exists():
        try:
            return stats_file.read_text()
        except Exception as e:
            logger.warning(f"Failed to read GarminDB stats: {e}")
            return None
    return None


def backup_garmindb() -> Dict[str, Any]:
    """
    Backup GarminDB databases.
    
    Returns:
        Dictionary with backup results
    """
    try:
        result = subprocess.run(
            ["garmindb_cli.py", "--backup"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("GarminDB backup completed")
        return {
            "success": True,
            "message": "Backup completed successfully"
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"GarminDB backup failed: {e.stderr}")
        return {
            "success": False,
            "message": f"Backup failed: {e.stderr}"
        }


if __name__ == "__main__":
    # Example usage
    from garmin_analysis.logging_config import setup_logging
    
    setup_logging(level=logging.INFO)
    
    # Check if GarminDB is installed
    if check_garmindb_installed():
        logger.info("GarminDB is installed and available")
    else:
        logger.error("GarminDB not found. Install with: pip install garmindb")
        exit(1)
    
    # Check if config exists
    if GARMINDB_CONFIG_FILE.exists():
        logger.info(f"GarminDB config found at {GARMINDB_CONFIG_FILE}")
        
        # Sync latest data
        logger.info("Syncing latest Garmin data...")
        result = sync_garmin_data(latest=True)
        logger.info(f"Sync result: {result['message']}")
        
        # Show stats
        stats = get_garmindb_stats()
        if stats:
            logger.info(f"GarminDB Statistics:\n{stats}")
    else:
        logger.warning(f"GarminDB config not found at {GARMINDB_CONFIG_FILE}")
        logger.info("Create config with your Garmin Connect credentials")


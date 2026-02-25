"""
Export master dataset to Parquet and optionally DuckDB for faster analytics.

Parquet provides columnar storage for efficient downstream use in pandas,
DuckDB, Spark, etc. DuckDB enables SQL queries without loading full dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from garmin_analysis.config import (
    MASTER_CSV,
    DATA_DIR,
    EXPORT_DIR,
    DAILY_DATA_QUALITY_CSV,
)

logger = logging.getLogger(__name__)


def export_to_parquet(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    *,
    include_data_quality: bool = True,
) -> Path:
    """
    Export master dataset to Parquet format.

    Args:
        df: Master DataFrame. If None, loads from MASTER_CSV.
        output_path: Output path. Default: data/export/master_daily_summary.parquet
        include_data_quality: If True, merge daily data quality scores into export.

    Returns:
        Path to written Parquet file.
    """
    if df is None:
        df = pd.read_csv(MASTER_CSV, parse_dates=["day"])
        logger.info(f"Loaded master from {MASTER_CSV} ({len(df)} rows)")

    if output_path is None:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / "master_daily_summary.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_data_quality and DAILY_DATA_QUALITY_CSV.exists():
        try:
            dq = pd.read_csv(DAILY_DATA_QUALITY_CSV, parse_dates=["day"])
            df = df.merge(dq, on="day", how="left", suffixes=("", "_dq"))
            # Avoid duplicate columns
            df = df[[c for c in df.columns if not c.endswith("_dq")]]
            logger.info("Merged daily data quality into export")
        except Exception as e:
            logger.warning(f"Could not merge data quality: {e}")

    df.to_parquet(output_path, index=False)
    logger.info(f"Exported {len(df)} rows to {output_path}")
    return output_path


def export_to_duckdb(
    parquet_path: Optional[Path] = None,
    duckdb_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Export master dataset to DuckDB for SQL analytics.

    Creates a DuckDB database with a 'master' table. Requires duckdb package.

    Args:
        parquet_path: Path to Parquet file. If None, exports Parquet first.
        duckdb_path: Output DuckDB path. Default: data/export/master.duckdb

    Returns:
        Path to DuckDB file, or None if duckdb not installed.
    """
    try:
        import duckdb
    except ImportError:
        logger.warning("duckdb not installed. Install with: pip install duckdb")
        return None

    if parquet_path is None:
        parquet_path = export_to_parquet()

    if duckdb_path is None:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        duckdb_path = EXPORT_DIR / "master.duckdb"

    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(duckdb_path))
    try:
        conn.execute(f"CREATE OR REPLACE TABLE master AS SELECT * FROM read_parquet('{parquet_path}')")
        conn.close()
        logger.info(f"Exported to DuckDB: {duckdb_path}")
        return duckdb_path
    except Exception as e:
        conn.close()
        logger.error(f"DuckDB export failed: {e}")
        raise


def export_master(
    parquet: bool = True,
    duckdb: bool = False,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Export master dataset to requested formats.

    Args:
        parquet: Export to Parquet (default True).
        duckdb: Also export to DuckDB (optional, requires duckdb package).
        df: Master DataFrame. If None, loads from CSV.

    Returns:
        Dict with 'parquet_path' and optionally 'duckdb_path'.
    """
    result = {}
    parquet_path = None

    if parquet:
        parquet_path = export_to_parquet(df=df)
        result["parquet_path"] = str(parquet_path)

    if duckdb:
        db_path = export_to_duckdb(parquet_path=parquet_path)
        if db_path:
            result["duckdb_path"] = str(db_path)

    return result

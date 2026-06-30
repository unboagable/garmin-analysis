"""
Export master dataset to Parquet and optionally DuckDB for faster analytics.

Parquet provides columnar storage for efficient downstream use in pandas,
DuckDB, Spark, etc. DuckDB enables SQL queries without loading full dataset.
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from garmin_analysis.config import (
    DAILY_DATA_QUALITY_CSV,
    EXPORT_DIR,
    MASTER_CSV,
)
from garmin_analysis.utils.data_filtering import filter_by_date

logger = logging.getLogger(__name__)

SUPPORTED_EXPORT_FORMATS = ("csv", "parquet", "json", "duckdb")

DateLike = Union[str, date, datetime, pd.Timestamp]


def normalize_export_format(export_format: str) -> str:
    """Validate and normalize an export format name."""
    normalized = export_format.strip().lower()
    if normalized not in SUPPORTED_EXPORT_FORMATS:
        supported = ", ".join(SUPPORTED_EXPORT_FORMATS)
        raise ValueError(f"Unsupported export format '{export_format}'. Choose one of: {supported}")
    return normalized


def _format_date_label(value: DateLike) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def filter_master_by_date_range(
    df: pd.DataFrame,
    start_date: DateLike,
    end_date: DateLike,
    *,
    date_col: str = "day",
) -> pd.DataFrame:
    """
    Return master rows whose ``date_col`` falls within [start_date, end_date].
    """
    if df is None or df.empty:
        logger.warning("filter_master_by_date_range received empty dataframe")
        return df

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        raise ValueError(
            f"start_date ({_format_date_label(start)}) must be on or before end_date ({_format_date_label(end)})"
        )

    filtered = filter_by_date(
        df.copy(),
        date_col=date_col,
        from_date=start,
        to_date=end,
    )
    logger.info(
        "Filtered export to %s through %s: %d of %d rows",
        _format_date_label(start),
        _format_date_label(end),
        len(filtered),
        len(df),
    )
    return filtered


def build_export_output_path(
    export_format: str,
    *,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Build a default export path, optionally including the date range in the filename."""
    export_format = normalize_export_format(export_format)
    output_dir = Path(output_dir or EXPORT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if start_date is not None and end_date is not None:
        stem = (
            f"master_daily_summary_{_format_date_label(start_date)}_{_format_date_label(end_date)}"
        )
    else:
        stem = "master_daily_summary"

    extensions = {
        "csv": ".csv",
        "parquet": ".parquet",
        "json": ".json",
        "duckdb": ".duckdb",
    }
    return output_dir / f"{stem}{extensions[export_format]}"


def _load_master_for_export(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is not None:
        return df
    from garmin_analysis.utils.data_loading import load_master_dataframe

    loaded = load_master_dataframe()
    logger.info("Loaded master dataset (%d rows)", len(loaded))
    return loaded


def _merge_data_quality(df: pd.DataFrame, *, include_data_quality: bool) -> pd.DataFrame:
    if not include_data_quality or not DAILY_DATA_QUALITY_CSV.exists():
        return df

    try:
        dq = pd.read_csv(DAILY_DATA_QUALITY_CSV, parse_dates=["day"])
        merged = df.merge(dq, on="day", how="left", suffixes=("", "_dq"))
        merged = merged[[col for col in merged.columns if not col.endswith("_dq")]]
        logger.info("Merged daily data quality into export")
        return merged
    except Exception as exc:
        logger.warning("Could not merge data quality: %s", exc)
        return df


def export_to_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Exported %d rows to %s", len(df), output_path)
    return output_path


def export_to_json(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(df.to_json(orient="records", date_format="iso"))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    logger.info("Exported %d rows to %s", len(df), output_path)
    return output_path


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
        logger.info("Loaded master from %s (%d rows)", MASTER_CSV, len(df))

    if output_path is None:
        output_path = build_export_output_path("parquet")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _merge_data_quality(df, include_data_quality=include_data_quality)
    df.to_parquet(output_path, index=False)
    logger.info("Exported %d rows to %s", len(df), output_path)
    return output_path


def export_to_duckdb(
    parquet_path: Optional[Path] = None,
    duckdb_path: Optional[Path] = None,
    *,
    df: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    """
    Export master dataset to DuckDB for SQL analytics.

    Creates a DuckDB database with a 'master' table. Requires duckdb package.

    Args:
        parquet_path: Path to Parquet file. If None, exports Parquet first.
        duckdb_path: Output DuckDB path. Default: data/export/master.duckdb
        df: Optional DataFrame to export when parquet_path is not provided.

    Returns:
        Path to DuckDB file, or None if duckdb not installed.
    """
    try:
        import duckdb
    except ImportError:
        logger.warning("duckdb not installed. Install with: pip install duckdb")
        return None

    if parquet_path is None:
        parquet_path = export_to_parquet(df=df)

    if duckdb_path is None:
        duckdb_path = build_export_output_path("duckdb")

    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(duckdb_path))
    try:
        conn.execute(
            f"CREATE OR REPLACE TABLE master AS SELECT * FROM read_parquet('{parquet_path}')"
        )
        conn.close()
        logger.info("Exported to DuckDB: %s", duckdb_path)
        return duckdb_path
    except Exception as exc:
        conn.close()
        logger.error("DuckDB export failed: %s", exc)
        raise


def export_master_date_range(
    start_date: DateLike,
    end_date: DateLike,
    export_format: str,
    *,
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    include_data_quality: bool = True,
    date_col: str = "day",
    summary_only: bool = False,
) -> dict:
    """
    Export data for an inclusive date range in the requested format.

    By default exports all Garmin source tables plus the merged master dataset.
    Set ``summary_only=True`` to export only the merged master daily summary.
    """
    if not summary_only:
        from garmin_analysis.data_ingestion.export_all_data import export_all_data_date_range

        output_dir = Path(output_path) if output_path is not None else None
        if output_dir is not None and output_dir.suffix:
            raise ValueError(
                "All-data export writes a directory (or .duckdb file). "
                "Pass a directory path to --output, not a single-table filename."
            )
        return export_all_data_date_range(
            start_date,
            end_date,
            export_format,
            output_dir=output_dir,
            include_data_quality=include_data_quality,
        )

    export_format = normalize_export_format(export_format)
    master_df = _load_master_for_export(df)
    filtered = filter_master_by_date_range(
        master_df,
        start_date,
        end_date,
        date_col=date_col,
    )
    filtered = _merge_data_quality(filtered, include_data_quality=include_data_quality)

    if output_path is None:
        output_path = build_export_output_path(
            export_format,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        output_path = Path(output_path)

    if export_format == "csv":
        written = export_to_csv(filtered, output_path)
    elif export_format == "parquet":
        written = export_to_parquet(
            df=filtered,
            output_path=output_path,
            include_data_quality=False,
        )
    elif export_format == "json":
        written = export_to_json(filtered, output_path)
    else:
        parquet_path = output_path.with_suffix(".parquet")
        export_to_parquet(
            df=filtered,
            output_path=parquet_path,
            include_data_quality=False,
        )
        written = export_to_duckdb(parquet_path=parquet_path, duckdb_path=output_path)
        if written is None:
            raise RuntimeError("DuckDB export requires the optional duckdb package")

    return {
        "format": export_format,
        "output_path": str(written),
        "row_count": len(filtered),
        "start_date": _format_date_label(start_date),
        "end_date": _format_date_label(end_date),
        "scope": "summary",
    }


def export_master(
    parquet: bool = True,
    duckdb: bool = False,
    df: Optional[pd.DataFrame] = None,
    *,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    export_format: Optional[str] = None,
    output_path: Optional[Path] = None,
    include_data_quality: bool = True,
    summary_only: bool = False,
) -> dict:
    """
    Export master dataset to requested formats.

    When ``start_date`` and ``end_date`` are provided, only rows in that inclusive
    range are exported. If ``export_format`` is also provided, a single file is
    written in that format. Otherwise the legacy parquet/duckdb flags apply.
    """
    if start_date is not None or end_date is not None:
        if start_date is None or end_date is None:
            raise ValueError("Both start_date and end_date are required for date-range export")
        chosen_format = export_format or ("duckdb" if duckdb and not parquet else "parquet")
        return export_master_date_range(
            start_date,
            end_date,
            chosen_format,
            df=df,
            output_path=output_path,
            include_data_quality=include_data_quality,
            summary_only=summary_only,
        )

    result = {}
    parquet_path = None

    if parquet:
        parquet_path = export_to_parquet(df=df, include_data_quality=include_data_quality)
        result["parquet_path"] = str(parquet_path)

    if duckdb:
        db_path = export_to_duckdb(parquet_path=parquet_path, df=df)
        if db_path:
            result["duckdb_path"] = str(db_path)

    return result

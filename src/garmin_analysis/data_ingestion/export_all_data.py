"""
Export all Garmin SQLite tables and the merged master dataset for a date range.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from garmin_analysis.config import DAILY_DATA_QUALITY_CSV, DB_PATHS, EXPORT_DIR
from garmin_analysis.data_ingestion.export_master import (
    SUPPORTED_EXPORT_FORMATS,
    DateLike,
    _format_date_label,
    _merge_data_quality,
    export_to_csv,
    export_to_json,
    export_to_parquet,
    filter_master_by_date_range,
    normalize_export_format,
)
from garmin_analysis.data_ingestion.inspect_sqlite_schema import extract_schema
from garmin_analysis.data_ingestion.load_all_garmin_dbs import load_table
from garmin_analysis.utils.data_filtering import filter_by_date

logger = logging.getLogger(__name__)

TABLE_DATE_COLUMNS: Dict[Tuple[str, str], str] = {
    ("garmin", "daily_summary"): "day",
    ("garmin", "sleep"): "day",
    ("garmin", "stress"): "timestamp",
    ("garmin", "resting_hr"): "day",
    ("summary", "days_summary"): "day",
    ("activities", "activities"): "start_time",
    ("activities", "steps_activities"): "timestamp",
    ("monitoring", "monitoring_hr"): "timestamp",
}

DATE_COLUMN_CANDIDATES = ("day", "timestamp", "start_time", "date", "start_date")


@dataclass(frozen=True)
class ExportTableSpec:
    db_key: str
    db_path: Path
    table_name: str
    date_column: Optional[str]
    parse_dates: Optional[Tuple[str, ...]]


def _validate_date_range(
    start_date: DateLike, end_date: DateLike
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        raise ValueError(
            f"start_date ({_format_date_label(start)}) must be on or before end_date ({_format_date_label(end)})"
        )
    return start, end


def resolve_date_column(db_key: str, table_name: str, columns: Iterable[str]) -> Optional[str]:
    """Pick the column used to filter a table by calendar date."""
    registry_key = (db_key, table_name)
    if registry_key in TABLE_DATE_COLUMNS:
        candidate = TABLE_DATE_COLUMNS[registry_key]
        if candidate in columns:
            return candidate

    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def discover_export_tables(db_paths: Optional[Dict[str, Path]] = None) -> List[ExportTableSpec]:
    """List exportable tables across configured Garmin SQLite databases."""
    db_paths = db_paths or DB_PATHS
    specs: List[ExportTableSpec] = []

    for db_key, db_path in db_paths.items():
        path = Path(db_path)
        if not path.exists():
            logger.warning("Skipping missing database for export: %s", path)
            continue

        with sqlite3.connect(path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
                conn,
            )["name"].tolist()

        schema = extract_schema(str(path))
        for table_name in tables:
            col_names = [name for name, _ in schema.get(table_name, [])]
            date_column = resolve_date_column(db_key, table_name, col_names)
            parse_dates = (date_column,) if date_column else None
            specs.append(
                ExportTableSpec(
                    db_key=db_key,
                    db_path=path,
                    table_name=table_name,
                    date_column=date_column,
                    parse_dates=parse_dates,
                )
            )

    return specs


def filter_table_by_date_range(
    df: pd.DataFrame,
    start_date: DateLike,
    end_date: DateLike,
    *,
    date_column: Optional[str],
) -> pd.DataFrame:
    """Filter a source table to an inclusive date range."""
    if df is None or df.empty:
        return df
    if date_column is None or date_column not in df.columns:
        logger.warning(
            "No date column for table filter; exporting all %d rows unchanged",
            len(df),
        )
        return df

    start, end = _validate_date_range(start_date, end_date)
    end_of_day = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = filter_by_date(
        df.copy(),
        date_col=date_column,
        from_date=start,
        to_date=end_of_day if date_column == "timestamp" or "time" in date_column else end,
    )
    return filtered


def build_export_bundle_dir(
    start_date: DateLike,
    end_date: DateLike,
    *,
    output_dir: Optional[Path] = None,
) -> Path:
    bundle_dir = Path(
        output_dir
        or (
            EXPORT_DIR
            / f"garmin_export_{_format_date_label(start_date)}_{_format_date_label(end_date)}"
        )
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def _table_output_path(
    bundle_dir: Path,
    export_format: str,
    db_key: str,
    table_name: str,
) -> Path:
    table_dir = bundle_dir / db_key
    table_dir.mkdir(parents=True, exist_ok=True)
    extensions = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}
    return table_dir / f"{table_name}{extensions[export_format]}"


def _write_export_file(
    df: pd.DataFrame,
    output_path: Path,
    export_format: str,
) -> Path:
    export_format = normalize_export_format(export_format)
    if export_format == "csv":
        return export_to_csv(df, output_path)
    if export_format == "parquet":
        return export_to_parquet(df=df, output_path=output_path, include_data_quality=False)
    if export_format == "json":
        return export_to_json(df, output_path)
    raise ValueError(f"Unsupported bundle table format: {export_format}")


def _export_tables_to_duckdb(
    tables: Dict[str, pd.DataFrame],
    duckdb_path: Path,
) -> Path:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB export requires the optional duckdb package") from exc

    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(duckdb_path))
    try:
        for table_id, df in tables.items():
            conn.register("export_df", df)
            conn.execute(f'CREATE OR REPLACE TABLE "{table_id}" AS SELECT * FROM export_df')
            conn.unregister("export_df")
        conn.close()
        return duckdb_path
    except Exception:
        conn.close()
        raise


def export_all_data_date_range(
    start_date: DateLike,
    end_date: DateLike,
    export_format: str,
    *,
    output_dir: Optional[Path] = None,
    include_master: bool = True,
    include_data_quality: bool = True,
    include_daily_data_quality_file: bool = True,
    db_paths: Optional[Dict[str, Path]] = None,
    table_specs: Optional[List[ExportTableSpec]] = None,
) -> dict:
    """
    Export all Garmin source tables plus the merged master dataset for a date range.

    Returns metadata including output directory and per-table row counts.
    """
    export_format = normalize_export_format(export_format)
    if export_format == "duckdb":
        bundle_dir = build_export_bundle_dir(start_date, end_date, output_dir=output_dir)
        duckdb_path = bundle_dir / "garmin_export.duckdb"
    else:
        bundle_dir = build_export_bundle_dir(start_date, end_date, output_dir=output_dir)
        duckdb_path = None

    exported_tables: Dict[str, int] = {}
    duckdb_tables: Dict[str, pd.DataFrame] = {}

    specs = table_specs if table_specs is not None else discover_export_tables(db_paths)
    for spec in specs:
        df = load_table(spec.db_path, spec.table_name, parse_dates=list(spec.parse_dates or []))
        filtered = filter_table_by_date_range(
            df,
            start_date,
            end_date,
            date_column=spec.date_column,
        )
        table_id = f"{spec.db_key}__{spec.table_name}"
        exported_tables[table_id] = len(filtered)

        if export_format == "duckdb":
            duckdb_tables[table_id] = filtered
        else:
            output_path = _table_output_path(
                bundle_dir, export_format, spec.db_key, spec.table_name
            )
            _write_export_file(filtered, output_path, export_format)

    if include_master:
        from garmin_analysis.utils.data_loading import load_master_dataframe

        master_df = _merge_data_quality(
            filter_master_by_date_range(
                load_master_dataframe(),
                start_date,
                end_date,
            ),
            include_data_quality=include_data_quality,
        )
        exported_tables["master__daily_summary"] = len(master_df)
        if export_format == "duckdb":
            duckdb_tables["master__daily_summary"] = master_df
        else:
            master_name = {
                "csv": "master_daily_summary.csv",
                "parquet": "master_daily_summary.parquet",
                "json": "master_daily_summary.json",
            }[export_format]
            _write_export_file(master_df, bundle_dir / master_name, export_format)

    if include_daily_data_quality_file and DAILY_DATA_QUALITY_CSV.exists():
        dq_df = filter_master_by_date_range(
            pd.read_csv(DAILY_DATA_QUALITY_CSV, parse_dates=["day"]),
            start_date,
            end_date,
        )
        exported_tables["master__daily_data_quality"] = len(dq_df)
        if export_format == "duckdb":
            duckdb_tables["master__daily_data_quality"] = dq_df
        elif export_format != "duckdb":
            dq_ext = {"csv": ".csv", "parquet": ".parquet", "json": ".json"}[export_format]
            _write_export_file(dq_df, bundle_dir / f"daily_data_quality{dq_ext}", export_format)

    if export_format == "duckdb":
        _export_tables_to_duckdb(duckdb_tables, duckdb_path)
        output_path = duckdb_path
    else:
        output_path = bundle_dir

    return {
        "format": export_format,
        "output_path": str(output_path),
        "output_dir": str(bundle_dir),
        "table_count": len(exported_tables),
        "tables": exported_tables,
        "start_date": _format_date_label(start_date),
        "end_date": _format_date_label(end_date),
        "scope": "all",
    }

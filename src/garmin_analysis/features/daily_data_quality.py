"""
Daily data quality score computation and persistence.

Computes a composite daily data quality score from:
- 24-hour coverage (coverage_pct, has_24h_coverage)
- Completeness of key health metrics (steps, sleep score, stress, etc.)

Persists to CSV for downstream use and dashboard display.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from garmin_analysis.config import DAILY_DATA_QUALITY_CSV, DATA_DIR

logger = logging.getLogger(__name__)

# Key columns used to compute completeness (must exist in master)
KEY_METRIC_COLUMNS = [
    "steps",
    "score",  # sleep score
    "stress_avg",
    "resting_heart_rate",
    "rhr",
    "calories_total",
    "body_battery_max",
    "bb_max",
    "body_battery_min",
    "bb_min",
]


def compute_daily_data_quality_score(
    master_df: pd.DataFrame,
    *,
    coverage_weight: float = 0.5,
    completeness_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Compute a daily data quality score (0-100) for each day.

    Score combines:
    - coverage_score: from coverage_pct (0-100) or has_24h_coverage (100/0)
    - completeness_score: % of available key metrics that are non-null

    Args:
        master_df: Master daily summary DataFrame
        coverage_weight: Weight for coverage component (default 0.5)
        completeness_weight: Weight for completeness component (default 0.5)

    Returns:
        DataFrame with columns: day, data_quality_score, coverage_score,
        completeness_score, key_metrics_count, key_metrics_total
    """
    if master_df is None or master_df.empty or "day" not in master_df.columns:
        logger.warning("compute_daily_data_quality_score: empty or invalid master_df")
        return pd.DataFrame()

    df = master_df.copy()
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()

    # Coverage score (0-100)
    if "coverage_pct" in df.columns:
        coverage_score = df["coverage_pct"].fillna(0).clip(0, 100)
    elif "has_24h_coverage" in df.columns:
        coverage_score = df["has_24h_coverage"].map(lambda x: 100.0 if x else 0.0)
    else:
        coverage_score = pd.Series(50.0, index=df.index)  # neutral default

    # Completeness: which key columns exist and are non-null
    available = [c for c in KEY_METRIC_COLUMNS if c in df.columns]
    if not available:
        completeness_score = pd.Series(0.0, index=df.index)
        key_total = 0
        non_null = pd.Series(0, index=df.index)
    else:
        non_null = df[available].notna().sum(axis=1)
        key_total = len(available)
        completeness_score = (non_null / key_total * 100).clip(0, 100)

    # Composite score
    total_weight = coverage_weight + completeness_weight
    data_quality_score = (
        coverage_weight * coverage_score + completeness_weight * completeness_score
    ) / total_weight

    result = pd.DataFrame({
        "day": df["day"],
        "data_quality_score": data_quality_score.round(1),
        "coverage_score": coverage_score.round(1),
        "completeness_score": completeness_score.round(1),
        "key_metrics_count": non_null,
        "key_metrics_total": key_total,
    })

    return result


def persist_daily_data_quality(
    quality_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Persist daily data quality CSV. Returns path written."""
    if output_path is None:
        output_path = DAILY_DATA_QUALITY_CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(output_path, index=False)
    logger.info(f"Saved daily data quality to {output_path} ({len(quality_df)} rows)")
    return output_path


def load_daily_data_quality(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load daily data quality CSV. Returns empty DataFrame if not found."""
    if path is None:
        path = DAILY_DATA_QUALITY_CSV
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["day"])
    return df


def compute_and_persist_daily_data_quality(
    master_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute daily data quality from master, persist to CSV, return the quality DataFrame.
    """
    if master_df is None:
        from garmin_analysis.utils.data_loading import load_master_dataframe
        master_df = load_master_dataframe()
    quality_df = compute_daily_data_quality_score(master_df)
    if quality_df.empty:
        return quality_df
    persist_daily_data_quality(quality_df, output_path)
    return quality_df

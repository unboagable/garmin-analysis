import pandas as pd
import logging
from pathlib import Path

from garmin_analysis.config import MASTER_CSV, MODELING_CSV


logger = logging.getLogger(__name__)
# Logging is configured at package level

def prepare_modeling_dataset(
    input_path: str = None,
    output_path: str = None,
    required_features: list = None,
    missing_threshold: float = 0.5,
    min_coverage_pct: float = None,
    require_24h_coverage: bool = False
):
    if input_path is None:
        input_path = str(MASTER_CSV)
    if output_path is None:
        output_path = str(MODELING_CSV)
    if required_features is None:
        required_features = [
            "score",
            "stress_avg",
            "yesterday_activity_minutes"
        ]

    # Load dataset
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path, parse_dates=["day"])
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Filter by 24-hour coverage if requested
    if require_24h_coverage and "has_24h_coverage" in df.columns:
        before_rows = len(df)
        df = df[df["has_24h_coverage"] == True]
        after_rows = len(df)
        logger.info(f"Dropped {before_rows - after_rows} rows without 24-hour coverage")
    elif min_coverage_pct is not None and "coverage_pct" in df.columns:
        before_rows = len(df)
        df = df[df["coverage_pct"] >= min_coverage_pct]
        after_rows = len(df)
        logger.info(f"Dropped {before_rows - after_rows} rows with coverage < {min_coverage_pct}%")

    # Drop rows missing critical features (skip features not present in data)
    available_required = [f for f in required_features if f in df.columns]
    missing_required = [f for f in required_features if f not in df.columns]
    if missing_required:
        logger.warning(f"Required features not found in data: {missing_required}")
    before_rows = len(df)
    if available_required:
        df = df.dropna(subset=available_required)
    after_rows = len(df)
    logger.info(f"Dropped {before_rows - after_rows} rows missing required features")

    # Drop columns with too much missingness
    col_threshold = df.isnull().mean() < missing_threshold
    kept_cols = df.columns[col_threshold].tolist()
    dropped_cols = df.columns[~col_threshold].tolist()
    df = df[kept_cols]
    logger.info(f"Dropped {len(dropped_cols)} columns with > {int(missing_threshold*100)}% missing values")

    # Drop metadata and flags
    to_drop = [col for col in df.columns if col.startswith("missing_") or "Unnamed" in col]
    df.drop(columns=to_drop, inplace=True, errors="ignore")
    logger.info(f"Dropped {len(to_drop)} metadata/flag columns")

    # Save result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned modeling dataset to: {output_path}")

if __name__ == "__main__":
    prepare_modeling_dataset()

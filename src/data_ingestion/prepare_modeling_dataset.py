import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_modeling_dataset(
    input_path: str = "data/master_daily_summary.csv",
    output_path: str = "data/modeling_ready_dataset.csv",
    required_features: list = None,
    missing_threshold: float = 0.5
):
    if required_features is None:
        required_features = [
            "score",
            "stress_avg",
            "yesterday_activity_minutes"
        ]

    # Load dataset
    if not Path(input_path).exists():
        logging.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path, parse_dates=["day"])
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # Drop rows missing critical features
    before_rows = len(df)
    df = df.dropna(subset=required_features)
    after_rows = len(df)
    logging.info(f"Dropped {before_rows - after_rows} rows missing required features")

    # Drop columns with too much missingness
    col_threshold = df.isnull().mean() < missing_threshold
    kept_cols = df.columns[col_threshold].tolist()
    dropped_cols = df.columns[~col_threshold].tolist()
    df = df[kept_cols]
    logging.info(f"Dropped {len(dropped_cols)} columns with > {int(missing_threshold*100)}% missing values")

    # Drop metadata and flags
    to_drop = [col for col in df.columns if col.startswith("missing_") or "Unnamed" in col]
    df.drop(columns=to_drop, inplace=True, errors="ignore")
    logging.info(f"Dropped {len(to_drop)} metadata/flag columns")

    # Save result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned modeling dataset to: {output_path}")

if __name__ == "__main__":
    prepare_modeling_dataset()

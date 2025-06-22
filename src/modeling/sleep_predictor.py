import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from src.utils import filter_required_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "data/master_daily_summary.csv"

def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        logging.error("Missing master dataset — please run load_all_garmin_dbs.py first.")
        return None, None

    df = pd.read_csv(DATA_PATH, parse_dates=["day"])
    df = df[df["score"].notnull()]  # Only include rows with a sleep score

    # Filter out days missing essential predictors
    df = filter_required_columns(df, ["yesterday_activity_minutes", "stress_avg"])

    drop_cols = ["day"]
    target = "score"

    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target])
    y = df[target]

    # Handle timedelta conversion and filter only numeric columns
    for col in X.select_dtypes(include=["timedelta64[ns]"]):
        X[col] = X[col].dt.total_seconds()
    X = X.select_dtypes(include=["number"])
    X = X.fillna(X.median(numeric_only=True))

    if X.empty or X.shape[1] < 5:
        logging.warning(f"Too few usable features for training ({X.shape[1]} columns). Check input data.")

    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logging.info("R^2 Score: %.4f", r2_score(y_test, y_pred))
    logging.info("MSE: %.4f", mean_squared_error(y_test, y_pred))
    return model

def plot_feature_importance(model, X, show=False):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(
        kind="barh",
        figsize=(10, 8),
        title="Feature Importance for Predicting Sleep Score"
    )
    plt.tight_layout()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"feature_importance_{timestamp_str}.png"
    plt.savefig(out_path)
    logging.info(f"Saved feature importance plot to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    X, y = load_and_prepare_data()
    if X is None or y is None:
        return
    model = train_and_evaluate(X, y)
    plot_feature_importance(model, X)

if __name__ == "__main__":
    main()

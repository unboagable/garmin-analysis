import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import filter_required_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_path = "data/master_daily_summary.csv"

def load_and_prepare_data(lagged=False):
    if not os.path.exists(data_path):
        logging.error("Missing master dataset — please run load_all_garmin_dbs.py first.")
        return None, None, None

    df = pd.read_csv(data_path, parse_dates=["day"])
    df = df[df["score"].notnull()]

    # Add missing value indicators
    df["missing_yesterday_activity_minutes"] = df["yesterday_activity_minutes"].isna()
    df["missing_stress_avg"] = df["stress_avg"].isna()

    # Impute missing values
    df["yesterday_activity_minutes"] = df["yesterday_activity_minutes"].fillna(0)
    df["stress_avg"] = df["stress_avg"].fillna(df["stress_avg"].median())

    if lagged:
        df["score_tomorrow"] = df["score"].shift(-1)
        df = df.dropna(subset=["score_tomorrow"])
        target = "score_tomorrow"
    else:
        df = filter_required_columns(df, ["yesterday_activity_minutes", "stress_avg"])
        target = "score"

    drop_cols = ["day"]
    features = [
        "yesterday_activity_minutes", "stress_avg",
        "missing_yesterday_activity_minutes", "missing_stress_avg"
    ]

    X = df[features]
    y = df[target]

    return X, y, lagged

def train_and_evaluate(X, y, lagged=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if lagged:
        logging.info("R^2 Score (Lagged): %.4f", r2_score(y_test, y_pred))
        logging.info("MSE (Lagged): %.4f", mean_squared_error(y_test, y_pred))
    else:
        logging.info("R^2 Score: %.4f", r2_score(y_test, y_pred))
        logging.info("MSE: %.4f", mean_squared_error(y_test, y_pred))

    return model, X

def plot_feature_importance(model, X, lagged=False, show=False):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(
        kind="barh",
        figsize=(10, 8),
        title="Feature Importance for Predicting {} Sleep Score".format("Next Day" if lagged else "Sleep")
    )
    plt.tight_layout()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    suffix = "lagged_" if lagged else ""
    out_path = plots_dir / f"feature_importance_{suffix}{timestamp_str}.png"
    plt.savefig(out_path)
    logging.info(f"Saved feature importance plot to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

def main():
    for lagged_mode in [False, True]:
        X, y, lagged = load_and_prepare_data(lagged=lagged_mode)
        if X is None or y is None:
            continue
        model, feature_X = train_and_evaluate(X, y, lagged)
        plot_feature_importance(model, feature_X, lagged)

if __name__ == "__main__":
    main()

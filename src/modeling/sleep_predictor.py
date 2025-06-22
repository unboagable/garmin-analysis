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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_path = "data/modeling_ready_dataset.csv"

def convert_time_columns(df):
    def to_minutes(x):
        try:
            parts = x.split(":")
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 60 + m + s / 60
        except:
            return np.nan
        return np.nan

    time_cols = [col for col in df.columns if df[col].dtype == "object" and df[col].str.contains(":", na=False).any()]
    for col in time_cols:
        df[col] = df[col].apply(to_minutes)
        logging.info(f"Converted time column to minutes: {col}")
    return df

def load_and_prepare_data(lagged=False):
    if not os.path.exists(data_path):
        fallback_path = "data/master_daily_summary.csv"
        if not os.path.exists(fallback_path):
            logging.error("Missing both modeling-ready and master datasets.")
            return None, None, None
        logging.warning("Modeling-ready dataset not found — falling back to master_daily_summary.csv")
        df = pd.read_csv(fallback_path, parse_dates=["day"])
        df = df[df["score"].notnull()]
        df = convert_time_columns(df)

        df["missing_yesterday_activity_minutes"] = df["yesterday_activity_minutes"].isna()
        df["missing_stress_avg"] = df["stress_avg"].isna()
        df["yesterday_activity_minutes"] = df["yesterday_activity_minutes"].fillna(0)
        df["stress_avg"] = df["stress_avg"].fillna(df["stress_avg"].median())
    else:
        df = pd.read_csv(data_path, parse_dates=["day"])
        df = convert_time_columns(df)

    if lagged:
        df["score_tomorrow"] = df["score"].shift(-1)
        df = df.dropna(subset=["score_tomorrow"])
        target = "score_tomorrow"
    else:
        target = "score"

    drop_cols = ["day"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in [target]]

    X = df[features]
    y = df[target]

    return X, y, lagged

def train_and_evaluate(X, y, lagged=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    if lagged:
        logging.info("R^2 Score (Lagged): %.4f", r2)
        logging.info("MSE (Lagged): %.4f", mse)
    else:
        logging.info("R^2 Score: %.4f", r2)
        logging.info("MSE: %.4f", mse)

    return model, X, r2, mse

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

    return str(out_path)

def run_sleep_model(df=None):
    if df is None:
        X, y, lagged = load_and_prepare_data(lagged=False)
    else:
        df = convert_time_columns(df)
        df = df[df["score"].notnull()]
        target = "score"
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != target]
        X = df[features]
        y = df[target]
        lagged = False

    if X is None or y is None:
        return {"r2": None, "mse": None, "plot_path": None}

    model, feature_X, r2, mse = train_and_evaluate(X, y, lagged)
    plot_path = plot_feature_importance(model, feature_X, lagged)
    return {"r2": r2, "mse": mse, "plot_path": plot_path}

def main():
    for lagged_mode in [False, True]:
        X, y, lagged = load_and_prepare_data(lagged=lagged_mode)
        if X is None or y is None:
            continue
        model, feature_X, r2, mse = train_and_evaluate(X, y, lagged)
        plot_feature_importance(model, feature_X, lagged)

if __name__ == "__main__":
    main()

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils import load_garmin_tables, filter_by_date, normalize_dates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(days_back=60):
    if not os.path.exists("garmin.db"):
        logging.error("Missing garmin.db — please run garmindb_cli.py or place the database in the root directory.")
        return None, None

    try:
        tables = load_garmin_tables()
    except Exception as e:
        logging.exception("Failed to load tables: %s", e)
        return None, None

    daily = filter_by_date(tables["daily"], days_back=days_back)
    sleep = filter_by_date(tables["sleep"], days_back=days_back)
    stress = filter_by_date(tables["stress"], date_col="timestamp", days_back=days_back)
    rest_hr = filter_by_date(tables["rest_hr"], days_back=days_back)

    if any(df.empty for df in [daily, sleep, stress, rest_hr]):
        logging.warning("One or more tables returned empty after filtering. Check data coverage.")

    stress["day"] = pd.to_datetime(stress["timestamp"]).dt.normalize()
    stress_daily = stress.groupby("day")["stress"].mean().reset_index()

    df = sleep[["day", "total_sleep"]].merge(daily, on="day", how="left")
    df = df.merge(rest_hr, on="day", how="left").merge(stress_daily, on="day", how="left")

    df = df[df["total_sleep"].notnull()]
    df = df.dropna(thresh=int(0.7 * df.shape[1]))

    if pd.api.types.is_timedelta64_dtype(df["total_sleep"]):
        df["total_sleep"] = df["total_sleep"].dt.total_seconds() / 3600
    else:
        df["total_sleep"] = pd.to_timedelta(df["total_sleep"]).dt.total_seconds() / 3600

    drop_cols = ["day", "calendar_date", "user_profile_pk"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["total_sleep"])
    y = df["total_sleep"]

    for col in X.select_dtypes(include=["timedelta64[ns]"]):
        X[col] = X[col].dt.total_seconds()

    X = X.select_dtypes(include=["number"])
    X = X.fillna(X.median(numeric_only=True))

    return X, y

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logging.info("R^2 Score: %.4f", r2_score(y_test, y_pred))
    logging.info("MSE: %.4f", mean_squared_error(y_test, y_pred))
    return model

def plot_feature_importance(model, X):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values(ascending=True).plot(kind="barh", figsize=(10, 8), title="Feature Importance for Predicting Sleep")
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_and_prepare_data(days_back=60)
    if X is None or y is None:
        return
    model = train_and_evaluate(X, y)
    plot_feature_importance(model, X)

if __name__ == "__main__":
    main()




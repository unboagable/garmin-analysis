import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Connect to the Garmin DB
conn = sqlite3.connect("garmin.db")

# Load relevant tables
daily = pd.read_sql_query("SELECT * FROM daily_summary", conn, parse_dates=["day"])
sleep = pd.read_sql_query("SELECT * FROM sleep", conn, parse_dates=["day"])
stress = pd.read_sql_query("SELECT * FROM stress", conn, parse_dates=["timestamp"])
rest_hr = pd.read_sql_query("SELECT * FROM resting_hr", conn, parse_dates=["day"])

conn.close()

# Preprocess
stress["day"] = pd.to_datetime(stress["timestamp"]).dt.normalize()
stress_daily = stress.groupby("day")["stress"].mean().reset_index()

# Merge all on day
df = sleep[["day", "total_sleep"]].merge(
    daily, on="day", how="left"
).merge(
    rest_hr, on="day", how="left"
).merge(
    stress_daily, on="day", how="left"
)

# Drop rows with missing target or too many nulls
df = df[df["total_sleep"].notnull()]
df = df.dropna(thresh=int(0.7 * df.shape[1]))  # Keep rows with at least 70% data

# Convert total_sleep to hours if in timedelta or string format
if pd.api.types.is_timedelta64_dtype(df["total_sleep"]):
    df["total_sleep"] = df["total_sleep"].dt.total_seconds() / 3600
else:
    df["total_sleep"] = pd.to_timedelta(df["total_sleep"]).dt.total_seconds() / 3600

# Drop non-numeric or unhelpful columns
drop_cols = ["day", "calendar_date", "user_profile_pk"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["total_sleep"])
y = df["total_sleep"]

# Convert any remaining timedelta columns to seconds
for col in X.select_dtypes(include=["timedelta64[ns]"]):
    X[col] = X[col].dt.total_seconds()

# Drop non-numeric columns
X = X.select_dtypes(include=["number"])

# Fill remaining NaNs with median
X = X.fillna(X.median(numeric_only=True))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind="barh", figsize=(10, 8), title="Feature Importance for Predicting Sleep")
plt.tight_layout()
plt.show()


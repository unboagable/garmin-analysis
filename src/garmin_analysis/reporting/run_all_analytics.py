import pandas as pd
import os
from datetime import datetime
import logging
from garmin_analysis.reporting.generate_trend_summary import generate_trend_summary
from garmin_analysis.modeling.anomaly_detection import run_anomaly_detection

# Logging is configured at package level

def run_all_analytics(df: pd.DataFrame, date_col='day', output_dir='reports', as_html=False, monthly=False):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = 'html' if as_html else 'md'
    report_type = 'monthly' if monthly else 'full'
    report_path = os.path.join(output_dir, f"{report_type}_report_{timestamp}.{ext}")

    if monthly:
        # Filter to only the most recent full calendar month
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        last_month = (df[date_col].max().replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        this_month = last_month + pd.DateOffset(months=1)
        df = df[(df[date_col] >= last_month) & (df[date_col] < this_month)]
        logging.info(f"Filtered data to monthly range: {last_month.date()} - {this_month.date()}")

    # Step 1: Trend Summary
    trend_path = os.path.join(output_dir, f"trend_summary_{timestamp}.{ext}")
    generate_trend_summary(df, date_col=date_col, output_dir=output_dir)

    # Step 2: Anomaly Detection
    logging.info("Running anomaly detection...")
    anomalies_df, anomaly_plot_path = run_anomaly_detection(df)

    # Build Report
    with open(report_path, 'w') as f:
        f.write(f"# {'Monthly' if monthly else 'Full'} Health Report\n\nGenerated: {timestamp}\n\n")

        f.write("## \U0001F4CA Trend Summary\n")
        f.write(f"See: `{os.path.basename(trend_path)}`\n\n")

        f.write("## \U0001F6A8 Anomaly Detection\n")
        if anomalies_df.empty:
            f.write("No anomalies detected.\n")
        else:
            f.write(f"Detected {len(anomalies_df)} anomalies.\n")
            f.write(f"- Plot: `{anomaly_plot_path}`\n\n")

    logging.info(f"✅ {report_type.capitalize()} report saved to {report_path}")
    return report_path

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("data/master_daily_summary.csv")
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day")
    run_all_analytics(df)  # full report by default

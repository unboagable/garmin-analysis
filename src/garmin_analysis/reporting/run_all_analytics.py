import pandas as pd
import os
from datetime import datetime
import logging
import argparse
from garmin_analysis.reporting.generate_trend_summary import generate_trend_summary
from garmin_analysis.modeling.anomaly_detection import run_anomaly_detection
from garmin_analysis.features.coverage import filter_by_24h_coverage

# Logging is configured at package level

def run_all_analytics(df: pd.DataFrame, date_col='day', output_dir='reports', as_html=False, monthly=False, filter_24h_coverage=False, max_gap_minutes=2, day_edge_tolerance_minutes=2, coverage_allowance_minutes=0):
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

    # Apply 24-hour coverage filtering if requested
    if filter_24h_coverage:
        logging.info("Filtering to days with 24-hour continuous coverage...")
        max_gap = pd.Timedelta(minutes=max_gap_minutes)
        day_edge_tolerance = pd.Timedelta(minutes=day_edge_tolerance_minutes)
        total_missing_allowance = pd.Timedelta(minutes=max(0, min(coverage_allowance_minutes, 300)))
        df = filter_by_24h_coverage(df, max_gap=max_gap, day_edge_tolerance=day_edge_tolerance, total_missing_allowance=total_missing_allowance)
        logging.info(f"After 24h coverage filtering: {len(df)} days remaining")

    # Step 1: Trend Summary
    trend_path = generate_trend_summary(df, date_col=date_col, output_dir=output_dir, 
                                      filter_24h_coverage=filter_24h_coverage,
                                      max_gap_minutes=max_gap_minutes,
                                      day_edge_tolerance_minutes=day_edge_tolerance_minutes,
                                      coverage_allowance_minutes=coverage_allowance_minutes,
                                      timestamp=timestamp)

    # Step 2: Anomaly Detection
    logging.info("Running anomaly detection...")
    anomalies_df, anomaly_plot_path = run_anomaly_detection(df)

    # Build Report
    with open(report_path, 'w') as f:
        f.write(f"# {'Monthly' if monthly else 'Full'} Health Report\n\nGenerated: {timestamp}\n\n")
        
        if filter_24h_coverage:
            f.write("**Note:** This analysis is filtered to days with 24-hour continuous coverage.\n\n")

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
    parser = argparse.ArgumentParser(description='Run comprehensive analytics on Garmin data')
    parser.add_argument('--monthly', action='store_true', 
                       help='Generate monthly report instead of full report')
    parser.add_argument('--as-html', action='store_true', 
                       help='Generate HTML report instead of Markdown')
    parser.add_argument('--filter-24h-coverage', action='store_true', 
                       help='Filter to only days with 24-hour continuous coverage')
    parser.add_argument('--max-gap', type=int, default=2,
                       help='Maximum gap in minutes for continuous coverage (default: 2)')
    parser.add_argument('--day-edge-tolerance', type=int, default=2,
                       help='Day edge tolerance in minutes for continuous coverage (default: 2)')
    parser.add_argument('--coverage-allowance-minutes', type=int, default=0,
                       help='Total allowed missing minutes within a day (0-300, default: 0)')
    
    args = parser.parse_args()
    
    # Load data
    from garmin_analysis.utils.data_loading import load_master_dataframe
    df = load_master_dataframe()
    
    # Run analytics
    report_path = run_all_analytics(
        df, 
        monthly=args.monthly,
        as_html=args.as_html,
        filter_24h_coverage=args.filter_24h_coverage,
        max_gap_minutes=args.max_gap,
        day_edge_tolerance_minutes=args.day_edge_tolerance,
        coverage_allowance_minutes=args.coverage_allowance_minutes
    )
    
    print(f"Report generated: {report_path}")

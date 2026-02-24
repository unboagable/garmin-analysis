#!/usr/bin/env python3
"""
Quick data quality check script for Garmin health data.

This script provides fast, command-line access to key data quality metrics.

Usage:
    python src/features/quick_data_check.py                    # Full analysis
    python src/features/quick_data_check.py --completeness    # Just completeness
    python src/features/quick_data_check.py --features        # Just feature suitability
    python src/features/quick_data_check.py --summary         # Just summary
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path

logger = logging.getLogger(__name__)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from garmin_analysis.features.data_quality_analysis import GarminDataQualityAnalyzer


def quick_completeness_check(df):
    """Quick check of data completeness."""
    logger.info("📊 QUICK COMPLETENESS CHECK")
    logger.info("=" * 50)
    
    total_cols = len(df.columns)
    total_count = len(df)
    sufficient_cols = 0
    adequate_cols = 0
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        completeness_pct = non_null_count / total_count if total_count > 0 else 0.0
        
        if non_null_count >= 50:
            sufficient_cols += 1
        if completeness_pct >= 0.1:
            adequate_cols += 1
    
    logger.info(f"Total columns: {total_cols}")
    suf_pct = f"{sufficient_cols/total_cols:.1%}" if total_cols > 0 else "N/A"
    adeq_pct = f"{adequate_cols/total_cols:.1%}" if total_cols > 0 else "N/A"
    logger.info(f"Columns with ≥50 non-null values: {sufficient_cols} ({suf_pct})")
    logger.info(f"Columns with ≥10% completeness: {adequate_cols} ({adeq_pct})")
    
    # Show worst columns
    logger.info("\n🔴 WORST 10 COLUMNS (by completeness):")
    completeness_data = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        completeness_pct = non_null_count / total_count if total_count > 0 else 0.0
        completeness_data.append((col, completeness_pct, non_null_count))
    
    # Sort by completeness (ascending)
    completeness_data.sort(key=lambda x: x[1])
    
    for i, (col, pct, count) in enumerate(completeness_data[:10]):
        status = "🔴" if pct < 0.1 else "🟡" if pct < 0.5 else "🟢"
        logger.info(f"  {i+1:2d}. {status} {col:<30} {pct:6.1%} ({count:4d}/{len(df)})")


def quick_feature_check(df):
    """Quick check of feature suitability for modeling."""
    logger.info("🔍 QUICK FEATURE SUITABILITY CHECK")
    logger.info("=" * 50)
    
    suitable_features = []
    unsuitable_features = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        is_numeric = df[col].dtype.kind in 'ifc'  # integer, float, complex
        
        if non_null_count >= 50 and is_numeric:
            suitable_features.append(col)
        else:
            unsuitable_features.append(col)
    
    logger.info(f"Suitable for modeling: {len(suitable_features)} features")
    logger.info(f"Unsuitable for modeling: {len(unsuitable_features)} features")
    
    if suitable_features:
        logger.info(f"\n✅ TOP 10 SUITABLE FEATURES:")
        # Sort by completeness
        feature_completeness = []
        total_count = len(df)
        for col in suitable_features:
            non_null_count = df[col].notna().sum()
            completeness_pct = non_null_count / total_count if total_count > 0 else 0.0
            feature_completeness.append((col, completeness_pct))
        
        feature_completeness.sort(key=lambda x: x[1], reverse=True)
        
        for i, (col, pct) in enumerate(feature_completeness[:10]):
            logger.info(f"  {i+1:2d}. {col:<30} {pct:6.1%}")
    
    if unsuitable_features:
        logger.info(f"\n❌ SAMPLE UNSUITABLE FEATURES:")
        total_count = len(df)
        for i, col in enumerate(unsuitable_features[:10]):
            non_null_count = df[col].notna().sum()
            completeness_pct = non_null_count / total_count if total_count > 0 else 0.0
            dtype = df[col].dtype
            logger.info(f"  {i+1:2d}. {col:<30} {completeness_pct:6.1%} (type: {dtype})")


def quick_summary(df):
    """Quick summary of the dataset."""
    logger.info("📈 QUICK DATASET SUMMARY")
    logger.info("=" * 50)
    
    logger.info(f"Dataset size: {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Date range
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
    if date_columns:
        for col in date_columns:
            try:
                if df[col].dtype.kind in 'M':  # datetime
                    min_date = df[col].min()
                    max_date = df[col].max()
                    unique_days = df[col].nunique()
                    logger.info(f"Date range: {min_date} to {max_date} ({unique_days} unique days)")
                    break
            except Exception as e:
                logger.debug(f"Error processing datetime column {col}: {e}")
                continue
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    logger.info(f"\nData types:")
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} columns")
    
    # Missing data overview
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_percentage = missing_cells / total_cells if total_cells > 0 else 0.0

    logger.info(f"\nMissing data: {missing_cells:,} cells ({missing_percentage:.1%})")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Quick Garmin data quality check')
    parser.add_argument('--completeness', action='store_true', help='Show only completeness analysis')
    parser.add_argument('--features', action='store_true', help='Show only feature suitability analysis')
    parser.add_argument('--summary', action='store_true', help='Show only dataset summary')
    parser.add_argument('--full', action='store_true', help='Run full analysis (default)')
    parser.add_argument('--continuous-24h', action='store_true', help='List days with 24h continuous timeseries coverage (based on stress table)')
    
    args = parser.parse_args()
    
    try:
        # Import and load data
        from garmin_analysis.utils.data_loading import load_master_dataframe
        from garmin_analysis.logging_config import setup_logging
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
        logger.info("📥 Loading Garmin data...")
        df = load_master_dataframe()
        logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns\n")
        
        # Run requested analysis
        if args.continuous_24h:
            # Load raw stress table to assess coverage
            from garmin_analysis.config import DB_PATHS
            from garmin_analysis.data_ingestion.load_all_garmin_dbs import load_table
            from garmin_analysis.features.coverage import days_with_continuous_coverage

            stress = load_table(DB_PATHS["garmin"], "stress", parse_dates=["timestamp"])
            if stress is None or stress.empty:
                logger.error("❌ No stress timeseries available to compute coverage.")
                sys.exit(2)

            days = days_with_continuous_coverage(stress, timestamp_col="timestamp")
            logger.info("📅 Days with 24h continuous coverage (no gap >2min):")
            for d in days:
                logger.info(f"  - {d.date()}")
            logger.info(f"Total: {len(days)} days")
        elif args.completeness:
            quick_completeness_check(df)
        elif args.features:
            quick_feature_check(df)
        elif args.summary:
            quick_summary(df)
        else:
            # Default: run all quick checks
            quick_summary(df)
            logger.info("")
            quick_completeness_check(df)
            logger.info("")
            quick_feature_check(df)
            
            logger.info("\n" + "=" * 50)
            logger.info("💡 For detailed analysis, run: python src/features/data_quality_analysis.py")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

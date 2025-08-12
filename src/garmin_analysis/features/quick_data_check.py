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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from garmin_analysis.features.data_quality_analysis import GarminDataQualityAnalyzer


def quick_completeness_check(df):
    """Quick check of data completeness."""
    print("📊 QUICK COMPLETENESS CHECK")
    print("=" * 50)
    
    total_cols = len(df.columns)
    sufficient_cols = 0
    adequate_cols = 0
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        total_count = len(df)
        completeness_pct = non_null_count / total_count
        
        if non_null_count >= 50:
            sufficient_cols += 1
        if completeness_pct >= 0.1:
            adequate_cols += 1
    
    print(f"Total columns: {total_cols}")
    print(f"Columns with ≥50 non-null values: {sufficient_cols} ({sufficient_cols/total_cols:.1%})")
    print(f"Columns with ≥10% completeness: {adequate_cols} ({adequate_cols/total_cols:.1%})")
    
    # Show worst columns
    print("\n🔴 WORST 10 COLUMNS (by completeness):")
    completeness_data = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        total_count = len(df)
        completeness_pct = non_null_count / total_count
        completeness_data.append((col, completeness_pct, non_null_count))
    
    # Sort by completeness (ascending)
    completeness_data.sort(key=lambda x: x[1])
    
    for i, (col, pct, count) in enumerate(completeness_data[:10]):
        status = "🔴" if pct < 0.1 else "🟡" if pct < 0.5 else "🟢"
        print(f"  {i+1:2d}. {status} {col:<30} {pct:6.1%} ({count:4d}/{len(df)})")


def quick_feature_check(df):
    """Quick check of feature suitability for modeling."""
    print("🔍 QUICK FEATURE SUITABILITY CHECK")
    print("=" * 50)
    
    suitable_features = []
    unsuitable_features = []
    
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        is_numeric = df[col].dtype.kind in 'ifc'  # integer, float, complex
        
        if non_null_count >= 50 and is_numeric:
            suitable_features.append(col)
        else:
            unsuitable_features.append(col)
    
    print(f"Suitable for modeling: {len(suitable_features)} features")
    print(f"Unsuitable for modeling: {len(unsuitable_features)} features")
    
    if suitable_features:
        print(f"\n✅ TOP 10 SUITABLE FEATURES:")
        # Sort by completeness
        feature_completeness = []
        for col in suitable_features:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness_pct = non_null_count / total_count
            feature_completeness.append((col, completeness_pct))
        
        feature_completeness.sort(key=lambda x: x[1], reverse=True)
        
        for i, (col, pct) in enumerate(feature_completeness[:10]):
            print(f"  {i+1:2d}. {col:<30} {pct:6.1%}")
    
    if unsuitable_features:
        print(f"\n❌ SAMPLE UNSUITABLE FEATURES:")
        for i, col in enumerate(unsuitable_features[:10]):
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness_pct = non_null_count / total_count
            dtype = df[col].dtype
            print(f"  {i+1:2d}. {col:<30} {pct:6.1%} (type: {dtype})")


def quick_summary(df):
    """Quick summary of the dataset."""
    print("📈 QUICK DATASET SUMMARY")
    print("=" * 50)
    
    print(f"Dataset size: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Date range
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
    if date_columns:
        for col in date_columns:
            try:
                if df[col].dtype.kind in 'M':  # datetime
                    min_date = df[col].min()
                    max_date = df[col].max()
                    unique_days = df[col].nunique()
                    print(f"Date range: {min_date} to {max_date} ({unique_days} unique days)")
                    break
            except:
                continue
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    print(f"\nData types:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Missing data overview
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_percentage = missing_cells / total_cells
    
    print(f"\nMissing data: {missing_cells:,} cells ({missing_percentage:.1%})")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Quick Garmin data quality check')
    parser.add_argument('--completeness', action='store_true', help='Show only completeness analysis')
    parser.add_argument('--features', action='store_true', help='Show only feature suitability analysis')
    parser.add_argument('--summary', action='store_true', help='Show only dataset summary')
    parser.add_argument('--full', action='store_true', help='Run full analysis (default)')
    
    args = parser.parse_args()
    
    try:
        # Import and load data
        from garmin_analysis.utils import load_master_dataframe
        
        print("📥 Loading Garmin data...")
        df = load_master_dataframe()
        print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns\n")
        
        # Run requested analysis
        if args.completeness:
            quick_completeness_check(df)
        elif args.features:
            quick_feature_check(df)
        elif args.summary:
            quick_summary(df)
        else:
            # Default: run all quick checks
            quick_summary(df)
            print()
            quick_completeness_check(df)
            print()
            quick_feature_check(df)
            
            print("\n" + "=" * 50)
            print("💡 For detailed analysis, run: python src/features/data_quality_analysis.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

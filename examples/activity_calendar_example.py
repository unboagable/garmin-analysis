#!/usr/bin/env python3
"""
Example script demonstrating how to use the activity calendar plotting feature.

This script shows different ways to create activity calendar visualizations
from your Garmin data.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from garmin_analysis.viz.plot_activity_calendar import load_activities_data, plot_activity_calendar
import logging

def main():
    """Demonstrate activity calendar plotting."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🏃‍♂️ Garmin Activity Calendar Example")
    print("=" * 50)
    
    # Load activities data
    print("\n1. Loading activities data...")
    try:
        activities_df = load_activities_data("db/garmin_activities.db")
        print(f"✅ Loaded {len(activities_df)} activities")
        
        if activities_df.empty:
            print("❌ No activities found in the database")
            return
            
        # Show activity summary
        sport_counts = activities_df['sport'].value_counts()
        print(f"\n📊 Activity types found:")
        for sport, count in sport_counts.items():
            print(f"   - {sport.replace('_', ' ').title()}: {count} activities")
            
    except FileNotFoundError:
        print("❌ Database not found. Please ensure 'db/garmin_activities.db' exists.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Example 1: Full calendar
    print(f"\n2. Creating full activity calendar...")
    try:
        plot_activity_calendar(activities_df, output_dir="plots")
        print("✅ Full calendar created successfully!")
    except Exception as e:
        print(f"❌ Error creating full calendar: {e}")
    
    # Example 2: Last 6 months
    print(f"\n3. Creating calendar for last 6 months...")
    try:
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months
        
        plot_activity_calendar(
            activities_df,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            output_dir="plots",
            figsize=(14, 8)
        )
        print("✅ 6-month calendar created successfully!")
    except Exception as e:
        print(f"❌ Error creating 6-month calendar: {e}")
    
    # Example 3: Specific year (2024)
    print(f"\n4. Creating calendar for 2024...")
    try:
        plot_activity_calendar(
            activities_df,
            start_date='2024-01-01',
            end_date='2024-12-31',
            output_dir="plots",
            figsize=(16, 12)
        )
        print("✅ 2024 calendar created successfully!")
    except Exception as e:
        print(f"❌ Error creating 2024 calendar: {e}")
    
    print(f"\n🎉 All examples completed!")
    print(f"📁 Check the 'plots' directory for generated calendar images.")
    print(f"\n💡 Tips:")
    print(f"   - Different colors represent different activity types")
    print(f"   - Darker colors indicate multiple activities on the same day")
    print(f"   - Use the CLI tool for more customization: python src/garmin_analysis/viz/cli_activity_calendar.py --help")

if __name__ == "__main__":
    main()

"""
Time-of-Day Stress Analysis for Garmin Health Data

This module analyzes stress patterns by time of day, showing:
- Average stress levels by hour of day
- Stress patterns by day of week
- Peak stress times and low-stress periods
- Statistical summaries and visualizations

Usage:
    python -m garmin_analysis.cli_time_of_day_stress
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
from datetime import datetime
from garmin_analysis.config import PLOTS_DIR, DB_DIR
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def load_stress_data(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load stress data from the Garmin database.
    
    Args:
        db_path: Path to garmin.db. If None, uses default from config.
        
    Returns:
        pd.DataFrame: Stress data with timestamp and stress columns
    """
    if db_path is None:
        db_path = DB_DIR / "garmin.db"
    
    if not Path(db_path).exists():
        logger.error(f"Database not found at {db_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading stress data from {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Load stress data
        query = "SELECT timestamp, stress FROM stress WHERE stress IS NOT NULL"
        df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
        
        conn.close()
        
        logger.info(f"Loaded {len(df):,} stress measurements")
        return df
        
    except Exception as e:
        logger.exception(f"Error loading stress data: {e}")
        return pd.DataFrame()


def calculate_hourly_stress_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average stress by hour of day.
    
    Args:
        df: DataFrame with timestamp and stress columns
        
    Returns:
        pd.DataFrame: Hourly stress statistics
    """
    if df.empty:
        logger.warning("DataFrame is empty")
        return pd.DataFrame()
    
    # Extract hour of day
    df['hour'] = df['timestamp'].dt.hour
    
    # Calculate statistics by hour
    hourly_stats = df.groupby('hour')['stress'].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Add confidence interval (95%)
    hourly_stats['ci_lower'] = hourly_stats['mean'] - 1.96 * hourly_stats['std'] / (hourly_stats['count'] ** 0.5)
    hourly_stats['ci_upper'] = hourly_stats['mean'] + 1.96 * hourly_stats['std'] / (hourly_stats['count'] ** 0.5)
    
    logger.info(f"Calculated hourly averages for {len(hourly_stats)} hours")
    return hourly_stats


def calculate_hourly_stress_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average stress by hour of day and day of week.
    
    Args:
        df: DataFrame with timestamp and stress columns
        
    Returns:
        pd.DataFrame: Hourly stress statistics by day of week
    """
    if df.empty:
        logger.warning("DataFrame is empty")
        return pd.DataFrame()
    
    # Extract hour and day of week
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['day_of_week_num'] = (df['timestamp'].dt.dayofweek + 1) % 7  # Sunday = 0
    
    # Calculate statistics by hour and day of week
    hourly_weekday_stats = df.groupby(['hour', 'day_of_week', 'day_of_week_num'])['stress'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Set proper ordering for days (Sunday first)
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    hourly_weekday_stats['day_of_week'] = pd.Categorical(
        hourly_weekday_stats['day_of_week'],
        categories=day_order,
        ordered=True
    )
    
    hourly_weekday_stats = hourly_weekday_stats.sort_values(['hour', 'day_of_week_num'])
    
    logger.info(f"Calculated hourly averages by weekday: {len(hourly_weekday_stats)} records")
    return hourly_weekday_stats


def plot_hourly_stress_pattern(hourly_stats: pd.DataFrame, save_plots: bool = True, 
                               show_plots: bool = False) -> Dict[str, str]:
    """
    Create visualizations for hourly stress patterns.
    
    Args:
        hourly_stats: DataFrame with hourly statistics
        save_plots: Whether to save plots to file
        show_plots: Whether to display plots
        
    Returns:
        dict: Dictionary with plot filenames
    """
    if hourly_stats.empty:
        logger.warning("No data to plot")
        return {}
    
    plot_files = {}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: Line plot with confidence interval
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = hourly_stats['hour']
    means = hourly_stats['mean']
    ci_lower = hourly_stats['ci_lower']
    ci_upper = hourly_stats['ci_upper']
    
    # Plot mean line
    ax.plot(hours, means, 'b-', linewidth=2, label='Mean Stress', marker='o')
    
    # Plot confidence interval
    ax.fill_between(hours, ci_lower, ci_upper, alpha=0.3, label='95% CI')
    
    # Highlight typical sleep hours (23:00 - 6:00)
    sleep_hours = [23, 0, 1, 2, 3, 4, 5, 6]
    for hour in sleep_hours:
        if hour in hours.values:
            ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.1, color='purple')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Stress Level', fontsize=12)
    ax.set_title('Average Stress Levels by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{timestamp_str}_stress_by_hour.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plot_files['hourly_stress'] = str(filepath)
        logger.info(f"Saved hourly stress plot to {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Plot 2: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(hours, means, yerr=hourly_stats['std'], capsize=3, alpha=0.7,
                  color=sns.color_palette("RdYlGn_r", len(hours)))
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Stress Level', fontsize=12)
    ax.set_title('Stress Distribution by Hour (Mean ± Std Dev)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{timestamp_str}_stress_by_hour_bars.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plot_files['hourly_stress_bars'] = str(filepath)
        logger.info(f"Saved hourly stress bar chart to {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def plot_stress_heatmap_by_weekday(hourly_weekday_stats: pd.DataFrame, 
                                   save_plots: bool = True, 
                                   show_plots: bool = False) -> Dict[str, str]:
    """
    Create a heatmap showing stress patterns by hour and day of week.
    
    Args:
        hourly_weekday_stats: DataFrame with hourly statistics by weekday
        save_plots: Whether to save plots to file
        show_plots: Whether to display plots
        
    Returns:
        dict: Dictionary with plot filenames
    """
    if hourly_weekday_stats.empty:
        logger.warning("No data to plot")
        return {}
    
    plot_files = {}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Pivot data for heatmap
    heatmap_data = hourly_weekday_stats.pivot(
        index='day_of_week', 
        columns='hour', 
        values='mean'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', annot=True, fmt='.1f', 
                linewidths=0.5, cbar_kws={'label': 'Average Stress Level'},
                ax=ax)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    ax.set_title('Stress Levels Heatmap: Hour of Day × Day of Week', 
                fontsize=14, fontweight='bold')
    
    # Format x-axis labels
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{timestamp_str}_stress_heatmap_weekday_hour.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plot_files['stress_heatmap'] = str(filepath)
        logger.info(f"Saved stress heatmap to {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Create line plot by day of week
    fig, ax = plt.subplots(figsize=(14, 8))
    
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    colors = sns.color_palette("husl", 7)
    
    for i, day in enumerate(day_order):
        day_data = hourly_weekday_stats[hourly_weekday_stats['day_of_week'] == day]
        if not day_data.empty:
            ax.plot(day_data['hour'], day_data['mean'], 
                   marker='o', linewidth=2, label=day, color=colors[i])
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Average Stress Level', fontsize=12)
    ax.set_title('Stress Patterns by Day of Week', fontsize=14, fontweight='bold')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"{timestamp_str}_stress_by_weekday_hour.png"
        filepath = PLOTS_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plot_files['stress_weekday_lines'] = str(filepath)
        logger.info(f"Saved stress by weekday plot to {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return plot_files


def print_stress_summary(hourly_stats: pd.DataFrame, 
                        hourly_weekday_stats: Optional[pd.DataFrame] = None) -> None:
    """
    Print a detailed summary of stress patterns.
    
    Args:
        hourly_stats: DataFrame with hourly statistics
        hourly_weekday_stats: Optional DataFrame with weekday-specific statistics
    """
    if hourly_stats.empty:
        logger.warning("No data to summarize")
        return
    
    logger.info("\n" + "="*70)
    logger.info("STRESS ANALYSIS BY TIME OF DAY")
    logger.info("="*70)
    
    # Overall statistics
    logger.info("\n📊 Overall Stress Statistics:")
    logger.info("-" * 70)
    logger.info(f"  Total measurements: {hourly_stats['count'].sum():,}")
    logger.info(f"  Overall mean stress: {hourly_stats['mean'].mean():.1f}")
    logger.info(f"  Overall std dev: {hourly_stats['std'].mean():.1f}")
    
    # Peak stress times
    logger.info("\n⬆️  Peak Stress Times:")
    logger.info("-" * 70)
    top_5_hours = hourly_stats.nlargest(5, 'mean')
    for _, row in top_5_hours.iterrows():
        hour = int(row['hour'])
        logger.info(f"  {hour:02d}:00 - {(hour+1):02d}:00: {row['mean']:5.1f} ± {row['std']:4.1f} "
                   f"(n={row['count']:,})")
    
    # Low stress times
    logger.info("\n⬇️  Low Stress Times:")
    logger.info("-" * 70)
    bottom_5_hours = hourly_stats.nsmallest(5, 'mean')
    for _, row in bottom_5_hours.iterrows():
        hour = int(row['hour'])
        logger.info(f"  {hour:02d}:00 - {(hour+1):02d}:00: {row['mean']:5.1f} ± {row['std']:4.1f} "
                   f"(n={row['count']:,})")
    
    # Hourly breakdown
    logger.info("\n📈 Hourly Stress Breakdown:")
    logger.info("-" * 70)
    logger.info(f"{'Hour':>6} | {'Mean':>6} | {'Median':>6} | {'Std':>6} | {'Min':>5} | "
               f"{'Max':>5} | {'Count':>10}")
    logger.info("-" * 70)
    
    for _, row in hourly_stats.iterrows():
        hour = int(row['hour'])
        logger.info(f"{hour:02d}:00 | {row['mean']:6.1f} | {row['median']:6.1f} | "
                   f"{row['std']:6.1f} | {row['min']:5.0f} | {row['max']:5.0f} | "
                   f"{row['count']:10,}")
    
    # Time period analysis
    logger.info("\n🕐 Time Period Analysis:")
    logger.info("-" * 70)
    
    morning_hours = hourly_stats[hourly_stats['hour'].between(6, 11)]
    afternoon_hours = hourly_stats[hourly_stats['hour'].between(12, 17)]
    evening_hours = hourly_stats[hourly_stats['hour'].between(18, 22)]
    night_hours = hourly_stats[hourly_stats['hour'].isin([23, 0, 1, 2, 3, 4, 5])]
    
    logger.info(f"  Morning (06:00-11:59):   {morning_hours['mean'].mean():5.1f} ± "
               f"{morning_hours['std'].mean():4.1f}")
    logger.info(f"  Afternoon (12:00-17:59): {afternoon_hours['mean'].mean():5.1f} ± "
               f"{afternoon_hours['std'].mean():4.1f}")
    logger.info(f"  Evening (18:00-22:59):   {evening_hours['mean'].mean():5.1f} ± "
               f"{evening_hours['std'].mean():4.1f}")
    logger.info(f"  Night (23:00-05:59):     {night_hours['mean'].mean():5.1f} ± "
               f"{night_hours['std'].mean():4.1f}")
    
    # Day of week breakdown if available
    if hourly_weekday_stats is not None and not hourly_weekday_stats.empty:
        logger.info("\n📅 Average Stress by Day of Week:")
        logger.info("-" * 70)
        
        day_averages = hourly_weekday_stats.groupby('day_of_week')['mean'].mean()
        day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        for day in day_order:
            if day in day_averages.index:
                logger.info(f"  {day:>9}: {day_averages[day]:5.1f}")


def main(db_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to run time-of-day stress analysis.
    
    Args:
        db_path: Optional path to garmin.db
        
    Returns:
        Tuple of (hourly_stats, hourly_weekday_stats)
    """
    try:
        # Load stress data
        logger.info("Loading stress data from database...")
        stress_df = load_stress_data(db_path)
        
        if stress_df.empty:
            logger.error("No stress data loaded")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate hourly averages
        logger.info("Calculating hourly stress averages...")
        hourly_stats = calculate_hourly_stress_averages(stress_df)
        
        # Calculate hourly averages by weekday
        logger.info("Calculating hourly stress averages by weekday...")
        hourly_weekday_stats = calculate_hourly_stress_by_weekday(stress_df)
        
        # Print summary
        print_stress_summary(hourly_stats, hourly_weekday_stats)
        
        # Create visualizations
        logger.info("\nGenerating stress pattern visualizations...")
        plot_files = plot_hourly_stress_pattern(hourly_stats, save_plots=True, show_plots=False)
        
        logger.info("Generating stress heatmap by weekday and hour...")
        weekday_plot_files = plot_stress_heatmap_by_weekday(
            hourly_weekday_stats, 
            save_plots=True, 
            show_plots=False
        )
        
        all_plot_files = {**plot_files, **weekday_plot_files}
        
        if all_plot_files:
            logger.info(f"\n✅ Generated {len(all_plot_files)} plots:")
            for plot_name, filepath in all_plot_files.items():
                logger.info(f"  {plot_name}: {filepath}")
        
        return hourly_stats, hourly_weekday_stats
        
    except Exception as e:
        logger.exception(f"Error in time-of-day stress analysis: {e}")
        raise


if __name__ == "__main__":
    main()


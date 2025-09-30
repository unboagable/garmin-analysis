import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from ..utils import load_master_dataframe

# Logging is configured at package level

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

def get_day_order():
    """Return day names in Sunday-first order"""
    return ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

def calculate_day_of_week_averages(df):
    """
    Calculate day-of-week averages for sleep score, body battery, and water intake.
    
    Args:
        df (pd.DataFrame): Master dataframe with daily data
        
    Returns:
        pd.DataFrame: DataFrame with day-of-week averages
    """
    if df.empty:
        logging.warning("DataFrame is empty")
        return pd.DataFrame()
    
    # Ensure day column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['day']):
        df['day'] = pd.to_datetime(df['day'])
    
    # Add day of week column
    df['day_of_week'] = df['day'].dt.day_name()
    # Convert to Sunday-first ordering (Sunday=0, Monday=1, ..., Saturday=6)
    df['day_of_week_num'] = (df['day'].dt.dayofweek + 1) % 7
    
    # Define the metrics we want to analyze
    metrics = {
        'sleep_score': 'score',
        'body_battery_max': 'bb_max', 
        'body_battery_min': 'bb_min',
        'water_intake': 'hydration_intake'
    }
    
    # Calculate averages by day of week
    results = []
    
    for metric_name, column_name in metrics.items():
        if column_name not in df.columns:
            logging.warning(f"Column '{column_name}' not found in dataframe for {metric_name}")
            continue
            
        # Filter out null values for this metric
        metric_data = df[df[column_name].notna()].copy()
        
        if metric_data.empty:
            logging.warning(f"No valid data found for {metric_name}")
            continue
            
        # Calculate averages by day of week
        day_averages = metric_data.groupby(['day_of_week', 'day_of_week_num'])[column_name].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        day_averages['metric'] = metric_name
        day_averages['column'] = column_name
        results.append(day_averages)
    
    if not results:
        logging.warning("No valid metrics found to analyze")
        return pd.DataFrame()
    
    # Combine all results
    combined_results = pd.concat(results, ignore_index=True)
    
    # Create a categorical column with proper ordering (Sunday first)
    day_order = get_day_order()
    combined_results['day_of_week'] = pd.Categorical(
        combined_results['day_of_week'], 
        categories=day_order, 
        ordered=True
    )
    
    # Sort by metric and day of week
    combined_results = combined_results.sort_values(['metric', 'day_of_week'])
    
    return combined_results

def plot_day_of_week_averages(df, save_plots=True, show_plots=False):
    """
    Create visualizations for day-of-week averages.
    
    Args:
        df (pd.DataFrame): Master dataframe with daily data
        save_plots (bool): Whether to save plots to file
        show_plots (bool): Whether to display plots
        
    Returns:
        dict: Dictionary with plot filenames
    """
    if df.empty:
        logging.warning("DataFrame is empty")
        return {}
    
    # Calculate day-of-week averages
    day_averages = calculate_day_of_week_averages(df)
    
    if day_averages.empty:
        logging.warning("No day-of-week averages calculated")
        return {}
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    plot_files = {}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create individual plots for each metric
    metrics = day_averages['metric'].unique()
    
    for metric in metrics:
        metric_data = day_averages[day_averages['metric'] == metric].copy()
        
        if metric_data.empty:
            continue
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bar chart of means
        bars = ax1.bar(metric_data['day_of_week'], metric_data['mean'], 
                      color=sns.color_palette("husl", len(metric_data)))
        ax1.set_title(f'{metric.replace("_", " ").title()} - Daily Averages')
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel('Average Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_data['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 2: Box plot style with error bars (mean ± std)
        ax2.errorbar(range(len(metric_data)), metric_data['mean'], 
                    yerr=metric_data['std'], fmt='o', capsize=5, capthick=2)
        ax2.set_xticks(range(len(metric_data)))
        ax2.set_xticklabels(metric_data['day_of_week'], rotation=45)
        ax2.set_title(f'{metric.replace("_", " ").title()} - Mean ± Std Dev')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            filename = f"{timestamp_str}_day_of_week_{metric}.png"
            filepath = PLOTS_DIR / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plot_files[metric] = str(filepath)
            logging.info(f"Saved {metric} plot to {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # Create a combined plot showing all metrics
    if len(metrics) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for combined plot
        pivot_data = day_averages.pivot(index='day_of_week', columns='metric', values='mean')
        
        # Create grouped bar chart
        pivot_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Day-of-Week Averages: Sleep Score, Body Battery, and Water Intake')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Value')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save combined plot
        if save_plots:
            filename = f"{timestamp_str}_day_of_week_combined.png"
            filepath = PLOTS_DIR / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plot_files['combined'] = str(filepath)
            logging.info(f"Saved combined plot to {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return plot_files

def print_day_of_week_summary(df):
    """
    Print a summary of day-of-week averages.
    
    Args:
        df (pd.DataFrame): Master dataframe with daily data
    """
    day_averages = calculate_day_of_week_averages(df)
    
    if day_averages.empty:
        logging.warning("No day-of-week averages to summarize")
        return
    
    logging.info("\n" + "="*60)
    logging.info("DAY-OF-WEEK AVERAGES SUMMARY")
    logging.info("="*60)
    
    for metric in day_averages['metric'].unique():
        metric_data = day_averages[day_averages['metric'] == metric]
        
        logging.info(f"\n{metric.replace('_', ' ').title()}:")
        logging.info("-" * 40)
        
        for _, row in metric_data.iterrows():
            logging.info(f"{row['day_of_week']:>12}: {row['mean']:6.1f} ± {row['std']:5.1f} (n={row['count']})")
        
        # Find best and worst days
        best_day = metric_data.loc[metric_data['mean'].idxmax()]
        worst_day = metric_data.loc[metric_data['mean'].idxmin()]
        
        logging.info(f"\nBest day:  {best_day['day_of_week']} ({best_day['mean']:.1f})")
        logging.info(f"Worst day: {worst_day['day_of_week']} ({worst_day['mean']:.1f})")
        logging.info(f"Difference: {best_day['mean'] - worst_day['mean']:.1f}")

def main():
    """Main function to run day-of-week analysis."""
    try:
        # Load the master dataframe
        logging.info("Loading master daily summary data...")
        df = load_master_dataframe()
        
        if df.empty:
            logging.error("No data loaded")
            return
        
        logging.info(f"Loaded {len(df)} days of data")
        
        # Print summary
        print_day_of_week_summary(df)
        
        # Create visualizations
        logging.info("\nGenerating day-of-week visualizations...")
        plot_files = plot_day_of_week_averages(df, save_plots=True, show_plots=False)
        
        if plot_files:
            logging.info(f"\nGenerated {len(plot_files)} plots:")
            for metric, filepath in plot_files.items():
                logging.info(f"  {metric}: {filepath}")
        
    except Exception as e:
        logging.exception("Error in day-of-week analysis: %s", e)
        raise

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Logging is configured at package level

# Import activity mapping utilities
from garmin_analysis.utils.activity_mappings import (
    load_activity_mappings, 
    get_display_name, 
    get_activity_color,
    map_activity_dataframe
)

def plot_activity_calendar(activities_df: pd.DataFrame, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          output_dir: str = "plots",
                          figsize: tuple = (16, 10),
                          use_mappings: bool = True,
                          mappings_config_path: str = "config/activity_type_mappings.json"):
    """
    Create a calendar-style plot showing days with activities, colored by activity type.
    
    Args:
        activities_df (pd.DataFrame): DataFrame with columns 'start_time' and 'sport'
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, uses data range.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. If None, uses data range.
        output_dir (str): Directory to save the plot.
        figsize (tuple): Figure size for the plot.
        use_mappings (bool): Whether to use activity type mappings for display names and colors.
        mappings_config_path (str): Path to the activity mappings configuration file.
    """
    
    # Validate required columns
    required_cols = ['start_time', 'sport']
    missing_cols = [col for col in required_cols if col not in activities_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert start_time to datetime
    activities_df = activities_df.copy()
    activities_df['start_time'] = pd.to_datetime(activities_df['start_time'])
    
    # Filter by date range if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        activities_df = activities_df[activities_df['start_time'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        activities_df = activities_df[activities_df['start_time'] <= end_date]
    
    if activities_df.empty:
        logging.warning("No activities found in the specified date range")
        return
    
    # Create date column for grouping
    activities_df['date'] = activities_df['start_time'].dt.date
    
    # Load activity mappings if enabled
    mappings = load_activity_mappings(mappings_config_path) if use_mappings else {}
    
    # Get unique sports and create color mapping
    unique_sports = activities_df['sport'].dropna().unique()
    
    if use_mappings:
        # Use mapping-aware color assignment
        sport_colors = {}
        default_colors = _get_sport_colors(unique_sports)
        for sport in unique_sports:
            sport_colors[sport] = get_activity_color(sport, mappings, default_colors)
    else:
        # Use original color mapping
        sport_colors = _get_sport_colors(unique_sports)
    
    # Group activities by date and get the primary sport for each day
    daily_activities = activities_df.groupby('date').agg({
        'sport': lambda x: x.iloc[0] if len(x) > 0 else None,  # Take first sport if multiple
        'start_time': 'count'  # Count of activities
    }).rename(columns={'start_time': 'activity_count'})
    
    # Create date range
    date_range = pd.date_range(
        start=activities_df['start_time'].min().date(),
        end=activities_df['start_time'].max().date(),
        freq='D'
    )
    
    # Create the calendar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate grid dimensions (weeks x days)
    start_date = date_range[0]
    end_date = date_range[-1]
    
    # Adjust start date to beginning of week (Monday)
    start_week = start_date - timedelta(days=start_date.weekday())
    # Adjust end date to end of week (Sunday)
    end_week = end_date + timedelta(days=6-end_date.weekday())
    
    # Create full date range including empty weeks
    full_range = pd.date_range(start=start_week, end=end_week, freq='D')
    
    # Create grid with spacing between weeks
    weeks = len(full_range) // 7
    days_in_week = 7
    
    # Spacing parameters
    day_width = 1.0
    day_height = 1.0
    week_spacing = 0.3  # Space between weeks
    day_spacing = 0.05  # Small spacing between days
    
    # Day labels
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create grid for the calendar
    for i, date in enumerate(full_range):
        week = i // 7
        day = i % 7
        
        # Calculate position with spacing
        x = day * (day_width + day_spacing)
        y = week * (day_height + week_spacing)
        
        # Check if this date has activities
        date_str = date.date()
        if date_str in daily_activities.index:
            sport = daily_activities.loc[date_str, 'sport']
            activity_count = daily_activities.loc[date_str, 'activity_count']
            
            if sport and sport in sport_colors:
                color = sport_colors[sport]
                # Make color darker if multiple activities
                if activity_count > 1:
                    color = _darken_color(color, 0.3)
            else:
                color = '#E0E0E0'  # Light gray for unknown sports
        else:
            color = '#FFFFFF'  # White for no activities
        
        # Create rectangle for the day
        rect = patches.Rectangle((x, y), day_width, day_height, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Add date number
        ax.text(x + day_width/2, y + day_height/2, str(date.day), 
               ha='center', va='center', fontsize=8, 
               color='black' if color != '#FFFFFF' else 'gray')
    
    # Calculate total dimensions with spacing
    total_width = 7 * day_width + 6 * day_spacing
    total_height = weeks * day_height + (weeks - 1) * week_spacing
    
    # Set up the plot with proper limits
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-0.5, total_height + 0.5)
    
    # Set tick positions and labels
    day_tick_positions = [i * (day_width + day_spacing) + day_width/2 for i in range(7)]
    week_tick_positions = [i * (day_height + week_spacing) + day_height/2 for i in range(weeks)]
    
    ax.set_xticks(day_tick_positions)
    ax.set_xticklabels(day_labels)
    ax.set_yticks(week_tick_positions)
    ax.set_yticklabels([f"Week {i+1}" for i in range(weeks)])
    
    # Maintain aspect ratio but allow for spacing
    ax.set_aspect('equal')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add title
    plt.title(f"Activity Calendar\n{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", 
              fontsize=14, pad=20)
    
    # Create legend with mapped names
    legend_elements = []
    for sport in unique_sports:
        if sport in sport_colors:
            if use_mappings:
                display_name, _ = get_display_name(sport, mappings)
                label = display_name
            else:
                label = sport.replace('_', ' ').title()
            legend_elements.append(patches.Patch(color=sport_colors[sport], label=label))
    
    # Add legend for multiple activities
    legend_elements.append(patches.Patch(color='#666666', label='Multiple Activities'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"activity_calendar_{timestamp}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved activity calendar to {out_path}")
    
    # Print summary statistics
    total_days = len(date_range)
    active_days = len(daily_activities)
    logging.info(f"Activity summary: {active_days}/{total_days} days with activities ({active_days/total_days*100:.1f}%)")
    
    sport_summary = daily_activities['sport'].value_counts()
    logging.info("Activities by sport:")
    for sport, count in sport_summary.items():
        if use_mappings:
            display_name, _ = get_display_name(sport, mappings)
            logging.info(f"  - {display_name}: {count} days")
        else:
            logging.info(f"  - {sport.replace('_', ' ').title()}: {count} days")


def _get_sport_colors(sports: List[str]) -> Dict[str, str]:
    """Create a color mapping for different sports."""
    
    # Define color palette for sports
    sport_color_map = {
        'running': '#FF6B6B',           # Red
        'cycling': '#4ECDC4',           # Teal
        'fitness_equipment': '#45B7D1',  # Blue
        'training': '#96CEB4',          # Green
        'soccer': '#FFEAA7',            # Yellow
        'rowing': '#DDA0DD',            # Plum
        'walking': '#98D8C8',           # Mint
        'generic': '#F7DC6F',           # Light Yellow
        'UnknownEnumValue_67': '#BDC3C7'  # Silver
    }
    
    # Assign colors to available sports
    colors = {}
    available_colors = ['#FF9F43', '#10AC84', '#EE5A24', '#0984e3', '#6c5ce7', 
                       '#a29bfe', '#fd79a8', '#fdcb6e', '#e17055', '#00b894']
    color_idx = 0
    
    for sport in sports:
        if sport in sport_color_map:
            colors[sport] = sport_color_map[sport]
        else:
            # Assign from available colors for unknown sports
            colors[sport] = available_colors[color_idx % len(available_colors)]
            color_idx += 1
    
    return colors


def _darken_color(color: str, factor: float) -> str:
    """Darken a hex color by the given factor."""
    # Convert hex to RGB
    hex_color = color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Darken each component
    darkened_rgb = tuple(max(0, int(c * (1 - factor))) for c in rgb)
    
    # Convert back to hex
    return '#%02x%02x%02x' % darkened_rgb


def load_activities_data(db_path: str = "db/garmin_activities.db") -> pd.DataFrame:
    """Load activities data from the SQLite database."""
    import sqlite3
    
    with sqlite3.connect(db_path) as conn:
        query = """
        SELECT activity_id, start_time, sport, name, description
        FROM activities 
        WHERE sport IS NOT NULL AND start_time IS NOT NULL
        ORDER BY start_time
        """
        df = pd.read_sql_query(query, conn)
    
    logging.info(f"Loaded {len(df)} activities from database")
    return df

def suggest_activity_mappings(activities_df: pd.DataFrame, 
                            mappings_config_path: str = "config/activity_type_mappings.json"):
    """
    Suggest mappings for unknown activity types found in the data.
    
    Args:
        activities_df (pd.DataFrame): DataFrame with activity data containing 'sport' column.
        mappings_config_path (str): Path to the mappings configuration file.
    """
    from garmin_analysis.utils.activity_mappings import list_unknown_activities
    
    mappings = load_activity_mappings(mappings_config_path)
    unmapped_counts = list_unknown_activities(activities_df, mappings)
    
    if unmapped_counts:
        logging.info("Found unmapped activity types:")
        for activity_type, count in sorted(unmapped_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  - {activity_type}: {count} activities")
            logging.info(f"    Consider adding mapping: add_activity_mapping('{activity_type}', 'Your Display Name')")
    else:
        logging.info("All activity types have mappings!")


# Example usage
if __name__ == "__main__":
    # Load data and create calendar
    activities_df = load_activities_data()
    
    if not activities_df.empty:
        # Create calendar for the last 3 months
        end_date = activities_df['start_time'].max()
        start_date = end_date - timedelta(days=90)
        
        plot_activity_calendar(
            activities_df, 
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    else:
        logging.warning("No activities data found")

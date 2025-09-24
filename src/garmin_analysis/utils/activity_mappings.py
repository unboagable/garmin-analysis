"""
Utility functions for managing activity type mappings and transformations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Logging is configured at package level

def load_activity_mappings(config_path: str = "config/activity_type_mappings.json") -> Dict:
    """
    Load activity type mappings from the configuration file.
    
    Args:
        config_path (str): Path to the mappings configuration file.
        
    Returns:
        Dict: Activity mappings configuration.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logging.warning(f"Activity mappings config not found at {config_path}")
        return {"unknown_activity_mappings": {}}
    
    try:
        with open(config_file, 'r') as f:
            mappings = json.load(f)
        logging.info(f"Loaded activity mappings from {config_path}")
        return mappings
    except Exception as e:
        logging.error(f"Error loading activity mappings from {config_path}: {e}")
        return {"unknown_activity_mappings": {}}

def get_display_name(activity_type: str, mappings: Dict) -> Tuple[str, Optional[str]]:
    """
    Get the display name and description for an activity type.
    
    Args:
        activity_type (str): The original activity type name.
        mappings (Dict): Activity mappings configuration.
        
    Returns:
        Tuple[str, Optional[str]]: (display_name, description)
    """
    unknown_mappings = mappings.get("unknown_activity_mappings", {})
    
    if activity_type in unknown_mappings:
        mapping = unknown_mappings[activity_type]
        display_name = mapping.get("display_name", activity_type)
        description = mapping.get("description")
        logging.debug(f"Mapped '{activity_type}' to '{display_name}'")
        return display_name, description
    
    # If no mapping found, return the original name formatted nicely
    display_name = activity_type.replace('_', ' ').title()
    return display_name, None

def get_activity_color(activity_type: str, mappings: Dict, default_colors: Dict[str, str]) -> str:
    """
    Get the color for an activity type, using mappings if available.
    
    Args:
        activity_type (str): The activity type name.
        mappings (Dict): Activity mappings configuration.
        default_colors (Dict[str, str]): Default color mapping.
        
    Returns:
        str: Hex color code for the activity type.
    """
    unknown_mappings = mappings.get("unknown_activity_mappings", {})
    
    if activity_type in unknown_mappings:
        custom_color = unknown_mappings[activity_type].get("color")
        if custom_color:
            return custom_color
    
    # Fall back to default colors
    return default_colors.get(activity_type, "#BDC3C7")  # Default silver color

def add_activity_mapping(activity_type: str, display_name: str, 
                        description: str = None, category: str = None, 
                        color: str = None, config_path: str = "config/activity_type_mappings.json") -> bool:
    """
    Add a new activity mapping to the configuration file.
    
    Args:
        activity_type (str): The original activity type name.
        display_name (str): The display name for the activity.
        description (str, optional): Description of the activity.
        category (str, optional): Category for the activity.
        color (str, optional): Hex color code for the activity.
        config_path (str): Path to the mappings configuration file.
        
    Returns:
        bool: True if successfully added, False otherwise.
    """
    config_file = Path(config_path)
    
    # Load existing mappings
    mappings = load_activity_mappings(config_path)
    
    # Create the new mapping
    new_mapping = {
        "display_name": display_name,
        "description": description,
        "category": category,
        "color": color
    }
    
    # Remove None values
    new_mapping = {k: v for k, v in new_mapping.items() if v is not None}
    
    # Add to mappings
    if "unknown_activity_mappings" not in mappings:
        mappings["unknown_activity_mappings"] = {}
    
    mappings["unknown_activity_mappings"][activity_type] = new_mapping
    
    # Ensure config directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        logging.info(f"Added mapping for '{activity_type}' -> '{display_name}'")
        return True
    except Exception as e:
        logging.error(f"Error saving activity mappings to {config_path}: {e}")
        return False

def list_unknown_activities(df, mappings: Dict) -> Dict[str, int]:
    """
    List activities that don't have mappings and their counts.
    
    Args:
        df: DataFrame with activity data containing 'sport' column.
        mappings (Dict): Activity mappings configuration.
        
    Returns:
        Dict[str, int]: Count of unmapped activity types.
    """
    unknown_mappings = mappings.get("unknown_activity_mappings", {})
    mapped_types = set(unknown_mappings.keys())
    
    # Get all unique activity types
    all_types = df['sport'].dropna().unique()
    unmapped_types = [t for t in all_types if t not in mapped_types]
    
    # Count unmapped activities
    unmapped_counts = {}
    for activity_type in unmapped_types:
        count = (df['sport'] == activity_type).sum()
        unmapped_counts[activity_type] = count
    
    return unmapped_counts

def suggest_mappings(df, mappings: Dict) -> None:
    """
    Suggest mappings for unknown activity types found in the data.
    
    Args:
        df: DataFrame with activity data containing 'sport' column.
        mappings (Dict): Activity mappings configuration.
    """
    unmapped_counts = list_unknown_activities(df, mappings)
    
    if unmapped_counts:
        logging.info("Found unmapped activity types:")
        for activity_type, count in sorted(unmapped_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  - {activity_type}: {count} activities")
            logging.info(f"    Consider adding mapping: add_activity_mapping('{activity_type}', 'Your Display Name')")
    else:
        logging.info("All activity types have mappings!")

# Example usage functions
def map_activity_dataframe(df, mappings: Dict) -> 'pd.DataFrame':
    """
    Apply activity mappings to a DataFrame, creating display_name column.
    
    Args:
        df: DataFrame with 'sport' column.
        mappings (Dict): Activity mappings configuration.
        
    Returns:
        pd.DataFrame: DataFrame with added 'display_name' column.
    """
    import pandas as pd
    
    df = df.copy()
    
    # Create display names
    display_names = []
    descriptions = []
    
    for activity_type in df['sport']:
        if pd.isna(activity_type):
            display_names.append(None)
            descriptions.append(None)
        else:
            display_name, description = get_display_name(activity_type, mappings)
            display_names.append(display_name)
            descriptions.append(description)
    
    df['display_name'] = display_names
    df['description'] = descriptions
    
    return df

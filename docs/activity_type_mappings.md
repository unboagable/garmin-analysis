# Activity Type Mappings

This document describes the activity type mapping system for the Garmin Analysis project, which allows you to customize how unknown or poorly named activity types are displayed in visualizations.

## Overview

Garmin devices sometimes record activities with technical names or unknown enum values that aren't user-friendly. The mapping system allows you to:

- Map unknown activity types to meaningful display names
- Assign custom colors to specific activity types
- Categorize activities for better organization
- Maintain a centralized configuration for all mappings

## Configuration File

The mappings are stored in `config/activity_type_mappings.json`. Here's the structure:

```json
{
  "unknown_activity_mappings": {
    "UnknownEnumValue_67": {
      "display_name": "Training Assessment",
      "description": "Automatic fitness assessments, recovery measurements, or VO2 Max estimations",
      "category": "assessment",
      "color": "#9B59B6"
    }
  },
  "activity_categories": {
    "cardio": {
      "description": "Cardiovascular activities",
      "color_scheme": "warm"
    },
    "strength": {
      "description": "Strength and resistance training", 
      "color_scheme": "cool"
    },
    "assessment": {
      "description": "Fitness assessments and measurements",
      "color_scheme": "purple"
    }
  }
}
```

## Adding New Mappings

### Method 1: Direct JSON Editing

Edit `config/activity_type_mappings.json` directly:

```json
{
  "unknown_activity_mappings": {
    "UnknownEnumValue_67": {
      "display_name": "Training Assessment",
      "description": "Automatic fitness assessments",
      "category": "assessment",
      "color": "#9B59B6"
    },
    "your_new_activity_type": {
      "display_name": "Your Display Name",
      "description": "Description of what this activity represents",
      "category": "cardio",
      "color": "#E74C3C"
    }
  }
}
```

### Method 2: Programmatic Addition

Use the utility functions in your code:

```python
from garmin_analysis.utils.activity_mappings import add_activity_mapping

# Add a new mapping
add_activity_mapping(
    activity_type="UnknownEnumValue_68",
    display_name="Recovery Check",
    description="Automatic recovery and wellness measurements",
    category="assessment",
    color="#3498DB"
)
```

### Method 3: CLI Suggestions

The CLI can help you identify unmapped activity types:

```bash
# See what activity types need mappings
poetry run python src/garmin_analysis/viz/cli_activity_calendar.py --suggest-mappings
```

## Field Descriptions

### Mapping Fields

- **`display_name`**: The human-readable name shown in plots and logs
- **`description`**: Optional description of what the activity represents
- **`category`**: Optional categorization (cardio, strength, assessment, etc.)
- **`color`**: Optional hex color code for the activity type

### Categories

Categories help organize activities and can be used for color schemes:

- **`cardio`**: Running, cycling, rowing, etc.
- **`strength`**: Weight training, resistance exercises
- **`assessment`**: Fitness tests, recovery measurements
- **`sports`**: Team sports, competitive activities
- **`outdoor`**: Hiking, trail running, outdoor activities

## Color Guidelines

When choosing colors for activity mappings:

- **Use high contrast** colors for better visibility
- **Group related activities** with similar color families
- **Avoid colors** that are too similar to existing ones
- **Test accessibility** for colorblind users

Suggested color palettes:
- **Warm**: `#E74C3C`, `#F39C12`, `#F1C40F`
- **Cool**: `#3498DB`, `#2ECC71`, `#1ABC9C`
- **Purple**: `#9B59B6`, `#8E44AD`, `#663399`
- **Earth**: `#8B4513`, `#228B22`, `#556B2F`

## Usage in Visualizations

### Activity Calendar

The activity calendar automatically uses mappings when enabled:

```python
from garmin_analysis.viz.plot_activity_calendar import plot_activity_calendar

# With mappings (default)
plot_activity_calendar(activities_df, use_mappings=True)

# Without mappings (raw names)
plot_activity_calendar(activities_df, use_mappings=False)
```

### CLI Usage

```bash
# Create calendar with mappings (default)
poetry run python src/garmin_analysis/viz/cli_activity_calendar.py

# Create calendar without mappings
poetry run python src/garmin_analysis/viz/cli_activity_calendar.py --no-mappings

# Check for unmapped activities
poetry run python src/garmin_analysis/viz/cli_activity_calendar.py --suggest-mappings
```

## Common Unknown Activity Types

Here are some common Garmin activity types that might need mapping:

- **`UnknownEnumValue_67`**: Training Effect assessments → "Training Assessment"
- **`UnknownEnumValue_68`**: Recovery measurements → "Recovery Check"
- **`UnknownEnumValue_69`**: VO2 Max tests → "Fitness Test"
- **`generic`**: Unspecified activities → "General Activity"

## Maintenance

### Regular Updates

1. **Check for new unmapped types** monthly:
   ```bash
   poetry run python src/garmin_analysis/viz/cli_activity_calendar.py --suggest-mappings
   ```

2. **Review existing mappings** for accuracy and clarity

3. **Update colors** if needed for better visual distinction

### Version Control

The `config/activity_type_mappings.json` file should be committed to version control so that:

- Team members share the same mappings
- Mappings are preserved across environments
- Changes are tracked and reviewable

## Troubleshooting

### Mapping Not Applied

1. Check that the mapping file exists at the correct path
2. Verify the JSON syntax is valid
3. Ensure the activity type name matches exactly (case-sensitive)
4. Check logs for any loading errors

### Color Not Showing

1. Verify the color is a valid hex code (e.g., `#FF0000`)
2. Check that the color isn't being overridden by default colors
3. Ensure the activity type is properly mapped

### Performance Issues

For large datasets, consider:
- Using simpler display names
- Limiting the number of custom colors
- Caching mappings if loading repeatedly

## Examples

### Complete Example

Here's a complete example of a well-configured mappings file:

```json
{
  "unknown_activity_mappings": {
    "UnknownEnumValue_67": {
      "display_name": "Training Assessment",
      "description": "Automatic fitness assessments and recovery measurements",
      "category": "assessment",
      "color": "#9B59B6"
    },
    "UnknownEnumValue_68": {
      "display_name": "Recovery Check", 
      "description": "Wellness and recovery status measurements",
      "category": "assessment",
      "color": "#3498DB"
    },
    "generic": {
      "display_name": "General Activity",
      "description": "Unspecified or mixed activities",
      "category": "cardio",
      "color": "#95A5A6"
    }
  },
  "activity_categories": {
    "cardio": {
      "description": "Cardiovascular activities",
      "color_scheme": "warm"
    },
    "assessment": {
      "description": "Fitness assessments and measurements",
      "color_scheme": "purple"
    }
  }
}
```

This configuration will make your activity visualizations much more readable and professional-looking!

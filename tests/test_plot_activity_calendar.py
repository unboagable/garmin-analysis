import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from garmin_analysis.viz.plot_activity_calendar import plot_activity_calendar, _get_sport_colors, _darken_color


class TestActivityCalendar:
    """Test cases for activity calendar plotting functionality."""
    
    def test_get_sport_colors(self):
        """Test sport color mapping."""
        sports = ['running', 'cycling', 'fitness_equipment', 'unknown_sport']
        colors = _get_sport_colors(sports)
        
        # Check that all sports have colors
        assert len(colors) == len(sports)
        for sport in sports:
            assert sport in colors
            assert colors[sport].startswith('#')
            assert len(colors[sport]) == 7  # Hex color format
    
    def test_darken_color(self):
        """Test color darkening function."""
        original_color = '#FF0000'  # Red
        darkened = _darken_color(original_color, 0.5)
        
        # Should be darker (lower RGB values)
        assert darkened != original_color
        assert darkened.startswith('#')
        assert len(darkened) == 7
    
    def test_plot_activity_calendar_basic(self):
        """Test basic calendar plotting functionality."""
        # Create sample data
        base_date = datetime(2024, 1, 1)
        activities_data = []
        
        # Add some activities over a few days
        for i in range(5):
            date = base_date + timedelta(days=i*2)  # Every other day
            activities_data.append({
                'start_time': date,
                'sport': 'running' if i % 2 == 0 else 'cycling'
            })
        
        df = pd.DataFrame(activities_data)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise any exceptions
            plot_activity_calendar(df, output_dir=temp_dir, figsize=(10, 6))
            
            # Check that plot file was created
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_empty_data(self):
        """Test handling of empty data."""
        df = pd.DataFrame(columns=['start_time', 'sport'])
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle empty data gracefully
            plot_activity_calendar(df, output_dir=temp_dir)
            
            # No plot should be created
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 0
    
    def test_plot_activity_calendar_date_filtering(self):
        """Test date range filtering."""
        # Create data spanning multiple months
        start_date = datetime(2024, 1, 1)
        activities_data = []
        
        for i in range(90):  # 3 months of data
            date = start_date + timedelta(days=i)
            activities_data.append({
                'start_time': date,
                'sport': 'running'
            })
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Filter to just February
            plot_activity_calendar(
                df, 
                start_date='2024-02-01',
                end_date='2024-02-29',
                output_dir=temp_dir
            )
            
            # Should create a plot
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_multiple_activities_per_day(self):
        """Test handling of multiple activities on the same day."""
        date = datetime(2024, 1, 1)
        activities_data = [
            {'start_time': date, 'sport': 'running'},
            {'start_time': date, 'sport': 'cycling'},
            {'start_time': date, 'sport': 'fitness_equipment'}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_activity_calendar(df, output_dir=temp_dir)
            
            # Should create a plot
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_missing_columns(self):
        """Test handling of missing required columns."""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_activity_calendar(df)
    
    def test_plot_activity_calendar_unknown_sports(self):
        """Test handling of unknown sport types."""
        activities_data = [
            {'start_time': datetime(2024, 1, 1), 'sport': 'running'},
            {'start_time': datetime(2024, 1, 2), 'sport': 'unknown_sport'},
            {'start_time': datetime(2024, 1, 3), 'sport': None}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle unknown sports gracefully
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1


@pytest.mark.integration
class TestActivityCalendarIntegration:
    """Integration tests for activity calendar with real database."""
    
    def test_load_and_plot_real_data(self, tmp_db):
        """Test loading and plotting with real database structure."""
        import sqlite3
        
        # Create test activities in the tmp_db
        activities_db = tmp_db["activities"]
        
        with sqlite3.connect(activities_db) as conn:
            # Insert some test activities
            activities = [
                ('test1', '2024-01-01 10:00:00', 'running'),
                ('test2', '2024-01-02 11:00:00', 'cycling'),
                ('test3', '2024-01-03 12:00:00', 'fitness_equipment'),
                ('test4', '2024-01-01 15:00:00', 'running'),  # Multiple activities same day
            ]
            
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO activities (activity_id, start_time, sport) VALUES (?, ?, ?)",
                activities
            )
            conn.commit()
        
        # Import the load function
        from garmin_analysis.viz.plot_activity_calendar import load_activities_data
        
        # Load data
        df = load_activities_data(activities_db)
        
        assert len(df) == 4
        assert 'start_time' in df.columns
        assert 'sport' in df.columns
        
        # Test plotting
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1

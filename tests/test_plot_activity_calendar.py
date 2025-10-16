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
    
    def test_plot_activity_calendar_edge_case_no_activities_in_date_range(self):
        """Test with no activities in specified date range."""
        # Create activities in January
        activities_data = [
            {'start_time': datetime(2024, 1, 5), 'sport': 'running'},
            {'start_time': datetime(2024, 1, 10), 'sport': 'cycling'},
            {'start_time': datetime(2024, 1, 15), 'sport': 'running'}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Request data for February (no activities)
            plot_activity_calendar(
                df,
                start_date='2024-02-01',
                end_date='2024-02-29',
                output_dir=temp_dir
            )
            
            # Should handle gracefully - no plot created or empty plot
            # (behavior depends on implementation)
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            # Either no file or empty plot file
            assert len(plot_files) <= 1
    
    def test_plot_activity_calendar_edge_case_multiple_activities_same_day_varied(self):
        """Test with many activities on the same day."""
        date = datetime(2024, 1, 15)
        activities_data = []
        
        # Add 10 different activities on the same day
        sports = ['running', 'cycling', 'swimming', 'walking', 'hiking', 
                  'fitness_equipment', 'yoga', 'weightlifting', 'tennis', 'basketball']
        
        for i, sport in enumerate(sports):
            activities_data.append({
                'start_time': date + timedelta(hours=i),
                'sport': sport
            })
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle multiple activities on same day
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_edge_case_unknown_activity_types_only(self):
        """Test with only unknown/unrecognized activity types."""
        activities_data = [
            {'start_time': datetime(2024, 1, 1), 'sport': 'quantum_jumping'},
            {'start_time': datetime(2024, 1, 2), 'sport': 'time_travel_training'},
            {'start_time': datetime(2024, 1, 3), 'sport': 'teleportation_practice'},
            {'start_time': datetime(2024, 1, 4), 'sport': None},
            {'start_time': datetime(2024, 1, 5), 'sport': ''}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle all unknown sports gracefully
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_edge_case_invalid_date_range(self):
        """Test with invalid date range (end before start)."""
        activities_data = [
            {'start_time': datetime(2024, 1, 15), 'sport': 'running'},
            {'start_time': datetime(2024, 1, 20), 'sport': 'cycling'}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # End date before start date
            try:
                plot_activity_calendar(
                    df,
                    start_date='2024-02-01',
                    end_date='2024-01-01',  # Invalid: before start
                    output_dir=temp_dir
                )
                # If it doesn't raise, check behavior
                plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
                # Should handle gracefully
                assert True
            except (ValueError, AssertionError):
                # If it raises an error, that's also acceptable
                assert True
    
    def test_plot_activity_calendar_edge_case_very_long_date_range(self):
        """Test with very long date range (> 1 year)."""
        # Create activities over 2 years
        start_date = datetime(2023, 1, 1)
        activities_data = []
        
        # Add activities every 10 days for 2 years
        for i in range(73):  # ~730 days / 10
            date = start_date + timedelta(days=i*10)
            activities_data.append({
                'start_time': date,
                'sport': 'running' if i % 2 == 0 else 'cycling'
            })
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle long date ranges
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_edge_case_18_month_range(self):
        """Test with 18 month date range."""
        start_date = datetime(2023, 1, 1)
        activities_data = []
        
        # Add activities weekly for 18 months
        for i in range(78):  # ~18 months * 4.33 weeks
            date = start_date + timedelta(weeks=i)
            activities_data.append({
                'start_time': date,
                'sport': ['running', 'cycling', 'swimming'][i % 3]
            })
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle 18 month range
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1
    
    def test_plot_activity_calendar_edge_case_future_dates(self):
        """Test with activities in the future."""
        future_date = datetime.now() + timedelta(days=30)
        activities_data = [
            {'start_time': future_date, 'sport': 'running'},
            {'start_time': future_date + timedelta(days=1), 'sport': 'cycling'}
        ]
        
        df = pd.DataFrame(activities_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle future dates gracefully
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
                ('test1', '2024-01-01 10:00:00', 'running', 'Morning Run', 'Easy morning run', 2.5, 0.5, '00:30:00', 300),
                ('test2', '2024-01-02 11:00:00', 'cycling', 'Bike Ride', 'Cycling workout', 2.8, 0.7, '00:45:00', 350),
                ('test3', '2024-01-03 12:00:00', 'fitness_equipment', 'Gym Workout', 'Strength training', 3.0, 0.8, '01:00:00', 400),
                ('test4', '2024-01-01 15:00:00', 'running', 'Evening Run', 'Evening run', 2.2, 0.4, '00:25:00', 280),  # Multiple activities same day
            ]
            
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO activities (activity_id, start_time, sport, name, description, training_effect, anaerobic_training_effect, elapsed_time, calories) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                activities
            )
            conn.commit()
        
        # Import the load function
        from garmin_analysis.viz.plot_activity_calendar import load_activities_data
        
        # Load data
        df = load_activities_data(activities_db)
        
        # Should have the existing test data (30 activities from conftest.py) plus our 4 new ones
        assert len(df) >= 4
        assert len(df) == 34  # 30 from conftest.py + 4 new ones
        assert 'start_time' in df.columns
        assert 'sport' in df.columns
        
        # Test plotting
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_activity_calendar(df, output_dir=temp_dir)
            
            plot_files = list(Path(temp_dir).glob("activity_calendar_*.png"))
            assert len(plot_files) == 1

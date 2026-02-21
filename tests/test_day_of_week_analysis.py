import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from garmin_analysis.features.day_of_week_analysis import (
    calculate_day_of_week_averages,
    plot_day_of_week_averages,
    print_day_of_week_summary
)

@pytest.fixture
def sample_data():
    """Create sample data for testing day-of-week analysis."""
    # Create 30 days of sample data
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Create sample data with some day-of-week patterns
    data = []
    for i, date in enumerate(dates):
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        
        # Sleep score: higher on weekends
        sleep_score = 70 + (5 if day_of_week >= 5 else 0) + np.random.normal(0, 5)
        
        # Body battery: higher on weekdays
        bb_max = 80 + (10 if day_of_week < 5 else 0) + np.random.normal(0, 8)
        bb_min = 20 + (5 if day_of_week < 5 else 0) + np.random.normal(0, 5)
        
        # Water intake: some variation by day
        water_intake = 2000 + (200 if day_of_week < 5 else 0) + np.random.normal(0, 300)
        
        data.append({
            'day': date,
            'score': max(0, sleep_score),
            'bb_max': max(0, bb_max),
            'bb_min': max(0, bb_min),
            'hydration_intake': max(0, water_intake)
        })
    
    return pd.DataFrame(data)


class TestCalculateDayOfWeekAverages:

    def test_basic(self, sample_data):
        """Test day-of-week averages calculation."""
        result = calculate_day_of_week_averages(sample_data)
        
        # Check that we get results for all metrics
        assert not result.empty
        assert 'metric' in result.columns
        assert 'day_of_week' in result.columns
        assert 'mean' in result.columns
        
        # Check that we have data for all days of the week
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        result_days = result['day_of_week'].unique()
        for day in expected_days:
            assert day in result_days
        
        # Check that we have the expected metrics
        expected_metrics = ['sleep_score', 'body_battery_max', 'body_battery_min', 'water_intake']
        result_metrics = result['metric'].unique()
        for metric in expected_metrics:
            assert metric in result_metrics

    def test_empty_data(self):
        """Test day-of-week averages with empty data."""
        empty_df = pd.DataFrame()
        result = calculate_day_of_week_averages(empty_df)
        assert result.empty

    def test_missing_columns(self):
        """Test day-of-week averages with missing columns."""
        # Create data with only some columns
        data = pd.DataFrame({
            'day': [datetime(2024, 1, 1)],
            'score': [70]  # Only sleep score, missing body battery and water intake
        })
        
        result = calculate_day_of_week_averages(data)
        
        # Should still work and only include available metrics
        assert not result.empty
        assert 'sleep_score' in result['metric'].values
        assert 'body_battery_max' not in result['metric'].values


class TestPlotDayOfWeek:

    def test_plot_averages(self, sample_data, tmp_path):
        """Test plotting functionality."""
        # Mock the plots directory to use tmp_path
        import garmin_analysis.features.day_of_week_analysis as module
        original_plots_dir = module.PLOTS_DIR
        module.PLOTS_DIR = tmp_path
        
        try:
            # Test plotting
            plot_files = plot_day_of_week_averages(sample_data, save_plots=True, show_plots=False)
            
            # Check that plots were created
            assert plot_files
            assert len(plot_files) >= 4  # At least 4 individual plots + combined
            
            # Check that files exist
            for metric, filepath in plot_files.items():
                assert Path(filepath).exists()
                
        finally:
            # Restore original plots directory
            module.PLOTS_DIR = original_plots_dir


class TestPrintDayOfWeekSummary:

    def test_summary(self, sample_data, caplog):
        """Test summary printing functionality."""
        with caplog.at_level("INFO"):
            print_day_of_week_summary(sample_data)
        
        # Check that summary was logged
        assert "DAY-OF-WEEK AVERAGES SUMMARY" in caplog.text
        assert "Sleep Score" in caplog.text
        assert "Body Battery" in caplog.text

    def test_summary_empty_data(self, caplog):
        """Test summary printing with empty data."""
        empty_df = pd.DataFrame()
        
        with caplog.at_level("WARNING"):
            print_day_of_week_summary(empty_df)
        
        # Check that warning was logged
        assert "No day-of-week averages to summarize" in caplog.text


class TestDayOfWeekIntegration:

    @pytest.mark.integration
    def test_end_to_end(self, tmp_db):
        """Integration test with real database structure."""
        # This test uses the integration marker and tmp_db fixture
        # to test with actual database structure
        
        # tmp_db is a dict with database paths, get the main garmin database
        garmin_db_path = tmp_db['garmin']
        
        import sqlite3
        conn = sqlite3.connect(garmin_db_path)
        cursor = conn.cursor()
        
        # Insert sample data into daily_summary (using available columns)
        for i in range(14):  # 2 weeks of data
            date = f"2024-01-{i+1:02d}"
            cursor.execute("""
                INSERT INTO daily_summary 
                (day, steps, calories_total) 
                VALUES (?, ?, ?)
            """, (date, 5000 + i*100, 2000 + i*50))
        
        # Insert sample data into sleep
        for i in range(14):
            date = f"2024-01-{i+1:02d}"
            cursor.execute("""
                INSERT INTO sleep 
                (day, score) 
                VALUES (?, ?)
            """, (date, 60 + i))
        
        conn.commit()
        conn.close()
        
        # Load data using the real data loading function
        from garmin_analysis.data_ingestion.load_all_garmin_dbs import summarize_and_merge
        
        # Mock the database paths to use our test database
        import garmin_analysis.data_ingestion.load_all_garmin_dbs as load_module
        original_paths = load_module.DB_PATHS
        load_module.DB_PATHS = {
            "garmin": garmin_db_path,
            "activities": tmp_db['activities'],
            "summary": tmp_db['summary'],
            "summary2": tmp_db['summary'],
        }
        
        try:
            # Get the merged data
            df = summarize_and_merge(return_df=True)
            
            # Test our day-of-week analysis
            result = calculate_day_of_week_averages(df)
            
            # Should have results (at least sleep score should be available)
            assert not result.empty
            assert 'sleep_score' in result['metric'].values
            
            # Note: body battery and water intake may not be available in test schema
            # but the function should handle missing columns gracefully
            
        finally:
            # Restore original paths
            load_module.DB_PATHS = original_paths

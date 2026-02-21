import pytest
from src.garmin_analysis.dashboard.app import app, create_layout, update_day_of_week_charts
from src.garmin_analysis.utils import load_master_dataframe


class TestDashboardIntegration:

    def test_app_imports(self):
        """Test that dashboard imports successfully."""
        assert app is not None
        assert app.title == "Garmin Health Dashboard"

    def test_layout_structure(self):
        """Test that dashboard layout can be created."""
        df = load_master_dataframe()
        layout = create_layout(df)
        
        # Check that layout has the expected structure
        assert layout is not None
        assert len(layout.children) >= 2  # H1 title and Tabs
        
        # Check that we have the expected tabs
        tabs = layout.children[1].children
        tab_labels = [tab.label for tab in tabs]
        
        expected_tabs = ['📅 Day of Week Analysis', '📊 30-Day Health Overview', '📈 Metric Trends']
        for expected_tab in expected_tabs:
            assert expected_tab in tab_labels

    def test_charts_with_selected_metrics(self):
        """Test the day-of-week analysis callback."""
        df = load_master_dataframe()
        
        # Test with sleep score and body battery metrics
        selected_metrics = ['sleep_score', 'body_battery_max']
        coverage_filter = []
        
        bar_fig, combined_fig = update_day_of_week_charts(selected_metrics, coverage_filter, 2)
        
        # Check that figures were created
        assert bar_fig is not None
        assert combined_fig is not None
        
        # Check that we have traces for the selected metrics
        assert len(bar_fig.data) >= 1  # At least one metric should have data
        assert len(combined_fig.data) >= 1

    def test_charts_with_coverage_filter(self):
        """Test the day-of-week analysis callback with coverage filter."""
        df = load_master_dataframe()
        
        # Test with coverage filter
        selected_metrics = ['sleep_score']
        coverage_filter = ['filter']
        
        bar_fig, combined_fig = update_day_of_week_charts(selected_metrics, coverage_filter, 2)
        
        # Check that figures were created (may be empty if no data after filtering)
        assert bar_fig is not None
        assert combined_fig is not None

    def test_charts_with_empty_metrics(self):
        """Test the day-of-week analysis callback with no metrics selected."""
        df = load_master_dataframe()
        
        # Test with no metrics selected
        selected_metrics = []
        coverage_filter = []
        
        bar_fig, combined_fig = update_day_of_week_charts(selected_metrics, coverage_filter, 2)
        
        # Should still return figures (may be empty)
        assert bar_fig is not None
        assert combined_fig is not None

    @pytest.mark.integration
    def test_end_to_end(self, tmp_db):
        """Integration test for dashboard with test database."""
        # This test uses the integration marker and tmp_db fixture
        # to test with actual database structure
        
        # tmp_db is a dict with database paths, get the main garmin database
        garmin_db_path = tmp_db['garmin']
        
        import sqlite3
        conn = sqlite3.connect(garmin_db_path)
        cursor = conn.cursor()
        
        # Insert sample data into sleep table
        for i in range(14):  # 2 weeks of data
            date = f"2024-01-{i+1:02d}"
            cursor.execute("""
                INSERT INTO sleep 
                (day, score) 
                VALUES (?, ?)
            """, (date, 60 + i))
        
        conn.commit()
        conn.close()
        
        # Mock the database paths to use our test database
        import src.garmin_analysis.data_ingestion.load_all_garmin_dbs as load_module
        original_paths = load_module.DB_PATHS
        load_module.DB_PATHS = {
            "garmin": garmin_db_path,
            "activities": tmp_db['activities'],
            "summary": tmp_db['summary'],
            "summary2": tmp_db['summary'],
        }
        
        try:
            # Test dashboard layout creation with test data
            from src.garmin_analysis.data_ingestion.load_all_garmin_dbs import summarize_and_merge
            df = summarize_and_merge(return_df=True)
            
            layout = create_layout(df)
            assert layout is not None
            
            # Test day-of-week callback
            bar_fig, combined_fig = update_day_of_week_charts(['sleep_score'], [], 2)
            assert bar_fig is not None
            assert combined_fig is not None
            
        finally:
            # Restore original paths
            load_module.DB_PATHS = original_paths

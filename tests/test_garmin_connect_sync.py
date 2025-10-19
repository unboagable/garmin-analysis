"""
Integration tests for GarminDB synchronization functionality.

Tests the integration with GarminDB for automated Garmin Connect data download.
"""

import pytest
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from garmin_analysis.data_ingestion.garmin_connect_sync import (
    check_garmindb_installed,
    create_garmindb_config,
    sync_garmin_data,
    get_garmindb_stats,
    backup_garmindb,
    GARMINDB_CONFIG_FILE
)
from garmin_analysis.utils.error_handling import ConfigurationError, DataLoadingError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_garmindb_config_dir(tmp_path):
    """Create a temporary GarminDB config directory."""
    config_dir = tmp_path / ".GarminDb"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def mock_db_dir(tmp_path):
    """Create a temporary db directory."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    return db_dir


@pytest.fixture
def sample_garmindb_config():
    """Sample GarminDB configuration."""
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=365)).date().isoformat()
    
    return {
        "credentials": {
            "user": "test@example.com",
            "password": "testpassword123"
        },
        "data": {
            "download_days": 365,
            "sleep_download_days": 365,
            "weight_download_days": 365
        },
        "enabled_stats": {
            "monitoring": True,
            "sleep": True,
            "rhr": True,
            "weight": True,
            "activities": True
        },
        "stat_start_dates": {
            "monitoring": start_date,
            "sleep": start_date,
            "weight": start_date,
            "rhr": start_date,
            "activities": start_date
        },
        "db_type": "sqlite"
    }


@pytest.fixture
def mock_garmindb_stats():
    """Sample GarminDB stats output."""
    return """
GarminDB Statistics
===================
Date Range: 2024-01-01 to 2025-10-18

Daily Monitoring:
  - Total days: 292
  - Days with data: 285
  - Coverage: 97.6%

Sleep:
  - Total nights: 285
  - Average score: 72.5

Activities:
  - Total activities: 156
  - Running: 78
  - Cycling: 45
  - Other: 33
"""


# ============================================================================
# Test: check_garmindb_installed
# ============================================================================

class TestCheckGarminDBInstalled:
    """Test GarminDB installation check."""
    
    def test_garmindb_installed(self):
        """Test when garmindb_cli.py is available."""
        with patch('shutil.which', return_value='/usr/local/bin/garmindb_cli.py'):
            assert check_garmindb_installed() is True
    
    def test_garmindb_not_installed(self):
        """Test when garmindb_cli.py is not available."""
        with patch('shutil.which', return_value=None):
            assert check_garmindb_installed() is False


# ============================================================================
# Test: create_garmindb_config
# ============================================================================

class TestCreateGarminDBConfig:
    """Test GarminDB configuration creation."""
    
    def test_create_config_with_valid_credentials(self, tmp_path):
        """Test creating config with valid credentials."""
        config_path = tmp_path / "test_config.json"
        
        result = create_garmindb_config(
            username="test@example.com",
            password="testpass123",
            start_date="01/01/2024",
            config_path=config_path
        )
        
        assert result == config_path
        assert config_path.exists()
        
        # Verify config contents
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["credentials"]["user"] == "test@example.com"
        assert config["credentials"]["password"] == "testpass123"
        assert config["data"]["monitoring_start_date"] == "01/01/2024"
    
    def test_create_config_with_custom_activities(self, tmp_path):
        """Test creating config with custom activity counts."""
        config_path = tmp_path / "test_config.json"
        
        create_garmindb_config(
            username="test@example.com",
            password="testpass123",
            download_latest_activities=50,
            download_all_activities=2000,
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["data"]["download_latest_activities"] == 50
        assert config["data"]["download_all_activities"] == 2000
    
    def test_create_config_creates_directory(self, tmp_path):
        """Test that config creation creates parent directory."""
        config_path = tmp_path / "subdir" / "config.json"
        
        create_garmindb_config(
            username="test@example.com",
            password="testpass123",
            config_path=config_path
        )
        
        assert config_path.parent.exists()
        assert config_path.exists()
    
    def test_create_config_overwrites_existing(self, tmp_path):
        """Test that config creation overwrites existing config."""
        config_path = tmp_path / "config.json"
        
        # Create initial config
        create_garmindb_config(
            username="old@example.com",
            password="oldpass",
            config_path=config_path
        )
        
        # Overwrite with new config
        create_garmindb_config(
            username="new@example.com",
            password="newpass",
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["credentials"]["user"] == "new@example.com"
        assert config["credentials"]["password"] == "newpass"
    
    def test_create_config_default_structure(self, tmp_path):
        """Test that created config has all required fields."""
        config_path = tmp_path / "config.json"
        
        create_garmindb_config(
            username="test@example.com",
            password="testpass123",
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Check all required top-level fields exist
        assert "db" in config
        assert "garmin" in config
        assert "credentials" in config
        assert "data" in config
        assert "directories" in config
        assert "enabled_stats" in config
        assert "course_views" in config
        assert "modes" in config
        assert "activities" in config
        assert "settings" in config
        assert "checkup" in config
        
        # Check db section
        assert config["db"]["type"] == "sqlite"
        
        # Check garmin section
        assert config["garmin"]["domain"] == "garmin.com"
        
        # Check credentials
        assert "user" in config["credentials"]
        assert "password" in config["credentials"]
        assert "secure_password" in config["credentials"]
        
        # Check data start dates (MM/DD/YYYY format)
        assert "monitoring_start_date" in config["data"]
        assert "sleep_start_date" in config["data"]
        assert "rhr_start_date" in config["data"]
        assert "weight_start_date" in config["data"]
        assert "download_latest_activities" in config["data"]
        assert "download_all_activities" in config["data"]
        
        # Check enabled_stats
        assert config["enabled_stats"]["monitoring"] is True
        assert config["enabled_stats"]["sleep"] is True
        assert config["enabled_stats"]["rhr"] is True
        assert config["enabled_stats"]["weight"] is True
        assert config["enabled_stats"]["activities"] is True
        assert config["enabled_stats"]["steps"] is True
        assert config["enabled_stats"]["itime"] is True
        
        # Check directories
        assert config["directories"]["relative_to_home"] is True
        assert config["directories"]["base_dir"] == "HealthData"
        
        # Check settings
        assert "metric" in config["settings"]
        assert "default_display_activities" in config["settings"]


# ============================================================================
# Test: sync_garmin_data
# ============================================================================

class TestSyncGarminData:
    """Test Garmin data synchronization."""
    
    @pytest.mark.integration
    def test_sync_with_garmindb_not_installed(self):
        """Test sync fails when GarminDB not installed."""
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=False):
            with pytest.raises(DataLoadingError, match="GarminDB not found"):
                sync_garmin_data()
    
    @pytest.mark.integration
    def test_sync_without_config_file(self, tmp_path):
        """Test sync fails when config file doesn't exist."""
        fake_config = tmp_path / "nonexistent_config.json"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', fake_config):
                with pytest.raises(DataLoadingError, match="config not found"):
                    sync_garmin_data()
    
    @pytest.mark.integration
    def test_sync_successful_all_data(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test successful sync with all data."""
        # Setup mock config
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {"user": "test", "password": "test"}}')
        
        # Setup mock source database
        source_db = mock_garmindb_config_dir / "garmin.db"
        source_db.write_text("mock database content")
        
        # Mock subprocess.run
        mock_result = MagicMock()
        mock_result.stdout = "Sync completed successfully"
        mock_result.returncode = 0
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result) as mock_run:
                            result = sync_garmin_data(
                                download=True,
                                import_data=True,
                                analyze=True,
                                latest=False,
                                all_data=True
                            )
        
        # Verify subprocess was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "garmindb_cli.py" in call_args
        assert "--all" in call_args
        assert "--download" in call_args
        assert "--import" in call_args
        assert "--analyze" in call_args
        
        # Verify result
        assert result['success'] is True
        assert "synced successfully" in result['message']
        assert result['db_path'].name == "garmin.db"
    
    @pytest.mark.integration
    def test_sync_latest_only(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test sync with latest flag."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        source_db = mock_garmindb_config_dir / "garmin.db"
        source_db.write_text("mock db")
        
        mock_result = MagicMock()
        mock_result.stdout = "Latest sync completed"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result) as mock_run:
                            sync_garmin_data(latest=True)
        
        call_args = mock_run.call_args[0][0]
        assert "--latest" in call_args
    
    @pytest.mark.integration
    def test_sync_without_download(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test sync without download flag."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        source_db = mock_garmindb_config_dir / "garmin.db"
        source_db.write_text("mock db")
        
        mock_result = MagicMock()
        mock_result.stdout = "Import completed"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result) as mock_run:
                            sync_garmin_data(download=False)
        
        call_args = mock_run.call_args[0][0]
        assert "--download" not in call_args
    
    @pytest.mark.integration
    def test_sync_subprocess_error(self, tmp_path, mock_garmindb_config_dir):
        """Test sync handles subprocess errors."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd', stderr='Error')):
                    with pytest.raises(DataLoadingError, match="GarminDB sync failed"):
                        sync_garmin_data()
    
    @pytest.mark.integration
    def test_sync_missing_database_after_run(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test sync fails when expected database not created."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        # Don't create source database - simulate failure
        mock_result = MagicMock()
        mock_result.stdout = "Completed"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result):
                            with pytest.raises(DataLoadingError, match="Expected database not found"):
                                sync_garmin_data()
    
    @pytest.mark.integration
    def test_sync_copies_database_to_project(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test that sync copies database to project directory."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        # Create source database with content
        source_db = mock_garmindb_config_dir / "garmin.db"
        test_content = "test database content"
        source_db.write_text(test_content)
        
        mock_result = MagicMock()
        mock_result.stdout = "Success"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result):
                            result = sync_garmin_data()
        
        # Verify database was copied
        target_db = result['db_path']
        assert target_db.exists()
        assert target_db.read_text() == test_content


# ============================================================================
# Test: get_garmindb_stats
# ============================================================================

class TestGetGarminDBStats:
    """Test GarminDB statistics retrieval."""
    
    def test_get_stats_file_exists(self, tmp_path, mock_garmindb_stats):
        """Test reading stats when file exists."""
        stats_file = tmp_path / ".GarminDb" / "stats.txt"
        stats_file.parent.mkdir()
        stats_file.write_text(mock_garmindb_stats)
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            result = get_garmindb_stats()
        
        assert result == mock_garmindb_stats
        assert "Date Range" in result
        assert "Daily Monitoring" in result
    
    def test_get_stats_file_not_exists(self, tmp_path):
        """Test when stats file doesn't exist."""
        with patch('pathlib.Path.home', return_value=tmp_path):
            result = get_garmindb_stats()
        
        assert result is None
    
    def test_get_stats_read_error(self, tmp_path, caplog):
        """Test handling of read errors."""
        stats_file = tmp_path / ".GarminDb" / "stats.txt"
        stats_file.parent.mkdir()
        stats_file.write_text("test")
        
        with patch('pathlib.Path.home', return_value=tmp_path):
            with patch('pathlib.Path.read_text', side_effect=PermissionError("Access denied")):
                result = get_garmindb_stats()
        
        assert result is None
        assert "Failed to read GarminDB stats" in caplog.text


# ============================================================================
# Test: backup_garmindb
# ============================================================================

class TestBackupGarminDB:
    """Test GarminDB backup functionality."""
    
    def test_backup_successful(self):
        """Test successful backup."""
        mock_result = MagicMock()
        mock_result.stdout = "Backup completed"
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            result = backup_garmindb()
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "garmindb_cli.py" in call_args
        assert "--backup" in call_args
        
        assert result['success'] is True
        assert "completed successfully" in result['message']
    
    def test_backup_failure(self):
        """Test backup failure."""
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd', stderr='Backup failed')):
            result = backup_garmindb()
        
        assert result['success'] is False
        assert "Backup failed" in result['message']


# ============================================================================
# Test: Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.integration
    def test_complete_setup_and_sync_workflow(self, tmp_path, mock_db_dir):
        """Test complete workflow: setup config -> sync data."""
        # Step 1: Create config
        config_file = tmp_path / "config.json"
        create_garmindb_config(
            username="integration@test.com",
            password="testpass",
            start_date="01/01/2024",
            config_path=config_file
        )
        
        # Verify config was created correctly
        with open(config_file) as f:
            config = json.load(f)
        assert config["credentials"]["user"] == "integration@test.com"
        assert config["data"]["monitoring_start_date"] == "01/01/2024"
        
        # Step 2: Sync data (mocked)
        source_db = tmp_path / ".GarminDb" / "garmin.db"
        source_db.parent.mkdir()
        source_db.write_text("integration test database")
        
        mock_result = MagicMock()
        mock_result.stdout = "Integration test sync"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result):
                            result = sync_garmin_data()
        
        # Verify sync was successful
        assert result['success'] is True
        assert result['db_path'].exists()
        assert result['db_path'].read_text() == "integration test database"
    
    @pytest.mark.integration
    def test_daily_update_workflow(self, tmp_path, mock_db_dir):
        """Test daily update workflow with --latest flag."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"credentials": {"user": "daily@test.com", "password": "test"}}')
        
        source_db = tmp_path / ".GarminDb" / "garmin.db"
        source_db.parent.mkdir()
        source_db.write_text("daily update database")
        
        mock_result = MagicMock()
        mock_result.stdout = "Latest data synced"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result) as mock_run:
                            result = sync_garmin_data(latest=True)
        
        # Verify latest flag was used
        call_args = mock_run.call_args[0][0]
        assert "--latest" in call_args
        assert result['success'] is True


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_config_with_empty_credentials(self, tmp_path):
        """Test config creation with empty credentials."""
        config_path = tmp_path / "config.json"
        
        create_garmindb_config(
            username="",
            password="",
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["credentials"]["user"] == ""
        assert config["credentials"]["password"] == ""
    
    def test_config_with_special_characters(self, tmp_path):
        """Test config with special characters in password."""
        config_path = tmp_path / "config.json"
        
        special_password = 'p@$$w0rd!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        
        create_garmindb_config(
            username="test@example.com",
            password=special_password,
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["credentials"]["password"] == special_password
    
    def test_sync_with_very_long_output(self, tmp_path, mock_garmindb_config_dir, mock_db_dir):
        """Test sync handles very long subprocess output."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        source_db = mock_garmindb_config_dir / "garmin.db"
        source_db.write_text("db")
        
        # Create very long output
        long_output = "Processing...\n" * 10000
        mock_result = MagicMock()
        mock_result.stdout = long_output
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result):
                            result = sync_garmin_data()
        
        assert result['success'] is True
    
    def test_config_with_max_activities(self, tmp_path):
        """Test config with maximum activity counts."""
        config_path = tmp_path / "config.json"
        
        create_garmindb_config(
            username="test@example.com",
            password="test",
            download_all_activities=9999,
            config_path=config_path
        )
        
        with open(config_path) as f:
            config = json.load(f)
        
        assert config["data"]["download_all_activities"] == 9999


# ============================================================================
# Test: Logging
# ============================================================================

class TestLogging:
    """Test logging functionality."""
    
    def test_sync_logs_success(self, tmp_path, mock_garmindb_config_dir, mock_db_dir, caplog):
        """Test that successful sync logs appropriate messages."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        source_db = mock_garmindb_config_dir / "garmin.db"
        source_db.write_text("db")
        
        mock_result = MagicMock()
        mock_result.stdout = "Success"
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('garmin_analysis.data_ingestion.garmin_connect_sync.DB_DIR', mock_db_dir):
                    with patch('pathlib.Path.home', return_value=tmp_path):
                        with patch('subprocess.run', return_value=mock_result):
                            sync_garmin_data()
        
        assert "GarminDB sync completed successfully" in caplog.text
        assert "Copied garmin.db" in caplog.text
    
    def test_sync_logs_error(self, tmp_path, mock_garmindb_config_dir, caplog):
        """Test that failed sync logs error messages."""
        config_file = mock_garmindb_config_dir / "GarminConnectConfig.json"
        config_file.write_text('{"credentials": {}}')
        
        with patch('garmin_analysis.data_ingestion.garmin_connect_sync.check_garmindb_installed', return_value=True):
            with patch('garmin_analysis.data_ingestion.garmin_connect_sync.GARMINDB_CONFIG_FILE', config_file):
                with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd', stderr='Test error')):
                    try:
                        sync_garmin_data()
                    except DataLoadingError:
                        pass
        
        assert "GarminDB sync failed" in caplog.text
    
    def test_config_creation_logs(self, tmp_path, caplog):
        """Test config creation logging."""
        config_path = tmp_path / "config.json"
        
        create_garmindb_config(
            username="test@example.com",
            password="test",
            config_path=config_path
        )
        
        assert "Created GarminDB config" in caplog.text


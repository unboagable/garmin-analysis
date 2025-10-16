"""
Unit tests for HR Activity Sleep Model.

Tests cover:
- Data loading and validation
- Feature preparation
- Lag feature creation
- Model training and evaluation
- Feature importance analysis
- Visualization generation
- Report generation
- End-to-end analysis pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
import warnings

from garmin_analysis.modeling.hr_activity_sleep_model import HRActivitySleepModel

# Suppress expected sklearn convergence warnings in tests
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='Objective did not converge')


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    # Create sample features
    data = {
        'day': dates,
        # HR features
        'hr_min': np.random.randint(45, 55, n_samples),
        'hr_max': np.random.randint(110, 140, n_samples),
        'rhr': np.random.randint(48, 60, n_samples),
        'hr_avg': np.random.randint(65, 85, n_samples),
        'inactive_hr_avg': np.random.randint(60, 75, n_samples),
        'inactive_hr_min': np.random.randint(50, 65, n_samples),
        'inactive_hr_max': np.random.randint(85, 100, n_samples),
        
        # Activity features
        'steps': np.random.randint(3000, 15000, n_samples),
        'moderate_activity_time': np.random.randint(0, 60, n_samples),
        'vigorous_activity_time': np.random.randint(0, 30, n_samples),
        'intensity_time': np.random.randint(0, 90, n_samples),
        'distance': np.random.uniform(2, 10, n_samples),
        'calories_active': np.random.randint(200, 800, n_samples),
        'yesterday_had_workout': np.random.randint(0, 2, n_samples),
        'yesterday_activity_minutes': np.random.randint(0, 180, n_samples),
        'yesterday_activity_calories': np.random.randint(0, 1000, n_samples),
        'yesterday_training_effect': np.random.uniform(0, 3, n_samples),
        'yesterday_anaerobic_te': np.random.uniform(0, 1.5, n_samples),
        'floors': np.random.randint(0, 30, n_samples),
        
        # Contextual features
        'stress_avg': np.random.randint(15, 45, n_samples),
        'bb_max': np.random.randint(60, 100, n_samples),
        'bb_min': np.random.randint(5, 30, n_samples),
        'steps_avg_7d': np.random.randint(5000, 12000, n_samples),
        
        # Target variable
        'score': np.random.randint(40, 95, n_samples),
    }
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing robustness."""
    np.random.seed(42)
    n_samples = 100
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    data = {
        'day': dates,
        'hr_min': np.random.randint(45, 55, n_samples),
        'hr_max': np.random.randint(110, 140, n_samples),
        'rhr': np.random.randint(48, 60, n_samples),
        'hr_avg': np.random.randint(65, 85, n_samples),
        'steps': np.random.randint(3000, 15000, n_samples),
        'calories_active': np.random.randint(200, 800, n_samples),
        'stress_avg': np.random.randint(15, 45, n_samples),
        'bb_max': np.random.randint(60, 100, n_samples),
        'bb_min': np.random.randint(5, 30, n_samples),
        'score': np.random.randint(40, 95, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[0:10, 'hr_avg'] = np.nan
    df.loc[20:25, 'steps'] = np.nan
    df.loc[50:55, 'stress_avg'] = np.nan
    
    return df


@pytest.fixture
def temp_data_file(sample_data, tmp_path):
    """Create a temporary CSV file with sample data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    data_file = data_dir / "modeling_ready_dataset.csv"
    sample_data.to_csv(data_file, index=False)
    
    return str(data_file)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


class TestHRActivitySleepModelInit:
    """Test model initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = HRActivitySleepModel()
        
        # Check that data_path ends with the expected file (may be absolute or relative)
        assert model.data_path.endswith("data/modeling_ready_dataset.csv")
        assert model.random_state == 42
        assert model.df is None
        assert model.models == {}
        assert model.scaler is not None
        assert model.feature_importance == {}
        assert model.results == {}
    
    def test_init_custom_path(self):
        """Test initialization with custom data path."""
        custom_path = "custom/path/data.csv"
        model = HRActivitySleepModel(data_path=custom_path)
        
        assert model.data_path == custom_path
    
    def test_init_custom_random_state(self):
        """Test initialization with custom random state."""
        model = HRActivitySleepModel(random_state=123)
        
        assert model.random_state == 123


class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_success(self, temp_data_file):
        """Test successful data loading."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        df = model.load_data()
        
        assert df is not None
        assert len(df) == 100
        assert 'day' in df.columns
        assert 'score' in df.columns
        assert model.df is not None
    
    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        model = HRActivitySleepModel(data_path="nonexistent/file.csv")
        
        with pytest.raises(FileNotFoundError):
            model.load_data()
    
    def test_load_data_parses_dates(self, temp_data_file):
        """Test that dates are parsed correctly."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        df = model.load_data()
        
        assert pd.api.types.is_datetime64_any_dtype(df['day'])


class TestFeaturePreparation:
    """Test feature preparation."""
    
    def test_prepare_features_success(self, temp_data_file):
        """Test successful feature preparation."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        
        assert X is not None
        assert y is not None
        assert len(feature_names) > 0
        assert len(X) == len(y)
        assert X.shape[1] == len(feature_names)
    
    def test_prepare_features_median_imputation(self, sample_data_with_missing, tmp_path):
        """Test median imputation strategy."""
        data_file = tmp_path / "data.csv"
        sample_data_with_missing.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        X, y, feature_names = model.prepare_features(imputation_strategy='median')
        
        # Should not have any NaN values after median imputation
        X_df = pd.DataFrame(X, columns=feature_names)
        assert not X_df.isna().any().any()
        assert not pd.Series(y).isna().any()
    
    def test_prepare_features_mean_imputation(self, sample_data_with_missing, tmp_path):
        """Test mean imputation strategy."""
        data_file = tmp_path / "data.csv"
        sample_data_with_missing.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        X, y, feature_names = model.prepare_features(imputation_strategy='mean')
        
        # Should not have any NaN values after mean imputation
        X_df = pd.DataFrame(X, columns=feature_names)
        assert not X_df.isna().any().any()
        assert not pd.Series(y).isna().any()
    
    def test_prepare_features_drop_imputation(self, sample_data_with_missing, tmp_path):
        """Test drop rows imputation strategy."""
        data_file = tmp_path / "data.csv"
        sample_data_with_missing.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        X, y, feature_names = model.prepare_features(imputation_strategy='drop')
        
        # Should have no NaN values
        X_df = pd.DataFrame(X, columns=feature_names)
        assert not X_df.isna().any().any()
        assert not pd.Series(y).isna().any()
        # Should have fewer samples than median/mean imputation
        assert len(X) < 100
    
    def test_prepare_features_invalid_strategy(self, temp_data_file):
        """Test that invalid imputation strategy raises error."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        with pytest.raises(ValueError, match="Invalid imputation strategy"):
            model.prepare_features(imputation_strategy='invalid_strategy')
    
    def test_prepare_features_has_hr_features(self, temp_data_file):
        """Test that HR features are included."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        
        hr_features = [f for f in feature_names if 'hr' in f.lower() or 'rhr' in f.lower()]
        assert len(hr_features) > 0
    
    def test_prepare_features_has_activity_features(self, temp_data_file):
        """Test that activity features are included."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        
        activity_keywords = ['activity', 'steps', 'workout', 'calories', 'distance', 'floors']
        activity_features = [f for f in feature_names 
                            if any(kw in f.lower() for kw in activity_keywords)]
        assert len(activity_features) > 0
    
    
    def test_prepare_features_target_in_range(self, temp_data_file):
        """Test that target values are in expected range."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        
        # Sleep scores typically range from 0-100
        assert y.min() >= 0
        assert y.max() <= 100


class TestLagFeatures:
    """Test lag feature creation."""
    
    def test_create_lag_features(self, temp_data_file):
        """Test lag feature creation."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_df = pd.DataFrame(X, columns=feature_names)
        
        X_with_lags, new_feature_names = model.create_lag_features(X_df, feature_names)
        
        assert len(new_feature_names) > len(feature_names)
        assert X_with_lags.shape[1] > X.shape[1]
    
    def test_lag_features_have_lag_suffix(self, temp_data_file):
        """Test that lag features have '_lag1' suffix."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_df = pd.DataFrame(X, columns=feature_names)
        
        X_with_lags, new_feature_names = model.create_lag_features(X_df, feature_names)
        
        lag_features = [f for f in new_feature_names if '_lag1' in f]
        assert len(lag_features) > 0
    
    def test_lag_features_no_nan(self, temp_data_file):
        """Test that lag features don't introduce NaN values."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_df = pd.DataFrame(X, columns=feature_names)
        
        X_with_lags, new_feature_names = model.create_lag_features(X_df, feature_names)
        
        # Should fill NaN values
        assert not X_with_lags.isna().any().any()


class TestModelTraining:
    """Test model training."""
    
    def test_train_models_success(self, temp_data_file):
        """Test successful model training."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        
        assert len(results) > 0
        assert 'ElasticNet' in results
        assert 'Ridge Regression' in results
        assert 'Random Forest' in results
    
    def test_train_models_has_metrics(self, temp_data_file):
        """Test that training results include all metrics."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        
        for model_name, result in results.items():
            assert 'train_mse' in result
            assert 'test_mse' in result
            assert 'train_mae' in result
            assert 'test_mae' in result
            assert 'train_r2' in result
            assert 'test_r2' in result
            assert 'cv_mse' in result
            assert 'cv_std' in result
            assert 'feature_importance' in result
    
    def test_train_models_stores_models(self, temp_data_file):
        """Test that trained models are stored."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        
        assert len(model.models) > 0
    
    def test_train_models_predictions_shape(self, temp_data_file):
        """Test that predictions have correct shape."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        
        for model_name, result in results.items():
            # Test set is 20% of data
            expected_test_size = int(0.2 * len(X))
            assert len(result['y_test']) == expected_test_size
            assert len(result['y_test_pred']) == expected_test_size


class TestFeatureImportance:
    """Test feature importance analysis."""
    
    def test_analyze_feature_importance(self, temp_data_file):
        """Test feature importance analysis."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        assert len(importance_df) > 0
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_feature_importance_sorted(self, temp_data_file):
        """Test that feature importance is sorted descending."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        # Check that importance values are in descending order
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_feature_importance_all_positive(self, temp_data_file):
        """Test that all importance values are non-negative."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        assert (importance_df['importance'] >= 0).all()


class TestVisualizations:
    """Test visualization generation."""
    
    def test_create_visualizations(self, temp_data_file, temp_output_dir):
        """Test visualization creation."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        X_df = pd.DataFrame(X, columns=feature_names)
        plot_paths = model.create_visualizations(
            results, importance_df, X_df, y, feature_names,
            output_dir=temp_output_dir
        )
        
        assert len(plot_paths) > 0
        
        # Check that files were created
        for path in plot_paths:
            assert Path(path).exists()
    
    def test_visualizations_include_key_plots(self, temp_data_file, temp_output_dir):
        """Test that key visualizations are created."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        X_df = pd.DataFrame(X, columns=feature_names)
        plot_paths = model.create_visualizations(
            results, importance_df, X_df, y, feature_names,
            output_dir=temp_output_dir
        )
        
        # Check for specific plot types
        plot_names = [Path(p).name for p in plot_paths]
        
        assert any('model_performance' in name for name in plot_names)
        assert any('feature_importance' in name for name in plot_names)
        assert any('predictions' in name for name in plot_names)
        assert any('feature_correlations' in name for name in plot_names)


class TestReportGeneration:
    """Test report generation."""
    
    def test_generate_summary_report(self, temp_data_file, temp_output_dir):
        """Test summary report generation."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        report_path = model.generate_summary_report(
            results, importance_df,
            output_dir=temp_output_dir
        )
        
        assert Path(report_path).exists()
    
    def test_report_contains_key_sections(self, temp_data_file, temp_output_dir):
        """Test that report contains all key sections."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        report_path = model.generate_summary_report(
            results, importance_df,
            output_dir=temp_output_dir
        )
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert 'HR & ACTIVITY IMPACT ON SLEEP QUALITY' in content
        assert 'MODEL PERFORMANCE SUMMARY' in content
        assert 'BEST MODEL' in content
        assert 'TOP 20 MOST IMPORTANT FEATURES' in content
        assert 'FEATURE CATEGORY ANALYSIS' in content
        assert 'KEY INSIGHTS' in content
        assert 'RECOMMENDATIONS' in content
    
    def test_report_has_model_metrics(self, temp_data_file, temp_output_dir):
        """Test that report includes model metrics."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        results = model.train_models(X_scaled, y, feature_names)
        importance_df = model.analyze_feature_importance(results)
        
        report_path = model.generate_summary_report(
            results, importance_df,
            output_dir=temp_output_dir
        )
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        # Check for metrics
        assert 'R² Score' in content
        assert 'MAE' in content
        assert 'MSE' in content


class TestImputationStrategies:
    """Test different imputation strategies."""
    
    def test_median_vs_mean_imputation(self, sample_data_with_missing, tmp_path):
        """Test that median and mean imputation produce different results."""
        data_file = tmp_path / "data.csv"
        sample_data_with_missing.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        # Get results with median
        X_median, y_median, features = model.prepare_features(imputation_strategy='median')
        
        # Get results with mean
        model.load_data()  # Reload to get fresh data
        X_mean, y_mean, _ = model.prepare_features(imputation_strategy='mean')
        
        # Both should have same shape (no rows dropped)
        assert X_median.shape == X_mean.shape
        
        # But values should differ (median != mean for skewed distributions)
        # At least some values should be different
        assert not np.allclose(X_median.values, X_mean.values)
    
    def test_drop_strategy_reduces_samples(self, sample_data_with_missing, tmp_path):
        """Test that drop strategy reduces sample count."""
        data_file = tmp_path / "data.csv"
        sample_data_with_missing.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        # Get results with median (should keep all rows)
        X_median, _, _ = model.prepare_features(imputation_strategy='median')
        
        # Get results with drop (should have fewer rows)
        model.load_data()
        X_drop, _, _ = model.prepare_features(imputation_strategy='drop')
        
        # Drop should have fewer samples
        assert len(X_drop) < len(X_median)


class TestEndToEndAnalysis:
    """Test end-to-end analysis pipeline."""
    
    def test_run_analysis_success(self, temp_data_file, temp_output_dir):
        """Test successful end-to-end analysis."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        results = model.run_analysis(
            use_lag_features=True,
            output_dir=temp_output_dir
        )
        
        assert results is not None
        assert 'best_model' in results
        assert 'best_r2' in results
        assert 'best_mae' in results
        assert 'all_results' in results
        assert 'feature_importance' in results
        assert 'plot_paths' in results
        assert 'report_path' in results
        assert 'n_samples' in results
        assert 'n_features' in results
    
    def test_run_analysis_with_different_imputation(self, temp_data_file, temp_output_dir):
        """Test analysis with different imputation strategies."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        # Test with mean imputation
        results = model.run_analysis(
            use_lag_features=False,
            imputation_strategy='mean',
            output_dir=temp_output_dir / "mean"
        )
        
        assert results is not None
        assert results['n_samples'] > 0
    
    def test_run_analysis_without_lag_features(self, temp_data_file, temp_output_dir):
        """Test analysis without lag features."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        results = model.run_analysis(
            use_lag_features=False,
            output_dir=temp_output_dir
        )
        
        assert results is not None
        # Should have fewer features without lag features
        X, y, base_features = model.prepare_features()
        assert results['n_features'] == len(base_features)
    
    def test_run_analysis_creates_outputs(self, temp_data_file, temp_output_dir):
        """Test that analysis creates all expected outputs."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        results = model.run_analysis(
            use_lag_features=True,
            output_dir=temp_output_dir
        )
        
        # Check that plots were created
        assert len(results['plot_paths']) > 0
        for plot_path in results['plot_paths']:
            assert Path(plot_path).exists()
        
        # Check that report was created
        assert Path(results['report_path']).exists()
    
    def test_run_analysis_best_model_identified(self, temp_data_file, temp_output_dir):
        """Test that best model is correctly identified."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        results = model.run_analysis(
            use_lag_features=True,
            output_dir=temp_output_dir
        )
        
        # Best model should be one of the trained models
        assert results['best_model'] in results['all_results']
        
        # Best model should have the highest R² score
        best_r2 = results['best_r2']
        all_r2s = [res['test_r2'] for res in results['all_results'].values()]
        assert best_r2 == max(all_r2s)
    
    def test_run_analysis_metrics_reasonable(self, temp_data_file, temp_output_dir):
        """Test that analysis produces reasonable metrics."""
        model = HRActivitySleepModel(data_path=temp_data_file)
        
        results = model.run_analysis(
            use_lag_features=True,
            output_dir=temp_output_dir
        )
        
        # R² should be between -inf and 1 (typically between -1 and 1 in practice)
        assert results['best_r2'] <= 1.0
        
        # MAE should be positive
        assert results['best_mae'] > 0
        
        # Number of samples should be positive
        assert results['n_samples'] > 0
        
        # Number of features should be positive
        assert results['n_features'] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_minimal_data(self, tmp_path):
        """Test with minimal data (just enough samples)."""
        # Create minimal dataset
        n_samples = 30  # Just above minimum threshold
        data = {
            'day': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'hr_min': np.random.randint(45, 55, n_samples),
            'hr_max': np.random.randint(110, 140, n_samples),
            'rhr': np.random.randint(48, 60, n_samples),
            'steps': np.random.randint(3000, 15000, n_samples),
            'stress_avg': np.random.randint(15, 45, n_samples),
            'bb_max': np.random.randint(60, 100, n_samples),
            'score': np.random.randint(40, 95, n_samples),
        }
        df = pd.DataFrame(data)
        
        data_file = tmp_path / "minimal_data.csv"
        df.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        X, y, feature_names = model.prepare_features()
        
        assert len(X) > 0
        assert len(y) > 0
    
    def test_constant_target(self, tmp_path):
        """Test with constant target values."""
        n_samples = 100
        data = {
            'day': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'hr_min': np.random.randint(45, 55, n_samples),
            'hr_max': np.random.randint(110, 140, n_samples),
            'rhr': np.random.randint(48, 60, n_samples),
            'steps': np.random.randint(3000, 15000, n_samples),
            'stress_avg': np.random.randint(15, 45, n_samples),
            'bb_max': np.random.randint(60, 100, n_samples),
            'score': np.full(n_samples, 70),  # Constant target
        }
        df = pd.DataFrame(data)
        
        data_file = tmp_path / "constant_target.csv"
        df.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        # Should still train without errors (suppress expected convergence warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)
            results = model.train_models(X_scaled, y, feature_names)
        assert len(results) > 0
    
    def test_high_correlation_features(self, tmp_path):
        """Test with highly correlated features."""
        n_samples = 100
        base_hr = np.random.randint(60, 80, n_samples)
        
        data = {
            'day': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'hr_min': base_hr - 10,
            'hr_max': base_hr + 30,
            'rhr': base_hr - 5,
            'hr_avg': base_hr,
            'steps': np.random.randint(3000, 15000, n_samples),
            'stress_avg': np.random.randint(15, 45, n_samples),
            'bb_max': np.random.randint(60, 100, n_samples),
            'score': np.random.randint(40, 95, n_samples),
        }
        df = pd.DataFrame(data)
        
        data_file = tmp_path / "correlated_features.csv"
        df.to_csv(data_file, index=False)
        
        model = HRActivitySleepModel(data_path=str(data_file))
        model.load_data()
        
        X, y, feature_names = model.prepare_features()
        X_scaled = model.scaler.fit_transform(X)
        
        # Should handle correlated features with regularized models
        results = model.train_models(X_scaled, y, feature_names)
        assert 'Ridge Regression' in results
        assert 'Lasso Regression' in results
        assert 'ElasticNet' in results


class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_same_random_state_same_results(self, temp_data_file, temp_output_dir):
        """Test that same random state produces same results."""
        # Run analysis twice with same random state
        model1 = HRActivitySleepModel(data_path=temp_data_file, random_state=42)
        results1 = model1.run_analysis(use_lag_features=False, output_dir=temp_output_dir / "run1")
        
        model2 = HRActivitySleepModel(data_path=temp_data_file, random_state=42)
        results2 = model2.run_analysis(use_lag_features=False, output_dir=temp_output_dir / "run2")
        
        # Results should be very similar (allowing for small floating point differences)
        assert abs(results1['best_r2'] - results2['best_r2']) < 1e-10
        assert abs(results1['best_mae'] - results2['best_mae']) < 1e-10
    
    def test_different_random_state_may_differ(self, temp_data_file, temp_output_dir):
        """Test that different random states may produce different results."""
        # Run analysis with different random states
        model1 = HRActivitySleepModel(data_path=temp_data_file, random_state=42)
        results1 = model1.run_analysis(use_lag_features=False, output_dir=temp_output_dir / "run1")
        
        model2 = HRActivitySleepModel(data_path=temp_data_file, random_state=123)
        results2 = model2.run_analysis(use_lag_features=False, output_dir=temp_output_dir / "run2")
        
        # Results may differ slightly due to random initialization
        # Just check that both runs completed successfully
        assert results1 is not None
        assert results2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


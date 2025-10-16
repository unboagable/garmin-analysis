"""
Tests for comprehensive modeling pipeline.

These tests verify:
- Full pipeline execution end-to-end
- Individual analysis phases (anomaly detection, clustering, predictive modeling)
- Recommendation generation
- Results saving functionality
- Error handling for edge cases
- 24h coverage filtering integration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

from garmin_analysis.modeling.comprehensive_modeling_pipeline import ComprehensiveModelingPipeline


@pytest.fixture
def sample_health_data():
    """Create sample health data for testing."""
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    return pd.DataFrame({
        'day': dates,
        'score': np.random.randint(60, 100, n_days),
        'steps': np.random.randint(5000, 15000, n_days),
        'resting_heart_rate': np.random.randint(50, 75, n_days),
        'calories_total': np.random.randint(1800, 2500, n_days),
        'stress_avg': np.random.randint(20, 60, n_days),
        'body_battery_max': np.random.randint(70, 100, n_days),
        'body_battery_min': np.random.randint(20, 50, n_days),
    })


@pytest.fixture
def sample_health_data_with_missing():
    """Create sample health data with missing values."""
    np.random.seed(42)
    n_days = 50
    
    dates = pd.date_range('2024-01-01', periods=n_days)
    
    df = pd.DataFrame({
        'day': dates,
        'score': np.random.randint(60, 100, n_days),
        'steps': np.random.randint(5000, 15000, n_days),
        'resting_heart_rate': np.random.randint(50, 75, n_days),
    })
    
    # Add some missing values
    df.loc[df.sample(10, random_state=42).index, 'score'] = np.nan
    df.loc[df.sample(15, random_state=43).index, 'steps'] = np.nan
    
    return df


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_pipeline_runs_end_to_end(sample_health_data, tmp_path):
    """Test that the full pipeline runs without errors."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Run with minimal options to speed up test
    results = pipeline.run_full_pipeline(
        sample_health_data,
        filter_24h_coverage=False,
        tune_hyperparameters=False
    )
    
    # Verify results structure
    assert isinstance(results, dict)
    assert results['status'] == 'success'
    assert 'results' in results
    assert 'saved_files' in results
    
    # Verify pipeline results contain key sections
    pipeline_results = results['results']
    assert 'summary' in pipeline_results
    
    # Verify summary contains key sections
    summary = pipeline_results['summary']
    assert 'timestamp' in summary
    assert 'overview' in summary
    assert 'recommendations' in summary


def test_run_anomaly_detection_with_valid_data(sample_health_data, tmp_path):
    """Test anomaly detection phase with valid data."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the anomaly detector to avoid long-running operations
    with patch.object(pipeline.anomaly_detector, 'run_comprehensive_analysis') as mock_analysis:
        mock_analysis.return_value = {
            'ensemble_results': {
                'ensemble': {
                    'n_anomalies': 5,
                    'anomaly_score': 0.05,
                    'anomalies': [1, 5, 10, 15, 20]
                }
            },
            'n_samples': len(sample_health_data)
        }
        
        results = pipeline.run_anomaly_detection(sample_health_data, tune_hyperparameters=False)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'ensemble_results' in results
        assert results['ensemble_results']['ensemble']['n_anomalies'] == 5
        
        # Verify results are stored in pipeline
        assert 'anomaly_detection' in pipeline.results


def test_run_anomaly_detection_with_error_handling(sample_health_data, tmp_path):
    """Test anomaly detection error handling."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the anomaly detector to raise an exception
    with patch.object(pipeline.anomaly_detector, 'run_comprehensive_analysis') as mock_analysis:
        mock_analysis.side_effect = ValueError("Invalid data")
        
        results = pipeline.run_anomaly_detection(sample_health_data)
        
        # Should return empty dict on error
        assert results == {}


def test_run_clustering_analysis_with_valid_data(sample_health_data, tmp_path):
    """Test clustering phase with valid data."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the clusterer to avoid long-running operations
    with patch.object(pipeline.clusterer, 'run_comprehensive_clustering') as mock_clustering:
        mock_clustering.return_value = {
            'best_algorithm': 'kmeans',
            'best_model': {
                'evaluation': {
                    'n_clusters': 3,
                    'silhouette_score': 0.45
                }
            },
            'n_samples': len(sample_health_data)
        }
        
        results = pipeline.run_clustering_analysis(
            sample_health_data,
            algorithms=['kmeans'],
            tune_hyperparameters=False
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert results['best_algorithm'] == 'kmeans'
        assert results['best_model']['evaluation']['n_clusters'] == 3
        
        # Verify results are stored in pipeline
        assert 'clustering' in pipeline.results


def test_run_clustering_analysis_with_error_handling(sample_health_data, tmp_path):
    """Test clustering analysis error handling."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the clusterer to raise an exception
    with patch.object(pipeline.clusterer, 'run_comprehensive_clustering') as mock_clustering:
        mock_clustering.side_effect = RuntimeError("Clustering failed")
        
        results = pipeline.run_clustering_analysis(sample_health_data)
        
        # Should return empty dict on error
        assert results == {}


def test_run_predictive_modeling_with_valid_data(sample_health_data, tmp_path):
    """Test predictive modeling phase with valid data."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the predictor to avoid long-running operations
    with patch.object(pipeline.predictor, 'run_comprehensive_prediction') as mock_prediction:
        mock_prediction.return_value = {
            'best_algorithm': 'random_forest',
            'best_score': 0.75,
            'n_features': 7,
            'train_size': 80,
            'test_size': 20,
            'all_results': {
                'random_forest': {'r2': 0.75, 'mse': 25.0}
            }
        }
        
        results = pipeline.run_predictive_modeling(
            sample_health_data,
            target_col='score',
            algorithms=['random_forest'],
            tune_hyperparameters=False
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert results['best_algorithm'] == 'random_forest'
        assert results['best_score'] == 0.75
        
        # Verify results are stored in pipeline
        assert 'predictive_modeling' in pipeline.results


def test_run_predictive_modeling_with_error_handling(sample_health_data, tmp_path):
    """Test predictive modeling error handling."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the predictor to raise an exception
    with patch.object(pipeline.predictor, 'run_comprehensive_prediction') as mock_prediction:
        mock_prediction.side_effect = KeyError("Target column not found")
        
        results = pipeline.run_predictive_modeling(sample_health_data)
        
        # Should return empty dict on error
        assert results == {}


def test_generate_recommendations_with_high_anomaly_rate(tmp_path):
    """Test recommendation generation with high anomaly rate."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up results with high anomaly rate
    pipeline.results = {
        'anomaly_detection': {
            'ensemble_results': {
                'ensemble': {
                    'anomaly_score': 0.15  # 15% anomaly rate
                }
            }
        }
    }
    
    recommendations = pipeline._generate_recommendations()
    
    # Should recommend reviewing data quality
    assert any('High anomaly rate' in rec for rec in recommendations)


def test_generate_recommendations_with_low_clustering_quality(tmp_path):
    """Test recommendation generation with low clustering quality."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up results with low silhouette score
    pipeline.results = {
        'clustering': {
            'best_model': {
                'evaluation': {
                    'silhouette_score': 0.15  # Low score
                }
            }
        }
    }
    
    recommendations = pipeline._generate_recommendations()
    
    # Should recommend feature engineering
    assert any('Low clustering quality' in rec for rec in recommendations)


def test_generate_recommendations_with_high_missing_data(tmp_path):
    """Test recommendation generation with high missing data."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up results with high missing data
    pipeline.results = {
        'feature_analysis': {
            'basic_stats': {
                'col1': {'missing_pct': 40},
                'col2': {'missing_pct': 50},
                'col3': {'missing_pct': 10}
            }
        }
    }
    
    recommendations = pipeline._generate_recommendations()
    
    # Should recommend handling missing data
    assert any('missing data' in rec for rec in recommendations)


def test_save_results_creates_files(sample_health_data, tmp_path):
    """Test that save_results creates expected output files."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up some results with all required keys for markdown report
    pipeline.results = {
        'summary': {
            'timestamp': pipeline.timestamp,
            'overview': {'total_samples': len(sample_health_data)},
            'anomaly_detection_summary': {},
            'clustering_summary': {},
            'predictive_modeling_summary': {},
            'feature_analysis_summary': {},
            'recommendations': ['Test recommendation']
        },
        'feature_analysis': {
            'basic_stats': {'score': {'mean': 75.5}},
            'n_features': 8,
            'n_samples': len(sample_health_data)
        }
    }
    
    saved_files = pipeline.save_results()
    
    # Verify files were created
    assert len(saved_files) > 0
    
    # Verify summary file exists
    summary_files = [f for f in saved_files if 'modeling_summary' in f]
    assert len(summary_files) == 1
    
    summary_path = Path(summary_files[0])
    assert summary_path.exists()
    
    # Verify summary content
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        assert 'timestamp' in summary
        assert 'recommendations' in summary


def test_save_results_with_markdown_report(sample_health_data, tmp_path):
    """Test that markdown report is generated."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up complete results with all required keys and proper types
    pipeline.results = {
        'summary': {
            'timestamp': pipeline.timestamp,
            'overview': {'total_samples': 100, 'total_features': 8, 'analysis_timestamp': pipeline.timestamp},
            'anomaly_detection_summary': {
                'n_anomalies': 5, 
                'anomaly_rate': 0.05,
                'best_algorithm': 'ensemble',
                'n_samples_analyzed': 100
            },
            'clustering_summary': {
                'best_algorithm': 'kmeans', 
                'n_clusters': 3,
                'silhouette_score': 0.45,
                'n_samples_clustered': 100
            },
            'predictive_modeling_summary': {
                'best_algorithm': 'random_forest', 
                'best_mse': 25.0,
                'n_features_used': 7,
                'train_test_split': '80/20'
            },
            'feature_analysis_summary': {
                'top_features': ['score', 'steps'],
                'features_with_missing_data': []
            },
            'recommendations': ['Test recommendation 1', 'Test recommendation 2']
        }
    }
    
    saved_files = pipeline.save_results()
    
    # Verify markdown report was created
    md_files = [f for f in saved_files if f.endswith('.md')]
    assert len(md_files) == 1
    
    md_path = Path(md_files[0])
    assert md_path.exists()
    
    # Verify markdown content
    with open(md_path, 'r') as f:
        content = f.read()
        assert '# 🧠 Comprehensive Modeling Analysis Report' in content or 'Comprehensive Modeling' in content
        assert 'Anomaly Detection' in content
        assert 'Clustering' in content
        assert 'Predictive Modeling' in content or 'Predictive' in content
        assert 'Recommendations' in content or 'recommendation' in content.lower()


def test_error_handling_in_submodules_with_all_failures(sample_health_data, tmp_path):
    """Test pipeline handles multiple submodule failures gracefully."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock all submodules to fail
    with patch.object(pipeline.anomaly_detector, 'run_comprehensive_analysis') as mock_ad, \
         patch.object(pipeline.clusterer, 'run_comprehensive_clustering') as mock_cl, \
         patch.object(pipeline.predictor, 'run_comprehensive_prediction') as mock_pm:
        
        mock_ad.side_effect = Exception("Anomaly detection failed")
        mock_cl.side_effect = Exception("Clustering failed")
        mock_pm.side_effect = Exception("Prediction failed")
        
        # Should not raise exception
        results = pipeline.run_full_pipeline(sample_health_data, tune_hyperparameters=False)
        
        # Pipeline continues even if modules fail, so it may succeed
        assert isinstance(results, dict)
        # Pipeline can succeed if at least feature_analysis works
        # Even with module failures, it will complete and return success
        assert results['status'] in ['success', 'failed']
        
        # Verify that module failures were logged but pipeline continued
        if results['status'] == 'success':
            # Results should exist even if some modules failed
            assert 'results' in results


def test_24h_coverage_filtering_integration(sample_health_data, tmp_path):
    """Test 24h coverage filtering integration in pipeline."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Mock the filter function - it's imported inside the function
    with patch('garmin_analysis.features.coverage.filter_by_24h_coverage') as mock_filter, \
         patch.object(pipeline.anomaly_detector, 'run_comprehensive_analysis') as mock_ad, \
         patch.object(pipeline.clusterer, 'run_comprehensive_clustering') as mock_cl, \
         patch.object(pipeline.predictor, 'run_comprehensive_prediction') as mock_pm:
        
        # Mock filter to return subset of data
        filtered_data = sample_health_data.head(50)
        mock_filter.return_value = filtered_data
        
        # Mock analysis methods to return minimal results
        mock_ad.return_value = {'n_samples': len(filtered_data)}
        mock_cl.return_value = {'n_samples': len(filtered_data)}
        mock_pm.return_value = {'n_samples': len(filtered_data)}
        
        results = pipeline.run_full_pipeline(
            sample_health_data,
            filter_24h_coverage=True,
            max_gap_minutes=5,
            day_edge_tolerance_minutes=3,
            coverage_allowance_minutes=15,
            tune_hyperparameters=False
        )
        
        # Verify filter was called
        mock_filter.assert_called_once()
        
        # Verify filtering parameters were passed correctly
        call_args = mock_filter.call_args
        assert call_args[1]['max_gap'] == pd.Timedelta(minutes=5)
        assert call_args[1]['day_edge_tolerance'] == pd.Timedelta(minutes=3)
        assert call_args[1]['total_missing_allowance'] == pd.Timedelta(minutes=15)


def test_create_feature_analysis_with_valid_data(sample_health_data, tmp_path):
    """Test feature analysis creation."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    feature_analysis = pipeline.create_feature_analysis(sample_health_data)
    
    # Verify structure
    assert isinstance(feature_analysis, dict)
    assert 'basic_stats' in feature_analysis
    assert 'correlation_matrix' in feature_analysis
    assert 'feature_ranking' in feature_analysis
    assert 'n_features' in feature_analysis
    assert 'n_samples' in feature_analysis
    
    # Verify stats for each numeric column
    numeric_cols = sample_health_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert col in feature_analysis['basic_stats']
        stats = feature_analysis['basic_stats'][col]
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'missing_pct' in stats


def test_create_modeling_summary_with_all_components(tmp_path):
    """Test modeling summary creation with all components."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Set up comprehensive results
    pipeline.results = {
        'feature_analysis': {
            'n_samples': 100,
            'n_features': 8,
            'basic_stats': {
                'score': {'mean': 75.5, 'missing_pct': 5},
                'steps': {'mean': 10000, 'missing_pct': 25}
            },
            'feature_ranking': [('score', 150.2), ('steps', 120.5)]
        },
        'anomaly_detection': {
            'ensemble_results': {
                'ensemble': {
                    'n_anomalies': 5,
                    'anomaly_score': 0.05
                }
            },
            'n_samples': 100
        },
        'clustering': {
            'best_algorithm': 'kmeans',
            'best_model': {
                'evaluation': {
                    'n_clusters': 3,
                    'silhouette_score': 0.45
                }
            },
            'n_samples': 100
        },
        'predictive_modeling': {
            'best_algorithm': 'random_forest',
            'best_score': 25.0,
            'n_features': 7,
            'train_size': 80,
            'test_size': 20,
            'all_results': {
                'random_forest': {'r2': 0.75, 'mse': 25.0}
            }
        }
    }
    
    summary = pipeline.create_modeling_summary()
    
    # Verify structure
    assert isinstance(summary, dict)
    assert 'overview' in summary
    assert 'anomaly_detection_summary' in summary
    assert 'clustering_summary' in summary
    assert 'predictive_modeling_summary' in summary
    assert 'feature_analysis_summary' in summary
    assert 'recommendations' in summary
    
    # Verify content
    assert summary['overview']['total_samples'] == 100
    assert summary['overview']['total_features'] == 8
    assert summary['anomaly_detection_summary']['n_anomalies'] == 5
    assert summary['clustering_summary']['best_algorithm'] == 'kmeans'
    assert summary['predictive_modeling_summary']['best_algorithm'] == 'random_forest'


def test_pipeline_with_minimal_data(tmp_path):
    """Test pipeline with minimal data points."""
    pipeline = ComprehensiveModelingPipeline(output_dir=tmp_path, random_state=42)
    
    # Create very small dataset
    minimal_data = pd.DataFrame({
        'day': pd.date_range('2024-01-01', periods=10),
        'score': np.random.randint(60, 100, 10),
        'steps': np.random.randint(5000, 15000, 10)
    })
    
    # Mock analysis methods to avoid errors with small dataset
    with patch.object(pipeline.anomaly_detector, 'run_comprehensive_analysis') as mock_ad, \
         patch.object(pipeline.clusterer, 'run_comprehensive_clustering') as mock_cl, \
         patch.object(pipeline.predictor, 'run_comprehensive_prediction') as mock_pm:
        
        mock_ad.return_value = {'n_samples': len(minimal_data)}
        mock_cl.return_value = {'n_samples': len(minimal_data)}
        mock_pm.return_value = {'n_samples': len(minimal_data)}
        
        results = pipeline.run_full_pipeline(minimal_data, tune_hyperparameters=False)
        
        # Should complete - may succeed or fail gracefully
        assert isinstance(results, dict)
        # Either success with summary in results, or failure with partial_results
        if results['status'] == 'success':
            assert 'results' in results
            assert 'summary' in results['results']
        else:
            assert 'error' in results or 'partial_results' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


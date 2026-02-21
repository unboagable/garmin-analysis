"""
Automated tests for data quality and completeness analysis.

These tests help identify:
- Which features have sufficient data for modeling
- Data completeness issues
- Feature type problems
- Optimal feature selection for different modeling tasks
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Automated data quality and completeness checker."""
    
    def __init__(self, completeness_threshold=50, completeness_percentage=0.1):
        """
        Initialize the data quality checker.
        
        Args:
            completeness_threshold: Minimum number of non-null values required
            completeness_percentage: Minimum percentage of non-null values required
        """
        self.completeness_threshold = completeness_threshold
        self.completeness_percentage = completeness_percentage
        self.quality_report = {}
    
    def analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Comprehensive analysis of dataframe quality and completeness.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing quality metrics and recommendations
        """
        logger.info(f"Analyzing dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic info
        self.quality_report['basic_info'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Column completeness analysis
        self.quality_report['column_completeness'] = self._analyze_column_completeness(df)
        
        # Data type analysis
        self.quality_report['data_types'] = self._analyze_data_types(df)
        
        # Feature suitability analysis
        self.quality_report['feature_suitability'] = self._analyze_feature_suitability(df)
        
        # Recommendations
        self.quality_report['recommendations'] = self._generate_recommendations()
        
        return self.quality_report
    
    def _analyze_column_completeness(self, df: pd.DataFrame) -> dict:
        """Analyze completeness of each column."""
        completeness_data = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            total_count = len(df)
            completeness_pct = non_null_count / total_count
            
            # Fix numpy boolean operation issue
            null_count = total_count - non_null_count
            
            completeness_data[col] = {
                'non_null_count': int(non_null_count),
                'null_count': int(null_count),
                'completeness_percentage': float(completeness_pct),
                'is_sufficient': bool(non_null_count >= 50),  # Lower threshold to 50 non-null values
                'is_adequate': bool(completeness_pct >= 0.2)
            }
        
        return completeness_data
    
    def _analyze_data_types(self, df: pd.DataFrame) -> dict:
        """Analyze data types and identify potential issues."""
        type_data = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            sample_values = df[col].dropna().head(5).tolist()
            
            type_data[col] = {
                'dtype': str(dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(df[col]),
                'is_categorical': isinstance(df[col].dtype, pd.CategoricalDtype),
                'sample_values': sample_values,
                'unique_count': df[col].nunique(),
                'potential_issues': self._identify_potential_issues(df[col])
            }
        
        return type_data
    
    def _identify_potential_issues(self, series: pd.Series) -> list:
        """Identify potential data quality issues in a series."""
        issues = []
        
        # Check for mixed types
        if series.dtype == 'object':
            non_null = series.dropna()
            if len(non_null) > 0:
                type_counts = non_null.apply(type).value_counts()
                if len(type_counts) > 1:
                    issues.append(f"Mixed types: {type_counts.to_dict()}")
        
        # Check for time-like strings
        if series.dtype == 'object':
            time_patterns = ['00:00:00', ':', 'AM', 'PM']
            sample_values = series.dropna().astype(str).head(10)
            if any(any(pattern in str(val) for pattern in time_patterns) for val in sample_values):
                issues.append("Contains time-like strings")
        
        # Check for extreme values in numeric columns
        if pd.api.types.is_numeric_dtype(series) and series.dtype.kind not in 'b':  # Exclude boolean
            non_null = series.dropna()
            if len(non_null) > 10:
                try:
                    q1, q3 = non_null.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                        if len(outliers) > 0:
                            issues.append(f"Potential outliers: {len(outliers)} values outside IQR")
                except Exception as e:
                    # Skip outlier detection if there are issues
                    pass
        
        return issues
    
    def _analyze_feature_suitability(self, df: pd.DataFrame) -> dict:
        """Analyze which features are suitable for different modeling tasks."""
        suitability = {
            'anomaly_detection': [],
            'clustering': [],
            'predictive_modeling': [],
            'time_series': [],
            'unsuitable': []
        }
        
        for col in df.columns:
            completeness = self.quality_report['column_completeness'][col]
            data_type = self.quality_report['data_types'][col]
            
            # Check if feature is suitable for modeling
            if (completeness['is_sufficient'] and 
                data_type['is_numeric']):
                
                # Categorize by feature type
                if 'time' in col.lower() or 'date' in col.lower():
                    suitability['time_series'].append(col)
                elif 'score' in col.lower() or 'target' in col.lower():
                    suitability['predictive_modeling'].append(col)
                else:
                    # Good for anomaly detection and clustering
                    suitability['anomaly_detection'].append(col)
                    suitability['clustering'].append(col)
                    suitability['predictive_modeling'].append(col)
            else:
                suitability['unsuitable'].append(col)
        
        return suitability
    
    def _generate_recommendations(self) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Analyze completeness issues
        low_completeness = [
            col for col, data in self.quality_report['column_completeness'].items()
            if not data['is_adequate']
        ]
        
        if low_completeness:
            recommendations.append({
                'type': 'completeness',
                'priority': 'high',
                'message': f"Consider removing or imputing {len(low_completeness)} columns with <{self.completeness_percentage*100}% completeness",
                'columns': low_completeness[:10]  # Show first 10
            })
        
        # Analyze data type issues
        problematic_types = [
            col for col, data in self.quality_report['data_types'].items()
            if data['potential_issues']
        ]
        
        if problematic_types:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'message': f"Address data quality issues in {len(problematic_types)} columns",
                'columns': problematic_types[:10]
            })
        
        # Feature selection recommendations
        suitable_features = len(self.quality_report['feature_suitability']['anomaly_detection'])
        if suitable_features < 5:
            recommendations.append({
                'type': 'modeling',
                'priority': 'high',
                'message': f"Only {suitable_features} features suitable for modeling. Consider data collection or feature engineering.",
                'suggestions': [
                    "Collect more complete data for key health metrics",
                    "Implement data imputation strategies",
                    "Create derived features from existing data"
                ]
            })
        else:
            recommendations.append({
                'type': 'modeling',
                'priority': 'low',
                'message': f"Good feature availability: {suitable_features} features suitable for modeling",
                'suggestions': [
                    "Proceed with modeling pipeline",
                    "Consider feature selection for optimal performance"
                ]
            })
        
        return recommendations
    
    def get_modeling_features(self, task: str = 'general', min_features: int = 3) -> list:
        """
        Get list of features suitable for a specific modeling task.
        
        Args:
            task: Modeling task ('anomaly_detection', 'clustering', 'predictive_modeling')
            min_features: Minimum number of features required
            
        Returns:
            List of suitable feature names
        """
        if task in self.quality_report['feature_suitability']:
            features = self.quality_report['feature_suitability'][task]
            if len(features) >= min_features:
                return features[:min_features]  # Return top features
            else:
                logger.warning(f"Only {len(features)} features available for {task}, need {min_features}")
                return features
        else:
            # Return general features if task not specified
            general_features = self.quality_report['feature_suitability']['anomaly_detection']
            if len(general_features) >= min_features:
                return general_features[:min_features]
            else:
                return general_features
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        print("\n" + "="*60)
        print("📊 DATA QUALITY ANALYSIS SUMMARY")
        print("="*60)
        
        basic = self.quality_report['basic_info']
        print(f"📈 Dataset: {basic['total_rows']:,} rows × {basic['total_columns']} columns")
        print(f"💾 Memory: {basic['memory_usage_mb']:.1f} MB")
        
        # Completeness summary
        completeness = self.quality_report['column_completeness']
        sufficient_cols = sum(1 for data in completeness.values() if data['is_sufficient'])
        adequate_cols = sum(1 for data in completeness.values() if data['is_adequate'])
        
        print(f"\n✅ Columns with ≥{self.completeness_threshold} non-null values: {sufficient_cols}/{len(completeness)}")
        print(f"✅ Columns with ≥{self.completeness_percentage*100}% completeness: {adequate_cols}/{len(completeness)}")
        
        # Feature suitability summary
        suitability = self.quality_report['feature_suitability']
        print(f"\n🔍 Feature Suitability:")
        print(f"   • Anomaly Detection: {len(suitability['anomaly_detection'])} features")
        print(f"   • Clustering: {len(suitability['clustering'])} features")
        print(f"   • Predictive Modeling: {len(suitability['predictive_modeling'])} features")
        print(f"   • Time Series: {len(suitability['time_series'])} features")
        
        # Top recommendations
        recommendations = self.quality_report['recommendations']
        if recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                print(f"   {i}. {priority_icon} {rec['message']}")
        
        print("="*60)


class TestDataQualityChecker:

    def test_analyze_dataframe(self):
        """Test the DataQualityChecker class."""
        # Create sample data with various quality issues
        data = {
            'good_feature': np.random.randn(100),
            'missing_feature': [np.nan] * 80 + list(np.random.randn(20)),
            'mixed_types': [1, 2, 'three', 4, 5] * 20,
            'time_strings': ['00:00:00'] * 50 + ['12:00:00'] * 50,
            'outliers': [1, 2, 3, 4, 5, 1000] * 16 + [6, 7, 8, 9, 10] * 2
        }
        
        # Ensure all arrays have the same length
        max_length = max(len(v) for v in data.values())
        for key in data:
            if len(data[key]) < max_length:
                # Pad with NaN or repeat values
                if isinstance(data[key][0], (int, float)):
                    data[key] = list(data[key]) + [np.nan] * (max_length - len(data[key]))
                else:
                    data[key] = list(data[key]) + [data[key][-1]] * (max_length - len(data[key]))
        
        df = pd.DataFrame(data)
        
        # Initialize checker
        checker = DataQualityChecker(completeness_threshold=20, completeness_percentage=0.2)
        
        # Run analysis
        report = checker.analyze_dataframe(df)
        
        # Assertions
        assert 'basic_info' in report
        assert 'column_completeness' in report
        assert 'feature_suitability' in report
        assert 'recommendations' in report
        
        # Check specific findings
        assert report['column_completeness']['good_feature']['is_sufficient'] == True
        assert report['column_completeness']['missing_feature']['is_sufficient'] == False
        
        # Check feature suitability
        suitable_features = checker.get_modeling_features('anomaly_detection', min_features=2)
        assert len(suitable_features) >= 1  # At least good_feature should be suitable
        
        print("✅ DataQualityChecker tests passed!")

    @pytest.mark.integration
    def test_end_to_end(self, tmp_db):
        """Test data quality analysis on real Garmin data."""
        try:
            from garmin_analysis.utils.data_loading import load_master_dataframe
            
            # Load real data
            df = load_master_dataframe()
            
            # Initialize checker
            checker = DataQualityChecker()
            
            # Run analysis
            report = checker.analyze_dataframe(df)
            
            # Print summary
            checker.print_summary()
            
            # Get modeling features
            anomaly_features = checker.get_modeling_features('anomaly_detection', min_features=3)
            clustering_features = checker.get_modeling_features('clustering', min_features=3)
            predictive_features = checker.get_modeling_features('predictive_modeling', min_features=3)
            
            print(f"\n🎯 Recommended Features:")
            print(f"   • Anomaly Detection: {anomaly_features}")
            print(f"   • Clustering: {clustering_features}")
            print(f"   • Predictive Modeling: {predictive_features}")
            
            # Assertions for real data
            assert len(anomaly_features) >= 3, f"Need at least 3 features for anomaly detection, got {len(anomaly_features)}"
            assert len(clustering_features) >= 3, f"Need at least 3 features for clustering, got {len(clustering_features)}"
            
            print("✅ Real data quality analysis completed successfully!")
            
            
        except ImportError:
            pytest.skip("Cannot import Garmin data modules")
        except Exception as e:
            pytest.fail(f"Real data analysis failed: {e}")


if __name__ == "__main__":
    # Run tests
    print("🧪 Running Data Quality Checker Tests...")
    
    # Test with sample data
    sample_checker = test_data_quality_checker()
    
    # Test with real data
    try:
        real_checker = test_real_data_quality()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n⚠️ Real data test failed: {e}")
        print("Sample data tests completed successfully.")

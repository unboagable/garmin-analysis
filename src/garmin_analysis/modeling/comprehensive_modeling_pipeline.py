"""
Comprehensive Modeling Pipeline for Garmin Health Data

This module orchestrates all modeling activities:
- Anomaly detection
- Clustering analysis
- Predictive modeling
- Results compilation and reporting
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

from garmin_analysis.utils import load_master_dataframe
from garmin_analysis.utils_cleaning import clean_data
from garmin_analysis.modeling.enhanced_anomaly_detection import EnhancedAnomalyDetector
from garmin_analysis.modeling.enhanced_clustering import EnhancedClusterer
from garmin_analysis.modeling.predictive_modeling import HealthPredictor
from garmin_analysis.features.coverage import filter_by_24h_coverage

logger = logging.getLogger(__name__)

class ComprehensiveModelingPipeline:
    """Orchestrates comprehensive modeling analysis."""
    
    def __init__(self, output_dir: Path = None, random_state: int = 42):
        self.output_dir = output_dir or Path("modeling_results")
        self.random_state = random_state
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize modeling components
        self.anomaly_detector = EnhancedAnomalyDetector(random_state=random_state)
        self.clusterer = EnhancedClusterer(random_state=random_state)
        self.predictor = HealthPredictor(random_state=random_state)
        
    def run_anomaly_detection(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Run anomaly detection analysis."""
        logger.info("Starting anomaly detection analysis...")
        
        try:
            results = self.anomaly_detector.run_comprehensive_analysis(
                df, 
                tune_hyperparameters=kwargs.get('tune_hyperparameters', True),
                output_dir=self.output_dir / "anomaly_detection"
            )
            
            self.results['anomaly_detection'] = results
            logger.info("Anomaly detection completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {}
    
    def run_clustering_analysis(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Run clustering analysis."""
        logger.info("Starting clustering analysis...")
        
        try:
            results = self.clusterer.run_comprehensive_clustering(
                df,
                algorithms=kwargs.get('algorithms', ['kmeans', 'gaussian_mixture', 'hierarchical']),
                tune_hyperparameters=kwargs.get('tune_hyperparameters', True),
                output_dir=self.output_dir / "clustering"
            )
            
            self.results['clustering'] = results
            logger.info("Clustering analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return {}
    
    def run_predictive_modeling(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Run predictive modeling analysis."""
        logger.info("Starting predictive modeling analysis...")
        
        try:
            results = self.predictor.run_comprehensive_prediction(
                df,
                target_col=kwargs.get('target_col', 'score'),
                algorithms=kwargs.get('algorithms', ['random_forest', 'gradient_boosting', 'linear_models']),
                tune_hyperparameters=kwargs.get('tune_hyperparameters', True),
                output_dir=self.output_dir / "predictive_modeling"
            )
            
            self.results['predictive_modeling'] = results
            logger.info("Predictive modeling completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Predictive modeling failed: {e}")
            return {}
    
    def create_feature_analysis(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive feature analysis."""
        logger.info("Creating feature analysis...")
        
        df_clean = clean_data(df)
        
        # Basic statistics
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        feature_stats = {}
        
        for col in numeric_cols:
            values = df_clean[col].dropna()
            if len(values) > 0:
                feature_stats[col] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median()),
                    'missing_pct': float(df_clean[col].isna().sum() / len(df_clean) * 100)
                }
        
        # Correlation analysis
        corr_matrix = df_clean[numeric_cols].corr()
        
        # Feature importance ranking (using variance)
        feature_variance = {}
        for col in numeric_cols:
            values = df_clean[col].dropna()
            if len(values) > 0:
                feature_variance[col] = float(values.var())
        
        # Sort by variance
        sorted_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)
        
        feature_analysis = {
            'basic_stats': feature_stats,
            'correlation_matrix': corr_matrix.to_dict(),
            'feature_ranking': sorted_features,
            'n_features': len(numeric_cols),
            'n_samples': len(df_clean)
        }
        
        self.results['feature_analysis'] = feature_analysis
        return feature_analysis
    
    def create_modeling_summary(self) -> Dict:
        """Create comprehensive modeling summary."""
        logger.info("Creating modeling summary...")
        
        summary = {
            'timestamp': self.timestamp,
            'overview': {},
            'anomaly_detection_summary': {},
            'clustering_summary': {},
            'predictive_modeling_summary': {},
            'feature_analysis_summary': {},
            'recommendations': []
        }
        
        # Overview
        if 'feature_analysis' in self.results:
            fa = self.results['feature_analysis']
            summary['overview'] = {
                'total_samples': fa['n_samples'],
                'total_features': fa['n_features'],
                'analysis_timestamp': self.timestamp
            }
        
        # Anomaly detection summary
        if 'anomaly_detection' in self.results:
            ad = self.results['anomaly_detection']
            summary['anomaly_detection_summary'] = {
                'n_anomalies': ad.get('ensemble_results', {}).get('ensemble', {}).get('n_anomalies', 0),
                'anomaly_rate': ad.get('ensemble_results', {}).get('ensemble', {}).get('anomaly_score', 0),
                'best_algorithm': 'ensemble',
                'n_samples_analyzed': ad.get('n_samples', 0)
            }
        
        # Clustering summary
        if 'clustering' in self.results:
            cl = self.results['clustering']
            summary['clustering_summary'] = {
                'best_algorithm': cl.get('best_algorithm', 'unknown'),
                'n_clusters': cl.get('best_model', {}).get('evaluation', {}).get('n_clusters', 0),
                'silhouette_score': cl.get('best_model', {}).get('evaluation', {}).get('silhouette_score', 0),
                'n_samples_clustered': cl.get('n_samples', 0)
            }
        
        # Predictive modeling summary
        if 'predictive_modeling' in self.results:
            pm = self.results['predictive_modeling']
            summary['predictive_modeling_summary'] = {
                'best_algorithm': pm.get('best_algorithm', 'unknown'),
                'best_mse': pm.get('best_score', float('inf')),
                'n_features_used': pm.get('n_features', 0),
                'train_test_split': f"{pm.get('train_size', 0)}/{pm.get('test_size', 0)}"
            }
        
        # Feature analysis summary
        if 'feature_analysis' in self.results:
            fa = self.results['feature_analysis']
            summary['feature_analysis_summary'] = {
                'top_features': [f[0] for f in fa.get('feature_ranking', [])[:5]],
                'features_with_missing_data': [col for col, stats in fa.get('basic_stats', {}).items() 
                                             if stats.get('missing_pct', 0) > 20]
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        summary['recommendations'] = recommendations
        
        self.results['summary'] = summary
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Feature quality recommendations
        if 'feature_analysis' in self.results:
            fa = self.results['feature_analysis']
            high_missing_features = [col for col, stats in fa.get('basic_stats', {}).items() 
                                   if stats.get('missing_pct', 0) > 30]
            
            if high_missing_features:
                recommendations.append(f"Consider removing or imputing features with >30% missing data: {high_missing_features[:3]}")
        
        # Anomaly detection recommendations
        if 'anomaly_detection' in self.results:
            ad = self.results['anomaly_detection']
            anomaly_rate = ad.get('ensemble_results', {}).get('ensemble', {}).get('anomaly_score', 0)
            
            if anomaly_rate > 0.1:
                recommendations.append(f"High anomaly rate detected ({anomaly_rate:.1%}). Review data quality and consider additional cleaning.")
            elif anomaly_rate < 0.01:
                recommendations.append("Very low anomaly rate. Consider adjusting detection sensitivity.")
        
        # Clustering recommendations
        if 'clustering' in self.results:
            cl = self.results['clustering']
            silhouette_score = cl.get('best_model', {}).get('evaluation', {}).get('silhouette_score', 0)
            
            if silhouette_score < 0.2:
                recommendations.append("Low clustering quality (silhouette score < 0.2). Consider feature engineering or different algorithms.")
            elif silhouette_score > 0.6:
                recommendations.append("Excellent clustering quality. Clusters are well-separated and meaningful.")
        
        # Predictive modeling recommendations
        if 'predictive_modeling' in self.results:
            pm = self.results['predictive_modeling']
            best_r2 = max([metrics.get('r2', 0) for metrics in pm.get('all_results', {}).values()])
            
            if best_r2 < 0.3:
                recommendations.append("Low predictive power (R² < 0.3). Consider feature engineering, more data, or different algorithms.")
            elif best_r2 > 0.7:
                recommendations.append("Strong predictive power. Model captures most variance in the target variable.")
        
        # General recommendations
        if 'feature_analysis' in self.results:
            fa = self.results['feature_analysis']
            if fa.get('n_features', 0) > 50:
                recommendations.append("High-dimensional dataset. Consider dimensionality reduction techniques.")
        
        if not recommendations:
            recommendations.append("Analysis completed successfully. No specific recommendations at this time.")
        
        return recommendations
    
    def save_results(self) -> List[str]:
        """Save all results to files."""
        logger.info("Saving modeling results...")
        
        saved_files = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        if 'summary' in self.results:
            summary_path = self.output_dir / f"modeling_summary_{self.timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(self.results['summary'], f, indent=2, default=str)
            saved_files.append(str(summary_path))
        
        # Save detailed results
        for analysis_type, results in self.results.items():
            if analysis_type != 'summary' and results:
                result_path = self.output_dir / f"{analysis_type}_results_{self.timestamp}.json"
                
                # Convert numpy types to Python types for JSON serialization
                serializable_results = self._make_json_serializable(results)
                
                with open(result_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2, default=str)
                saved_files.append(str(result_path))
        
        # Create markdown report
        markdown_path = self.output_dir / f"modeling_report_{self.timestamp}.md"
        self._create_markdown_report(markdown_path)
        saved_files.append(str(markdown_path))
        
        logger.info(f"Results saved to {len(saved_files)} files")
        return saved_files
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _create_markdown_report(self, output_path: Path):
        """Create a comprehensive markdown report."""
        if 'summary' not in self.results:
            return
        
        summary = self.results['summary']
        
        with open(output_path, 'w') as f:
            f.write("# 🧠 Comprehensive Modeling Analysis Report\n\n")
            f.write(f"**Generated:** {summary['timestamp']}\n\n")
            
            # Overview
            f.write("## 📊 Analysis Overview\n\n")
            f.write(f"- **Total Samples:** {summary['overview'].get('total_samples', 'N/A')}\n")
            f.write(f"- **Total Features:** {summary['overview'].get('total_features', 'N/A')}\n")
            f.write(f"- **Analysis Timestamp:** {summary['overview'].get('analysis_timestamp', 'N/A')}\n\n")
            
            # Anomaly Detection
            if summary['anomaly_detection_summary']:
                f.write("## 🚨 Anomaly Detection Results\n\n")
                ad = summary['anomaly_detection_summary']
                f.write(f"- **Anomalies Detected:** {ad.get('n_anomalies', 'N/A')}\n")
                f.write(f"- **Anomaly Rate:** {ad.get('anomaly_rate', 'N/A'):.2%}\n")
                f.write(f"- **Best Algorithm:** {ad.get('best_algorithm', 'N/A')}\n")
                f.write(f"- **Samples Analyzed:** {ad.get('n_samples_analyzed', 'N/A')}\n\n")
            
            # Clustering
            if summary['clustering_summary']:
                f.write("## 🔗 Clustering Analysis Results\n\n")
                cl = summary['clustering_summary']
                f.write(f"- **Best Algorithm:** {cl.get('best_algorithm', 'N/A')}\n")
                f.write(f"- **Number of Clusters:** {cl.get('n_clusters', 'N/A')}\n")
                f.write(f"- **Silhouette Score:** {cl.get('silhouette_score', 'N/A'):.3f}\n")
                f.write(f"- **Samples Clustered:** {cl.get('n_samples_clustered', 'N/A')}\n\n")
            
            # Predictive Modeling
            if summary['predictive_modeling_summary']:
                f.write("## 🔮 Predictive Modeling Results\n\n")
                pm = summary['predictive_modeling_summary']
                f.write(f"- **Best Algorithm:** {pm.get('best_algorithm', 'N/A')}\n")
                f.write(f"- **Best MSE:** {pm.get('best_mse', 'N/A'):.4f}\n")
                f.write(f"- **Features Used:** {pm.get('n_features_used', 'N/A')}\n")
                f.write(f"- **Train/Test Split:** {pm.get('train_test_split', 'N/A')}\n\n")
            
            # Feature Analysis
            if summary['feature_analysis_summary']:
                f.write("## 🔍 Feature Analysis Summary\n\n")
                fa = summary['feature_analysis_summary']
                f.write(f"- **Top Features:** {', '.join(fa.get('top_features', []))}\n")
                if fa.get('features_with_missing_data'):
                    f.write(f"- **Features with Missing Data:** {', '.join(fa['features_with_missing_data'][:5])}\n")
                f.write("\n")
            
            # Recommendations
            if summary['recommendations']:
                f.write("## 💡 Recommendations\n\n")
                for i, rec in enumerate(summary['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*Report generated automatically by Comprehensive Modeling Pipeline*\n")
    
    def run_full_pipeline(self, df: pd.DataFrame, filter_24h_coverage=False, max_gap_minutes=2, day_edge_tolerance_minutes=2, **kwargs) -> Dict:
        """Run the complete modeling pipeline."""
        logger.info("Starting comprehensive modeling pipeline...")
        
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply 24-hour coverage filtering if requested
            if filter_24h_coverage:
                logger.info("Filtering to days with 24-hour continuous coverage...")
                max_gap = pd.Timedelta(minutes=max_gap_minutes)
                day_edge_tolerance = pd.Timedelta(minutes=day_edge_tolerance_minutes)
                df = filter_by_24h_coverage(df, max_gap=max_gap, day_edge_tolerance=day_edge_tolerance)
                logger.info(f"After 24h coverage filtering: {len(df)} days remaining")
            
            # Run all analyses
            self.run_anomaly_detection(df, **kwargs)
            self.run_clustering_analysis(df, **kwargs)
            self.run_predictive_modeling(df, **kwargs)
            self.create_feature_analysis(df)
            
            # Create summary and save results
            self.create_modeling_summary()
            saved_files = self.save_results()
            
            logger.info("Comprehensive modeling pipeline completed successfully!")
            logger.info(f"Results saved to: {saved_files}")
            
            return {
                'status': 'success',
                'results': self.results,
                'saved_files': saved_files,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': self.results
            }

def main():
    """Main function to run the comprehensive modeling pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive modeling pipeline on Garmin data')
    parser.add_argument('--filter-24h-coverage', action='store_true', 
                       help='Filter to only days with 24-hour continuous coverage')
    parser.add_argument('--max-gap', type=int, default=2,
                       help='Maximum gap in minutes for continuous coverage (default: 2)')
    parser.add_argument('--day-edge-tolerance', type=int, default=2,
                       help='Day edge tolerance in minutes for continuous coverage (default: 2)')
    parser.add_argument('--target-col', type=str, default='score',
                       help='Target column for predictive modeling (default: score)')
    parser.add_argument('--tune-hyperparameters', action='store_true', default=True,
                       help='Tune hyperparameters (default: True)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_master_dataframe()
        
        # Initialize pipeline
        pipeline = ComprehensiveModelingPipeline(
            output_dir=Path("modeling_results"),
            random_state=42
        )
        
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            df,
            filter_24h_coverage=args.filter_24h_coverage,
            max_gap_minutes=args.max_gap,
            day_edge_tolerance_minutes=args.day_edge_tolerance,
            tune_hyperparameters=args.tune_hyperparameters,
            target_col=args.target_col
        )
        
        if results['status'] == 'success':
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"Results saved to: {results['output_dir']}")
        else:
            logger.error(f"❌ Pipeline failed: {results['error']}")
        
    except Exception as e:
        logger.error(f"Failed to run modeling pipeline: {e}")

if __name__ == "__main__":
    main()

"""
Standalone data quality analysis script for Garmin health data.

This script provides comprehensive analysis of:
- Data completeness and missing values
- Feature suitability for different modeling tasks
- Data type issues and quality problems
- Automated recommendations for data improvement

Usage:
    python src/features/data_quality_analysis.py
    python -m src.features.data_quality_analysis
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GarminDataQualityAnalyzer:
    """Specialized data quality analyzer for Garmin health data."""
    
    def __init__(self, output_dir: str = "data_quality_reports"):
        """
        Initialize the Garmin data quality analyzer.
        
        Args:
            output_dir: Directory to save analysis reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_results = {}
        
        # Garmin-specific feature categories
        self.feature_categories = {
            'activity': ['steps', 'distance', 'calories', 'floors', 'activity'],
            'heart_rate': ['hr', 'rhr', 'heart_rate', 'resting_heart_rate'],
            'sleep': ['sleep', 'rem', 'deep', 'awake', 'light_sleep'],
            'stress': ['stress', 'recovery', 'body_battery'],
            'fitness': ['vo2_max', 'training_effect', 'fitness_age'],
            'body': ['weight', 'bmi', 'body_fat', 'muscle_mass'],
            'nutrition': ['hydration', 'sweat_loss', 'calories_consumed'],
            'time': ['time', 'duration', 'start', 'end', 'date']
        }
    
    def analyze_garmin_data(self, df: pd.DataFrame) -> dict:
        """
        Comprehensive analysis of Garmin health data quality.
        
        Args:
            df: Garmin health data dataframe
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info(f"Starting comprehensive Garmin data quality analysis...")
        logger.info(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
        
        try:
            # Basic dataset information
            logger.info("Analyzing dataset info...")
            self.analysis_results['dataset_info'] = self._analyze_dataset_info(df)
            
            # Column completeness analysis
            logger.info("Analyzing completeness...")
            self.analysis_results['completeness'] = self._analyze_completeness(df)
            
            # Data quality issues
            logger.info("Identifying quality issues...")
            self.analysis_results['quality_issues'] = self._identify_quality_issues(df)
            
            # Feature categorization (after completeness is available)
            logger.info("Categorizing features...")
            self.analysis_results['feature_categories'] = self._categorize_features(df)
            
            # Modeling suitability
            logger.info("Analyzing modeling suitability...")
            self.analysis_results['modeling_suitability'] = self._analyze_modeling_suitability(df)
            
            # Recommendations
            logger.info("Generating recommendations...")
            self.analysis_results['recommendations'] = self._generate_recommendations()
            
            # Timestamp
            self.analysis_results['analysis_timestamp'] = datetime.now().isoformat()
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _analyze_dataset_info(self, df: pd.DataFrame) -> dict:
        """Analyze basic dataset information."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': self._get_date_range(df),
            'unique_days': self._count_unique_days(df)
        }
    
    def _get_date_range(self, df: pd.DataFrame) -> dict:
        """Get the date range of the dataset."""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        
        if date_columns:
            for col in date_columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        min_date = df[col].min()
                        max_date = df[col].max()
                        return {
                            'column': col,
                            'start_date': str(min_date),
                            'end_date': str(max_date),
                            'duration_days': (max_date - min_date).days
                        }
                except:
                    continue
        
        return {'column': None, 'start_date': None, 'end_date': None, 'duration_days': None}
    
    def _count_unique_days(self, df: pd.DataFrame) -> int:
        """Count unique days in the dataset."""
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        
        for col in date_columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return df[col].nunique()
            except:
                continue
        
        return 0
    
    def _analyze_completeness(self, df: pd.DataFrame) -> dict:
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
                'completeness_level': self._get_completeness_level(completeness_pct),
                'is_sufficient_for_modeling': bool(non_null_count >= 50),  # At least 50 non-null values
                'is_adequate_for_analysis': bool(completeness_pct >= 0.1)  # At least 10% complete
            }
        
        return completeness_data
    
    def _get_completeness_level(self, completeness_pct: float) -> str:
        """Get human-readable completeness level."""
        if completeness_pct >= 0.9:
            return "Excellent (≥90%)"
        elif completeness_pct >= 0.7:
            return "Good (70-89%)"
        elif completeness_pct >= 0.5:
            return "Fair (50-69%)"
        elif completeness_pct >= 0.3:
            return "Poor (30-49%)"
        elif completeness_pct >= 0.1:
            return "Very Poor (10-29%)"
        else:
            return "Critical (<10%)"
    
    def _categorize_features(self, df: pd.DataFrame) -> dict:
        """Categorize features by Garmin health domains."""
        categorized_features = {category: [] for category in self.feature_categories.keys()}
        categorized_features['other'] = []
        
        for col in df.columns:
            col_lower = col.lower()
            categorized = False
            
            for category, keywords in self.feature_categories.items():
                if any(keyword in col_lower for keyword in keywords):
                    categorized_features[category].append(col)
                    categorized = True
                    break
            
            if not categorized:
                categorized_features['other'].append(col)
        
        # Add counts and completeness info
        for category, features in categorized_features.items():
            if features:
                category_completeness = []
                for feature in features:
                    completeness = self.analysis_results['completeness'][feature]
                    category_completeness.append({
                        'feature': feature,
                        'completeness_pct': completeness['completeness_percentage'],
                        'completeness_level': completeness['completeness_level'],
                        'is_sufficient': completeness['is_sufficient_for_modeling']
                    })
                
                # Sort by completeness
                category_completeness.sort(key=lambda x: x['completeness_pct'], reverse=True)
                categorized_features[category] = category_completeness
        
        return categorized_features
    
    def _identify_quality_issues(self, df: pd.DataFrame) -> dict:
        """Identify data quality issues."""
        issues = {
            'mixed_types': [],
            'time_strings': [],
            'extreme_outliers': [],
            'inconsistent_formats': [],
            'duplicate_columns': []
        }
        
        for col in df.columns:
            # Check for mixed types
            if df[col].dtype == 'object':
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    type_counts = non_null.apply(type).value_counts()
                    if len(type_counts) > 1:
                        issues['mixed_types'].append({
                            'column': col,
                            'type_distribution': type_counts.to_dict()
                        })
            
            # Check for time-like strings
            if df[col].dtype == 'object':
                time_patterns = ['00:00:00', ':', 'AM', 'PM', 'am', 'pm']
                sample_values = df[col].dropna().astype(str).head(10)
                if any(any(pattern in str(val) for pattern in time_patterns) for val in sample_values):
                    issues['time_strings'].append({
                        'column': col,
                        'sample_values': sample_values.tolist()
                    })
            
            # Check for extreme outliers in numeric columns
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype.kind not in 'b':  # Exclude boolean
                non_null = df[col].dropna()
                if len(non_null) > 10:
                    try:
                        q1, q3 = non_null.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        if iqr > 0:
                            lower_bound = q1 - 3 * iqr  # More conservative than 1.5
                            upper_bound = q3 + 3 * iqr
                            outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
                            if len(outliers) > 0:
                                issues['extreme_outliers'].append({
                                    'column': col,
                                    'outlier_count': int(len(outliers)),
                                    'outlier_percentage': float(len(outliers) / len(non_null)),
                                    'outlier_range': [float(outliers.min()), float(outliers.max())]
                                })
                    except Exception as e:
                        logger.warning(f"Could not analyze outliers for column {col}: {e}")
                        continue
        
        # Check for duplicate columns (same data, different names)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        potential_duplicates = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if df[col1].equals(df[col2]):
                    potential_duplicates.append([col1, col2])
        
        if potential_duplicates:
            issues['duplicate_columns'] = potential_duplicates
        
        return issues
    
    def _analyze_modeling_suitability(self, df: pd.DataFrame) -> dict:
        """Analyze which features are suitable for different modeling tasks."""
        suitability = {
            'anomaly_detection': [],
            'clustering': [],
            'predictive_modeling': [],
            'time_series': [],
            'unsuitable': []
        }
        
        for col in df.columns:
            completeness = self.analysis_results['completeness'][col]
            
            # Check if feature is suitable for modeling
            if (completeness['is_sufficient_for_modeling'] and 
                pd.api.types.is_numeric_dtype(df[col])):
                
                # Categorize by feature type
                if any(time_keyword in col.lower() for time_keyword in ['time', 'date', 'duration']):
                    suitability['time_series'].append(col)
                elif any(target_keyword in col.lower() for target_keyword in ['score', 'target', 'outcome']):
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
        
        # Completeness recommendations
        low_completeness = [
            col for col, data in self.analysis_results['completeness'].items()
            if not data['is_adequate_for_analysis']
        ]
        
        if low_completeness:
            recommendations.append({
                'type': 'completeness',
                'priority': 'high',
                'message': f"Address data completeness issues in {len(low_completeness)} columns",
                'details': f"Columns with <10% completeness: {low_completeness[:10]}",
                'actions': [
                    "Review data collection processes",
                    "Implement data imputation strategies",
                    "Consider removing columns with <5% completeness"
                ]
            })
        
        # Data quality recommendations
        quality_issues = self.analysis_results['quality_issues']
        total_issues = sum(len(issues) for issues in quality_issues.values())
        
        if total_issues > 0:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'message': f"Address {total_issues} data quality issues",
                'details': f"Issues found: Mixed types ({len(quality_issues['mixed_types'])}), Time strings ({len(quality_issues['time_strings'])}), Outliers ({len(quality_issues['extreme_outliers'])})",
                'actions': [
                    "Convert time strings to proper datetime format",
                    "Standardize data types across columns",
                    "Review and handle extreme outliers"
                ]
            })
        
        # Modeling recommendations
        suitability = self.analysis_results['modeling_suitability']
        suitable_features = len(suitability['anomaly_detection'])
        
        if suitable_features < 5:
            recommendations.append({
                'type': 'modeling',
                'priority': 'high',
                'message': f"Insufficient features for modeling: only {suitable_features} suitable features",
                'details': f"Need at least 5 features for robust modeling",
                'actions': [
                    "Focus on data collection for key health metrics",
                    "Implement feature engineering from existing data",
                    "Consider data imputation for important features"
                ]
            })
        else:
            recommendations.append({
                'type': 'modeling',
                'priority': 'low',
                'message': f"Good feature availability: {suitable_features} features suitable for modeling",
                'details': f"Proceed with modeling pipeline",
                'actions': [
                    "Run comprehensive modeling analysis",
                    "Consider feature selection for optimal performance",
                    "Monitor data quality over time"
                ]
            })
        
        return recommendations
    
    def print_summary(self):
        """Print a comprehensive summary of the analysis."""
        print("\n" + "="*80)
        print("🏃‍♂️ GARMIN DATA QUALITY ANALYSIS SUMMARY")
        print("="*80)
        
        # Dataset info
        info = self.analysis_results['dataset_info']
        print(f"📊 Dataset Overview:")
        print(f"   • Size: {info['total_rows']:,} rows × {info['total_columns']} columns")
        print(f"   • Memory: {info['memory_usage_mb']:.1f} MB")
        if info['date_range']['column']:
            print(f"   • Date Range: {info['date_range']['start_date']} to {info['date_range']['end_date']}")
            print(f"   • Duration: {info['date_range']['duration_days']} days")
        print(f"   • Unique Days: {info['unique_days']}")
        
        # Completeness summary
        completeness = self.analysis_results['completeness']
        sufficient_cols = sum(1 for data in completeness.values() if data['is_sufficient_for_modeling'])
        adequate_cols = sum(1 for data in completeness.values() if data['is_adequate_for_analysis'])
        
        print(f"\n✅ Data Completeness:")
        print(f"   • Columns with ≥50 non-null values: {sufficient_cols}/{len(completeness)}")
        print(f"   • Columns with ≥10% completeness: {adequate_cols}/{len(completeness)}")
        
        # Feature categories
        categories = self.analysis_results['feature_categories']
        print(f"\n🏷️  Feature Categories:")
        for category, features in categories.items():
            if features:
                sufficient_count = sum(1 for f in features if f['is_sufficient'])
                print(f"   • {category.title()}: {len(features)} features ({sufficient_count} sufficient)")
        
        # Modeling suitability
        suitability = self.analysis_results['modeling_suitability']
        print(f"\n🔍 Modeling Suitability:")
        print(f"   • Anomaly Detection: {len(suitability['anomaly_detection'])} features")
        print(f"   • Clustering: {len(suitability['clustering'])} features")
        print(f"   • Predictive Modeling: {len(suitability['predictive_modeling'])} features")
        print(f"   • Time Series: {len(suitability['time_series'])} features")
        
        # Top recommendations
        recommendations = self.analysis_results['recommendations']
        if recommendations:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                print(f"   {i}. {priority_icon} {rec['message']}")
                if 'details' in rec:
                    print(f"      Details: {rec['details']}")
        
        print("="*80)
    
    def save_report(self, filename: str = None):
        """Save the analysis report to JSON and Markdown files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"garmin_data_quality_report_{timestamp}"
        
        # Save JSON report
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save Markdown report
        md_path = self.output_dir / f"{filename}.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_report())
        
        logger.info(f"Reports saved to:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")
        
        return json_path, md_path
    
    def _generate_markdown_report(self) -> str:
        """Generate a Markdown report from the analysis results."""
        md_content = []
        
        # Header
        md_content.append("# 🏃‍♂️ Garmin Data Quality Analysis Report")
        md_content.append(f"**Generated:** {self.analysis_results['analysis_timestamp']}")
        md_content.append("")
        
        # Dataset Overview
        info = self.analysis_results['dataset_info']
        md_content.append("## 📊 Dataset Overview")
        md_content.append(f"- **Total Rows:** {info['total_rows']:,}")
        md_content.append(f"- **Total Columns:** {info['total_columns']}")
        md_content.append(f"- **Memory Usage:** {info['memory_usage_mb']:.1f} MB")
        if info['date_range']['column']:
            md_content.append(f"- **Date Range:** {info['date_range']['start_date']} to {info['date_range']['end_date']}")
            md_content.append(f"- **Duration:** {info['date_range']['duration_days']} days")
        md_content.append("")
        
        # Completeness Summary
        completeness = self.analysis_results['completeness']
        sufficient_cols = sum(1 for data in completeness.values() if data['is_sufficient_for_modeling'])
        adequate_cols = sum(1 for data in completeness.values() if data['is_adequate_for_analysis'])
        
        md_content.append("## ✅ Data Completeness Summary")
        md_content.append(f"- **Columns with ≥50 non-null values:** {sufficient_cols}/{len(completeness)}")
        md_content.append(f"- **Columns with ≥10% completeness:** {adequate_cols}/{len(completeness)}")
        md_content.append("")
        
        # Feature Categories
        categories = self.analysis_results['feature_categories']
        md_content.append("## 🏷️ Feature Categories")
        for category, features in categories.items():
            if features:
                sufficient_count = sum(1 for f in features if f['is_sufficient'])
                md_content.append(f"### {category.title()}")
                md_content.append(f"- **Total Features:** {len(features)}")
                md_content.append(f"- **Sufficient for Modeling:** {sufficient_count}")
                md_content.append("")
        
        # Modeling Suitability
        suitability = self.analysis_results['modeling_suitability']
        md_content.append("## 🔍 Modeling Suitability")
        md_content.append(f"- **Anomaly Detection:** {len(suitability['anomaly_detection'])} features")
        md_content.append(f"- **Clustering:** {len(suitability['clustering'])} features")
        md_content.append(f"- **Predictive Modeling:** {len(suitability['predictive_modeling'])} features")
        md_content.append(f"- **Time Series:** {len(suitability['time_series'])} features")
        md_content.append("")
        
        # Recommendations
        recommendations = self.analysis_results['recommendations']
        if recommendations:
            md_content.append("## 💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                md_content.append(f"### {i}. {priority_icon} {rec['message']}")
                if 'details' in rec:
                    md_content.append(f"**Details:** {rec['details']}")
                if 'actions' in rec:
                    md_content.append("**Recommended Actions:**")
                    for action in rec['actions']:
                        md_content.append(f"- {action}")
                md_content.append("")
        
        return "\n".join(md_content)


def main():
    """Main function to run Garmin data quality analysis."""
    try:
        # Import Garmin data loading function
        from src.utils import load_master_dataframe
        
        print("🏃‍♂️ Starting Garmin Data Quality Analysis...")
        
        # Load data
        print("📥 Loading Garmin data...")
        df = load_master_dataframe()
        print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Initialize analyzer
        analyzer = GarminDataQualityAnalyzer()
        
        # Run analysis
        print("🔍 Running comprehensive analysis...")
        results = analyzer.analyze_garmin_data(df)
        
        # Print summary
        analyzer.print_summary()
        
        # Save reports
        print("\n💾 Saving reports...")
        json_path, md_path = analyzer.save_report()
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Reports saved to: {analyzer.output_dir}")
        
        return analyzer
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return None
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


if __name__ == "__main__":
    main()

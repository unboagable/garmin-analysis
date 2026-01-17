"""
Enhanced Anomaly Detection for Garmin Health Data

This module provides multiple anomaly detection algorithms with:
- Hyperparameter tuning
- Cross-validation
- Multiple evaluation metrics
- Ensemble methods
- Interpretable results
"""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.utils.data_filtering import standardize_features
from garmin_analysis.utils_cleaning import clean_data
from garmin_analysis.utils.imputation import impute_missing_values
from garmin_analysis.config import PLOTS_DIR

logger = logging.getLogger(__name__)

class EnhancedAnomalyDetector:
    """Enhanced anomaly detection with multiple algorithms and evaluation metrics."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                        imputation_strategy: str = 'median') -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns
            imputation_strategy: Strategy for handling missing values
                               - 'median': Fill with median (default, robust)
                               - 'mean': Fill with mean
                               - 'drop': Drop rows with missing values
                               - 'none': No imputation
        
        Returns:
            Tuple of (X_scaled, feature_names)
        """
        if feature_cols is None:
            # Use features that actually have data in the dataset
            feature_cols = [
                "hr_min", "hr_max", "rhr", "steps", "calories_total", "distance",
                "moderate_activity_time", "vigorous_activity_time", "intensity_time",
                "stress_avg", "resting_heart_rate", "hr_avg", "steps_avg_7d"
            ]
        
        # Clean data
        df_clean = clean_data(df)
        
        # Filter to features that exist, have sufficient data, and are numeric
        available_features = []
        for col in feature_cols:
            if col in df_clean.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    non_null_count = df_clean[col].notna().sum()
                    if non_null_count >= 50:  # Lower threshold to 50 non-null values
                        available_features.append(col)
                else:
                    logger.warning(f"Feature '{col}' is not numeric (type: {df_clean[col].dtype})")
        
        missing_features = set(feature_cols) - set(available_features)
        
        if missing_features:
            logger.warning(f"Features with insufficient data or wrong type: {missing_features}")
        
        if len(available_features) < 3:
            # Fallback to any numeric columns with sufficient data
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            fallback_features = []
            for col in numeric_cols:
                if col not in ['day', 'missing_score', 'missing_training_effect']:
                    non_null_count = df_clean[col].notna().sum()
                    if non_null_count >= 50:
                        fallback_features.append(col)
                        if len(fallback_features) >= 5:  # Get at least 5 features
                            break
            
            if len(fallback_features) >= 3:
                available_features = fallback_features
                logger.info(f"Using fallback features: {fallback_features}")
            else:
                raise ValueError(f"Need at least 3 features, but only {len(available_features)} available: {available_features}")
        
        # Check data completeness for each feature
        feature_completeness = {}
        for col in available_features:
            non_null_count = df_clean[col].notna().sum()
            total_count = len(df_clean)
            completeness = non_null_count / total_count
            feature_completeness[col] = completeness
            logger.info(f"Feature '{col}': {non_null_count}/{total_count} ({completeness:.1%} complete)")
        
        # Impute missing values using specified strategy
        df_clean = impute_missing_values(df_clean, available_features, strategy=imputation_strategy, copy=False)
        
        if df_clean.empty:
            raise ValueError("No data left after handling missing values")
        
        if len(df_clean) < 10:
            raise ValueError(f"Not enough rows for anomaly detection. Need at least 10, got {len(df_clean)}")
        
        # Extract features
        X = df_clean[available_features].values
        
        # Handle different scaling strategies
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Prepared {len(df_clean)} samples with {len(available_features)} features")
        logger.info(f"Feature completeness: {feature_completeness}")
        return X_scaled, available_features
    
    def fit_isolation_forest(self, X: np.ndarray, **kwargs) -> Dict:
        """Fit Isolation Forest with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'contamination': [0.01, 0.05, 0.1],
            'max_samples': ['auto', 100, 200]
        }
        
        # Use default parameters if no tuning
        if not kwargs.get('tune_hyperparameters', False):
            model = IsolationForest(
                n_estimators=100,
                contamination=0.05,
                random_state=self.random_state,
                **kwargs
            )
            model.fit(X)
        else:
            # Grid search for best parameters
            model = GridSearchCV(
                IsolationForest(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            model.fit(X)
            logger.info(f"Best Isolation Forest params: {model.best_params_}")
        
        self.models['isolation_forest'] = model
        return {'model': model, 'algorithm': 'Isolation Forest'}
    
    def fit_local_outlier_factor(self, X: np.ndarray, **kwargs) -> Dict:
        """Fit Local Outlier Factor."""
        param_grid = {
            'n_neighbors': [10, 20, 30],
            'contamination': [0.01, 0.05, 0.1]
        }
        
        if not kwargs.get('tune_hyperparameters', False):
            model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,
                novelty=False,
                **kwargs
            )
            model.fit_predict(X)  # LOF doesn't have a separate fit method
        else:
            # For LOF, we need to use a different approach since it doesn't support GridSearchCV directly
            best_score = -np.inf
            best_params = {}
            
            for n_neighbors in param_grid['n_neighbors']:
                for contamination in param_grid['contamination']:
                    model = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination,
                        novelty=False
                    )
                    labels = model.fit_predict(X)
                    score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'n_neighbors': n_neighbors, 'contamination': contamination}
            
            model = LocalOutlierFactor(**best_params, novelty=False)
            model.fit_predict(X)
            logger.info(f"Best LOF params: {best_params}")
        
        self.models['local_outlier_factor'] = model
        return {'model': model, 'algorithm': 'Local Outlier Factor'}
    
    def fit_one_class_svm(self, X: np.ndarray, **kwargs) -> Dict:
        """Fit One-Class SVM."""
        param_grid = {
            'nu': [0.1, 0.2, 0.3],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
        
        if not kwargs.get('tune_hyperparameters', False):
            model = OneClassSVM(
                nu=0.1,
                gamma='scale',
                random_state=self.random_state,
                **kwargs
            )
            model.fit(X)
        else:
            model = GridSearchCV(
                OneClassSVM(),
                param_grid,
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            model.fit(X)
            logger.info(f"Best One-Class SVM params: {model.best_params_}")
        
        self.models['one_class_svm'] = model
        return {'model': model, 'algorithm': 'One-Class SVM'}
    
    def ensemble_detection(self, X: np.ndarray, algorithms: List[str] = None) -> Dict:
        """Ensemble anomaly detection using multiple algorithms."""
        if algorithms is None:
            algorithms = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
        
        results = {}
        predictions = {}
        
        for algo in algorithms:
            if algo in self.models:
                model = self.models[algo]
                
                if algo == 'local_outlier_factor':
                    pred = model.fit_predict(X)
                else:
                    pred = model.predict(X)
                
                # Convert to binary (1 for normal, -1 for anomaly)
                pred_binary = (pred == 1).astype(int)
                predictions[algo] = pred_binary
                
                # Calculate anomaly score
                anomaly_score = 1 - np.mean(pred_binary)
                results[algo] = {
                    'anomaly_score': anomaly_score,
                    'n_anomalies': np.sum(pred_binary == 0),
                    'n_normal': np.sum(pred_binary == 1)
                }
        
        # Ensemble prediction (majority voting)
        if len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            
            results['ensemble'] = {
                'anomaly_score': 1 - np.mean(ensemble_binary),
                'n_anomalies': np.sum(ensemble_binary == 0),
                'n_normal': np.sum(ensemble_binary == 1)
            }
        
        return results
    
    def evaluate_clustering_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate clustering quality metrics."""
        if len(np.unique(labels)) < 2:
            return {'silhouette_score': -1, 'calinski_harabasz_score': -1}
        
        try:
            silhouette = silhouette_score(X, labels)
        except Exception as e:
            logger.debug(f"Error calculating silhouette score: {e}")
            silhouette = -1
        
        try:
            calinski_harabasz = calinski_harabasz_score(X, labels)
        except Exception as e:
            logger.debug(f"Error calculating Calinski-Harabasz score: {e}")
            calinski_harabasz = -1
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz
        }
    
    def create_visualizations(self, X: np.ndarray, labels: np.ndarray, 
                            feature_names: List[str], output_dir: Path) -> List[str]:
        """Create comprehensive visualizations."""
        output_dir.mkdir(exist_ok=True)
        plot_paths = []
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Anomaly detection plot
        plt.figure(figsize=(12, 8))
        
        # Main plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                            cmap='coolwarm', alpha=0.7, s=50)
        plt.title('Anomaly Detection Results (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, label='Anomaly Label')
        
        # Feature importance (if available)
        plt.subplot(2, 2, 2)
        if hasattr(self.models.get('isolation_forest'), 'feature_importances_'):
            importances = self.models['isolation_forest'].feature_importances_
            plt.barh(range(len(feature_names)), importances)
            plt.yticks(range(len(feature_names)), feature_names)
            plt.title('Feature Importance (Isolation Forest)')
            plt.xlabel('Importance')
        else:
            plt.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        # Anomaly distribution
        plt.subplot(2, 2, 3)
        # Handle both positive and negative labels
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            # Binary case: normal vs anomaly
            if -1 in unique_labels and 1 in unique_labels:
                # Convert -1 to 0 for counting
                labels_for_counting = np.where(labels == -1, 0, labels)
                anomaly_counts = np.bincount(labels_for_counting)
                plt.pie(anomaly_counts, labels=['Anomaly', 'Normal'], autopct='%1.1f%%')
            else:
                anomaly_counts = np.bincount(labels)
                plt.pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%')
        else:
            # Multiple classes or single class
            anomaly_counts = np.bincount(labels)
            plt.pie(anomaly_counts, labels=[f'Class {i}' for i in range(len(anomaly_counts))], autopct='%1.1f%%')
        plt.title('Anomaly Distribution')
        
        # Anomaly scores histogram
        plt.subplot(2, 2, 4)
        if hasattr(self.models.get('isolation_forest'), 'score_samples'):
            scores = self.models['isolation_forest'].score_samples(X)
            plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
            plt.title('Anomaly Scores Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
        else:
            plt.text(0.5, 0.5, 'Anomaly scores not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Anomaly Scores Distribution')
        
        plt.tight_layout()
        
        # Save main plot
        main_plot_path = output_dir / "enhanced_anomaly_detection.png"
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(main_plot_path))
        
        # Time series plot (if we have temporal data)
        if len(labels) > 1:
            plt.figure(figsize=(12, 6))
            anomaly_scores = 1 - labels  # Convert to anomaly scores
            plt.plot(range(len(anomaly_scores)), anomaly_scores, 'o-', alpha=0.7)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            plt.title('Anomaly Scores Over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Anomaly Score')
            plt.tight_layout()
            
            time_plot_path = output_dir / "anomaly_scores_time.png"
            plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(str(time_plot_path))
        
        return plot_paths
    
    def run_comprehensive_analysis(self, df: pd.DataFrame, 
                                 tune_hyperparameters: bool = False,
                                 output_dir: Path = None) -> Dict:
        """Run comprehensive anomaly detection analysis."""
        if output_dir is None:
            output_dir = PLOTS_DIR
        
        try:
            # Prepare features
            X, feature_names = self.prepare_features(df)
            
            # Fit models
            logger.info("Fitting Isolation Forest...")
            self.fit_isolation_forest(X, tune_hyperparameters=tune_hyperparameters)
            
            logger.info("Fitting Local Outlier Factor...")
            self.fit_local_outlier_factor(X, tune_hyperparameters=tune_hyperparameters)
            
            logger.info("Fitting One-Class SVM...")
            self.fit_one_class_svm(X, tune_hyperparameters=tune_hyperparameters)
            
            # Get predictions from Isolation Forest (main model)
            main_model = self.models['isolation_forest']
            if hasattr(main_model, 'best_estimator_'):
                main_model = main_model.best_estimator_
            
            labels = main_model.predict(X)
            
            # Evaluate ensemble
            ensemble_results = self.ensemble_detection(X)
            
            # Evaluate clustering quality
            clustering_metrics = self.evaluate_clustering_quality(X, labels)
            
            # Create visualizations
            plot_paths = self.create_visualizations(X, labels, feature_names, output_dir)
            
            # Compile results
            results = {
                'ensemble_results': ensemble_results,
                'clustering_metrics': clustering_metrics,
                'plot_paths': plot_paths,
                'n_samples': len(X),
                'n_features': len(feature_names),
                'feature_names': feature_names,
                'anomaly_labels': labels
            }
            
            logger.info(f"Analysis complete. Results saved to {output_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            raise

def main():
    """Main function to run enhanced anomaly detection."""
    try:
        # Load data
        df = load_master_dataframe()
        
        # Initialize detector
        detector = EnhancedAnomalyDetector()
        
        # Run analysis
        results = detector.run_comprehensive_analysis(
            df, 
            tune_hyperparameters=True,
            output_dir=PLOTS_DIR
        )
        
        # Print summary
        logger.info("=== ANOMALY DETECTION SUMMARY ===")
        logger.info(f"Total samples: {results['n_samples']}")
        logger.info(f"Features used: {results['n_features']}")
        
        logger.info("\n=== ENSEMBLE RESULTS ===")
        for algo, metrics in results['ensemble_results'].items():
            logger.info(f"{algo}: {metrics['n_anomalies']} anomalies "
                       f"({metrics['anomaly_score']:.2%} anomaly rate)")
        
        logger.info("\n=== CLUSTERING QUALITY ===")
        logger.info(f"Silhouette Score: {results['clustering_metrics']['silhouette_score']:.3f}")
        logger.info(f"Calinski-Harabasz Score: {results['clustering_metrics']['calinski_harabasz_score']:.3f}")
        
        logger.info(f"\nPlots saved to: {results['plot_paths']}")
        
    except Exception as e:
        logger.error(f"Failed to run anomaly detection: {e}")

if __name__ == "__main__":
    main()

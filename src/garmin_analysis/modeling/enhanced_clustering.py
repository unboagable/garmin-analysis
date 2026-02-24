"""
Enhanced Clustering for Garmin Health Data

This module provides multiple clustering algorithms with:
- Optimal cluster number selection
- Multiple evaluation metrics
- Feature importance analysis
- Interpretable cluster profiles
- Temporal clustering analysis
"""

import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from garmin_analysis.utils.data_loading import load_master_dataframe
from garmin_analysis.utils.data_filtering import standardize_features
from garmin_analysis.utils_cleaning import clean_data
from garmin_analysis.utils.imputation import impute_missing_values
from garmin_analysis.config import PLOTS_DIR

logger = logging.getLogger(__name__)

class EnhancedClusterer:
    """Enhanced clustering with multiple algorithms and evaluation metrics."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_params = {}
        
    def prepare_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                        imputation_strategy: str = 'median') -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare features for clustering.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns
            imputation_strategy: Strategy for handling missing values
                               - 'median': Fill with median (default, robust)
                               - 'mean': Fill with mean
                               - 'drop': Drop rows with missing values
                               - 'none': No imputation
        
        Returns:
            Tuple of (X_scaled, feature_names, df_clean)
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
        
        # Filter to features that exist and have sufficient data
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
        total_count = len(df_clean)
        for col in available_features:
            non_null_count = df_clean[col].notna().sum()
            completeness = non_null_count / total_count if total_count > 0 else 0.0
            feature_completeness[col] = completeness
            logger.info(f"Feature '{col}': {non_null_count}/{total_count} ({completeness:.1%} complete)")
        
        # Impute missing values using specified strategy
        df_clean = impute_missing_values(df_clean, available_features, strategy=imputation_strategy, copy=False)
        
        if df_clean.empty:
            raise ValueError("No data left after handling missing values")
        
        if len(df_clean) < 10:
            raise ValueError(f"Not enough rows for clustering. Need at least 10, got {len(df_clean)}")
        
        # Extract features
        X = df_clean[available_features].values
        
        # Handle different scaling strategies
        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Prepared {len(df_clean)} samples with {len(available_features)} features")
        logger.info(f"Feature completeness: {feature_completeness}")
        return X_scaled, available_features, df_clean
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Dict:
        """Find optimal number of clusters using multiple metrics."""
        if max_clusters > len(X) // 2:
            max_clusters = len(X) // 2
        
        metrics = {}
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Calculate metrics
                sil_score = silhouette_score(X, labels)
                cal_score = calinski_harabasz_score(X, labels)
                dav_score = davies_bouldin_score(X, labels)
                
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
                davies_scores.append(dav_score)
                
                metrics[k] = {
                    'silhouette': sil_score,
                    'calinski_harabasz': cal_score,
                    'davies_bouldin': dav_score,
                    'inertia': kmeans.inertia_
                }
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {k} clusters: {e}")
                continue
        
        # Find optimal k for each metric
        optimal_k = {}
        if silhouette_scores:
            optimal_k['silhouette'] = np.argmax(silhouette_scores) + 2
        if calinski_scores:
            optimal_k['calinski_harabasz'] = np.argmax(calinski_scores) + 2
        if davies_scores:
            optimal_k['davies_bouldin'] = np.argmin(davies_scores) + 2
        
        # Elbow method for inertia
        if len(metrics) > 1:
            inertias = [metrics[k]['inertia'] for k in metrics.keys()]
            # Simple elbow detection (second derivative)
            if len(inertias) > 2:
                second_deriv = np.diff(np.diff(inertias))
                elbow_idx = np.argmax(second_deriv) + 2
                optimal_k['elbow'] = elbow_idx
        
        return {
            'metrics': metrics,
            'optimal_k': optimal_k,
            'recommended_k': optimal_k.get('silhouette', 3)  # Default to silhouette
        }
    
    def fit_kmeans(self, X: np.ndarray, n_clusters: int = None, **kwargs) -> Dict:
        """Fit KMeans clustering."""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)['recommended_k']
        
        param_grid = {
            'n_clusters': [n_clusters],
            'n_init': [10, 20],
            'max_iter': [300, 500]
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                KMeans(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',  # Use valid scoring metric
                n_jobs=-1
            )
            model.fit(X)
            logger.info(f"Best KMeans params: {model.best_params_}")
        else:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                **kwargs
            )
            model.fit(X)
        
        self.models['kmeans'] = model
        return {'model': model, 'algorithm': 'KMeans', 'n_clusters': n_clusters}
    
    def fit_gaussian_mixture(self, X: np.ndarray, n_clusters: int = None, **kwargs) -> Dict:
        """Fit Gaussian Mixture Model."""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)['recommended_k']
        
        param_grid = {
            'n_components': [n_clusters],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'init_params': ['kmeans', 'random']
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                GaussianMixture(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',  # Use valid scoring metric
                n_jobs=-1
            )
            model.fit(X)
            logger.info(f"Best GMM params: {model.best_params_}")
        else:
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                **kwargs
            )
            model.fit(X)
        
        self.models['gaussian_mixture'] = model
        return {'model': model, 'algorithm': 'Gaussian Mixture', 'n_clusters': n_clusters}
    
    def fit_hierarchical(self, X: np.ndarray, n_clusters: int = None, **kwargs) -> Dict:
        """Fit Hierarchical Clustering."""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)['recommended_k']
        
        param_grid = {
            'n_clusters': [n_clusters],
            'linkage': ['ward', 'complete', 'average', 'single']
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                AgglomerativeClustering(),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',  # Use valid scoring metric
            )
            model.fit(X)
            logger.info(f"Best Hierarchical params: {model.best_params_}")
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                **kwargs
            )
            model.fit(X)
        
        self.models['hierarchical'] = model
        return {'model': model, 'algorithm': 'Hierarchical', 'n_clusters': n_clusters}
    
    def fit_dbscan(self, X: np.ndarray, **kwargs) -> Dict:
        """Fit DBSCAN clustering."""
        param_grid = {
            'eps': [0.1, 0.5, 1.0, 2.0],
            'min_samples': [5, 10, 20]
        }
        
        if kwargs.get('tune_hyperparameters', False):
            best_score = -np.inf
            best_params = {}
            
            for eps in param_grid['eps']:
                for min_samples in param_grid['min_samples']:
                    if min_samples < len(X):
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(X)
                        
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(X, labels)
                            if score > best_score:
                                best_score = score
                                best_params = {'eps': eps, 'min_samples': min_samples}
            
            model = DBSCAN(**best_params)
            model.fit(X)
            logger.info(f"Best DBSCAN params: {best_params}")
        else:
            model = DBSCAN(eps=0.5, min_samples=10, **kwargs)
            model.fit(X)
        
        self.models['dbscan'] = model
        n_clusters = len(np.unique(model.labels_))
        return {'model': model, 'algorithm': 'DBSCAN', 'n_clusters': n_clusters}
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate clustering quality using multiple metrics."""
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': -1,
                'n_clusters': 1
            }
        
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
        
        try:
            davies_bouldin = davies_bouldin_score(X, labels)
        except Exception as e:
            logger.debug(f"Error calculating Davies-Bouldin score: {e}")
            davies_bouldin = -1
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'n_clusters': len(np.unique(labels))
        }
    
    def analyze_cluster_profiles(self, df: pd.DataFrame, labels: np.ndarray, 
                               feature_names: List[str]) -> Dict:
        """Analyze and profile each cluster."""
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        cluster_profiles = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points (DBSCAN)
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_with_clusters) * 100,
                'features': {}
            }
            
            # Analyze each feature
            for feature in feature_names:
                if feature in cluster_data.columns:
                    values = cluster_data[feature].dropna()
                    if len(values) > 0:
                        profile['features'][feature] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(values.median())
                        }
            
            cluster_profiles[f'cluster_{cluster_id}'] = profile
        
        return cluster_profiles
    
    def create_visualizations(self, X: np.ndarray, labels: np.ndarray, 
                            feature_names: List[str], df: pd.DataFrame,
                            output_dir: Path) -> List[str]:
        """Create comprehensive clustering visualizations."""
        output_dir.mkdir(exist_ok=True)
        plot_paths = []
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Main clustering plot
        plt.figure(figsize=(15, 10))
        
        # PCA scatter plot
        plt.subplot(2, 3, 1)
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # Noise points
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c='black', marker='x', s=20, alpha=0.6, label='Noise')
            else:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title('Clustering Results (PCA)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        
        # Cluster size distribution
        plt.subplot(2, 3, 2)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
        cluster_labels = [f'Cluster {label}' for label in unique_labels if label != -1]
        
        if cluster_sizes:
            plt.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Cluster Size Distribution')
        
        # Feature importance by cluster
        plt.subplot(2, 3, 3)
        if len(unique_labels) > 1:
            # Calculate feature importance using variance between clusters
            feature_importance = []
            for i, feature in enumerate(feature_names):
                if feature in df.columns:
                    values = df[feature].dropna()
                    if len(values) > 0:
                        # Calculate F-statistic between clusters
                        cluster_means = []
                        for label in unique_labels:
                            if label != -1:
                                cluster_data = values[labels == label]
                                if len(cluster_data) > 0:
                                    cluster_means.append(cluster_data.mean())
                        
                        if len(cluster_means) > 1:
                            # Simple variance between cluster means
                            importance = np.var(cluster_means)
                        else:
                            importance = 0
                    else:
                        importance = 0
                else:
                    importance = 0
                feature_importance.append(importance)
            
            if any(feature_importance):
                # Normalize importance
                feature_importance = np.array(feature_importance) / np.max(feature_importance)
                plt.barh(range(len(feature_names)), feature_importance)
                plt.yticks(range(len(feature_names)), feature_names)
                plt.title('Feature Importance (Between-Cluster Variance)')
                plt.xlabel('Importance Score')
        
        # Feature correlation heatmap
        plt.subplot(2, 3, 4)
        df_features = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df_features.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
        
        # Cluster evaluation metrics
        plt.subplot(2, 3, 5)
        metrics = self.evaluate_clustering(X, labels)
        metric_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
        metric_values = [
            metrics['silhouette_score'],
            metrics['calinski_harabasz_score'] / 1000,  # Scale down for visualization
            metrics['davies_bouldin_score']
        ]
        
        colors = ['green' if v > 0 else 'red' for v in metric_values]
        plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
        plt.title('Clustering Quality Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # Temporal analysis (if date column exists)
        if 'day' in df.columns:
            plt.subplot(2, 3, 6)
            df_temp = df.copy()
            df_temp['cluster'] = labels
            df_temp['day'] = pd.to_datetime(df_temp['day'])
            
            # Plot cluster distribution over time
            for label in unique_labels:
                if label != -1:
                    cluster_data = df_temp[df_temp['cluster'] == label]
                    if len(cluster_data) > 0:
                        plt.scatter(cluster_data['day'], [label] * len(cluster_data), 
                                  alpha=0.7, s=30, label=f'Cluster {label}')
            
            plt.title('Cluster Distribution Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cluster')
            plt.legend()
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save main plot
        plot_path = output_dir / "enhanced_clustering.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def run_comprehensive_clustering(self, df: pd.DataFrame, 
                                   algorithms: List[str] = None,
                                   tune_hyperparameters: bool = False,
                                   output_dir: Path = None) -> Dict:
        """Run comprehensive clustering analysis."""
        if algorithms is None:
            algorithms = ['kmeans', 'gaussian_mixture', 'hierarchical', 'dbscan']
        
        if output_dir is None:
            output_dir = PLOTS_DIR
        
        try:
            # Prepare features
            X, feature_names, df_clean = self.prepare_features(df)
            
            # Find optimal number of clusters
            logger.info("Finding optimal number of clusters...")
            optimal_clusters = self.find_optimal_clusters(X)
            logger.info(f"Recommended number of clusters: {optimal_clusters['recommended_k']}")
            
            # Fit models
            results = {}
            best_score = -np.inf
            best_algorithm = None
            
            for algo in algorithms:
                logger.info(f"Fitting {algo}...")
                
                try:
                    if algo == 'kmeans':
                        result = self.fit_kmeans(X, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'gaussian_mixture':
                        result = self.fit_gaussian_mixture(X, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'hierarchical':
                        result = self.fit_hierarchical(X, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'dbscan':
                        result = self.fit_dbscan(X, tune_hyperparameters=tune_hyperparameters)
                    else:
                        logger.warning(f"Unknown algorithm: {algo}")
                        continue
                    
                    # Get predictions
                    model = result['model']
                    if hasattr(model, 'best_estimator_'):
                        model = model.best_estimator_
                    
                    if hasattr(model, 'predict'):
                        labels = model.predict(X)
                    else:
                        labels = model.labels_
                    
                    # Evaluate
                    evaluation = self.evaluate_clustering(X, labels)
                    result['evaluation'] = evaluation
                    result['labels'] = labels
                    
                    results[algo] = result
                    
                    # Track best model
                    if evaluation['silhouette_score'] > best_score:
                        best_score = evaluation['silhouette_score']
                        best_algorithm = algo
                        self.best_model = model
                        self.best_params = result
                    
                except Exception as e:
                    logger.warning(f"Failed to fit {algo}: {e}")
                    continue
            
            # Analyze best model
            if best_algorithm:
                best_result = results[best_algorithm]
                best_labels = best_result['labels']
                
                # Analyze cluster profiles
                cluster_profiles = self.analyze_cluster_profiles(df_clean, best_labels, feature_names)
                
                # Create visualizations
                plot_paths = self.create_visualizations(X, best_labels, feature_names, df_clean, output_dir)
                
                # Compile final results
                final_results = {
                    'best_algorithm': best_algorithm,
                    'best_model': best_result,
                    'all_results': results,
                    'cluster_profiles': cluster_profiles,
                    'plot_paths': plot_paths,
                    'optimal_clusters': optimal_clusters,
                    'n_samples': len(X),
                    'n_features': len(feature_names),
                    'feature_names': feature_names
                }
                
                logger.info(f"Clustering analysis complete. Results saved to {output_dir}")
                return final_results
            else:
                raise ValueError("No clustering algorithm succeeded")
                
        except Exception as e:
            logger.error(f"Error in comprehensive clustering: {e}")
            raise

def main():
    """Main function to run enhanced clustering."""
    try:
        # Load data
        df = load_master_dataframe()
        
        # Initialize clusterer
        clusterer = EnhancedClusterer()
        
        # Run analysis
        results = clusterer.run_comprehensive_clustering(
            df, 
            tune_hyperparameters=True,
            output_dir=PLOTS_DIR
        )
        
        # Print summary
        logger.info("=== CLUSTERING SUMMARY ===")
        logger.info(f"Best algorithm: {results['best_algorithm']}")
        logger.info(f"Total samples: {results['n_samples']}")
        logger.info(f"Features used: {results['n_features']}")
        
        best_eval = results['best_model']['evaluation']
        logger.info(f"Best silhouette score: {best_eval['silhouette_score']:.3f}")
        logger.info(f"Number of clusters: {best_eval['n_clusters']}")
        
        logger.info("\n=== CLUSTER PROFILES ===")
        for cluster_name, profile in results['cluster_profiles'].items():
            logger.info(f"{cluster_name}: {profile['size']} samples ({profile['percentage']:.1f}%)")
        
        logger.info(f"\nPlots saved to: {results['plot_paths']}")
        
    except Exception as e:
        logger.error(f"Failed to run clustering: {e}")

if __name__ == "__main__":
    main()

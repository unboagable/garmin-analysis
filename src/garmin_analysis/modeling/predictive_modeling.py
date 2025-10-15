"""
Predictive Modeling for Garmin Health Data

This module provides predictive modeling capabilities including:
- Time series forecasting
- Health outcome prediction
- Feature engineering
- Model evaluation and comparison
- Cross-validation strategies
"""

import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
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

logger = logging.getLogger(__name__)

class HealthPredictor:
    """Predictive modeling for health outcomes."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'score',
                        feature_cols: Optional[List[str]] = None,
                        lag_features: bool = True,
                        rolling_features: bool = True,
                        imputation_strategy: str = 'median') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for predictive modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature columns
            lag_features: Whether to create lag features
            rolling_features: Whether to create rolling features
            imputation_strategy: Strategy for handling missing values
                               - 'median': Fill with median (default, robust)
                               - 'mean': Fill with mean
                               - 'drop': Drop rows with missing values
                               - 'none': No imputation
        
        Returns:
            Tuple of (X_scaled, y, feature_names)
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
        
        # Ensure temporal ordering
        if 'day' in df_clean.columns:
            df_clean['day'] = pd.to_datetime(df_clean['day'])
            df_clean = df_clean.sort_values('day').reset_index(drop=True)
        
        # Create lag features
        if lag_features:
            for col in feature_cols:
                if col in df_clean.columns:
                    df_clean[f'{col}_lag1'] = df_clean[col].shift(1)
                    df_clean[f'{col}_lag7'] = df_clean[col].shift(7)
        
        # Create rolling features
        if rolling_features:
            for col in feature_cols:
                if col in df_clean.columns:
                    try:
                        # Ensure column is numeric before creating rolling features
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            df_clean[f'{col}_rolling_7d'] = df_clean[col].rolling(window=7, min_periods=1).mean()
                            df_clean[f'{col}_rolling_30d'] = df_clean[col].rolling(window=30, min_periods=1).mean()
                        else:
                            logger.warning(f"Skipping rolling features for non-numeric column: {col} (type: {df_clean[col].dtype})")
                    except Exception as e:
                        logger.warning(f"Failed to create rolling features for {col}: {e}")
        
        # Create interaction features
        if 'steps' in df_clean.columns and 'stress_avg' in df_clean.columns:
            try:
                if pd.api.types.is_numeric_dtype(df_clean['steps']) and pd.api.types.is_numeric_dtype(df_clean['stress_avg']):
                    df_clean['steps_stress_interaction'] = df_clean['steps'] * df_clean['stress_avg']
                else:
                    logger.warning("Skipping steps_stress_interaction - columns not numeric")
            except Exception as e:
                logger.warning(f"Failed to create steps_stress_interaction: {e}")
        
        if 'hr_avg' in df_clean.columns and 'stress_avg' in df_clean.columns:
            try:
                if pd.api.types.is_numeric_dtype(df_clean['hr_avg']) and pd.api.types.is_numeric_dtype(df_clean['stress_avg']):
                    df_clean['hr_stress_interaction'] = df_clean['hr_avg'] * df_clean['stress_avg']
                else:
                    logger.warning("Skipping hr_stress_interaction - columns not numeric")
            except Exception as e:
                logger.warning(f"Failed to create hr_stress_interaction: {e}")
        
        # Filter to features that exist and are truly numeric
        all_features = [col for col in df_clean.columns if col not in ['day', 'date', target_col]]
        available_features = []
        
        for col in all_features:
            if col in df_clean.columns:
                # Skip time-related columns that might be strings
                if any(time_keyword in col.lower() for time_keyword in ['time', 'start', 'end', 'duration']):
                    logger.info(f"Skipping time column: {col}")
                    continue
                
                # Check if column is numeric and has sufficient data
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    non_null_count = df_clean[col].notna().sum()
                    if non_null_count >= 20:  # Require at least 20 non-null values
                        available_features.append(col)
                    else:
                        logger.info(f"Feature '{col}' has insufficient data: {non_null_count} non-null values")
                else:
                    logger.warning(f"Feature '{col}' is not numeric (type: {df_clean[col].dtype})")
        
        if len(available_features) < 3:
            raise ValueError(f"Need at least 3 numeric features, but only {len(available_features)} available: {available_features}")
        
        # Check target column availability and find suitable target
        if target_col not in df_clean.columns or df_clean[target_col].notna().sum() < 20:
            # Look for columns with sufficient data that could be good targets
            potential_targets = []
            for col in df_clean.columns:
                if col not in ['day', 'date'] and pd.api.types.is_numeric_dtype(df_clean[col]):
                    non_null_count = df_clean[col].notna().sum()
                    if non_null_count >= 50:  # Require at least 50 non-null values
                        potential_targets.append((col, non_null_count))
            
            if potential_targets:
                # Sort by data completeness and choose the best one
                potential_targets.sort(key=lambda x: x[1], reverse=True)
                target_col = potential_targets[0][0]
                logger.info(f"Using '{target_col}' as target column (has {potential_targets[0][1]} non-null values)")
            else:
                raise ValueError(f"No suitable target columns found with sufficient data (need at least 50 non-null values)")
        
        # Check data completeness
        logger.info(f"Target column '{target_col}' completeness: {df_clean[target_col].notna().sum()}/{len(df_clean)} ({df_clean[target_col].notna().sum()/len(df_clean):.1%})")
        
        if df_clean[target_col].notna().sum() < 20:
            raise ValueError(f"Target column '{target_col}' has insufficient data: {df_clean[target_col].notna().sum()} non-null values (need at least 20)")
        
        # Drop rows with missing target (always required)
        df_clean = df_clean.dropna(subset=[target_col])
        
        # Impute missing feature values using specified strategy
        df_clean = impute_missing_values(df_clean, available_features, strategy=imputation_strategy, copy=False)
        
        if df_clean.empty:
            raise ValueError("No data left after handling missing values")
        
        if len(df_clean) < 20:
            raise ValueError(f"Not enough complete rows for predictive modeling. Need at least 20, got {len(df_clean)}")
        
        # Extract features and target
        X = df_clean[available_features].values
        y = df_clean[target_col].values
        
        # Handle different scaling strategies
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['feature_scaler'] = scaler
        
        logger.info(f"Prepared {len(df_clean)} samples with {len(available_features)} features")
        logger.info(f"Target range: {y.min():.2f} to {y.max():.2f}")
        return X_scaled, y, available_features
    
    def create_time_series_split(self, n_splits: int = 5) -> TimeSeriesSplit:
        """Create time series cross-validation split."""
        return TimeSeriesSplit(n_splits=n_splits)
    
    def fit_random_forest(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit Random Forest model."""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            model.fit(X, y)
            logger.info(f"Best Random Forest params: {model.best_params_}")
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=self.random_state,
                **kwargs
            )
            model.fit(X, y)
        
        self.models['random_forest'] = model
        return {'model': model, 'algorithm': 'Random Forest'}
    
    def fit_gradient_boosting(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit Gradient Boosting model."""
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6]
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                GradientBoostingRegressor(random_state=self.random_state),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            model.fit(X, y)
            logger.info(f"Best Gradient Boosting params: {model.best_params_}")
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                **kwargs
            )
            model.fit(X, y)
        
        self.models['gradient_boosting'] = model
        return {'model': model, 'algorithm': 'Gradient Boosting'}
    
    def fit_linear_models(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit linear models (Linear, Ridge, Lasso)."""
        models = {}
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        models['linear_regression'] = lr
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        ridge.fit(X, y)
        models['ridge'] = ridge
        
        # Lasso Regression
        lasso = Lasso(alpha=0.1, random_state=self.random_state)
        lasso.fit(X, y)
        models['lasso'] = lasso
        
        self.models.update(models)
        return {'models': models, 'algorithms': ['Linear', 'Ridge', 'Lasso']}
    
    def fit_svr(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit Support Vector Regression."""
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                SVR(),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            model.fit(X, y)
            logger.info(f"Best SVR params: {model.best_params_}")
        else:
            model = SVR(C=1.0, gamma='scale', kernel='rbf', **kwargs)
            model.fit(X, y)
        
        self.models['svr'] = model
        return {'model': model, 'algorithm': 'Support Vector Regression'}
    
    def fit_mlp(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit Multi-layer Perceptron."""
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        
        if kwargs.get('tune_hyperparameters', False):
            model = GridSearchCV(
                MLPRegressor(random_state=self.random_state, max_iter=500),
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            model.fit(X, y)
            logger.info(f"Best MLP params: {model.best_params_}")
        else:
            model = MLPRegressor(
                hidden_layer_sizes=(100,),
                alpha=0.001,
                learning_rate_init=0.01,
                random_state=self.random_state,
                max_iter=500,
                **kwargs
            )
            model.fit(X, y)
        
        self.models['mlp'] = model
        return {'model': model, 'algorithm': 'Multi-layer Perceptron'}
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      cv_splits: int = 5) -> Dict:
        """Evaluate model performance using multiple metrics."""
        # Time series cross-validation
        tscv = self.create_time_series_split(n_splits=cv_splits)
        
        # Cross-validation scores
        cv_mse = -cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        cv_mae = -cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        cv_r2 = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Predictions
        y_pred = model.predict(X)
        
        # Performance metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'cv_mse_mean': cv_mse.mean(),
            'cv_mse_std': cv_mse.std(),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance from model."""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = dict(zip(feature_names, np.abs(model.coef_)))
        else:
            # Other models - use permutation importance or default
            importance = dict(zip(feature_names, [0] * len(feature_names)))
        
        return importance
    
    def create_visualizations(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str], results: Dict,
                            output_dir: Path) -> List[str]:
        """Create comprehensive model evaluation visualizations."""
        output_dir.mkdir(exist_ok=True)
        plot_paths = []
        
        # Model performance comparison
        plt.figure(figsize=(15, 10))
        
        # Performance metrics comparison
        plt.subplot(2, 3, 1)
        algorithms = list(results.keys())
        mse_scores = [results[algo]['mse'] for algo in algorithms]
        r2_scores = [results[algo]['r2'] for algo in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        plt.bar(x - width/2, mse_scores, width, label='MSE', alpha=0.7)
        plt.bar(x + width/2, r2_scores, width, label='R²', alpha=0.7)
        plt.xlabel('Algorithm')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, algorithms, rotation=45)
        plt.legend()
        
        # Cross-validation stability
        plt.subplot(2, 3, 2)
        cv_mse_means = [results[algo]['cv_mse_mean'] for algo in algorithms]
        cv_mse_stds = [results[algo]['cv_mse_std'] for algo in algorithms]
        
        plt.errorbar(algorithms, cv_mse_means, yerr=cv_mse_stds, 
                    fmt='o', capsize=5, capthick=2)
        plt.title('Cross-Validation MSE Stability')
        plt.ylabel('MSE')
        plt.xticks(rotation=45)
        
        # Feature importance (best model)
        plt.subplot(2, 3, 3)
        best_algo = min(results.keys(), key=lambda x: results[x]['mse'])
        best_model = self.models[best_algo]
        
        if hasattr(best_model, 'best_estimator_'):
            best_model = best_model.best_estimator_
        
        feature_importance = self.get_feature_importance(best_model, feature_names)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:10]  # Top 10 features
        
        feature_names_top = [f[0] for f in top_features]
        importance_values = [f[1] for f in top_features]
        
        plt.barh(range(len(feature_names_top)), importance_values)
        plt.yticks(range(len(feature_names_top)), feature_names_top)
        plt.title(f'Feature Importance ({best_algo})')
        plt.xlabel('Importance Score')
        
        # Actual vs Predicted
        plt.subplot(2, 3, 4)
        y_pred = best_model.predict(X)
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted')
        
        # Residuals
        plt.subplot(2, 3, 5)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Feature correlation with target
        plt.subplot(2, 3, 6)
        df_features = pd.DataFrame(X, columns=feature_names)
        df_features['target'] = y
        
        # Calculate correlations
        correlations = []
        for feature in feature_names:
            corr = df_features[feature].corr(df_features['target'])
            correlations.append(abs(corr))
        
        # Sort by correlation
        sorted_corr = sorted(zip(feature_names, correlations), key=lambda x: x[1], reverse=True)
        top_corr_features = [f[0] for f in sorted_corr[:10]]
        top_corr_values = [f[1] for f in sorted_corr[:10]]
        
        plt.barh(range(len(top_corr_features)), top_corr_values)
        plt.yticks(range(len(top_corr_features)), top_corr_features)
        plt.title('Feature Correlation with Target')
        plt.xlabel('|Correlation|')
        
        plt.tight_layout()
        
        # Save main plot
        plot_path = output_dir / "predictive_modeling_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        return plot_paths
    
    def run_comprehensive_prediction(self, df: pd.DataFrame, 
                                   target_col: str = 'score',
                                   algorithms: List[str] = None,
                                   tune_hyperparameters: bool = False,
                                   output_dir: Path = None) -> Dict:
        """Run comprehensive predictive modeling analysis."""
        if algorithms is None:
            algorithms = ['random_forest', 'gradient_boosting', 'linear_models', 'svr', 'mlp']
        
        if output_dir is None:
            output_dir = Path("plots")
        
        try:
            # Prepare features
            X, y, feature_names = self.prepare_features(df, target_col)
            
            # Split data (maintaining temporal order)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            
            # Fit models
            results = {}
            
            for algo in algorithms:
                logger.info(f"Fitting {algo}...")
                
                try:
                    if algo == 'random_forest':
                        result = self.fit_random_forest(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'gradient_boosting':
                        result = self.fit_gradient_boosting(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'linear_models':
                        result = self.fit_linear_models(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'svr':
                        result = self.fit_svr(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
                    elif algo == 'mlp':
                        result = self.fit_mlp(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
                    else:
                        logger.warning(f"Unknown algorithm: {algo}")
                        continue
                    
                    # Evaluate model
                    if algo == 'linear_models':
                        # Evaluate each linear model separately
                        for model_name, model in result['models'].items():
                            eval_result = self.evaluate_model(model, X_test, y_test)
                            results[model_name] = eval_result
                    else:
                        model = result['model']
                        eval_result = self.evaluate_model(model, X_test, y_test)
                        results[algo] = eval_result
                    
                except Exception as e:
                    logger.warning(f"Failed to fit {algo}: {e}")
                    continue
            
            # Find best model
            best_algo = min(results.keys(), key=lambda x: results[x]['mse'])
            best_score = results[best_algo]['mse']
            
            logger.info(f"Best model: {best_algo} (MSE: {best_score:.4f})")
            
            # Create visualizations
            plot_paths = self.create_visualizations(X, y, feature_names, results, output_dir)
            
            # Compile final results
            final_results = {
                'best_algorithm': best_algo,
                'best_score': best_score,
                'all_results': results,
                'plot_paths': plot_paths,
                'n_samples': len(X),
                'n_features': len(feature_names),
                'feature_names': feature_names,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            logger.info(f"Predictive modeling analysis complete. Results saved to {output_dir}")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {e}")
            raise

def main():
    """Main function to run predictive modeling."""
    try:
        # Load data
        df = load_master_dataframe()
        
        # Initialize predictor
        predictor = HealthPredictor()
        
        # Run analysis
        results = predictor.run_comprehensive_prediction(
            df, 
            target_col='score',
            tune_hyperparameters=True,
            output_dir=Path("plots")
        )
        
        # Print summary
        logger.info("=== PREDICTIVE MODELING SUMMARY ===")
        logger.info(f"Best algorithm: {results['best_algorithm']}")
        logger.info(f"Best MSE: {results['best_score']:.4f}")
        logger.info(f"Total samples: {results['n_samples']}")
        logger.info(f"Features used: {results['n_features']}")
        logger.info(f"Train/Test split: {results['train_size']}/{results['test_size']}")
        
        logger.info("\n=== MODEL PERFORMANCE ===")
        for algo, metrics in results['all_results'].items():
            logger.info(f"{algo}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.3f}")
        
        logger.info(f"\nPlots saved to: {results['plot_paths']}")
        
    except Exception as e:
        logger.error(f"Failed to run predictive modeling: {e}")

if __name__ == "__main__":
    main()

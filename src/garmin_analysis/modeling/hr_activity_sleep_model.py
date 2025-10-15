"""
Heart Rate and Activity Impact on Sleep Quality Model

This module analyzes how heart rate metrics (min, max, resting) and 
physical activities affect sleep score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class HRActivitySleepModel:
    """Model to analyze HR and activity impact on sleep score."""
    
    def __init__(self, data_path: str = "data/modeling_ready_dataset.csv", random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            data_path: Path to the modeling dataset
            random_state: Random state for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['day'])
        logger.info(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        
        return self.df
    
    def _impute_missing_values(self, df_model: pd.DataFrame, features: List[str], 
                              strategy: str = 'median') -> pd.DataFrame:
        """
        Impute missing values in features.
        
        Args:
            df_model: DataFrame with features
            features: List of feature names to impute
            strategy: Imputation strategy - 'median', 'mean', 'drop', or 'none'
                     - 'median': Fill with column median (robust to outliers)
                     - 'mean': Fill with column mean
                     - 'drop': Drop rows with any missing values
                     - 'none': No imputation (keep NaN values)
        
        Returns:
            DataFrame with imputed values
        """
        df_result = df_model.copy()
        
        if strategy == 'drop':
            # Drop rows with any missing values in features
            df_result = df_result.dropna(subset=features)
            logger.info(f"Dropped rows with missing values. Remaining: {len(df_result)} samples")
        elif strategy == 'median':
            # Fill with median (robust to outliers)
            for col in features:
                if df_result[col].isna().any():
                    median_val = df_result[col].median()
                    df_result[col] = df_result[col].fillna(median_val)
                    logger.info(f"Filled {df_model[col].isna().sum()} missing values in {col} with median: {median_val}")
        elif strategy == 'mean':
            # Fill with mean
            for col in features:
                if df_result[col].isna().any():
                    mean_val = df_result[col].mean()
                    df_result[col] = df_result[col].fillna(mean_val)
                    logger.info(f"Filled {df_model[col].isna().sum()} missing values in {col} with mean: {mean_val}")
        elif strategy == 'none':
            # No imputation
            logger.info("No imputation applied. Missing values preserved.")
        else:
            raise ValueError(f"Invalid imputation strategy: {strategy}. Choose from 'median', 'mean', 'drop', or 'none'")
        
        return df_result
    
    def prepare_features(self, imputation_strategy: str = 'median') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for modeling.
        
        Args:
            imputation_strategy: Strategy for handling missing values
                               - 'median': Fill with column median (default, robust)
                               - 'mean': Fill with column mean
                               - 'drop': Drop rows with missing values
                               - 'none': No imputation
        
        Returns:
            X: Feature dataframe
            y: Target variable (sleep score)
            feature_names: List of feature names
        """
        if self.df is None:
            self.load_data()
        
        # Define HR features
        hr_features = [
            'hr_min',           # Minimum heart rate
            'hr_max',           # Maximum heart rate  
            'rhr',              # Resting heart rate
            'hr_avg',           # Average heart rate
            'inactive_hr_avg',  # Average inactive heart rate
            'inactive_hr_min',  # Minimum inactive heart rate
            'inactive_hr_max',  # Maximum inactive heart rate
        ]
        
        # Define activity features
        activity_features = [
            'steps',                          # Daily steps
            'moderate_activity_time',         # Moderate activity duration
            'vigorous_activity_time',         # Vigorous activity duration
            'intensity_time',                 # Total intensity time
            'distance',                       # Distance covered
            'calories_active',                # Active calories burned
            'yesterday_had_workout',          # Whether workout happened yesterday
            'yesterday_activity_minutes',     # Yesterday's activity minutes
            'yesterday_activity_calories',    # Yesterday's activity calories
            'yesterday_training_effect',      # Yesterday's training effect
            'yesterday_anaerobic_te',         # Yesterday's anaerobic training effect
            'floors',                         # Floors climbed
        ]
        
        # Additional contextual features that might affect sleep
        contextual_features = [
            'stress_avg',       # Average stress
            'bb_max',           # Max body battery
            'bb_min',           # Min body battery
            'steps_avg_7d',     # 7-day average steps
        ]
        
        # Combine all features
        all_features = hr_features + activity_features + contextual_features
        
        # Filter to features that exist in the dataset
        available_features = [f for f in all_features if f in self.df.columns]
        
        logger.info(f"Using {len(available_features)} features out of {len(all_features)} desired")
        logger.info(f"Features: {available_features}")
        
        # Target variable
        target = 'score'  # Sleep score
        
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
        
        # Create feature matrix and target
        df_model = self.df[available_features + [target]].copy()
        
        # Convert time duration columns to minutes (if they're in timedelta format)
        time_cols = ['moderate_activity_time', 'vigorous_activity_time', 'intensity_time']
        for col in time_cols:
            if col in df_model.columns:
                try:
                    # Check if it's a timedelta or string representation
                    if df_model[col].dtype == 'object':
                        df_model[col] = pd.to_timedelta(df_model[col]).dt.total_seconds() / 60
                    elif 'timedelta' in str(df_model[col].dtype):
                        df_model[col] = df_model[col].dt.total_seconds() / 60
                except Exception as e:
                    logger.warning(f"Could not convert {col} to minutes: {e}")
        
        # Drop rows with missing target (always required)
        df_model = df_model.dropna(subset=[target])
        
        # Impute missing feature values using specified strategy
        df_model = self._impute_missing_values(df_model, available_features, imputation_strategy)
        
        X = df_model[available_features]
        y = df_model[target]
        
        logger.info(f"Prepared dataset: {len(X)} samples, {len(available_features)} features")
        logger.info(f"Target (sleep score) range: {y.min():.1f} to {y.max():.1f}, mean: {y.mean():.1f}")
        
        return X, y, available_features
    
    def create_lag_features(self, X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create lagged features to capture temporal effects.
        
        Args:
            X: Feature dataframe
            feature_names: List of feature names
            
        Returns:
            X_with_lags: Feature dataframe with lag features
            all_feature_names: Updated list of feature names
        """
        X_with_lags = X.copy()
        new_features = []
        
        # Create 1-day lag for key HR and activity features
        lag_features = ['rhr', 'hr_avg', 'steps', 'stress_avg', 'bb_max']
        lag_features = [f for f in lag_features if f in feature_names]
        
        for col in lag_features:
            lag_col = f'{col}_lag1'
            X_with_lags[lag_col] = X_with_lags[col].shift(1)
            new_features.append(lag_col)
        
        # Fill NaN values in lag features with original values (for first row)
        for col in new_features:
            orig_col = col.replace('_lag1', '')
            X_with_lags[col] = X_with_lags[col].fillna(X_with_lags[orig_col])
        
        all_feature_names = feature_names + new_features
        logger.info(f"Created {len(new_features)} lag features")
        
        return X_with_lags, all_feature_names
    
    def train_models(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train multiple models to predict sleep score.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            Dictionary with model results
        """
        # Split data (80/20 train/test split with temporal ordering preserved)
        # For time series, we don't shuffle to maintain temporal order
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=0.1, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluation metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(len(feature_names))
            
            feature_importance_dict = dict(zip(feature_names, importance))
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mse': cv_mse,
                'cv_std': cv_std,
                'feature_importance': feature_importance_dict,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
            
            logger.info(f"{name} - Test R²: {test_r2:.3f}, Test MAE: {test_mae:.2f}, Test MSE: {test_mse:.2f}")
        
        self.results = results
        self.models = {name: res['model'] for name, res in results.items()}
        
        return results
    
    def analyze_feature_importance(self, results: Dict, top_n: int = 15) -> pd.DataFrame:
        """
        Analyze feature importance across models.
        
        Args:
            results: Dictionary of model results
            top_n: Number of top features to show
            
        Returns:
            DataFrame with feature importance
        """
        # Get feature importance from the best performing model (based on test R²)
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_model_importance = results[best_model_name]['feature_importance']
        
        # Sort by importance
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in best_model_importance.items()
        ]).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} most important features (from {best_model_name}):")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def create_visualizations(self, results: Dict, feature_importance_df: pd.DataFrame,
                            X: pd.DataFrame, y: pd.Series, feature_names: List[str],
                            output_dir: Path = Path("plots")) -> List[str]:
        """
        Create comprehensive visualizations.
        
        Args:
            results: Dictionary of model results
            feature_importance_df: DataFrame with feature importance
            X: Feature dataframe
            y: Target variable
            feature_names: List of feature names
            output_dir: Output directory for plots
            
        Returns:
            List of saved plot paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_paths = []
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Test R² scores
        ax = axes[0, 0]
        model_names = list(results.keys())
        r2_scores = [results[m]['test_r2'] for m in model_names]
        colors = ['#2ecc71' if r2 > 0.3 else '#e74c3c' for r2 in r2_scores]
        ax.barh(model_names, r2_scores, color=colors, alpha=0.7)
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title('Model Performance (Test R²)', fontsize=14, fontweight='bold')
        ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
        ax.legend()
        
        # Test MAE scores
        ax = axes[0, 1]
        mae_scores = [results[m]['test_mae'] for m in model_names]
        ax.barh(model_names, mae_scores, color='#3498db', alpha=0.7)
        ax.set_xlabel('Mean Absolute Error', fontsize=12)
        ax.set_title('Model Error (Test MAE)', fontsize=14, fontweight='bold')
        
        # Cross-validation scores
        ax = axes[1, 0]
        cv_scores = [results[m]['cv_mse'] for m in model_names]
        cv_stds = [results[m]['cv_std'] for m in model_names]
        ax.barh(model_names, cv_scores, xerr=cv_stds, color='#9b59b6', alpha=0.7, capsize=5)
        ax.set_xlabel('Cross-Validation MSE', fontsize=12)
        ax.set_title('Model Stability (CV MSE ± Std)', fontsize=14, fontweight='bold')
        
        # Train vs Test R² (overfitting check)
        ax = axes[1, 1]
        train_r2 = [results[m]['train_r2'] for m in model_names]
        test_r2 = [results[m]['test_r2'] for m in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.7)
        ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.7)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f"{timestamp}_model_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        logger.info(f"Saved model performance plot to {plot_path}")
        
        # 2. Feature Importance Analysis
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top features overall
        ax = axes[0]
        top_features = feature_importance_df.head(15)
        colors_feat = ['#e74c3c' if 'hr' in f.lower() or 'rhr' in f.lower() 
                      else '#3498db' if 'activity' in f.lower() or 'steps' in f.lower() or 'workout' in f.lower()
                      else '#2ecc71' 
                      for f in top_features['feature']]
        ax.barh(top_features['feature'], top_features['importance'], color=colors_feat, alpha=0.7)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # HR vs Activity feature importance
        ax = axes[1]
        hr_features = feature_importance_df[
            feature_importance_df['feature'].str.contains('hr|rhr', case=False, na=False)
        ]
        activity_features = feature_importance_df[
            feature_importance_df['feature'].str.contains('activity|steps|workout|calories|distance|floors', case=False, na=False)
        ]
        
        hr_total = hr_features['importance'].sum()
        activity_total = activity_features['importance'].sum()
        other_total = feature_importance_df['importance'].sum() - hr_total - activity_total
        
        categories = ['Heart Rate\nFeatures', 'Activity\nFeatures', 'Other\nFeatures']
        values = [hr_total, activity_total, other_total]
        colors_cat = ['#e74c3c', '#3498db', '#95a5a6']
        
        ax.bar(categories, values, color=colors_cat, alpha=0.7)
        ax.set_ylabel('Total Importance', fontsize=12)
        ax.set_title('Feature Category Importance', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = output_dir / f"{timestamp}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        logger.info(f"Saved feature importance plot to {plot_path}")
        
        # 3. Actual vs Predicted (best model)
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        best_results = results[best_model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax = axes[0]
        y_test = best_results['y_test']
        y_test_pred = best_results['y_test_pred']
        
        ax.scatter(y_test, y_test_pred, alpha=0.6, s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Sleep Score', fontsize=12)
        ax.set_ylabel('Predicted Sleep Score', fontsize=12)
        ax.set_title(f'Actual vs Predicted ({best_model_name})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add R² and MAE text
        r2 = best_results['test_r2']
        mae = best_results['test_mae']
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.2f}', 
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Residual plot
        ax = axes[1]
        residuals = y_test - y_test_pred
        ax.scatter(y_test_pred, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Sleep Score', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'Residual Plot ({best_model_name})', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / f"{timestamp}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        logger.info(f"Saved predictions plot to {plot_path}")
        
        # 4. Top Feature Correlations with Sleep Score
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        top_9_features = feature_importance_df.head(9)['feature'].tolist()
        
        for idx, feature in enumerate(top_9_features):
            ax = axes[idx]
            if feature in X.columns:
                # Create scatter plot with regression line
                feature_data = X[feature].values
                
                ax.scatter(feature_data, y, alpha=0.5, s=30)
                
                # Add regression line
                z = np.polyfit(feature_data, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(feature_data.min(), feature_data.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                
                # Calculate correlation
                corr = np.corrcoef(feature_data, y)[0, 1]
                
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Sleep Score', fontsize=10)
                ax.set_title(f'{feature}\n(corr: {corr:.3f})', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(top_9_features), 9):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plot_path = output_dir / f"{timestamp}_feature_correlations.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        logger.info(f"Saved feature correlations plot to {plot_path}")
        
        return plot_paths
    
    def generate_summary_report(self, results: Dict, feature_importance_df: pd.DataFrame,
                               output_dir: Path = Path("reports")) -> str:
        """
        Generate a text summary report.
        
        Args:
            results: Dictionary of model results
            feature_importance_df: DataFrame with feature importance
            output_dir: Output directory for report
            
        Returns:
            Path to saved report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"{timestamp}_hr_activity_sleep_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HR & ACTIVITY IMPACT ON SLEEP QUALITY - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance Summary
            f.write("\n" + "="*80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for model_name, res in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Test R² Score:    {res['test_r2']:>8.4f}\n")
                f.write(f"  Test MAE:         {res['test_mae']:>8.2f}\n")
                f.write(f"  Test MSE:         {res['test_mse']:>8.2f}\n")
                f.write(f"  Train R² Score:   {res['train_r2']:>8.4f}\n")
                f.write(f"  CV MSE (mean):    {res['cv_mse']:>8.2f}\n")
                f.write(f"  CV MSE (std):     {res['cv_std']:>8.2f}\n")
            
            # Best Model
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
            best_r2 = results[best_model_name]['test_r2']
            best_mae = results[best_model_name]['test_mae']
            
            f.write("\n" + "="*80 + "\n")
            f.write("BEST MODEL\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {best_model_name}\n")
            f.write(f"Test R² Score: {best_r2:.4f}\n")
            f.write(f"Test MAE: {best_mae:.2f} points\n")
            f.write(f"\nInterpretation: The model explains {best_r2*100:.1f}% of the variance in sleep scores.\n")
            f.write(f"On average, predictions are off by {best_mae:.1f} sleep score points.\n")
            
            # Feature Importance
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 20 MOST IMPORTANT FEATURES\n")
            f.write("="*80 + "\n\n")
            
            for rank, (idx, row) in enumerate(feature_importance_df.head(20).iterrows(), start=1):
                f.write(f"{rank:>2}. {row['feature']:<40} {row['importance']:>10.6f}\n")
            
            # Category Analysis
            hr_features = feature_importance_df[
                feature_importance_df['feature'].str.contains('hr|rhr', case=False, na=False)
            ]
            activity_features = feature_importance_df[
                feature_importance_df['feature'].str.contains('activity|steps|workout|calories|distance|floors', 
                                                             case=False, na=False)
            ]
            
            hr_total = hr_features['importance'].sum()
            activity_total = activity_features['importance'].sum()
            total_importance = feature_importance_df['importance'].sum()
            
            f.write("\n" + "="*80 + "\n")
            f.write("FEATURE CATEGORY ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Heart Rate Features:\n")
            f.write(f"  Total Importance: {hr_total:.4f} ({hr_total/total_importance*100:.1f}% of total)\n")
            f.write(f"  Number of Features: {len(hr_features)}\n\n")
            
            f.write(f"Activity Features:\n")
            f.write(f"  Total Importance: {activity_total:.4f} ({activity_total/total_importance*100:.1f}% of total)\n")
            f.write(f"  Number of Features: {len(activity_features)}\n\n")
            
            # Key Insights
            f.write("\n" + "="*80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("="*80 + "\n\n")
            
            top_hr = hr_features.head(3) if len(hr_features) > 0 else pd.DataFrame()
            top_activity = activity_features.head(3) if len(activity_features) > 0 else pd.DataFrame()
            
            if len(top_hr) > 0:
                f.write("Top HR Features Affecting Sleep:\n")
                for idx, row in top_hr.iterrows():
                    f.write(f"  - {row['feature']}: {row['importance']:.6f}\n")
                f.write("\n")
            
            if len(top_activity) > 0:
                f.write("Top Activity Features Affecting Sleep:\n")
                for idx, row in top_activity.iterrows():
                    f.write(f"  - {row['feature']}: {row['importance']:.6f}\n")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Based on the model analysis:\n\n")
            
            if hr_total > activity_total:
                f.write("1. Heart rate metrics have a STRONGER influence on sleep quality than activity.\n")
                f.write("   Focus on monitoring and managing heart rate, especially resting HR.\n\n")
            else:
                f.write("1. Physical activity has a STRONGER influence on sleep quality than heart rate.\n")
                f.write("   Maintain consistent exercise routines for better sleep.\n\n")
            
            f.write("2. Monitor the top features identified above to optimize sleep quality.\n\n")
            
            if best_r2 > 0.5:
                f.write("3. The model shows STRONG predictive power (R² > 0.5).\n")
                f.write("   These features reliably predict sleep quality.\n\n")
            elif best_r2 > 0.3:
                f.write("3. The model shows MODERATE predictive power (R² > 0.3).\n")
                f.write("   These features are useful but other factors also affect sleep.\n\n")
            else:
                f.write("3. The model shows LIMITED predictive power (R² < 0.3).\n")
                f.write("   Sleep quality may be influenced by factors not captured in this data.\n\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Saved report to {report_path}")
        return str(report_path)
    
    def run_analysis(self, use_lag_features: bool = True, 
                    imputation_strategy: str = 'median',
                    output_dir: Path = Path("modeling_results")) -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            use_lag_features: Whether to include lag features
            imputation_strategy: Strategy for handling missing values
                               - 'median': Fill with column median (default, robust)
                               - 'mean': Fill with column mean
                               - 'drop': Drop rows with missing values
                               - 'none': No imputation
            output_dir: Output directory for results
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting HR & Activity -> Sleep Quality Analysis")
        logger.info("="*80)
        
        # Load and prepare data
        X, y, feature_names = self.prepare_features(imputation_strategy=imputation_strategy)
        
        # Optionally add lag features
        if use_lag_features:
            X, feature_names = self.create_lag_features(X, feature_names)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        results = self.train_models(X_scaled, y.values, feature_names)
        
        # Analyze feature importance
        feature_importance_df = self.analyze_feature_importance(results)
        
        # Create visualizations
        plot_paths = self.create_visualizations(
            results, feature_importance_df, X, y, feature_names,
            output_dir=output_dir / "plots"
        )
        
        # Generate report
        report_path = self.generate_summary_report(
            results, feature_importance_df,
            output_dir=output_dir / "reports"
        )
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"\nPlots saved to: {output_dir / 'plots'}")
        logger.info(f"Report saved to: {report_path}")
        
        # Return summary
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        
        return {
            'best_model': best_model_name,
            'best_r2': results[best_model_name]['test_r2'],
            'best_mae': results[best_model_name]['test_mae'],
            'all_results': results,
            'feature_importance': feature_importance_df,
            'plot_paths': plot_paths,
            'report_path': report_path,
            'n_samples': len(X),
            'n_features': len(feature_names)
        }


def main():
    """Main function to run the analysis."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    model = HRActivitySleepModel()
    results = model.run_analysis(use_lag_features=True)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Test R² Score: {results['best_r2']:.4f}")
    print(f"Test MAE: {results['best_mae']:.2f} points")
    print(f"Samples: {results['n_samples']}")
    print(f"Features: {results['n_features']}")
    print(f"\nTop 5 Most Important Features:")
    for rank, (idx, row) in enumerate(results['feature_importance'].head(5).iterrows(), start=1):
        print(f"  {rank}. {row['feature']}: {row['importance']:.6f}")
    print(f"\nReport: {results['report_path']}")
    print("="*80)


if __name__ == "__main__":
    main()


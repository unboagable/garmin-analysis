import pandas as pd
import numpy as np
import logging

# Logging is configured at package level

def clean_data(df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
    """
    Cleans and standardizes the Garmin daily dataset.

    Parameters:
    - df (pd.DataFrame): Raw input DataFrame
    - remove_outliers (bool): If True, apply IQR-based outlier filtering
                              Default: False (to preserve health anomalies and extreme values)

    Returns:
    - pd.DataFrame: Cleaned DataFrame

    Cleaning Steps:
    1. Replace placeholder values with NaN
    2. Standardize numeric data types
    3. (Optional) Remove statistical outliers using IQR method
    4. Normalize column names

    Note:
    - Outlier removal may eliminate legitimate extreme values (e.g., high-intensity
      workouts, recovery days). Use with caution for health/fitness data.
    - Consider using imputation strategies from utils.imputation instead of
      dropping outliers for modeling tasks.

    Examples:
        >>> df_clean = clean_data(df)  # Basic cleaning, preserves all data
        >>> df_clean = clean_data(df, remove_outliers=True)  # Aggressive cleaning
    """

    df = df.copy()
    original_shape = df.shape
    
    # Handle empty DataFrame edge case
    if df.empty:
        logging.info("Empty DataFrame provided, returning as-is")
        return df
    
    logging.info(f"Starting data cleaning on DataFrame: {original_shape[0]} rows × {original_shape[1]} columns")

    # 1. Replace placeholder strings and invalid numeric values with NaN
    placeholders = ["", "NA", "null", "None", -1]
    placeholders_replaced = 0
    for placeholder in placeholders:
        placeholders_replaced += (df == placeholder).sum().sum()
    
    df = df.replace(placeholders, np.nan)
    if placeholders_replaced > 0:
        logging.info(f"Replaced {placeholders_replaced} placeholder values with NaN")

    # 2. Convert all numeric columns to float32 or int64 as appropriate
    conversions = {'int': 0, 'float': 0}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            is_integer_like = series.apply(
                lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())
            ).all()
            
            if is_integer_like:
                df[col] = df[col].astype('Int64')  # nullable integer
                conversions['int'] += 1
            else:
                df[col] = df[col].astype('float32')
                conversions['float'] += 1
    
    logging.info(f"Standardized data types: {conversions['int']} int columns, {conversions['float']} float columns")

    # 3. Optionally remove outliers using IQR filtering
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=['float32', 'Int64']).columns
        rows_before = len(df)
        outliers_by_col = {}
        
        for col in numeric_cols:
            col_before = len(df)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) | df[col].isna()]
            
            outliers_removed = col_before - len(df)
            if outliers_removed > 0:
                outliers_by_col[col] = outliers_removed
        
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            logging.warning(f"Outlier removal: removed {rows_removed} rows ({rows_removed/rows_before:.1%})")
            logging.warning(f"Outliers by column: {outliers_by_col}")
        else:
            logging.info("Outlier removal: no outliers detected")
    else:
        logging.debug("Outlier removal skipped (remove_outliers=False)")

    # 4. Clean column names
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    renamed = sum(1 for old, new in zip(original_cols, df.columns) if old != new)
    if renamed > 0:
        logging.info(f"Normalized {renamed} column names (lowercased, stripped, underscored)")
    
    logging.info(f"Data cleaning complete: {original_shape} → {df.shape}")
    return df


# Example usage: demonstrate cleaning functionality
if __name__ == "__main__":
    from garmin_analysis.logging_config import setup_logging
    from garmin_analysis.utils.data_loading import load_master_dataframe
    
    # Setup logging to see cleaning operations
    setup_logging(level=logging.INFO)
    
    # Load data
    logging.info("Loading Garmin data...")
    df_raw = load_master_dataframe()
    logging.info(f"Loaded raw data: {df_raw.shape}")
    
    # Clean without outlier removal (default, recommended)
    logging.info("\n" + "="*60)
    logging.info("CLEANING WITH OUTLIER PRESERVATION (default)")
    logging.info("="*60)
    df_clean = clean_data(df_raw)
    
    # Optional: Clean with outlier removal (aggressive)
    logging.info("\n" + "="*60)
    logging.info("CLEANING WITH OUTLIER REMOVAL (aggressive)")
    logging.info("="*60)
    df_clean_no_outliers = clean_data(df_raw, remove_outliers=True)
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info(f"Original shape: {df_raw.shape}")
    logging.info(f"After cleaning (preserve outliers): {df_clean.shape}")
    logging.info(f"After cleaning (remove outliers): {df_clean_no_outliers.shape}")
    logging.info(f"Data loss from outlier removal: {len(df_clean) - len(df_clean_no_outliers)} rows")

import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """
    Cleans and standardizes the Garmin daily dataset.

    Parameters:
    - df (pd.DataFrame): Raw input DataFrame
    - remove_outliers (bool): If True, apply IQR-based outlier filtering

    Returns:
    - pd.DataFrame: Cleaned DataFrame

    Steps:
    - Replace placeholder/missing values with NaN
    - Convert data types for numeric consistency
    - Optionally remove or flag outliers using IQR
    - Ensure column naming consistency
    """

    df = df.copy()

    # 1. Replace placeholder strings and invalid numeric values with NaN
    placeholders = ["", "NA", "null", "None", -1]
    df.replace(placeholders, np.nan, inplace=True)

    # 2. Convert all numeric columns to float32 or int64 as appropriate
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            is_integer_like = series.apply(
                lambda x: isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())
            ).all()
            if is_integer_like:
                df[col] = df[col].astype('Int64')  # nullable integer
            else:
                df[col] = df[col].astype('float32')

    # 3. Optionally remove outliers using IQR filtering
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=['float32', 'Int64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound) | df[col].isna()]

    # 4. Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


# Example integration: clean before modeling pipeline
if __name__ == "__main__":
    from data_ingestion.prepare_modeling_dataset import load_modeling_dataset

    df_raw = load_modeling_dataset()
    df_clean = clean_data(df_raw)

    print("Cleaned DataFrame shape:", df_clean.shape)
    print("Columns:", df_clean.columns.tolist())

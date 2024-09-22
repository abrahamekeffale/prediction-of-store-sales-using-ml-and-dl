import pandas as pd
import numpy as np
from scipy import stats

def handle_missing_data(df):
    if 'numeric_column' in df.columns:
        df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')
        df['numeric_column'].fillna(df['numeric_column'].median(), inplace=True)
    else:
        print("Column 'numeric_column' does not exist in the DataFrame")
    return df

def remove_outliers(df: pd.DataFrame, threshold: float = 3.0):
    """
    Removes outliers using z-score.
    :param df: DataFrame to clean
    :param threshold: z-score threshold for identifying outliers
    :return: DataFrame with outliers removed
    """
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
    return df[(z_scores < threshold).all(axis=1)]

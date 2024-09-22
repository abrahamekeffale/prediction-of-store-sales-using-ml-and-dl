import pandas as pd

def correlation_analysis(df: pd.DataFrame, cols: list):
    """
    Returns the correlation matrix of specified columns.
    :param df: DataFrame
    :param cols: List of columns to analyze
    :return: Correlation matrix
    """
    return df[cols].corr()

def sales_promo_analysis(df: pd.DataFrame):
    """
    Analyzes the effect of promotions on sales.
    :param df: DataFrame
    :return: Summary of the sales with and without promos
    """
    return df.groupby('Promo')['Sales'].describe()

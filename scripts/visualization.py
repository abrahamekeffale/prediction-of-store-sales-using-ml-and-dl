import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(df: pd.DataFrame, column: str, title: str):
    """
    Plots the distribution of a specified column.
    :param df: DataFrame
    :param column: Column to visualize
    :param title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.show()

def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str):
    """
    Plots a boxplot of x against y.
    :param df: DataFrame
    :param x: X-axis column
    :param y: Y-axis column
    :param title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, cols: list, title: str):
    """
    Plots a heatmap of correlations.
    :param df: DataFrame
    :param cols: List of columns to include in the heatmap
    :param title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

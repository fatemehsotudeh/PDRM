import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def calculate_correlation_matrix(dataframe, method='pearson'):
    correlation_matrix = dataframe.corr(method=method)
    return correlation_matrix


def get_top_correlations(correlation_matrix, n=20):
    # Exclude NaN values
    mask = ~correlation_matrix.isna()

    # Set the lower triangle of the correlation matrix to NaN
    lower_triangle = np.tril(mask)
    correlation_matrix.mask(lower_triangle, inplace=True)

    # Get the absolute correlations and sort
    correlations = correlation_matrix.abs().unstack().sort_values(ascending=False)

    # Drop NaN values and select the top correlations
    top_correlations = correlations.dropna()[:n]

    return top_correlations


def filter_dataframe(df, top_correlations):
    correlated_columns = [pair for pair, _ in top_correlations.index]
    return df[correlated_columns]


def create_heatmap(filtered_df):
    plt.figure(figsize=(24, 24))
    heatmap = sns.heatmap(filtered_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Top 100 Absolute Correlations Heatmap")
    plt.show()
    return heatmap


def filter_dataset_with_columns(df, selected_attributes):
    return df.loc[:, selected_attributes]

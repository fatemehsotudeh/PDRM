import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


def get_highly_correlated_attributes(correlation_matrix, threshold=0.5):
    highly_correlated_attributes = set()
    sorted_correlations = get_top_correlations(correlation_matrix, n=len(correlation_matrix) ** 2)

    for attr1, attr2 in sorted_correlations.index:
        if sorted_correlations[attr1, attr2] > threshold:
            highly_correlated_attributes.add(attr1)

    return highly_correlated_attributes


def plot_and_perform_dimensionality_reduction(X, y, target_names, num_components, method='PCA'):
    if method == 'PCA':
        reducer = PCA(num_components)
    elif method == 'LDA':
        reducer = LinearDiscriminantAnalysis(solver="svd")
    else:
        raise ValueError("Unsupported dimensionality reduction method. Use 'PCA' or 'LDA'.")

    X_r = reducer.fit_transform(X, y)

    df_X = pd.DataFrame(data=X_r)

    plt.figure()
    colors = sns.color_palette("husl", len(target_names))
    lw = 2

    for color, target, target_name in zip(colors, range(len(target_names)), target_names):
        plt.scatter(X_r[y == target, 0], X_r[y == target, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(f'{method} - Dimensionality Reduction')
    plt.show()

    return df_X


def find_optimal_num_components(X, y, method='PCA'):
    if method == 'PCA':
        reducer = PCA()
    elif method == 'LDA':
        reducer = LinearDiscriminantAnalysis()
    else:
        raise ValueError("Unsupported dimensionality reduction method. Use 'PCA' or 'LDA'.")

    # Fit the dimensionality reduction model
    reducer.fit(X, y)

    # Plot the explained variance ratio
    explained_variance_ratio = reducer.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    plt.plot(cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Explained Variance Ratio vs. Number of Components ({method})')
    plt.grid(True)
    plt.show()

    # Determine the number of components that capture a sufficient amount of variance
    threshold = 0.95
    optimal_num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

    print(f"Optimal Number of Components ({method}): {optimal_num_components}")
    print("Explained Variance Ratio:", explained_variance_ratio[:optimal_num_components])

    return optimal_num_components, explained_variance_ratio

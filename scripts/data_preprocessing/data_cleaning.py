import re
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
sys.path.append('../utilities')
from helper_functions import *


def extract_columns_by_threshold(df, threshold_percentage):
    """
    Extract columns from a DataFrame based on a threshold percentage of missing values.

    Parameters:
    - df: pandas DataFrame
    - threshold_percentage: float, the threshold percentage for missing values

    Returns:
    - df_extracted: pandas DataFrame, a new DataFrame containing only columns exceeding the threshold
    """

    missing_percentage = (df.isnull().sum() / df.shape[0]) * 100
    columns_to_extract = missing_percentage[missing_percentage > threshold_percentage].index
    df_extracted = df[columns_to_extract].copy()

    return df_extracted


def is_repeating_sequence(value, sequence_values):
    """
    Check if a value is a sequence of repeating specified values.

    Parameters:
    - value: The value to check.
    - sequence_values: An array of values to check for.

    Returns:
    - bool: True if the value is a sequence of repeating specified values, False otherwise.
    """
    pattern = f'^({"|".join(map(str, sequence_values))})+$'
    return bool(re.match(pattern, str(value)))


def replace_repeating_sequence(df, columns, strategy='median', sequence_values=None):
    """
    Replace repeating sequences in specified columns using statistical imputation.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - columns (list): List of columns to process.
    - strategy (str, optional): The imputation strategy ('median' or 'mean'). Default is 'median'.
    - sequence_values (list, optional): List of values to check for repeating sequences. Default is None.

    Returns:
    - df (DataFrame): The DataFrame with replaced values.
    """
    for col in columns:
        if strategy == 'median':
            # Replace repeating sequences with the median value (excluding nan values)
            df[col] = df[col].apply(lambda x: df[col].median() if is_repeating_sequence(convert_float_to_int(x), sequence_values) else x)
        elif strategy == 'mean':
            # Replace repeating sequences with the mean value (excluding nan values)
            df[col] = df[col].apply(lambda x: df[col].mean() if is_repeating_sequence(convert_float_to_int(x), sequence_values) else x)
        else:
            raise ValueError("Invalid strategy. Choose from 'median' or 'mean'.")

    return df


def statistical_imputer(df, columns, strategy='median'):
    """
    Impute missing values in specified columns using statistical imputation.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - columns (list): List of columns to process.
    - strategy (str, optional): The imputation strategy ('mean', 'median', 'most_frequent'). Default is 'median'.

    Returns:
    - df (DataFrame): The DataFrame with imputed values.
    """
    # Validate the imputation strategy
    valid_strategies = ['mean', 'median', 'most_frequent']
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy. Choose from {valid_strategies}.")

    # Impute missing values using statistical imputation
    if strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
        df[columns] = imputer.fit_transform(df[columns])
    elif strategy == 'most_frequent':
        # Impute missing values with the most frequent value in each column
        for col in columns:
            mode_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_value)

    return df
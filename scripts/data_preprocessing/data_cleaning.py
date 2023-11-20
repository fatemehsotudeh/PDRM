import re
import sys
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.append('../utilities')
from helper_functions import *


def extract_columns_by_threshold(df, threshold_percentage):
    missing_percentage = (df.isnull().sum() / df.shape[0]) * 100
    columns_to_extract = missing_percentage[missing_percentage > threshold_percentage].index
    df_extracted = df[columns_to_extract].copy()
    return df_extracted


def is_repeating_sequence(value, sequence_values):
    pattern = f'^({"|".join(map(str, sequence_values))})+$'
    return bool(re.match(pattern, str(value)))


def replace_repeating_sequence(df, columns, strategy='median', sequence_values=None):
    for col in columns:
        if strategy == 'median':
            # Replace repeating sequences with the median value (excluding nan values)
            df[col] = df[col].apply(
                lambda x: df[col].median() if is_repeating_sequence(convert_float_to_int(x), sequence_values) else x)
        elif strategy == 'mean':
            # Replace repeating sequences with the mean value (excluding nan values)
            df[col] = df[col].apply(
                lambda x: df[col].mean() if is_repeating_sequence(convert_float_to_int(x), sequence_values) else x)
        else:
            raise ValueError("Invalid strategy. Choose from 'median' or 'mean'.")

    return df


def replace_target_repeating_sequence_with_constant(df, columns, constant_value, sequence_values=None):
    for col in columns:
        df[col] = df[col].apply(
            lambda x: constant_value if is_repeating_sequence(convert_float_to_int(x), sequence_values) else x)
    return df


def statistical_imputer(df, columns, strategy='median'):
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


def constant_imputer(df, columns, constant_value):
    for col in columns:
        df[col] = df[col].fillna(constant_value)
    return df


def drop_constant_columns(df):
    # Identify constant columns
    constant_columns = df.columns[df.nunique() == 1]

    # Drop constant columns
    df_filtered = df.drop(columns=constant_columns)

    return df_filtered


def drop_duplicates(df):
    return df.drop_duplicates()

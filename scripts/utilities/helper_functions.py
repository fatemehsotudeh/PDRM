import os
import pandas as pd


def read_files(*file_names, base_path=None):
    """
    Read multiple files (CSV or Excel) and return a list of DataFrames.

    Parameters:
    - *file_names: Variable number of file names relative to the base path
    - base_path (str or None): Base path for the files. If None, uses a default base path.

    Returns:
    - dfs (list): List of DataFrames.
    """
    default_base_path = '../../data/raw_data/'

    if base_path is None:
        base_path = default_base_path

    data_frames = []
    for file_name in file_names:
        full_path = os.path.join(base_path, file_name)
        _, extension = os.path.splitext(file_name)

        if extension.lower() == '.csv':
            data_frames.append(pd.read_csv(full_path))
        elif extension.lower() == '.xlsx':
            data_frames.append(pd.read_excel(full_path))
        else:
            raise ValueError(f"Unsupported file format for {file_name}. Use '.csv' or '.xlsx'.")

    return data_frames


def save_files(data_frames, *file_names, base_path=None, file_format='csv'):
    """
    Save multiple DataFrames to files (CSV or Excel).

    Parameters:
    - data_frames (list): List of DataFrames to be saved.
    - *file_names: Variable number of file names relative to the base path.
    - base_path (str or None): Base path for the files. If None, uses a default base path.
    - file_format (str): File format for saving ('csv' or 'xlsx').

    """
    default_base_path = '../../data/processed_data/'

    if base_path is None:
        base_path = default_base_path

    for df, file_name in zip(data_frames, file_names):
        full_path = os.path.join(base_path, file_name)

        if file_format.lower() == 'csv':
            df.to_csv(full_path, index=False)
        elif file_format.lower() == 'xlsx':
            df.to_excel(full_path, index=False)
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'xlsx'.")


def extract_categorical_columns(df):
    """
    Extract categorical columns from the DataFrame.

    Parameters:
    - df: The DataFrame to extract categorical columns from.

    Returns:
    - categorical_columns: DataFrame containing only categorical columns.
    """
    df_categorical = df.select_dtypes(include=['object'])
    return list(df_categorical.columns)


def extract_numerical_columns(df):
    """
    Extract numerical columns from the DataFrame.

    Parameters:
    - df: The DataFrame to extract numerical columns from.

    Returns:
    - numerical_columns: DataFrame containing only numerical columns.
    """
    df_numerical = df.select_dtypes(include=['int', 'float'])
    return list(df_numerical.columns)


def convert_float_to_int(value):
    """
    Converts a floating-point number to an integer if it is a whole number.

    Parameters:
    - value (float): The input floating-point number.

    Returns:
    - int or float: If the input is a whole number, returns the integer representation; otherwise, returns the original value.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def extract_X_columns(df):
    """
    Extract numerical columns from the DataFrame.

    Parameters:
    - df: The DataFrame to extract numerical columns from.

    Returns:
    - numerical_columns: DataFrame containing only numerical columns.
    """
    df_numerical = df.select_dtypes(include=['int', 'float'])
    return list(df_numerical.columns)





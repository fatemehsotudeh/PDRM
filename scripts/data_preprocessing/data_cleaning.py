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

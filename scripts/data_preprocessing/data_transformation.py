from sklearn.preprocessing import LabelEncoder

def label_encode(df, columns_to_encode):
    """
    Perform label encoding on the specified columns in the DataFrame.

    Parameters:
    - df: DataFrame
    - columns_to_encode: list of column names to be label encoded

    Returns:
    - DataFrame with label encoded columns
    """
    df_encoded = df.copy()
    label_encoder = LabelEncoder()

    for column in columns_to_encode:
        df_encoded[column] = label_encoder.fit_transform(df[column])

    return df_encoded
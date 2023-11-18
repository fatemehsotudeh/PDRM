from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler

def label_encode(df, columns_to_encode):

    df_encoded = df.copy()
    label_encoder = LabelEncoder()

    for column in columns_to_encode:
        df_encoded[column] = label_encoder.fit_transform(df[column])

    return df_encoded

def normalize_data(df, columns=None, method='minmax'):

    if columns is None:
        columns = df.columns

    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid normalization method. Use 'minmax' or 'zscore'.")

    normalized_data = df.copy()
    normalized_data[columns] = scaler.fit_transform(df[columns])
    return normalized_data
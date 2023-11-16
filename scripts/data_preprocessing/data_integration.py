import pandas as pd
from functools import reduce


def merge_dataframes(data_frames, on_column, join_type):
    """
    Merge a list of DataFrames on a specified column using an outer join.

    Parameters:
    - data_frames (list): List of pandas DataFrames to be merged.
    - on_column (str): Column name to merge on.
    - join_type (str): join type (outer, inner)
    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    # Use reduce and lambda to iteratively merge DataFrames
    dfs_merged = reduce(lambda left, right: pd.merge(left, right, on=on_column, how=join_type), data_frames)

    return dfs_merged

import pandas as pd
from functools import reduce


def merge_dataframes(data_frames, on_column, join_type):

    # Use reduce and lambda to iteratively merge DataFrames
    dfs_merged = reduce(lambda left, right: pd.merge(left, right, on=on_column, how=join_type), data_frames)
    return dfs_merged

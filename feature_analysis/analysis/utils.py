from typing import List, Union

import numpy as np
import pandas as pd

__all__ = ['top_percent_choices', 'remove_outliers']


def top_percent_choices(df: pd.DataFrame,
                        feature: str,
                        percent: float = 0.8) -> list:
    """get top few items which cover percent of total counts

    Args:
        df: dataframe
        feature: feature name
        percent: how many percentage of total counts

    Returns:
        list: top items 
    """
    copy_df = df.copy()
    copy_df['count'] = 1
    copy_df = copy_df.groupby(feature).agg({'count': sum})
    sort_df = copy_df.sort_values('count', ascending=False).cumsum()
    percent_index = sum(sort_df['count'] /
                        sort_df['count'].max() < percent) + 1
    top_col_choices = sort_df.index.values[:percent_index].tolist()
    return top_col_choices


def remove_outliers(data: List[Union[int, float]]) -> list:
    """remove outliers using 1.5*IQR

    Args:
        data: numerical data

    Returns:
        list: remain data with original order
    """
    assert isinstance(data, (list, tuple))
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    lower, upper = np.percentile(data, 25) - 1.5 * IQR, np.percentile(
        data, 75) + 1.5 * IQR
    return [i for i in data if lower <= i <= upper]

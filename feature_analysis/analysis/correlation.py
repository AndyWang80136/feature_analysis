from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ['df_correlation']


def df_correlation(df: pd.DataFrame,
                   method: str = 'pearson',
                   show_image: bool = False,
                   features: Optional[List[str]] = None,
                   threshold: float = 0.2,
                   descending: bool = True) -> list:
    """correation of dataframe feature columns

    Args:
        df: dataframe
        method: 'pearson' or 'spearman'
        show_image: whether to show image
        features: List of feature name
        threshold: absolute threshold for valid high enough correlation
        descending: whether return in descending order

    Returns:
        list: [(correlation, feature1, feature2), ...]
    """
    assert isinstance(df,
                      pd.DataFrame), f'only accept dataframe, got {type(df)}'
    corr = df[features].corr(
        method=method) if features is not None else df.corr(method=method)
    if show_image:
        sns.heatmap(corr,
                    annot=True,
                    cmap='Blues',
                    fmt='.2f',
                    annot_kws={"fontsize": 8})
        plt.title(f'{method.capitalize()} Correlation')
        plt.show()

    r_feats = []
    corr_triu = np.triu(corr)
    high_corr_index = np.where(np.abs(corr_triu) >= threshold)
    for i, j in zip(*high_corr_index):
        if i != j:
            r_feats.append((corr_triu[i, j], corr.index[i], corr.index[j]))

    return sorted(r_feats, key=lambda a: abs(a[0]), reverse=descending)

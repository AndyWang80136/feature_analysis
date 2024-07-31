from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api
from scipy.stats import chi2_contingency

from .utils import remove_outliers

__all__ = ['hypothesis_test']


def hypothesis_test(user_df: pd.DataFrame,
                    feature: str,
                    label: str,
                    test_type: str,
                    times: int = 5,
                    random_seed: int = 42) -> dict:
    """hypothesis testing for Chi-Square test of indenpendence and z-test

    Args:
        user_df: user dataframe
        feature: feature name
        label: label name
        test_type: 'chi_square_independence' or 'ztest'
        times: repeat times
        random_seed: random seed

    Returns:
        dict: total times and significant counts
    """
    np.random.seed(random_seed)
    sample_flag = True
    df = user_df.copy()
    type_feature = type(df[feature].iloc[0])
    type_label = type(df[label].iloc[0])
    if type_feature == type_label == list:
        assert df[feature].apply(len).equals(df[label].apply(len))
        df['length'] = df[feature].apply(len)
    elif type_feature is list:
        df['length'] = df[feature].apply(len)
    elif type_label is list:
        df['length'] = df[label].apply(len)
    else:
        sample_flag = False

    p_values = []
    for _ in range(times):

        if sample_flag:
            df['sample_index'] = df['length'].apply(
                lambda a: np.random.randint(low=0, high=a))
            df['sample_feature'] = df.apply(
                lambda a: a[feature][a['sample_index']],
                axis=1) if type_feature is list else df[feature]
            df['sample_label'] = df.apply(
                lambda a: a[label][a['sample_index']],
                axis=1) if type_label is list else df[label]
        else:
            df['sample_feature'] = df[feature]
            df['sample_label'] = df[label]

        if test_type == 'chi_square_independence':
            contingency = df[['sample_feature',
                              'sample_label']].groupby('sample_label').agg(
                                  {'sample_feature':
                                   Counter})['sample_feature'].apply(pd.Series)
            # make sure each cell is larger than 5 counts
            contingency = contingency[contingency >= 5].dropna(axis=1)
            result = chi2_contingency(contingency.T)
            p_values.append(result.pvalue)
        elif test_type == 'ztest':
            value = df[['sample_feature',
                        'sample_label']].groupby('sample_label').agg(
                            {'sample_feature': list})['sample_feature'].values
            assert len(value) == 2
            # remove outliers with 1.5IQR
            value_0, value_1 = np.asarray(remove_outliers(
                value[0])), np.asarray(remove_outliers(value[1]))
            _, pvalue = statsmodels.stats.weightstats.ztest(value_0,
                                                            value_1,
                                                            value=0)
            p_values.append(pvalue)
        else:
            raise NotImplementedError(
                f'Not support {test_type}, choose from (chi_square_independence, ztest)'
            )

    if times > 1:
        # apply multipletests correction
        reject_array, *_ = statsmodels.stats.multitest.multipletests(
            p_values, method='fdr_bh')
        return dict(total_times=times, significant_count=reject_array.sum())
    else:
        return dict(total_times=times,
                    significant_count=(np.asarray(p_values) <= 0.05).sum())

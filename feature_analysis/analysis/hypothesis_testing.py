from collections import Counter
from typing import Optional

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
                    ztest_sample_size: Optional[int] = None,
                    times: int = 5,
                    random_seed: int = 42) -> dict:
    """hypothesis testing for Chi-Square test of indenpendence and z-test

    Args:
        user_df: user dataframe
        feature: feature name
        label: label name
        test_type: 'chi_square_independence' or 'ztest'
        ztest_sample_size": z-test sample_size per sample
        times: repeat times
        random_seed: random seed

    Returns:
        dict: total times and significant counts
    """
    if test_type == 'ztest' and ztest_sample_size is None:
        ztest_sample_size = int(
            statsmodels.stats.power.zt_ind_solve_power(
                effect_size=0.5,
                nobs1=None,
                alpha=0.05,
                power=0.8,
                ratio=1.0,
                alternative='two-sided')) + 1

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

    count = 0
    total_count = times
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
            count += (result.pvalue <= 0.05)
        elif test_type == 'ztest':
            value = df[['sample_feature',
                        'sample_label']].groupby('sample_label').agg(
                            {'sample_feature': list})['sample_feature'].values
            assert len(value) == 2
            # remove outliers with 1.5IQR
            value_0, value_1 = np.asarray(remove_outliers(
                value[0])), np.asarray(remove_outliers(value[1]))

            sample_0 = np.asarray(
                np.random.choice(value_0,
                                 size=ztest_sample_size,
                                 replace=False))
            sample_1 = np.asarray(
                np.random.choice(value_1,
                                 size=ztest_sample_size,
                                 replace=False))

            _, pvalue = statsmodels.stats.weightstats.ztest(sample_0,
                                                            sample_1,
                                                            value=0)
            count += (pvalue <= 0.05)
        else:
            raise NotImplementedError(
                f'Not support {test_type}, choose from (chi_square_independence, ztest)'
            )
    return dict(total_times=total_count, significant_count=count)
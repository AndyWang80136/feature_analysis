from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from feature_analysis.analysis import (df_correlation, hypothesis_test,
                                       random_forest_importance,
                                       remove_outliers, top_percent_choices)


@pytest.fixture
def random_df():
    return pd.DataFrame(
        [[1, 0, 0], [0, 2, 2], [0, 1, 1], [0, 0, 1], [0, 0, 2], [0, 0, 1]],
        columns=['colA', 'colB', 'colC'])


@pytest.fixture
def hypothesis_test_df(request):
    params = getattr(request, 'param', {})

    feature = [np.random.randint(2, size=(3, )).tolist()
               for _ in range(1000)] if params.get('feature') == 'list' else [
                   np.random.randint(2) for _ in range(1000)
               ]
    label = [np.random.randint(2, size=(3, )).tolist()
             for _ in range(1000)] if params.get('label') == 'list' else [
                 np.random.randint(2) for _ in range(1000)
             ]

    return pd.DataFrame(zip(feature, label), columns=['colA', 'colB'])


@pytest.mark.parametrize('kwargs, expected',
                         [(dict(feature='colA', percent=0.8), [0]),
                          (dict(feature='colA', percent=1.0), [0, 1]),
                          (dict(feature='colB', percent=0.5), [0]),
                          (dict(feature='colC', percent=0.8), [1, 2])])
def test_top_percent_choices(random_df, kwargs, expected):
    top_choices = top_percent_choices(random_df, **kwargs)
    assert isinstance(top_choices, list)
    assert top_choices == expected


@pytest.mark.parametrize('data, expected, raise_error',
                         [([-100, 1, 1, 1, 100], [1, 1, 1], nullcontext()),
                          ([-5, 1, 2, 3, 10], [1, 2, 3], nullcontext()),
                          ((-5, 1, 2, 3, 10), [1, 2, 3], nullcontext()),
                          (1, [1, 2, 3], pytest.raises(AssertionError))])
def test_remove_outliers(data, expected, raise_error):
    with raise_error:
        result = remove_outliers(data)
        assert result == expected


@pytest.mark.parametrize('kwargs', [
    dict(method='pearson', descending=True),
    dict(method='spearman', descending=True),
    dict(method='pearson', descending=False),
    dict(method='spearman', descending=False),
    dict(method='pearson', descending=False, features=['colA', 'colB']),
    dict(method='spearman', descending=False, features=['colC', 'colB']),
    dict(method='spearman',
         descending=False,
         features=['colC', 'colB'],
         threshold=0.1)
])
def test_df_correlation(random_df, kwargs):
    corr = df_correlation(random_df, **kwargs)
    assert isinstance(corr, list)
    assert all(isinstance(i, tuple) for i in corr)
    assert corr == sorted(corr,
                          key=lambda a: abs(a[0]),
                          reverse=kwargs.get('descending', True))
    if 'features' in kwargs:
        assert all(i[1] in kwargs['features'] and i[2] in kwargs['features']
                   for i in corr)


@pytest.mark.parametrize('hypothesis_test_df', [
    dict(feature='list', label='list'),
    dict(feature='list'),
    dict(label='list'),
    dict()
],
                         indirect=True)
def test_chi_square_hypothesis_test(hypothesis_test_df):
    result = hypothesis_test(hypothesis_test_df,
                             feature='colB',
                             label='colA',
                             times=10,
                             test_type='chi_square_independence')
    assert isinstance(
        result,
        dict) and 'significant_count' in result and 'total_times' in result
    assert result['total_times'] == 10


@pytest.mark.parametrize('hypothesis_test_df', [
    dict(feature='list', label='list'),
    dict(feature='list'),
    dict(label='list'),
    dict()
],
                         indirect=['hypothesis_test_df'])
def test_ztest_hypothesis_test(hypothesis_test_df):
    result = hypothesis_test(hypothesis_test_df,
                             feature='colB',
                             label='colA',
                             times=5,
                             test_type='ztest')
    assert isinstance(
        result,
        dict) and 'significant_count' in result and 'total_times' in result


def test_raise_error_hypothesis_test(hypothesis_test_df):
    with pytest.raises(NotImplementedError):
        result = hypothesis_test(hypothesis_test_df,
                                 feature='colB',
                                 label='colA',
                                 times=5,
                                 test_type='invalid_test')


@pytest.mark.parametrize('kwargs', [
    dict(model_kwargs=dict(min_samples_leaf=20, random_state=42)),
    dict(model_kwargs=dict(min_samples_leaf=20, random_state=42),
         apply_permutation_importance=True)
])
def test_random_forest_importance(random_df, kwargs):
    label = random_df.pop('colA')
    result = random_forest_importance(random_df, label, random_df, label,
                                      **kwargs)
    assert isinstance(result, dict)
    assert 'metric_name' in result
    assert 'metric_value' in result
    assert 'feature_importance' in result
    assert isinstance(result['feature_importance'], list)
    assert all(
        isinstance(i, tuple) and isinstance(i[0], str)
        and isinstance(i[1], float) for i in result['feature_importance'])

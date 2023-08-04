from typing import Optional

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

__init__ = ['random_forest_importance']


def random_forest_importance(
        train_df: pd.DataFrame,
        train_label: pd.DataFrame,
        test_df: pd.DataFrame,
        test_label: pd.DataFrame,
        model_kwargs: Optional[dict] = None,
        apply_permutation_importance: bool = False,
        permutation_importance_kwargs: Optional[dict] = None) -> dict:
    """get feature importances by random forest 

    Args:
        train_df: train dataframe
        train_label: train label
        test_df: test dataframe
        test_label: test label
        model_kwargs: random forest kwargs
        apply_permutation_importance: whether to apply permutation importance
        permutation_importance_kwargs: permutation importance kwargs

    Returns:
        dict: return dict with metric with random forest and feature importance from model
    """
    if model_kwargs is None: model_kwargs = {}
    if permutation_importance_kwargs is None:
        permutation_importance_kwargs = {}

    forest = RandomForestClassifier(**model_kwargs)
    forest.fit(train_df, train_label)
    pred = forest.predict_proba(test_df)
    auc = metrics.roc_auc_score(test_label, pred[:, 1])

    if apply_permutation_importance:
        result = permutation_importance(forest,
                                        test_df,
                                        test_label,
                                        scoring=['roc_auc'],
                                        **permutation_importance_kwargs)
        result = result['roc_auc']
        feature_importance = list(
            zip(forest.feature_names_in_, result['importances_mean']))
    else:
        feature_importance = list(
            zip(forest.feature_names_in_, forest.feature_importances_))
    feature_importance = sorted(feature_importance,
                                key=lambda a: a[-1],
                                reverse=True)
    return dict(metric_name='auc',
                metric_value=auc,
                feature_importance=feature_importance)

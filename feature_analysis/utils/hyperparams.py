from itertools import product
from typing import List

__all__ = ['parse_hyperparams']


def parse_hyperparams(hyperparams: dict) -> List[dict]:
    """parse hyperparameters with certain patterns

    Args:
        hyperparams: hyperparameters dict

    Returns:
        List[dict]: List of hyperparameter combinations
    """
    assert isinstance(hyperparams, dict)
    assert all(isinstance(v, (list, tuple)) for v in hyperparams.values())
    value_pairs = [i for i in product(*hyperparams.values())]
    value_pairs_with_key = [
        dict(zip(hyperparams.keys(), i)) for i in value_pairs
    ]
    return value_pairs_with_key

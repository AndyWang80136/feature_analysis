from contextlib import nullcontext
from pathlib import Path

import pytest

from feature_analysis.utils import flatten_dict, load_yaml, parse_hyperparams


def test_load_yaml(tmp_path):
    test_config = Path(tmp_path).joinpath('test.yaml')
    with open(test_config, 'w') as fp:
        fp.writelines('A: "A"\nB: 1\nC: [0, 1]')
    data = load_yaml(test_config)
    assert isinstance(data, dict)
    assert data == {'A': 'A', 'B': 1, 'C': [0, 1]}


@pytest.mark.parametrize('params, expected, raise_error', [
    (dict(param_a=[0.1, 0.01], param_b=['a', 'b']), [
        dict(param_a=0.1, param_b='a'),
        dict(param_a=0.1, param_b='b'),
        dict(param_a=0.2, param_b='a'),
        dict(param_a=0.2, param_b='b')
    ], nullcontext()),
    (dict(param_a=0.1, param_b=['a', 'b']), [
        dict(param_a=0.1, param_b='a'),
        dict(param_a=0.1, param_b='b'),
        dict(param_a=0.2, param_b='a'),
        dict(param_a=0.2, param_b='b')
    ], pytest.raises(AssertionError)),
    ([1, 2, 3], [1, 2, 3], pytest.raises(AssertionError))
])
def test_parse_hyperparams(params, expected, raise_error):
    with raise_error:
        parsed_params = parse_hyperparams(params)
        assert isinstance(parsed_params, list)
        assert all(isinstance(i, dict) for i in parsed_params)
        parsed_params == expected


def test_flatten_dict():
    data = {'a': 1, 'b': ['1', '2'], 'c': {'d': 2}}
    flatten_data = flatten_dict(data)
    assert flatten_data == {'a': 1, 'b': '1,2', 'c/d': 2}

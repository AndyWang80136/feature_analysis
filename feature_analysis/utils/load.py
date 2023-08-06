from pathlib import Path
from typing import Union

import yaml

__all__ = ['load_yaml']

def load_yaml(yaml_file: Union[str, Path]) -> dict:
    """load yaml file

    Args:
        yaml_file: yaml file path

    Returns:
        dict: yaml content in dict format
    """
    with open(yaml_file, 'r') as fp:
        data_dict = yaml.safe_load(fp)
    return data_dict

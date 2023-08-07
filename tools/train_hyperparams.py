import json
import random
from pathlib import Path
from typing import Union

import fire
import numpy as np
import torch
from loguru import logger

from feature_analysis.process import train_process
from feature_analysis.utils import load_yaml, parse_hyperparams


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config: str, log_dir: Union[Path, str]):
    config_dict = load_yaml(config)
    random_seed = config_dict.get('random_seed', 42)
    set_random_seed(seed=random_seed)

    results = []
    hyperparams = parse_hyperparams(config_dict.pop('hyperparams'))
    for hparam in hyperparams:
        input_dict = {**config_dict, **hparam}
        logger.info(input_dict)
        metric = train_process(**input_dict)
        info = {**input_dict, **metric}
        results.append(info)

    top_results = sorted(results, key=lambda r: r['best_value'], reverse=True)
    logger.info(f'The best params settings: {top_results[0]}')

    if not Path(log_dir).exists():
        Path(log_dir).mkdir(exist_ok=True, parents=True)

    with open(Path(log_dir).joinpath('hyperparam_results.json'), 'w') as fp:
        json.dump(top_results, fp)


if __name__ == '__main__':
    fire.Fire(train)

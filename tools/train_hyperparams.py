import json
import random
from pathlib import Path
from typing import Union

import fire
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from feature_analysis.process import train_process
from feature_analysis.utils import flatten_dict, load_yaml, parse_hyperparams


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config: str, log_dir: Union[Path, str]):
    config_dict = load_yaml(config)
    random_seed = config_dict.get('random_seed', 42)
    set_random_seed(seed=random_seed)

    writer = SummaryWriter(log_dir=log_dir, flush_secs=15)

    results = []
    hyperparams = parse_hyperparams(config_dict.pop('hyperparams'))
    for idx, hparam in enumerate(hyperparams):
        input_dict = {**config_dict, **hparam}
        logger.info(f'[{idx+1}/{len(hyperparams)}]: {input_dict}')
        metric = train_process(**input_dict)
        info = {**input_dict, **metric}
        writer.add_hparams(
            hparam_dict=flatten_dict(input_dict),
            metric_dict=flatten_dict(metric),
        )
        results.append(info)
    writer.close()
    
    top_results = sorted(results, key=lambda r: r['metrics']['auc'], reverse=True)
    logger.info(f'The best params settings: {top_results[0]}')

    if not Path(log_dir).exists():
        Path(log_dir).mkdir(exist_ok=True, parents=True)

    with open(Path(log_dir).joinpath('hyperparam_results.json'), 'w') as fp:
        json.dump(top_results, fp)


if __name__ == '__main__':
    fire.Fire(train)

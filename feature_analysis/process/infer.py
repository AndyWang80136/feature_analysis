from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from ..data import DatasetLoader
from .utils import load_dcn_model

__all__ = ['infer_df']


def infer_df(df: pd.DataFrame,
          ckpt: Union[Path, str],
          dataset_config: Union[Path, str],
          random_seed: int = 42,
          batch_size: int = 128) -> np.ndarray:
    """infer function

    Args:
        df: dataframe
        ckpt: checkpoint path
        dataset_config: dataset config path
        random_seed: random seed
        batch_size: inference batch size

    Returns:
        np.ndarray: prediction scores
    """
    config = joblib.load(dataset_config)
    dataset = DatasetLoader.load(**config)
    trns_df = dataset.transform_df(df=df)
    trns_data = {col: trns_df[col].values for col in trns_df.columns}

    model = load_dcn_model(ckpt=ckpt, random_seed=random_seed)
    pred_ans = model.predict(trns_data, batch_size)
    return pred_ans.ravel()

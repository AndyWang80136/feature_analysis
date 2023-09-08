from pathlib import Path
from typing import Union

import torch
from deepctr_torch.models import DCN

__all__ = ['load_dcn_model']


def load_dcn_model(ckpt: Union[Path, str], random_seed: int = 42):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')
    ckpt = torch.load(ckpt, map_location=device)
    model_cfg, feature_columns = ckpt['config']['model'], ckpt['config'][
        'feature_columns']
    dcn_model = DCN(feature_columns,
                    feature_columns,
                    task='binary',
                    cross_num=model_cfg.get('cross_num', 2),
                    dnn_hidden_units=model_cfg.get('dnn_hidden_units',
                                                   (128, 128)),
                    l2_reg_linear=0.0001,
                    l2_reg_embedding=0.0001,
                    l2_reg_cross=0.0001,
                    l2_reg_dnn=0.0001,
                    init_std=0.0001,
                    seed=random_seed,
                    dnn_dropout=0.2,
                    dnn_activation='relu',
                    dnn_use_bn=True,
                    cross_parameterization='matrix',
                    device=device)
    dcn_model.load_state_dict(ckpt['model'])
    return dcn_model

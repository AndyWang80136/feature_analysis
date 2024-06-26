import importlib
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import DenseFeat, SparseFeat
from deepctr_torch.models import DCN
from recommenders.utils.timer import Timer
from sklearn.metrics import log_loss, roc_auc_score

from ..data import DatasetLoader

__all__ = ['train_process']


def train_process(data: dict,
                  model: dict,
                  lr: float,
                  optimizer: str,
                  epochs: int,
                  features: dict,
                  random_seed: int = 42,
                  save_model_dir: Optional[Union[Path, str]] = None) -> dict:
    """train process

    Args:
        data: data kwargs
        model: model architecture kwargs
        lr: learning rate
        optimizer: optimizer
        epochs: number of epochs
        features: feature dict with numerical and categorical feature list
        random_seed: random seed
        save_model_dir: model saved directory

    Returns:
        dict: training metric and result 
    """

    dataset = DatasetLoader.load(**data, **features)
    phase_data = dataset.phase_data
    train_df, train_label = phase_data['train']
    val_df, val_label = phase_data['val']
    test_df, test_label = phase_data['test']

    num_features = dataset.num_features

    train_data = {col: train_df[col].values for col in train_df.columns}
    val_data = {col: val_df[col].values for col in val_df.columns}
    test_data = {col: test_df[col].values for col in test_df.columns}

    train_label, val_label, test_label = train_label.values, val_label.values, test_label.values

    # model features
    feature_columns = [
        SparseFeat(feat,
                   vocabulary_size=num_features[feat],
                   embedding_dim=model.get('embedding_dim', 4))
        for feat in dataset.categorical
    ] + [DenseFeat(
        feat,
        num_features[feat],
    ) for feat in dataset.numerical]
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda:0')

    dcn_model = DCN(feature_columns,
                    feature_columns,
                    task='binary',
                    cross_num=model.get('cross_num', 2),
                    dnn_hidden_units=model.get('dnn_hidden_units', (128, 128)),
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

    # optimizer
    optimizer = getattr(importlib.import_module('torch.optim'),
                        optimizer)(params=dcn_model.parameters(), lr=lr)
    dcn_model.compile(
        optimizer,
        "binary_crossentropy",
        metrics=["binary_crossentropy", "auc"],
    )

    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        tmp_ckpt = Path(tmp_dir).joinpath('val_best.pth')
        with Timer() as train_time:
            history = dcn_model.fit(train_data,
                                    train_label,
                                    batch_size=128,
                                    epochs=epochs,
                                    verbose=0,
                                    validation_data=(val_data, val_label),
                                    callbacks=[
                                        ModelCheckpoint(
                                            filepath=tmp_ckpt,
                                            monitor='val_auc',
                                            mode='max',
                                            save_best_only=True,
                                            save_weights_only=True),
                                        EarlyStopping(monitor='val_auc',
                                                      patience=10,
                                                      mode='max')
                                    ])
        if save_model_dir is not None:
            shutil.copytree(src=tmp_dir,
                            dst=save_model_dir,
                            dirs_exist_ok=True)
            dataset.save(save_dir=save_model_dir)
            torch.save(
                {
                    'model':
                    torch.load(Path(save_model_dir).joinpath('val_best.pth')),
                    'config': {
                        'feature_columns': feature_columns,
                        'model': model
                    }
                },
                Path(save_model_dir).joinpath('val_best.pth'))
        total_epoch = len(history.history['loss'])
        ckpt = torch.load(tmp_ckpt)
        dcn_model.load_state_dict(ckpt)

    pred_ans = dcn_model.predict(test_data, 128)
    test_loss = round(log_loss(test_label, pred_ans), 4)
    test_auc = round(roc_auc_score(test_label, pred_ans), 4)

    return dict(
        metrics={
            'auc': test_auc,
            'logloss': test_loss
        },
        train_time={
            'time': train_time.interval,
            'total_epoch': total_epoch
        },
    )

from pathlib import Path

import numpy as np
from deepctr_torch.models import DCN

from feature_analysis.data import RandomDataset
from feature_analysis.process import infer_df, load_dcn_model, train_process


def test_train_process(tmp_path):
    result = train_process(data={'name': 'RandomDataset'},
                           model={},
                           features={},
                           lr=0.01,
                           optimizer='SGD',
                           epochs=1,
                           save_model_dir=tmp_path)
    assert isinstance(result, dict)
    assert 'metrics' in result
    assert 'train_time' in result
    assert 'total_epoch' in result['train_time'] and 'time' in result[
        'train_time']
    assert result['train_time']['total_epoch'] == 1
    assert Path(tmp_path).joinpath('dataset.pkl').exists()
    assert Path(tmp_path).joinpath('val_best.pth').exists()


def test_load_dcn_model(tmp_path):
    _ = train_process(data={'name': 'RandomDataset'},
                      model={},
                      features={},
                      lr=0.01,
                      optimizer='SGD',
                      epochs=1,
                      save_model_dir=tmp_path)
    model = load_dcn_model(ckpt=Path(tmp_path).joinpath('val_best.pth'))
    assert isinstance(model, DCN)


def test_infer_df(tmp_path):
    phase_data = RandomDataset().phase_data
    test_df, _ = phase_data['test']
    _ = train_process(data={'name': 'RandomDataset'},
                      model={},
                      features={},
                      lr=0.01,
                      optimizer='SGD',
                      epochs=1,
                      save_model_dir=tmp_path)
    pred = infer_df(df=test_df,
                    ckpt=Path(tmp_path).joinpath('val_best.pth'),
                    dataset_config=Path(tmp_path).joinpath('dataset.pkl'))
    assert isinstance(pred, np.ndarray)

from pathlib import Path

from feature_analysis.process import train_process


def test_train_process(tmp_path):
    tmp_path = 'test_test'
    result = train_process(data={'name': 'RandomDataset'},
                           model={},
                           features={},
                           lr=0.01,
                           optimizer='SGD',
                           epochs=1,
                           save_model_dir=tmp_path)
    assert isinstance(result, dict)
    assert 'best_metric' in result
    assert 'best_value' in result
    assert 'other_metrics' in result
    assert 'train_time' in result
    assert 'total_epoch' in result['train_time'] and 'time' in result[
        'train_time']
    assert result['train_time']['total_epoch'] == 1
    assert Path(tmp_path).joinpath('dataset.pkl').exists()
    assert Path(tmp_path).joinpath('val_best.pth').exists()
    

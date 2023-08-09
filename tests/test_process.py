from feature_analysis.process import train_process


def test_train_process():
    result = train_process(data={'name': 'RandomDataset'},
                           model={},
                           features={},
                           lr=0.01,
                           optimizer='SGD',
                           epochs=1)
    assert isinstance(result, dict)
    assert 'best_metric' in result
    assert 'best_value' in result
    assert 'other_metrics' in result
    assert 'train_time' in result
    assert 'total_epoch' in result['train_time'] and 'time' in result[
        'train_time']
    assert result['train_time']['total_epoch'] == 1

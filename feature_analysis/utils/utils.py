__all__ = ['flatten_dict']


def flatten_dict(data: dict, prefix: str = ''):
    assert isinstance(data, dict)
    flat_dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(data=v, prefix=f'{k}'))
        elif isinstance(v, (list, tuple)):
            flat_dict[k if not prefix else f'{prefix}/{k}'] = ','.join(
                map(str, v))
        else:
            flat_dict[k if not prefix else f'{prefix}/{k}'] = v
    return flat_dict

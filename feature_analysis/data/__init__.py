from .dataset import *


class DatasetLoader:
    MODULES = {**globals()}

    @staticmethod
    def add(name, module):
        DatasetLoader.MODULES[name] = module

    @staticmethod
    def load(name: str, **kwargs):
        """load specific dataset with its name

        Args:
            name: dataset class name

        Returns:
            dataset
        """
        dataset = DatasetLoader.MODULES[name](**kwargs)
        assert hasattr(dataset, 'phase_data') and hasattr(
            dataset, 'numerical') and hasattr(
                dataset, 'categorical') and hasattr(dataset, 'num_features')
        return dataset

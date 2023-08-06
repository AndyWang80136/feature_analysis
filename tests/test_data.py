from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_analysis.data import (CATEGORICAL, ML100K, NUMERICAL,
                                   RandomDataset, load_dataset)

DATA_DIR = Path(__file__).parent.parent.joinpath('data')


def test_load_dataset():
    dataset = load_dataset(name='RandomDataset')
    assert isinstance(dataset, RandomDataset)


class TestML100K:

    @pytest.mark.skipif(not DATA_DIR.exists(), reason=f'{DATA_DIR} not exists')
    def setup_method(self):
        self.numerical = ['timestamp', 'year', 'age', 'freshness']
        self.categorical = [
            'user_id', 'item_id', 'gender', 'occupation', 'age_interval'
        ]
        self.dataset = ML100K(data_dir=DATA_DIR,
                              categorical=self.categorical,
                              numerical=self.numerical)

    @pytest.mark.skipif(not DATA_DIR.exists(), reason=f'{DATA_DIR} not exists')
    def test_load(self):
        df = self.dataset.load_df(data_dir=DATA_DIR)
        assert isinstance(df, pd.DataFrame)
        assert all(i in df.columns for i in NUMERICAL)
        assert all(i in df.columns for i in CATEGORICAL)

    @pytest.mark.skipif(not DATA_DIR.exists(), reason=f'{DATA_DIR} not exists')
    def test_load_ml100k_user(self):
        self.dataset.load_df(data_dir=DATA_DIR)
        user_df = ML100K.load_ml100k_user(data_dir=DATA_DIR)
        assert isinstance(user_df, pd.DataFrame)
        assert 'user_id' in user_df.columns
        assert 'gender' in user_df.columns
        assert 'occupation' in user_df.columns
        assert 'age' in user_df.columns

    def test_numerical_fillnan(self):
        test_df = pd.DataFrame(
            [[1, 2, 4], [2, 3, 4], [5, 4, np.nan], [None, np.nan, 4]],
            columns=NUMERICAL)
        processed_df = ML100K.numerical_fillnan(test_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert all(processed_df[NUMERICAL].notna())

    def test_categorical_fillnan(self):
        test_df = pd.DataFrame(
            [['a1', 'b1', 'c1', 'd1'], ['a2', 'b1', 'c2', 'd1'],
             ['a1', np.nan, None, 'd1'], ['a3', 'b2', None, 'd1']],
            columns=CATEGORICAL)
        processed_df = ML100K.categorical_fillnan(test_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert all(processed_df[CATEGORICAL].notna())

    def test_create_age_interval(self):
        test_df = pd.DataFrame([15, 40, 65], columns=['age'])
        processed_df = ML100K.create_age_interval(test_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert 'age_interval' in processed_df.columns
        assert list(processed_df['age_interval']) == ['15-39', '40-64', '65-']
        with pytest.raises(AssertionError):
            ML100K.create_age_interval(
                pd.DataFrame([21, 51, 61, 45],
                             columns=['column']))  # wrong column name

    def test_create_freshness(self):
        test_df = pd.DataFrame(
            [['2023', datetime(year=2023, month=1, day=1).timestamp()],
             ['2020', datetime(year=2022, month=1, day=1).timestamp()]],
            columns=['year', 'timestamp'])
        processed_df = ML100K.create_freshness(test_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert 'freshness' in processed_df.columns
        assert list(processed_df['freshness']) == [0, 2]
        with pytest.raises(AssertionError):
            ML100K.create_age_interval(
                pd.DataFrame([21, 51, 61, 45],
                             columns=['column']))  # wrong column name

    @pytest.mark.skipif(not DATA_DIR.exists(), reason=f'{DATA_DIR} not exists')
    def test_phase_data(self):
        phase_data1 = self.dataset.phase_data
        assert isinstance(phase_data1, dict)
        assert set(['train', 'val', 'test']) == phase_data1.keys()
        assert all(isinstance(data, tuple) for data in phase_data1.values())
        self.dataset._phase_data = None
        phase_data2 = self.dataset.phase_data
        assert {
            phase_data1[phase][0].equals(phase_data2[phase][0])
            and phase_data1[phase][1].equals(phase_data2[phase][1])
            for phase in ['train', 'val', 'test']
        }
        self.dataset._phase_data = None
        self.dataset.apply_preprocessing = False
        phase_data3 = self.dataset.phase_data
        assert isinstance(phase_data3, dict)
        assert set(['train', 'val', 'test']) == phase_data3.keys()
        assert all(isinstance(data, tuple) for data in phase_data3.values())

    @pytest.mark.skipif(not DATA_DIR.exists(), reason=f'{DATA_DIR} not exists')
    def test_num_features(self):
        _ = self.dataset.phase_data
        num_features = self.dataset.num_features
        assert isinstance(num_features, dict)
        assert set(self.categorical + self.numerical) == num_features.keys()
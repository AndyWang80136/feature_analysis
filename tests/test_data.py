import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from feature_analysis.data import (CATEGORICAL, ML100K, NUMERICAL,
                                   DatasetLoader, RandomDataset)

DATA = Path(__file__).parent.parent.joinpath('data').joinpath('ml-100k.zip')


def test_dataset_loader():
    dataset = DatasetLoader.load(name='RandomDataset')
    assert isinstance(dataset, RandomDataset)

    class CustomDatasest:

        @property
        def phase_data(self):
            return {}

        @property
        def num_features(self):
            return {}

        @property
        def categorical(self):
            return {}

        @property
        def numerical(self):
            return {}

    DatasetLoader.add(name='CustomDatasest', module=CustomDatasest)
    dataset = DatasetLoader.load(name='CustomDatasest')
    assert isinstance(dataset, CustomDatasest)


class TestML100K:

    @pytest.mark.skipif(not DATA.exists(), reason=f'{DATA} not exists')
    def setup_method(self):
        self.numerical = ['timestamp', 'year', 'age', 'freshness']
        self.categorical = [
            'user_id', 'item_id', 'gender', 'occupation', 'age_interval',
            'day', 'day_type', 'day_hour', 'day_interval'
        ]
        self.dataset = ML100K(data_dir=DATA.parent,
                              categorical=self.categorical,
                              numerical=self.numerical,
                              rating_threshold=3,
                              drop_threshold=True)

    @pytest.mark.skipif(not DATA.exists(), reason=f'{DATA} not exists')
    def test_load(self):
        df = self.dataset.load_df(data_dir=DATA.parent)
        assert isinstance(df, pd.DataFrame)
        assert all(i in df.columns for i in NUMERICAL)
        assert all(i in df.columns for i in CATEGORICAL)

    @pytest.mark.skipif(not DATA.exists(), reason=f'{DATA} not exists')
    def test_load_ml100k_user(self):
        self.dataset.load_df(data_dir=DATA.parent)
        user_df = ML100K.load_ml100k_user(data_dir=DATA.parent)
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
        test_df = pd.DataFrame([24, 40, 41], columns=['age'])
        processed_df = ML100K.create_age_interval(test_df)
        assert isinstance(processed_df, pd.DataFrame)
        assert 'age_interval' in processed_df.columns
        assert list(processed_df['age_interval']) == ['1-24', '31-40', '41-']
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

    def test_create_year(self):
        test_df = pd.DataFrame(
            [['movie_title1 (2023)', 2023], ['movie_title2 (2000)', 2000]],
            columns=['movie_title', 'year'])
        processed_df = ML100K.create_year(test_df)
        assert 'year' in processed_df.columns
        assert processed_df['year'].values.tolist() == [2023, 2000]
        test_df = test_df.drop(columns=['year'])
        processed_df = ML100K.create_year(test_df)
        assert 'year' in processed_df.columns
        assert processed_df['year'].values.tolist() == [2023, 2000]

    def test_create_datetime(self):
        test_df = pd.DataFrame([[time.time()]], columns=['timestamp'])
        processed_df = ML100K.create_datetime(test_df)
        assert 'datetime' in processed_df.columns
        assert all(isinstance(i, datetime) for i in processed_df['datetime'])
        with pytest.raises(AssertionError):
            test_df = test_df.drop(columns=['timestamp'])
            ML100K.create_datetime(test_df)

    def test_create_datetime_related_attr(self):
        now_datetime = datetime.now()
        test_df = pd.DataFrame([[now_datetime]], columns=['datetime'])
        assert ML100K.create_day(
            test_df)['day'].values[0] == now_datetime.weekday()
        assert ML100K.create_day_type(test_df)['day_type'].values[0] in [
            'weekday', 'weekend'
        ]
        assert 0 <= ML100K.create_day_hour(test_df)['day_hour'].values[0] <= 23
        assert ML100K.create_day_interval(
            test_df)['day_interval'].values[0] in [
                'early_morning', 'morning', 'afternoon', 'night'
            ]
        with pytest.raises(AssertionError):
            test_df = test_df.drop(columns=['datetime'])
            ML100K.create_day(test_df)
            ML100K.create_day_type(test_df)
            ML100K.create_day_hour(test_df)
            ML100K.create_day_interval(test_df)

    @pytest.mark.skipif(not DATA.exists(), reason=f'{DATA} not exists')
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
        self.dataset.numerical = []
        self.dataset.categorical = ['user_id', 'item_id']
        phase_data3 = self.dataset.phase_data
        assert isinstance(phase_data3, dict)
        assert set(['train', 'val', 'test']) == phase_data3.keys()
        assert all(isinstance(data, tuple) for data in phase_data3.values())
        assert {
            phase_data2[phase][0][['user_id', 'item_id']].equals(
                phase_data3[phase][0][['user_id', 'item_id']])
            and all(phase_data2[phase][0] == phase_data3[phase][0])
            and phase_data2[phase][1].equals(phase_data3[phase][1])
            and all(phase_data2[phase][1] == phase_data3[phase][1])
            for phase in ['train', 'val', 'test']
        }

    @pytest.mark.skipif(not DATA.exists(), reason=f'{DATA} not exists')
    def test_num_features(self):
        _ = self.dataset.phase_data
        num_features = self.dataset.num_features
        assert isinstance(num_features, dict)
        assert set(self.categorical + self.numerical) == num_features.keys()

    def test_save(self, tmp_path):
        self.dataset.save(save_dir=tmp_path)
        dataset_config = joblib.load(Path(tmp_path).joinpath('dataset.pkl'))
        assert isinstance(dataset_config, dict)
        dataset = DatasetLoader.load(**dataset_config)
        assert isinstance(dataset, ML100K)

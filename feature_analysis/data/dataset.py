import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from recommenders.datasets import movielens
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

__all__ = ['ML100K', 'RandomDataset', 'NUMERICAL', 'CATEGORICAL']


class RandomDataset:

    def __init__(self,
                 categorical: List[str] = None,
                 numerical: List[str] = None,
                 random_seed: int = 42):
        self.categorical = categorical if categorical is not None else [
            'categorical1'
        ]
        self.numerical = numerical if numerical is not None else ['numerical1']
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @property
    def phase_data(self):
        return {
            phase: (pd.DataFrame(
                [[np.random.rand(), np.random.randint(2)] for _ in range(10)],
                columns=['numerical1', 'categorical1']),
                    pd.DataFrame([np.random.randint(2) for _ in range(10)],
                                 columns=['label']))
            for phase in ['train', 'val', 'test']
        }

    @property
    def num_features(self):
        return {'numerical1': 1, 'categorical1': 2}

    def save(self, save_dir: Union[Path, str]):
        config = {
            'name': self.__class__.__name__,
            'categorical': self.categorical,
            'numerical': self.numerical,
            'random_seed': self.random_seed
        }
        joblib.dump(config, Path(save_dir).joinpath('dataset.pkl'))

    def transform_df(self, df):
        return df


NUMERICAL = ['timestamp', 'year', 'age']
CATEGORICAL = ['user_id', 'item_id', 'gender', 'occupation']
CUSTOM_FEATURES = ['year', 'freshness', 'age_interval']


class ML100K:

    def __init__(self,
                 data_dir: Optional[Union[str, Path]] = None,
                 rating_threshold: int = 3,
                 drop_threshold: bool = True,
                 categorical: Optional[List[str]] = None,
                 numerical: Optional[List[str]] = None,
                 custom_features: Optional[List[str]] = None,
                 apply_fillnan: bool = True,
                 apply_preprocessing: bool = True,
                 categorical_encoders: Optional[dict] = None,
                 numerical_encoders: Optional[dict] = None,
                 random_seed: int = 42,
                 inference: bool = False):
        assert (categorical is not None) or (numerical is not None)
        self.data_dir = data_dir
        self.rating_threshold = rating_threshold
        self.drop_threshold = drop_threshold
        self.categorical = categorical
        self.numerical = numerical
        self.apply_fillnan = apply_fillnan
        self.apply_preprocessing = apply_preprocessing
        self.random_seed = random_seed
        self._phase_data = None
        self._num_features = None
        self.categorical_encoders = defaultdict(lambda: LabelEncoder())
        self.numerical_encoders = defaultdict(lambda: StandardScaler())
        if categorical_encoders is not None:
            self.categorical_encoders.update(categorical_encoders)
        if numerical_encoders is not None:
            self.numerical_encoders.update(numerical_encoders)
        self.inference = inference
        if custom_features is None:
            self.custom_features = CUSTOM_FEATURES

    @staticmethod
    def load_ml100k_user(data_dir: Union[str, Path]) -> pd.DataFrame:
        """load user dataframe

        Args:
            data_dir: data directory

        Returns:
            pd.DataFrame: user dataframe
        """
        zip_file = Path(data_dir).joinpath('ml-100k.zip')
        assert zip_file.exists(), f'path: {data_dir} not exists'

        user_file_path = Path(data_dir).joinpath('u.user')
        if not user_file_path.exists():
            with ZipFile(zip_file, 'r') as f:
                with f.open('ml-100k/u.user') as src_user, open(
                        user_file_path, 'wb') as dst_user:
                    shutil.copyfileobj(src_user, dst_user)

        columns = ['user_id', 'age', 'gender', 'occupation']
        df = pd.read_csv(user_file_path,
                         sep='|',
                         engine='python',
                         names=columns,
                         usecols=list(range(len(columns))))
        return df

    def load_df(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """load dataframe from existed dataset

        Args:
            data_dir: data directory
        Returns:
            pd.DataFrame: dataframe
        """
        assert Path(data_dir).joinpath(
            'ml-100k.zip').exists(), f'{data_dir} not exists'
        df = movielens.load_pandas_df(size='100k',
                                      local_cache_path=data_dir,
                                      header=('user_id', 'item_id', 'rating',
                                              'timestamp'),
                                      title_col=None,
                                      genres_col='genres',
                                      year_col='year')
        user_df = self.load_ml100k_user(data_dir=data_dir)
        df = pd.merge(df, user_df, how='left', on='user_id')
        return df

    @staticmethod
    def numerical_fillnan(df: pd.DataFrame) -> pd.DataFrame:
        """numerical data fill in NaN 

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: processed dataframe on NUMERICAL features
        """
        columns = [i for i in NUMERICAL if i in df.columns]
        if columns:
            df[columns] = df[columns].where(df[columns].notna(), np.nan)
            for feat in columns:
                imputer = SimpleImputer(missing_values=np.nan,
                                        strategy='most_frequent')
                df[feat] = imputer.fit_transform(
                    df[[feat]].astype('object')).astype(float)
        return df

    @staticmethod
    def categorical_fillnan(df: pd.DataFrame) -> pd.DataFrame:
        """categorical data fill in NaN 

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: processed dataframe on CATEGORICAL features
        """
        columns = [i for i in NUMERICAL if i in df.columns]
        # None to np.nan
        if columns:
            df[columns] = df[columns].where(df[columns].notna(), np.nan)
            df[columns] = df[columns].fillna('nan')
        return df

    def generate_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'rating' not in df.columns:
            return df

        if self.drop_threshold:
            df = df[df['rating'] != self.rating_threshold]

        df.loc[df.index, ['label']] = (df['rating']
                                       >= self.rating_threshold).astype(int)
        return df

    def numerical_encoder_fit(self, df: pd.DataFrame):
        """encode on numerical features

        Args:
            df: dataframe
        """
        for feat in self.numerical:
            self.numerical_encoders[feat].fit(df[[feat]])

    def numerical_encoder_transform(self, df: pd.DataFrame):
        """transform on numerical features

        Args:
            df: dataframe
        """
        for feat in self.numerical:
            df[feat] = self.numerical_encoders[feat].transform(df[[feat]])
        return df

    def categorical_encoder_fit(self, df: pd.DataFrame):
        """encode on categorical features

        Args:
            df: dataframe
        """
        for feat in self.categorical:
            self.categorical_encoders[feat].fit(df[feat])

    def categorical_encoder_transform(self, df: pd.DataFrame):
        """encode on categorical features

        Args:
            df: dataframe
        """
        for feat in self.categorical:
            df[feat] = self.categorical_encoders[feat].transform(df[feat])

        return df

    @staticmethod
    def create_age_interval(df: pd.DataFrame) -> pd.DataFrame:
        """create age_interval based on numerical 'age' feature

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: dataframe with 'age_interval' column
        """
        assert 'age' in df.columns
        df['age_interval'] = pd.cut(df['age'],
                                    bins=[0, 24, 30, 40, 200],
                                    labels=['1-24', '25-30', '31-40', '41-'])
        return df

    @staticmethod
    def create_freshness(df: pd.DataFrame) -> pd.DataFrame:
        """create freshness based on numerical timestamp and year feature

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: dataframe with 'freshness' column
        """
        assert 'timestamp' in df.columns and 'year' in df.columns
        df['freshness'] = df['timestamp'].apply(
            lambda a: datetime.fromtimestamp(a).year) - df['year'].astype(int)
        return df

    @staticmethod
    def create_year(df: pd.DataFrame) -> pd.DataFrame:
        """create release year of the movie

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: dataframe with year column
        """
        if 'year' not in df.columns:
            df['year'] = df['movie_title'].apply(
                lambda a: re.search('(\d\d\d\d)', a).group(1)).astype(int)
        return df

    def create_new_features(self, df: pd.DataFrame,
                            features: List[str]) -> pd.DataFrame:
        """create new features on already existed feature function

        Args:
            df: dataframe
            features: new feature name list

        Returns:
            pd.DataFrame: dataframe with new feature columns
        """
        for feature in features:
            df = getattr(self, f'create_{feature}')(df)
        return df

    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame]:
        # 8:1:1 split
        train_df, val_df = train_test_split(df,
                                            test_size=0.2,
                                            stratify=df['label'],
                                            random_state=self.random_seed)
        val_df, test_df = train_test_split(val_df,
                                           test_size=0.5,
                                           stratify=val_df['label'],
                                           random_state=self.random_seed)
        return train_df, val_df, test_df

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: transformed dataframe
        """
        df = df.copy()
        df = self.generate_label(df)

        if self.apply_fillnan:
            df = self.numerical_fillnan(df)
            df = self.categorical_fillnan(df)

        # after fillnan, genreate freshness and age_interval
        df = self.create_new_features(df, features=self.custom_features)
        # preprocessing
        if self.apply_preprocessing:
            # train
            if not self.inference:
                self.categorical_encoder_fit(df)
                self.numerical_encoder_fit(df)
            # train and inference
            df = self.categorical_encoder_transform(df)
            df = self.numerical_encoder_transform(df)

        return df[[*self.numerical, *self.categorical, 'label'
                   ]] if 'label' in df.columns else df[[
                       *self.numerical, *self.categorical
                   ]]

    @property
    def phase_data(self):
        if self._phase_data is None and not self.inference:
            df = self.load_df(data_dir=self.data_dir)
            trns_df = self.transform_df(df)
            train_df, val_df, test_df = self.train_test_split(df=trns_df)
            train_label, val_label, test_label = train_df.pop(
                'label'), val_df.pop('label'), test_df.pop('label')
            self._phase_data = {
                'train': (train_df[[*self.numerical,
                                    *self.categorical]], train_label),
                'val': (val_df[[*self.numerical,
                                *self.categorical]], val_label),
                'test': (test_df[[*self.numerical,
                                  *self.categorical]], test_label)
            }
        return self._phase_data

    @property
    def num_features(self):
        if self._num_features is None:
            self._num_features = {}
            self._num_features.update({feat: 1 for feat in self.numerical})
            self._num_features.update({
                feat:
                len(self.categorical_encoders[feat].classes_)
                if feat in self.categorical_encoders else None
                for feat in self.categorical
            })
        return self._num_features

    def save(self, save_dir: Union[Path, str]):
        """save dataset configs

        Args:
            save_dir: save directory
        """
        config = {
            'name': self.__class__.__name__,
            'data_dir': self.data_dir,
            'categorical': self.categorical,
            'numerical': self.numerical,
            'rating_threshold': self.rating_threshold,
            'drop_threshold': self.drop_threshold,
            'custom_features': self.custom_features,
            'apply_fillnan': self.apply_fillnan,
            'apply_preprocessing': self.apply_preprocessing,
            'categorical_encoders': dict(self.categorical_encoders),
            'numerical_encoders': dict(self.numerical_encoders),
            'random_seed': self.random_seed,
            'inference': True
        }
        joblib.dump(config, Path(save_dir).joinpath('dataset.pkl'))

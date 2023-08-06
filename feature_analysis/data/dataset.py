import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from recommenders.datasets import movielens
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

__all__ = [
    'load_dataset', 'ML100K', 'RandomDataset', 'NUMERICAL', 'CATEGORICAL'
]


def load_dataset(name: str, **kwargs):
    """load specific dataset with its name

    Args:
        name: dataset class name

    Returns:
        dataset
    """
    dataset = globals()[name](**kwargs)
    assert hasattr(dataset, 'phase_data') and hasattr(
        dataset, 'numerical') and hasattr(dataset, 'categorical') and hasattr(
            dataset, 'num_features')
    return dataset


class RandomDataset:

    def __init__(self):
        pass

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

    @property
    def categorical(self):
        return ['categorical1']

    @property
    def numerical(self):
        return ['numerical1']


NUMERICAL = ['timestamp', 'year', 'age']
CATEGORICAL = ['user_id', 'item_id', 'gender', 'occupation']


class ML100K:

    def __init__(self,
                 data_dir: Optional[Union[str, Path]] = None,
                 categorical: Optional[List[str]] = None,
                 numerical: Optional[List[str]] = None,
                 apply_fillnan: bool = True,
                 apply_preprocessing: bool = True,
                 random_seed: int = 42):
        assert (categorical is not None) or (numerical is not None)
        self.data_dir = data_dir
        self.categorical = categorical
        self.numerical = numerical
        self.apply_fillnan = apply_fillnan
        self.apply_preprocessing = apply_preprocessing
        self.random_seed = random_seed
        self._phase_data = None
        self._num_features = None
        self.categorical_encoder = None
        self.numerical_encoder = None

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

        # drop rating: 3 and treat 4,5 as 1 and treat 1,2 as 0
        df = df[df['rating'] != 3]
        df['label'] = df['rating'].apply(lambda a: a > 3).astype(int)
        return df

    @staticmethod
    def numerical_fillnan(df: pd.DataFrame) -> pd.DataFrame:
        """numerical data fill in NaN 

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: processed dataframe on NUMERICAL features
        """
        df[NUMERICAL] = df[NUMERICAL].where(df[NUMERICAL].notna(), np.nan)
        for feat in NUMERICAL:
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

        # None to np.nan
        df[CATEGORICAL] = df[CATEGORICAL].where(df[CATEGORICAL].notna(),
                                                np.nan)
        df[CATEGORICAL] = df[CATEGORICAL].fillna('nan')
        return df

    def numerical_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """preprocess on numerical features

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: processed dataframe
        """
        if self.numerical:
            scaler = StandardScaler()
            df[self.numerical] = scaler.fit_transform(df[self.numerical])
            self.numerical_encoder = scaler
        return df

    def categorical_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """preprocess on categorical features

        Args:
            df: dataframe

        Returns:
            pd.DataFrame: processed dataframe
        """
        if self.categorical:
            self.categorical_encoder = {}
            for feat in self.categorical:
                encoder = LabelEncoder()
                df[feat] = encoder.fit_transform(df[feat])
                self.categorical_encoder[feat] = encoder
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
                                    bins=[0, 14, 39, 64, 200],
                                    labels=['0-14', '15-39', '40-64', '65-'])
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

    def create_new_features(self, df: pd.DataFrame,
                            features: List[str]) -> pd.DataFrame:
        """create new features on already existed feature function

        Args:
            df (pd.DataFrame): dataframe
            features (List[str]): new feature name list

        Returns:
            pd.DataFrame: dataframe with new feature columns
        """
        for feature in features:
            df = getattr(self, f'create_{feature}')(df)
        return df

    @property
    def phase_data(self):
        if self._phase_data is None:
            df = self.load_df(data_dir=self.data_dir)
            # fillnan
            if self.apply_fillnan:
                df = self.numerical_fillnan(df)
                df = self.categorical_fillnan(df)
            # add new features
            df_processed_new_feat = self.create_new_features(
                df, features=['freshness', 'age_interval'])
            # preprocessing
            if self.apply_preprocessing:
                df_processed_new_feat = self.categorical_preprocessing(
                    df_processed_new_feat)
                df_processed_new_feat = self.numerical_preprocessing(
                    df_processed_new_feat)

            merge_df = df_processed_new_feat[[
                *self.numerical, *self.categorical, 'label'
            ]]
            # 8:1:1 split
            train_df, val_df = train_test_split(merge_df,
                                                test_size=0.2,
                                                stratify=merge_df['label'],
                                                random_state=self.random_seed)
            val_df, test_df = train_test_split(val_df,
                                               test_size=0.5,
                                               stratify=val_df['label'],
                                               random_state=self.random_seed)
            train_label, val_label, test_label = train_df.pop(
                'label'), val_df.pop('label'), test_df.pop('label')
            self._phase_data = {
                'train': (train_df, train_label),
                'val': (val_df, val_label),
                'test': (test_df, test_label)
            }
        return self._phase_data

    @property
    def num_features(self):
        if self._num_features is None:
            self._num_features = {}
            self._num_features.update({feat: 1 for feat in self.numerical})
            self._num_features.update({
                feat:
                len(self.categorical_encoder[feat].classes_)
                for feat in self.categorical
            })
        return self._num_features
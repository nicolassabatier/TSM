import numpy as np
import pandas as pd
from typing import Union, List
import os

def data_subset_by_dict(df: pd.DataFrame, subset_dict: dict, features: list = None) \
        -> pd.DataFrame:
    """subset_dict is expected in format {"column_name": value, ... }"""
    if not features:
        features = [f for f in list(df) if f not in list(subset_dict.keys())]
    else:
        features = [f for f in features if f not in list(subset_dict.keys())]

    filtered_df = df.copy()
    for filter_col, value in subset_dict.items():
        filtered_df = filtered_df[filtered_df[filter_col] == value]
    return filtered_df[features]


def get_series_past_k_lags(s: pd.Series, k: Union[int, List] = 10) -> pd.DataFrame:
    df_with_lags = pd.DataFrame({})

    if type(k) == int:
        k_list = range(1, (k + 1))
    elif type(k) == list:
        k_list = k
    else:
        raise TypeError('K must be integer or list')

    for lag in k_list:
        df_with_lags['{}_t-{}'.format(s.name, lag)] = [np.nan for _ in range(lag)] + list(s.iloc[:-lag].values)

    return df_with_lags

def split_data_frame_by_column(data: pd.DataFrame, by: str, drop: bool = True):

    dataframes = []
    split_values = data[by].value_counts().index.values.tolist()
    print('Splitter will return list of', len(split_values), 'dataframe')
    for v in split_values:
        v_data = data[data[by] == v].copy()
        if drop:
            v_data.drop(by, axis=1, inplace=True)
        dataframes.append(v_data)
    del data
    return dataframes

def split_data_file_by_column(datapath: str, by: str, drop):
    filename, file_extension = os.path.splitext(datapath)

    if file_extension == '.pkl' or file_extension == '.pickle':
        data = pd.read_pickle(datapath)

    split_values = data[by].value_counts().index.values.tolist()
    for v in split_values:
        v_data = data[data[by] == v].copy()
        v_data.drop(by, axis=1, inplace=True)
        v_data.to_pickle(filename + '_' + by + '_' + str(v) + '.pkl')
        del v_data
    del data
    os.remove(datapath)
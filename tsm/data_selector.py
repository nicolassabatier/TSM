import numpy as np
import pandas as pd
from typing import Union, List

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
        df_with_lags[f'{s.name}_t-{lag}'] = [np.nan for _ in range(lag)] + list(s.iloc[:-lag].values)

    return df_with_lags

import logging
from math import ceil
from typing import Union, List, Tuple,Dict

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from pandas.api.types import is_datetime64_any_dtype as is_datetime

tqdm.pandas()


def compress_memory_usage(df_in: pd.DataFrame, replacer: dict = None):
    start_mem_usg = df_in.memory_usage().sum() / 1024 ** 2
    cols_with_nas = []
    df = df_in.copy()  # Avoid changing input df
    for col in tqdm(df.columns, "DataFrame: compress_memory_usage"):
        if df[col].dtype != object and not is_datetime(df[col]):
            # make variables for Int, max and min
            is_int = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                cols_with_nas.append(col)
                if replacer:
                    df[col].fillna(replacer[col], inplace=True)
                else:
                    df[col].fillna(mn - 1, inplace=True)

            as_int = df[col].fillna(0).astype(np.int64)
            result = (df[col] - as_int)
            result = result.sum()
            if -0.01 < result < 0.01:
                is_int = True
            # Make integer/unsigned integer types
            if is_int:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            # Make float data types 32 bit
            else:
                df[col] = df[col].astype(np.float32)

    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage pre-compression was {}'.format(start_mem_usg))
    print('Memory usage after-compression was {}'.format(mem_usg))
    print("This is  {}% of the initial size".format(100 * mem_usg / start_mem_usg))
    return df, cols_with_nas


def get_data_sample(df: pd.DataFrame, pct_size: float = 0.1):
    return df.sample(frac=pct_size)


def time_processing(df: pd.DataFrame, timestamp_key: str):
    df[timestamp_key] = pd.to_datetime(df[timestamp_key])
    df['dt_m'] = df[timestamp_key].dt.month.astype(np.int8)
    df['dt_w'] = df[timestamp_key].dt.weekofyear.astype(np.int8)
    df['dt_d'] = df[timestamp_key].dt.dayofyear.astype(np.int16)
    df['dt_hour'] = df[timestamp_key].dt.hour.astype(np.int8)
    df['dt_day_week'] = df[timestamp_key].dt.dayofweek.astype(np.int8)
    df['dt_day_month'] = df[timestamp_key].dt.day.astype(np.int8)
    df['dt_week_month'] = df[timestamp_key].dt.day / 7
    df['dt_week_month'] = df['dt_week_month'].progress_apply(lambda x: ceil(x)).astype(np.int8)
    return df


def ordinal2wave(col_name: str, df: pd.DataFrame):
    df['{}_sin'.format(col_name)] = np.sin(2 * np.pi * df[col_name] / df[col_name].max())
    df['{}_cos'.format(col_name)] = np.cos(2 * np.pi * df[col_name] / df[col_name].max())
    return df


def df_to_x_y(df: pd.DataFrame, x_indexes: Union[List[int], int], y_index: int = None):
    if type(x_indexes) == int:
        x_indexes = [x_indexes]
    x = np.array(df.iloc[:, x_indexes])
    if y_index is not None:
        y = np.array(df.iloc[:, y_index])
        return x, y
    return x


def train_dev_test_split(df: pd.DataFrame, train_pct: int = 0.9, dev_pct: int = 0.025) -> Dict:
    tr_idx = int(train_pct * len(df))
    if dev_pct > 0:
        dv_idx = int((train_pct + dev_pct) * len(df))
        dev_df = df.iloc[tr_idx:dv_idx, :]
        train_df = df.iloc[:tr_idx, :]
        test_df = df.iloc[dv_idx:, :]
        assert (len(train_df) + len(dev_df) + len(test_df)) == len(df), 'Mismatch in split'
        return {'train_df':train_df, 'dev_df':dev_df, 'test_df':test_df}
    else:
        train_df = df.iloc[:tr_idx, :]
        test_df = df.iloc[tr_idx:, :]
        assert (len(train_df) + len(test_df)) == len(df), 'Mismatch in split'
        return train_df, test_df

def train_dev_test_split_index(data_size, train_pct: int = 0.9, dev_pct: int = 0.025):
    tr_idx = int(train_pct * data_size)
    dv_idx = int(train_pct + dev_pct) * data_size
    return range(tr_idx), range(tr_idx, dv_idx),range(dv_idx,data_size)



def compute_and_add_mean_by_a_and_b(df: pd.DataFrame, mean_of: str, a: str, b: str) -> pd.DataFrame:
    mean_by =  df.groupby([a, b], as_index=False).agg({mean_of: 'mean'})
    means_by_a = {}
    for x in list(set(df[a])):
        x_data = mean_by[mean_by[a] == x][[b, mean_of]].to_dict('records')
        means_by_a[x] = x_data
    df_with_means = only_add_mean_by_a_and_b(df, mean_of, a, b, means_by_a)
    return df_with_means, means_by_a

def only_add_mean_by_a_and_b(df: pd.DataFrame, mean_of: str, a: str, b: str, means_by_a_and_b: dict) -> pd.DataFrame:
    means_by_and_b_df = pd.DataFrame(df[a].progress_apply(lambda x: [x[mean_of] for x in means_by_a_and_b[x]]).values.tolist(),
        columns=['{}_{}_{}_mean'.format(a, b, y) for y in [y[b] for y in means_by_a_and_b[df[a].values[0]]]]).reset_index(drop=True)
    return df.reset_index(drop=True).merge(means_by_and_b_df, left_index=True, right_index=True)

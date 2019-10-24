import logging
from math import ceil
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

tqdm.pandas()


def compress_memory_usage(df_in: pd.DataFrame, replacer: dict = None):
    start_mem_usg = df_in.memory_usage().sum() / 1024 ** 2
    cols_with_nas = []
    df = df_in.copy()  # Avoid changing input df
    for col in tqdm(df.columns, "DataFrame: compress_memory_usage"):
        if df[col].dtype != object:  # Exclude strings
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
    logging.info(f'Memory usage pre-compression was {start_mem_usg}')
    logging.info(f'Memory usage after-compression was {mem_usg}')
    logging.info(f"This is  {100 * mem_usg / start_mem_usg}% of the initial size")
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
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / df[col_name].max())
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / df[col_name].max())
    return df


def encode_categories(df: pd.DataFrame, cat_cols: Union[List[str], str]) -> pd.DataFrame:
    if type(cat_cols) == str:
        cat_cols = [cat_cols]
    for col in cat_cols:
        le = LabelEncoder()
        no_of_categories = len(set(df[col]))
        if no_of_categories < 255:
            df[col] = le.fit_transform(df[col]).astype(np.uint8)
        elif no_of_categories < 65535:
            df[col] = le.fit_transform(df[col]).astype(np.uint16)
        else:
            df[col] = le.fit_transform(df[col]).astype(np.uint32)
    return df


def df_to_x_y(df: pd.DataFrame, x_indexes: Union[List[int], int], y_index: int = None):
    if type(x_indexes) == int:
        x_indexes = [x_indexes]
    x = np.array(df.iloc[:, x_indexes])
    if y_index is not None:
        y = np.array(df.iloc[:, y_index])
        return x, y
    return x


def train_dev_test_split(df: pd.DataFrame, train_pct: int = 0.9, dev_pct: int = 0.025) -> Tuple:
    tr_idx = int(train_pct * len(df))
    if dev_pct > 0:
        dv_idx = int((train_pct + dev_pct) * len(df))
        dev_df = df.iloc[tr_idx:dv_idx, :]
        train_df = df.iloc[:tr_idx, :]
        test_df = df.iloc[dv_idx:, :]
        assert (len(train_df) + len(dev_df) + len(test_df)) == len(df), 'Mismatch in split'
        return train_df, dev_df, test_df
    else:
        train_df = df.iloc[:tr_idx, :]
        test_df = df.iloc[tr_idx:, :]
        assert (len(train_df) + len(test_df)) == len(df), 'Mismatch in split'
        return train_df, test_df

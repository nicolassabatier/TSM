import pandas as pd
from tsm.data_utils import time_processing, encode_categories, ordinal2wave
from tsm.data_selector import get_series_past_k_lags

if __name__ == '__main__':
    data_path = '../kaggle/input/ashrae-energy-prediction/train.pkl'
    data = pd.read_pickle(data_path)

    add_time = False
    add_categories = False
    add_ordinal2wave = True
    add_past_lags = True
    add_ewm = True
    
    if add_time:
        data = time_processing(data, timestamp_key='timestamp')

    if add_categories:
        data = encode_categories(data, cat_cols='primary_use')

    if add_ordinal2wave:
        ordinal2wave('dt_m', data)
        ordinal2wave('dt_w', data)
        ordinal2wave('dt_d', data)
        ordinal2wave('dt_hour', data)
        ordinal2wave('dt_day_week', data)
        ordinal2wave('dt_day_month', data)
        ordinal2wave('dt_week_month', data)
    
    if add_past_lags:
        k_air = get_series_past_k_lags(data['air_temperature'], k=[1, 2, 3, 6]).fillna(99).reset_index(drop=True)
        k_dew = get_series_past_k_lags(data['dew_temperature'], k=[1, 2, 3, 6]).fillna(99).reset_index(drop=True)
        data = pd.concat([data.reset_index(drop=True), k_air, k_dew], axis=1, sort=False)

    if add_ewm:
        k_air_e_01 = data.air_temperature.ewm(alpha=0.1).mean().reset_index(drop=True)
        k_air_e_01.name = 'k_air_e_01'
        k_dew_e_01 = data.dew_temperature.ewm(alpha=0.1).mean().reset_index(drop=True)
        k_dew_e_01.name = 'k_dew_e_01'
        k_air_e_025 = data.air_temperature.ewm(alpha=0.25).mean().reset_index(drop=True)
        k_air_e_025.name = 'k_air_e_025'
        k_dew_e_025 = data.dew_temperature.ewm(alpha=0.25).mean().reset_index(drop=True)
        k_dew_e_025.name = 'k_dew_e_025'
        data = pd.concat([data.reset_index(drop=True), k_air_e_01, k_dew_e_01, k_air_e_025, k_dew_e_025], axis=1,
                         sort=False)

    data.to_pickle(data_path)
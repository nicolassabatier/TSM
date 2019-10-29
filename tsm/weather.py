import pandas as pd
import numpy as np
from tsm.data_selector import split_data_frame_by_column

NA_REPLACER = {
    'air_temperature': 99,
    'cloud_coverage': 99,
    'dew_temperature': 99,
    'precip_depth_1_hr': 999,
    'sea_level_pressure': 9999,
    'wind_direction': 999,
    'wind_speed': 99
}

def interpolate_weather_data(data):
    
    sites_data = []
    for site_data in split_data_frame_by_column(data, by='site_id', drop=False):
        site_data['timestamp'] = pd.to_datetime(site_data['timestamp'])
        site_data.set_index('timestamp', inplace=True)
        for col in site_data.columns:
            if col != 'site_id':
                site_data[col].replace(NA_REPLACER[col], np.nan, inplace=True)
                if site_data[col].isna().sum() < len(site_data):
                    site_data[col] = site_data[col].interpolate(method='time')
        sites_data.append(site_data)
    
    return pd.concat(sites_data).reset_index()

def add_ewm_lags(data):
    
    sites_data = []
    for site_data in split_data_frame_by_column(data, by='site_id', drop=False):
        for col in site_data.columns:
            if col != 'site_id' and col != 'timestamp':
                
                site_data[col + '_ewm_001'] = site_data[col].ewm(alpha=0.01).mean()
                site_data[col + '_ewm_005'] = site_data[col].ewm(alpha=0.05).mean()
                site_data[col + '_ewm_01'] = site_data[col].ewm(alpha=0.1).mean()
                site_data[col + '_ewm_02'] = site_data[col].ewm(alpha=0.1).mean()
        
        sites_data.append(site_data)
        
    return pd.concat(sites_data)
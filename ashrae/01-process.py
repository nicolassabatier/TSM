import gc
from tsm.data_utils import compress_memory_usage
import pandas as pd

gc.enable()

NA_REPLACER = {
    'year_built': 9999,
    'floor_count': 99,
    'air_temperature': 99,
    'cloud_coverage': 99,
    'dew_temperature': 99,
    'precip_depth_1_hr': 999,
    'sea_level_pressure': 9999,
    'wind_direction': 999,
    'wind_speed': 99
}


def load_data():
    train_data = pd.read_csv('kaggle/input/ashrae-energy-prediction/train.csv')
    building = pd.read_csv('kaggle/input/ashrae-energy-prediction/building_metadata.csv')
    weather_train = pd.read_csv('kaggle/input/ashrae-energy-prediction/weather_train.csv')
    train_data = train_data.merge(building, on='building_id', how='left')
    train_data = train_data.merge(weather_train, on=['site_id', 'timestamp'], how='left')

    test_data = pd.read_csv('kaggle/input/ashrae-energy-prediction/test.csv')
    weather_test = pd.read_csv('kaggle/input/ashrae-energy-prediction/weather_test.csv')
    test_data = test_data.merge(building, on='building_id', how='left')
    test_data = test_data.merge(weather_test, on=['site_id', 'timestamp'], how='left')

    return train_data, test_data


if __name__ == '__main__':
    train, test = load_data()

    train, _ = compress_memory_usage(train, replacer=NA_REPLACER)
    train.to_pickle('kaggle/input/ashrae-energy-prediction/train.pkl')

    test, _ = compress_memory_usage(train, replacer=NA_REPLACER)
    test.to_pickle('kaggle/input/ashrae-energy-prediction/test.pkl')


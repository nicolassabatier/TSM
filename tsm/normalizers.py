from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle
import os

def scale_and_save_data_frame_columns(df: pd.DataFrame, columns: list, store_path: str, scaler = MinMaxScaler):
    for col in columns:
        col_scaler = scaler()
        col_scaler.fit(df[col].values.reshape(-1, 1)) 
        df[col] = col_scaler.transform(df[col].values.reshape(-1, 1))
        with open('{}_{}_normalizer.pkl'.format(store_path, col), 'wb') as f_out:
            pickle.dump(col_scaler, f_out)
    return df

def load_normalisers_and_scale_data_frame(df: pd.DataFrame, store_path: str):
    potential_normalisers = [os.path.join(store_path, n) for n in os.listdir(store_path)]
    for col in list(df):
        for normalizer in potential_normalisers:
            if col in normalizer:
                print('Found {} for {}'.format(normalizer, col))
                with open(normalizer, 'rb') as f_in:
                    col_scaler = pickle.load(f_in)

                df[col] = col_scaler.transform(df[col].values.reshape(-1, 1))
    return df
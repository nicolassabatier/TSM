from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def encode_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    le = LabelEncoder()
    no_of_categories = len(set(df[col]))
    if no_of_categories < 255:
        df[col] = le.fit_transform(df[col]).astype(np.uint8)
    elif no_of_categories < 65535:
        df[col] = le.fit_transform(df[col]).astype(np.uint16)
    else:
        df[col] = le.fit_transform(df[col]).astype(np.uint32)
    return df, le
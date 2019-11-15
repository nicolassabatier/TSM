import numpy as np
from sklearn.model_selection import KFold

def root_mean_squared_log_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(np.log(predicted + 1) - np.log(actual + 1), 2)) / len(actual))

def k_fold_validator(k: int, data, shuffle: bool = True, random_state: int = 42):
    np.random.seed(random_state)
    folder = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    return folder.split(data)
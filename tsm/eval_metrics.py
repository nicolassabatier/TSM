import numpy as np


def root_mean_squared_log_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(np.log(predicted + 1) - np.log(actual + 1), 2)) / len(actual))

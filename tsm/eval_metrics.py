import numpy as np
from typing import Callable


def root_mean_squared_log_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(np.log(predicted + 1) - np.log(actual + 1), 2)) / len(actual))


def scorer_root_mean_squared_log_error(estimator:Callable, X: np.ndarray, actual: np.ndarray) -> float:
    predicted=estimator.predict(X)
    return root_mean_squared_log_error(predicted=predicted,actual=actual)
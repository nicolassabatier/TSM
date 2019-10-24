import numpy as np
from unittest import TestCase
from tsm.eval_metrics import root_mean_squared_log_error


class TestEvalMetrics(TestCase):

    def test_root_mean_squared_log_error(self):
        y_hat = np.array([np.e - 1, np.e ** 2 - 1])  # np.log(y_hat + 1) = [1, 2]
        y_true = np.array([np.e ** 3 - 1, np.e ** 4 - 1])  # np.log(y_true + 1) = [3, 4]
        # np.log(y_hat + 1) - np.log(y_true + 1) = [2, 2]
        # np.power(np.log(y_hat + 1) - np.log(y_true + 1), 2) = [4, 4]
        # np.sum(np.power(np.log(y_hat + 1) - np.log(y_true + 1), 2)) = 8
        # np.sum(np.power(np.log(y_hat + 1) - np.log(y_true + 1), 2)) / len(actual)) = 8 / 2 = 4
        # np.sqrt(np.sum(np.power(np.log(predicted + 1) - np.log(actual + 1), 2)) / len(actual)) = sqrt(4) = 2
        self.assertEqual(root_mean_squared_log_error(predicted=y_hat, actual=y_true), 2.0)



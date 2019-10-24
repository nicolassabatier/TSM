from sklearn.dummy import DummyRegressor
from typing import Callable


def dummy_regressor_accuracy(x, y, evaluator: Callable, strategy: str = 'mean'):
    dummy = DummyRegressor(strategy)
    dummy.fit(x, y)
    y_hat = dummy.predict(x)
    print('DummyRegressor accuracy:', evaluator(y_hat, y))
    return dummy

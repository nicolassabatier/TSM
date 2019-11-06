import numpy as np
import pandas as pd
from unittest import TestCase
from tsm.model import Model
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass
def make_test_data():
    df = pd.DataFrame()
    df['target_variable'] = np.sin(np.arange(0, 10, 0.1))
    df['predictor_0'] = np.zeros(100)
    df['predictor_1'] = np.sin(np.arange(0, 10, 0.1))
    return df



class TestModel(TestCase):

    def test_update_best_model(self):
        test_data = make_test_data()
        mock_cv_score = np.array([0, 0, 1, 1])
        test_model = Model(model_regressors=LinearRegression(), data_train=test_data,
                           target_variable='target_variable')

        test_model._update_best_model(model=LinearRegression(), cross_val_score=mock_cv_score,
                                      variable_used=['predictor'])
        self.assertIsInstance(test_model.best_model, type(LinearRegression()))
        self.assertEqual(test_model.best_model_variables, ['predictor'])

    def test__fit(self):
        test_data = make_test_data()
        test_model = Model(model_regressors=LinearRegression(), data_train=test_data,
                           target_variable='target_variable')

        test_model._fit(LinearRegression(), variable_used='predictor_0')
        self.assertIsNot(test_model.best_model_score, np.inf)

        test_model._fit(LinearRegression(), variable_used='predictor_1')
        self.assertEqual(test_model.best_model_variables, ['predictor_1'])
        self.assertEqual(len(test_model.score), 2)
        self.assertEqual(set(list(test_model.score)), set(['score', 'std', 'variable_used', 'regressor']))

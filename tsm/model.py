from dataclasses import dataclass
from typing import Callable, List, Union, Dict
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from tsm.eval_metrics import root_mean_squared_log_error, scorer_root_mean_squared_log_error
import numpy as np
from copy import deepcopy
from tsm.data_selector import data_subset_by_dict


@dataclass
class Model:
    model_regressors: Union[BaseEstimator, List[BaseEstimator]]
    data_train: pd.DataFrame
    evaluator: Callable = root_mean_squared_log_error
    best_model: BaseEstimator = None
    best_model_variables: list = None
    target_variable: str = 'meter_reading'
    score = pd.DataFrame()
    best_model_score: float = np.inf

    def _update_best_model(self, model: Callable, cross_val_score: np.array, variable_used: list):
        score = cross_val_score.mean()
        if score < self.best_model_score:
            self.best_model_score = score
            self.best_model = deepcopy(model)
            self.best_model_variables = variable_used

        score = pd.DataFrame([[score, cross_val_score.var(), variable_used,
                               model.get_params()]],
                             columns=['score', 'std', 'variable_used', 'regressor'])
        self.score = pd.concat([self.score, score])

    def _fit(self, model: Callable, variable_used: Union[str, List[str]], **cross_vall_args):

        if isinstance(variable_used, str):
            variable_used = [variable_used]

        cv_val_score = cross_val_score(estimator=model,
                                       X=self.data_train[variable_used],
                                       y=self.data_train[self.target_variable],
                                       scoring=scorer_root_mean_squared_log_error,
                                       **cross_vall_args)
        model.fit(self.data_train[variable_used], self.data_train[self.target_variable])
        self.variable_used = variable_used
        self._update_best_model(model, cv_val_score, self.variable_used)

    def fit(self, variable_used: Union[str, List[str]], **cross_vall_args):

        if not isinstance(self.model_regressors, list):
            self.model_regressors = [self.model_regressors]

        for model in self.model_regressors:
            self._fit(model=model, variable_used=variable_used, **cross_vall_args)

    def predict(self, X, best_model: int = True):

        if self.best_model and best_model:
            model_to_use = self.best_model
            variable_to_used = self.best_model_variables


        else:
            model_to_use = self.model_regressor
            variable_to_used = self.variable_used

        if isinstance(X, pd.DataFrame):
            X = X[variable_to_used]

        return model_to_use.predict(X)

    def save_score(self, data_path: str):
        if self.score:
            self.score.to_csv(data_path)


def pipeline_train_by_meter_reading(ModelObject: Model, variable_used: Union[List[str], Dict[str, List[str]]],
                                    data_train: pd.DataFrame, **cross_vall_args) -> Dict[str, Model]:
    if ModelObject.data_train is not None:
        ModelObject.data_train = None

    dict_result = {}
    local_variables = variable_used
    for meter_reading in [0, 1, 2, 3]:
        if isinstance(variable_used, Dict):
            local_variables = variable_used[f'meter_{meter_reading}']

        local_data = data_subset_by_dict(data_train, {'meter': meter_reading},
                                         features=local_variables + [ModelObject.target_variable])

        if len(local_data)>0:
            local_model = deepcopy(ModelObject)
            local_model.data_train = local_data
            local_model.fit(variable_used=local_variables, **cross_vall_args)
            dict_result[f'meter_{meter_reading}'] = local_model
    return dict_result


def pipeline_predict_by_meter_reading(dict_result, data: pd.DataFrame) -> pd.DataFrame:
    df_results = pd.DataFrame()
    for meter_reading, local_model in dict_result.items():
        local_data = data_subset_by_dict(data, {'meter': int(meter_reading[-1])},
                                         features=local_model.best_model_variables + [local_model.target_variable,'row_id'])
        local_results = pd.DataFrame()
        local_results['row_id'] = local_data['row_id']
        local_results['meter_reading'] = local_model.predict(local_data[local_model.best_model_variables])
        df_results = pd.concat([df_results, local_results])
    return df_results

import pandas as pd
from tsm.evaluators import k_fold_validator
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import xgboost as xgb
import time

SEED = 42
ROUNDS = 10000
EARLY_STOP = 50
V_ROUNDS = 100

data = pd.read_pickle('data/prep/train_meter_0.pkl')
exclude = ['timestamp', 'suspicious_day', 'suspicious_month']
y_col = ['log_meter_reading']
x_cols = [x for x in list(data) if x not in exclude + y_col]
data = data[data.day_suspicious == False].reset_index(drop=True)

errors = []
for tr_idx, ts_idx in k_fold_validator(k=2, data=data.index.values, shuffle=True, random_state=SEED):
    
    x = data[x_cols].values
    y = data[y_col].values
    x_tr, x_ts, y_tr, y_ts = x[tr_idx], x[ts_idx], y[tr_idx], y[ts_idx]
    
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dtest = xgb.DMatrix(x_ts, label=y_ts)
    num_round = ROUNDS
    param = {}
    param = {'max_depth': 15, 'eta': 0.05, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    #param['gpu_id'] = 0
    #param['tree_method'] = 'gpu_hist'
    bst = xgb.train(param, dtrain, ROUNDS, evals=[(dtest, 'eval')], early_stopping_rounds=EARLY_STOP)
    y_hat = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    rmsle = sqrt(mean_squared_error(y_ts, y_hat))
    print('Fold rmsle:', rmsle)
    errors.append(rmsle)
    
    with open('data/objects/xgboost_met0_rmsle_{}_preds.pkl'.format(rmsle), 'wb') as fout:
        pickle.dump(list(zip(y_hat, y_ts)), fout)
    bst.save_model('data/objects/xgboost_met0_rmsle_{}.pkl'.format(rmsle))
    break

#overall_rmsle = round(sum(errors) / len(errors), 3)
#print('Overall rmsle:', overall_rmsle)


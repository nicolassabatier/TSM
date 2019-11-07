import pandas as pd
from tsm.evaluators import k_fold_validator
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import lightgbm as lgb

SEED = 42
ROUNDS = 10000
EARLY_STOP = 50
V_ROUNDS = 100
METER = 1

data = pd.read_pickle('data/prep/train_meter_{}.pkl'.format(METER))
exclude = ['timestamp', 'day_suspicious', 'month_suspicious']
y_col = ['log_meter_reading']
x_cols = [x for x in list(data) if x not in exclude + y_col]
if METER == 0:
    data = data[data.day_suspicious == False].reset_index(drop=True)
else:
    data = data[(data.day_suspicious == False) & ((data.day_suspicious == True) & (data.month_suspicious == True)].reset_index(drop=True)

lgb_reg_params = {'objective':'regression',  'boosting_type':'gbdt', 'metric':'rmse',
                  'n_jobs':-1, 'learning_rate':0.05, 'max_depth':-1,
                  'tree_learner':'serial', 'colsample_bytree': 0.7, 'subsample_freq': 1,
                  'subsample':0.5, 'max_bin': 255, 'verbose': 0, 'seed': SEED}
                  # 'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}

errors = []
for tr_idx, ts_idx in k_fold_validator(k=2, data=data.index.values, shuffle=True, random_state=42):
    
    x = data[x_cols].values
    y = data[y_col].values
    x_tr, x_ts, y_tr, y_ts = x[tr_idx], x[ts_idx], y[tr_idx], y[ts_idx]
    
    
    lgb_train = lgb.Dataset(x_tr, y_tr.ravel())
    lgb_eval = lgb.Dataset(x_ts, y_ts.ravel())
    lgb_reg = lgb.train(lgb_reg_params, lgb_train, valid_sets=(lgb_train, lgb_eval),
        num_boost_round=ROUNDS,early_stopping_rounds=EARLY_STOP,verbose_eval=V_ROUNDS)
    
    y_hat = lgb_reg.predict(x_ts, num_iteration=lgb_reg.best_iteration)
    rmsle = sqrt(mean_squared_error(y_ts, y_hat))
   
    print('Fold rmsle:', rmsle)
    errors.append(rmsle)

    with open('data/objects/lightgbm_met{}_rmsle_{}_preds.pkl'.format(METER, rmsle), 'wb') as fout:
        pickle.dump(list(zip(y_hat, y_ts)), fout)
    lgb_reg.save_model('data/objects/lightgbm_met{}_rmsle_{}.pkl'.format(METER, rmsle), num_iteration=model.best_iteration)
    break

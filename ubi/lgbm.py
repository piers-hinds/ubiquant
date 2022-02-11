import lightgbm as lgb
import gc
from .metrics import lgb_pearson, lgb_pearson_eval


def LightGBM(trial, params, train, features, target, splitter):
    scores = []
    for i, (train_index, val_index) in enumerate(splitter.split(train[features], train[target])):
        print('fold: {}'.format(i))
        # Set data
        lgb_train = lgb.Dataset(train[features].iloc[train_index], train[target].iloc[train_index])
        lgb_valid = lgb.Dataset(train[features].iloc[val_index], train[target].iloc[val_index], reference = lgb_train)
        gc.collect()
        # Training
        model = lgb.train(
            params,
            lgb_train,
            valid_sets = [lgb_train, lgb_valid],
            num_boost_round = 500,
            feval = lgb_pearson_eval,
            callbacks=[
                lgb.early_stopping(50)
            ]
        )
        # Prediction
        y_pred = model.predict(train[features].iloc[val_index], num_iteration = model.best_iteration)

        # Evaluation
        score = pearson(y_pred, train[target].iloc[val_index])
        scores.append(score)
        weights = np.array([0.125, 0.125, 0.25, 0.5])
    return (scores * weights).sum()

  
def objective(trial):
    params = {
        'task': 'train',
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'seed': SEED,
        'metric': 'pearson',
        
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.0001, 10),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.0001, 10),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        'max_depth': trial.suggest_int('max_depth', 3, 24),
        'max_bin': trial.suggest_int('max_bin', 24, 64),
        'min_data_in_leaf':trial.suggest_int('min_data_in_leaf', 8, 512),
        
        #'gpu_platform_id': 0,
        #'gpu_device_id': 0,
        #'device': 'gpu',
        'n_jobs': -1,
        #'gpu_use_dp': False,
        'verbose': -1,
        'force_col_wise': True
    }
    
    return LightGBM(trial, params, train, features, target, splitter)

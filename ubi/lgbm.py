import lightgbm as lgb
import gc
import numpy as np
from .metrics import lgb_pearson, lgb_pearson_eval


def lgb_score(trial, params, train, features, target, splitter, train_final=False, lgb_pearson_eval=None):
    scores = []
    preds = []

    for i, (train_index, val_index) in enumerate(splitter.split(train[features], train[target])):
        print('fold: {}'.format(i))
        # Set data
        lgb_train = lgb.Dataset(train[features].iloc[train_index], train[target].iloc[train_index])
        lgb_valid = lgb.Dataset(train[features].iloc[val_index], train[target].iloc[val_index], reference=lgb_train)
        gc.collect()
        # Training
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            feval=lgb_pearson_eval,
            callbacks=[
                lgb.early_stopping(100)
            ]
        )
        # Prediction
        y_pred = model.predict(train[features].iloc[val_index], num_iteration=model.best_iteration)
        preds.append(y_pred)

        # Evaluation
        score = lgb_pearson(y_pred, train[target].iloc[val_index])
        scores.append(score)
        weights = np.array([0.125, 0.125, 0.25, 0.5])

    if train_final:
        lgb_train = lgb.Dataset(train[features], train[target])
        gc.collect()

        # Training
        final_model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train],
            feval=lgb_pearson_eval
        )
        return (scores * weights).sum(), scores, preds, final_model

    return (scores * weights).sum()

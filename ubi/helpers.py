import torch
import numpy as np
import pandas as pd


def save_models(models, model_path):
    for i, model in enumerate(models):
        torch.save(model.state_dict(), model_path+'/mlp_'+str(i)+'.pkl')


def save_scores(score, weights, model_path):
    pd.DataFrame(np.concatenate([np.array([score]), weights])).to_csv(model_path+'/score.csv', index=False)


def save_models_and_score(models, score, weights, model_path):
    save_models(models, model_path)
    save_scores(score, weights, model_path)


def fold_weighted_mean(scores, weights):
    return (scores * weights).sum()
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import os
import pandas as pd
from .data import get_ubiquant_dataloaders


def train_model(model, dl, loss_fn, epochs, vdl=None, metrics=[], lr=0.0006):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    #scheduler = ExponentialLR(opt, gamma=gamma)
    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for x, y in dl:
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        losses.append(running_loss / len(dl))
        #scheduler.step()

        if vdl is not None:
            val_losses, _ = validate_model(model, vdl, metrics)
            model.train()
            print('Epoch: ', epoch, '    Train loss: ', round(running_loss / len(dl), 6),
                  '    Val loss: ', *[round(v, 6) for v in val_losses])
        else:
            print('Epoch: ', epoch, '    Train loss: ', round(running_loss / len(dl), 6))

    return 


def validate_model(model, dl, metrics, save_preds=False):
    model.eval()
    dfs = []
    with torch.inference_mode():
        running_loss = np.zeros(shape=len(metrics))
        for x, y in dl:
            preds = model(x)
            for i, metric in enumerate(metrics):
                loss = metric(preds, y)
                running_loss[i] += loss.item()
            if save_preds:
                dfs.append(pd.DataFrame({'pred':preds.cpu().numpy()}))
    if save_preds:
        all_preds = pd.concat(dfs)
        return running_loss / len(dl), all_preds
    else:
        return running_loss / len(dl), None

    
def cv(Model, criterion, metric, splitter, dir, file_names, epochs, device='cuda', save_preds=False):
    scores = []
    weights = []
    models = []
    dfs = []

    for train_index, val_index in splitter.split(file_names):
        model = Model().to(device)
        train_dl, val_dl = get_ubiquant_dataloaders(dir, file_names, train_index, val_index, device)
        _ = train_model(model, train_dl, criterion, epochs)
        fold_score, preds = validate_model(model, val_dl, [metric], save_preds=save_preds)
        dfs.append(preds)
        scores.append(fold_score[0])
        weights.append(len(train_index))
        print('Fold score: ', round(fold_score[0], 6))
        models.append(model)
    weights = np.array(weights)
    weights = weights / weights.sum()
    if save_preds:
        all_preds = pd.concat(dfs)

    return np.array(scores), weights, models, all_preds


def save_models(models, model_path):
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(model_path, 'fold_'+str(i)+'.pkl'))

        
def save_scores(score, weights, model_path):
    pd.DataFrame({'weight': weights, 'score': score}).to_csv(os.path.join(model_path, 'fold_scores.csv'), index=False)
    pd.DataFrame({'score': [(score * weights).sum()]}).to_csv(os.path.join(model_path, 'score.csv'), index=False)

    
def save_preds(preds, model_path):
    preds.to_csv(os.path.join(model_path, 'preds.csv'), index=False)

    
def save_cv_info(model_path, cv_output):
    """Saves score, weights, models and OOF preds from CV"""
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    score, weights, models, preds = cv_output
    save_models(models, model_path)
    save_scores(score, weights, model_path)
    save_preds(preds, model_path)
    return 0

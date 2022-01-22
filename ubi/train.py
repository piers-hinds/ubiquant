import torch
import numpy as np
from .data import get_ubiquant_dataloaders


def train_model(model, dl, loss_fn, epochs, vdl=None, metrics=[], lr=0.0006):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
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

        if vdl is not None:
            val_losses = validate_model(model, vdl, metrics)
            model.train()
            print('Epoch: ', epoch, '    Train loss: ', round(running_loss / len(dl), 6),
                  '    Val loss: ', *[round(v, 6) for v in val_losses])
        else:
            print('Epoch: ', epoch, '    Train loss: ', round(running_loss / len(dl), 6))

    return losses


def validate_model(model, dl, metrics):
    model.eval()
    with torch.inference_mode():
        running_loss = np.zeros(shape=len(metrics))
        for x, y in dl:
            preds = model(x)
            for i, metric in enumerate(metrics):
                loss = metric(preds, y)
                running_loss[i] += loss.item()
    return running_loss / len(dl)


def cv(Model, criterion, metric, splitter, dir, file_names, epochs, device='cuda'):
    scores = []
    weights = []
    models = []

    for train_index, val_index in splitter.split(file_names):
        model = Model().to(device)
        train_dl, val_dl = get_ubiquant_dataloaders(dir, file_names, train_index, val_index, device)
        _ = train_model(model, train_dl, criterion, epochs)
        fold_score = validate_model(model, val_dl, [metric])[0]
        scores.append(fold_score)
        weights.append(len(train_index))
        print('Fold score: ', round(fold_score, 6))
        models.append(model)
    weights = np.array(weights)
    weights = weights / weights.sum()

    return np.array(scores), weights, models
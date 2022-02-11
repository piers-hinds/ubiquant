import torch
import torch.nn as nn
import scipy


def pearson(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    return torch.sum(xc * yc) / (torch.sqrt(torch.sum(xc ** 2)) * torch.sqrt(torch.sum(yc ** 2)))


def lgb_pearson(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def lgb_pearson_eval(preds, train_data):
    targets = train_data.get_label()
    score = pearson(preds, targets)
    return 'pearson', score, True


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mu_x = x.mean()
        mu_y = y.mean()
        cx = x - mu_x
        cy = y - mu_y
        var_x = x.var()
        var_y = y.var()
        cor = (cx * cy).sum() / ((cx**2).sum() * (cy**2).sum()).sqrt()
        ccc = 2 * cor * var_x.sqrt() * var_y.sqrt() / (var_x + var_y + (mu_x - mu_y)**2)
        return 1 - ccc

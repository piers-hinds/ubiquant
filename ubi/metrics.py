import torch


def pearson(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    return torch.sum(xc * yc) / (torch.sqrt(torch.sum(xc ** 2)) * torch.sqrt(torch.sum(yc ** 2)))

import torch.nn as nn
import torch


class Scaler(nn.Module):
    """Min-max scaler"""
    def __init__(self, mins, maxs):
        super(Scaler, self).__init__()
        self.mins = torch.tensor(mins)
        self.maxs = torch.tensor(maxs)
        self.ranges = self.maxs - self.mins

    def forward(self, x):
        x_std = (x - self.mins) / self.ranges
        x_scaled = x_std * 2 - 1
        return x_scaled

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mins = self.mins.to(*args, **kwargs)
        self.maxs = self.maxs.to(*args, **kwargs)
        return self


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Linear(300, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)

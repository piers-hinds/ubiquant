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
        self.ranges = self.ranges.to(*args, **kwargs)
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
    

class Dcnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_ids = nn.Sequential(
            nn.Embedding(3773, 32),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        self.layers = nn.Sequential(
            nn.BatchNorm1d(300),
            nn.Dropout(0.25),
            nn.Linear(300, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.high_dim_map = nn.Sequential(nn.BatchNorm1d(300),
                                          nn.Dropout(0.2),
                                          nn.Linear(300, 1024),
                                          nn.ReLU())
        
        self.image_map_1 = nn.Sequential(nn.BatchNorm1d(64),
                                         nn.Dropout(0.2),
                                         nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
                                         nn.ReLU(),
                                         nn.AdaptiveAvgPool1d(output_size = int(1024 / 128)),
                                         nn.Flatten())

        self.final_mlp = nn.Sequential(
            nn.BatchNorm1d(64 + 1024),
            nn.Dropout(0.2),
            nn.Linear(64+1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        ids, new_x = x[:, 0].to(int), x[:, 1:]
        ids = self.embed_ids(ids)
        new_x = self.high_dim_map(new_x)
        new_x = new_x.reshape(new_x.shape[0], 64, 16)
        new_x = self.image_map_1(new_x)
        x = torch.cat([ids, new_x], dim=-1)

        return self.final_mlp(x).squeeze(-1)

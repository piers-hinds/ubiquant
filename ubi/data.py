import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


class UbiquantDataset(Dataset):
    def __init__(self, dir, fnames, device='cuda', invest_id=False):
        self.dir = dir
        self.fnames = fnames
        self.features = ['f_' + str(i) for i in range(300)]
        self.invest_id = invest_id
        if self.invest_id:
            self.features = ['investment_id'] + self.features
        self.target = 'target'
        self.device = device

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        data_path = os.path.join(self.dir, self.fnames.iloc[idx].values[0])
        data = pd.read_parquet(data_path)
        return (torch.tensor(data[self.features].to_numpy(), device=self.device),
                torch.tensor(data[self.target].to_numpy(), device=self.device))


def get_ubiquant_dataloaders(dir, fnames, train_index, val_index, device='cuda', min_max=False, invest_id=False):
    train_dset = UbiquantDataset(dir, fnames.iloc[train_index], device, invest_id)
    val_dset = UbiquantDataset(dir, fnames.iloc[val_index], device, invest_id)
    if min_max:
        return (DataLoader(train_dset, batch_size=None, batch_sampler=None),
                DataLoader(val_dset, batch_size=None, batch_sampler=None),
                get_min_max(dir, fnames.iloc[train_index]))
    else:
        return (DataLoader(train_dset, batch_size=None, batch_sampler=None),
                DataLoader(val_dset, batch_size=None, batch_sampler=None))


def get_min_max(dir, fnames):
    df = pd.concat((pd.read_parquet(os.path.join(dir, f)) for f in fnames.iloc[:, 0]))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df[['f_' + str(i) for i in range(300)]])
    return scaler.data_min_, scaler.data_max_

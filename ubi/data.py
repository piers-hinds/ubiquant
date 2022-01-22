import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


class UbiquantDataset(Dataset):
    def __init__(self, dir, fnames, device='cuda'):
        self.dir = dir
        self.fnames = fnames
        self.features = ['f_' + str(i) for i in range(300)]
        self.target = 'target'
        self.device = device

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        data_path = os.path.join(self.dir, self.fnames.iloc[idx].values[0])
        data = pd.read_parquet(data_path)
        return (torch.tensor(data[self.features].to_numpy(), device=self.device),
                torch.tensor(data[self.target].to_numpy(), device=self.device))


def get_ubiquant_dataloaders(dir, fnames, train_index, val_index, device='cuda'):
    train_dset = UbiquantDataset(dir, fnames.iloc[train_index], device)
    val_dset = UbiquantDataset(dir, fnames.iloc[val_index], device)
    return DataLoader(train_dset), DataLoader(val_dset)
import torch
from torch.utils.data import Dataset


class MenuDataset(Dataset):
    def __init__(self, features: list, labels: list):
        # feature[0] = token_ids
        # feature[1] = poss_ids
        # feature[2] = attention
        # feature[3] = segment

        self.x = torch.stack(features)
        self.y = torch.stack(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

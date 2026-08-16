import numpy as np
import torch
from torch.utils.data import Dataset

class BackboneDataset(Dataset):
    def __init__(self, path, split):
        with np.load(path, allow_pickle=False) as archive:
            coords = archive["coords"].astype(np.float32)
            splits = archive["splits"].astype(str)

        indices = np.flatnonzero(splits == split)
        self.coords = torch.from_numpy(coords[indices].copy())

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        return self.coords[index]


def load_datasets(path):
    return BackboneDataset(path, "train"), BackboneDataset(path, "val")

def coordinate_scale(dataset):
    return float(dataset.coords.doubles().square().mean().sqrt())

    
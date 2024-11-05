import torch as th
from torch.utils.data import Dataset, random_split

from src.util import normalize_range


class ToyDataset(Dataset):

    def __init__(self, density: int, split: str):
        x = th.rand(density)
        x_min, x_max = (-th.pi, th.pi)
        x = normalize_range(x, low=x_min, high=x_max)

        # Mask region out (epistemic)
        mask = th.logical_or(x < -1, x > 1)
        x = x[mask]

        # Increase noise variance with x (aleatoric)
        var = normalize_range(x.clip(0), low=0, high=0.04)
        y = th.sin(x) + th.sqrt(var) * th.randn(x.shape)

        # Rescale to [-1, 1]
        x = normalize_range(x)
        y = normalize_range(y)

        train_split, test_split = random_split(range(len(x)), [0.9, 0.1])
        if split == "train":
            self.y = y[train_split]
            self.x = x[train_split]
        elif split == "test":
            self.y = y[test_split]
            self.x = x[test_split]
        else:
            raise ValueError(f"Invalid split {split} provided.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.y[idx].reshape(1, 1, 1), self.x[idx].reshape(1, 1, 1)

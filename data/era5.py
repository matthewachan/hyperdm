from functools import partial

import cdsapi
import numpy as np
import torch as th
import xarray as xr
from cv2 import resize
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, random_split

from src.util import normalize_range


class ERA5(Dataset):

    def __init__(self, image_size: int, split: str, download: bool = False):
        if download:
            print("Downloading ERA5 (takes ~1hr)...")
            dataset_name = "reanalysis-era5-single-levels"
            request = {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature"],
                "year": list(range(1940, 2023)),
                "month": ["01"],
                "day": list(range(1, 32)),
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "grib",
                "area": [83, -169, 7, -35]
            }
            client = cdsapi.Client()
            client.retrieve(dataset_name, request, 'data/era5_t2m.grib')
        print("Loading dataset...")
        dataset = xr.open_dataset("data/era5_t2m.grib")["t2m"].values

        # Pre-processing
        resize_func = partial(resize, dsize=(image_size, image_size))
        dataset = np.array(list(map(resize_func, dataset)))
        dataset = normalize_range(dataset, low=-1, high=1)

        # Splits dataset into staggered time steps
        windows = sliding_window_view(dataset, window_shape=2, axis=0)
        y = windows[..., 0]  # time t
        x = windows[..., 1]  # time t+6hr
        assert len(x) == len(y), f"Size mismatch {len(x)} versus {len(y)}!"

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
        return th.from_numpy(self.x[idx:idx + 1]), th.from_numpy(
            self.y[idx:idx + 1])

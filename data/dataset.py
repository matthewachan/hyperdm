from enum import Enum


class Dataset(Enum):
    TOY = "toy"
    LUNA16 = "luna16"
    ERA5 = "era5"

    def __str__(self):
        return self.value

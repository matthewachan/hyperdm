from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np

from data.dataset import Dataset


def normalize_range(x, low=-1, high=1):
    """
    Normalizes values to a specified range.
    :param x: input value
    :param low: low end of the range
    :param high: high end of the range
    :return: normalized value
    """
    x = (x - x.min()) / (x.max() - x.min())
    x = ((high - low) * x) + low
    return x


def circular_mask(h: int, w: int, center: tuple, radius: int):
    """
    Creates a circular mask.
    :param h: target mask height
    :param w: target mask weight
    :param center: circle center
    :param radius: circle radius
    :return: circle mask
    """
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist <= radius
    return mask



def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset))
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument('--download', action=BooleanOptionalAction)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--hyper_net_dims", type=int, nargs="+")
    return parser.parse_args()


def parse_test_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset))
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument('--download', action=BooleanOptionalAction)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--hyper_net_dims", type=int, nargs="+")
    return parser.parse_args()

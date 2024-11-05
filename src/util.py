from argparse import ArgumentParser

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


def parse_train_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset))
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--hyper_net_input_dim", type=int, default=8)
    return parser.parse_args()


def parse_test_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset))
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--hyper_net_input_dim", type=int, default=8)
    return parser.parse_args()

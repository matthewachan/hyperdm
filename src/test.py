import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

from data.dataset import Dataset
from data.era5 import ERA5
from data.toy import ToyDataset
from guided_diffusion.script_util import create_gaussian_diffusion
from model.mlp import MLP
from model.unet import Unet
from src.hyperdm import HyperDM
from src.util import circular_mask, normalize_range, parse_test_args


def toy_test(args):
    device = "cuda" if th.cuda.is_available() else "cpu"
    primary_net = MLP([3, 8, 16, 8, 1])
    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True,
                                          timestep_respacing="ddim10")
    hyperdm = HyperDM(primary_net, args.hyper_net_dims, diffusion).to(device)
    hyperdm.load_state_dict(th.load(args.checkpoint, weights_only=True))
    hyperdm.print_stats()
    hyperdm.eval()

    eu = []
    au = []
    pred = []
    xs = th.linspace(-1.0, 1.0, 1000)
    for i in tqdm(xs):
        y = th.tensor([i]).reshape(1, 1, 1, 1)
        mean, var = hyperdm.get_mean_variance(M=args.M,
                                              N=args.N,
                                              condition=y,
                                              device=device)
        eu.append(mean.var())
        au.append(var.mean())
        pred.append(mean.mean())
    eu = th.vstack(eu).ravel()
    au = th.vstack(au).ravel()
    pred = th.vstack(pred).ravel()

    # Normalize uncertainty for visualization purposes
    eu_norm = normalize_range(eu, low=0, high=1)
    au_norm = normalize_range(au, low=0, high=1)

    dataset = ToyDataset(args.dataset_size, split="train")
    plt.scatter(x=dataset.x,
                y=dataset.y,
                s=5,
                c="gray",
                label="Train Data",
                alpha=0.5)
    plt.plot(xs, pred, c='black', label="Prediction")
    plt.fill_between(xs,
                     pred - au_norm,
                     pred + au_norm,
                     color='lightsalmon',
                     alpha=0.4,
                     label="AU")
    plt.fill_between(xs,
                     pred - eu_norm,
                     pred + eu_norm,
                     color='lightskyblue',
                     alpha=0.4,
                     label="EU")
    plt.legend()
    plt.title("HyperDM")
    plt.savefig("toy_result.pdf")


def era5_test(args):
    device = "cuda" if th.cuda.is_available() else "cpu"
    primary_net = Unet(dim=16,
                       dim_mults=(1, 2, 4, 8),
                       channels=1,
                       self_condition=True)
    dataset = ERA5(args.image_size, split="test", download=args.download)

    # Initialize network
    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True,
                                          timestep_respacing="ddim25")
    hyperdm = HyperDM(primary_net, args.hyper_net_dims, diffusion).to(device)
    hyperdm.load_state_dict(th.load(args.checkpoint, weights_only=True))
    hyperdm.print_stats()
    hyperdm.eval()
    random_idx = np.random.choice(range(len(dataset)))
    x, y = dataset[random_idx]

    # Create out-of-distribution image
    ood = y.squeeze().clone()
    mask = circular_mask(
        *ood.shape,
        center=[int(.88 * args.image_size),
                int(.1 * args.image_size)],
        radius=int(.03 * args.image_size))
    ood[mask] = 1
    ood = ood.reshape(1, 1, *ood.shape).to(device)
    mean, var = hyperdm.get_mean_variance(M=args.M,
                                          N=args.N,
                                          condition=ood,
                                          device=device,
                                          progress=True)
    pred = mean.mean(dim=0).squeeze()
    eu = mean.var(dim=0).squeeze()
    au = var.mean(dim=0).squeeze()

    _, axs = plt.subplots(1, 4, figsize=(25, 6))
    axs[0].imshow(pred, cmap="gray")
    axs[1].imshow(ood.squeeze().cpu(), cmap="gray")
    axs[2].imshow(eu, cmap="gray")
    axs[3].imshow(au, cmap="gray")
    axs[0].set_title("Prediction")
    axs[1].set_title("Anomalous Input")
    axs[2].set_title("EU")
    axs[3].set_title("AU")
    for ax in axs:
        ax.axis('off')
    plt.savefig("era5_result.pdf")


if __name__ == "__main__":
    args = parse_test_args()
    print(args)
    plt.rcParams['text.usetex'] = True

    if args.seed:
        rng = th.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.dataset == Dataset.TOY:
        toy_test(args)
    elif args.dataset == Dataset.ERA5:
        era5_test(args)
    else:
        raise NotImplementedError()

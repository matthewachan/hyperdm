import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

from data.dataset import Dataset
from data.toy import ToyDataset
from guided_diffusion.script_util import create_gaussian_diffusion
from model.mlp import MLP
from src.hyperdm import HyperDM
from src.util import normalize_range, parse_test_args

if __name__ == "__main__":
    args = parse_test_args()
    print(args)

    if args.seed:
        rng = th.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"

    if args.dataset == Dataset.TOY:
        primary_net = MLP([3, 8, 16, 8, 1])
        dataset = ToyDataset(args.dataset_size, split="train")
    else:
        raise NotImplementedError()

    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True,
                                          timestep_respacing="ddim10")
    hyperdm = HyperDM(primary_net, args.hyper_net_input_dim,
                      diffusion).to(device)
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

    plt.rcParams['text.usetex'] = True
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

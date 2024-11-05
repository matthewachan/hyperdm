from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.toy import ToyDataset
from guided_diffusion.script_util import create_gaussian_diffusion
from model.mlp import MLP
from src.util import normalize_range


def get_mean_variance(M,
                      N,
                      models,
                      diffusion,
                      condition,
                      device=None,
                      progress=False):
    """
    Sample the predictive distribution mean and variance. In the paper this is \mathbb{E}_{\hat{x} \sim p(x|y,\phi)}\[\hat{x}\] and \text{Var}_{\hat{x} \sim p(x|y,\phi)}\[\hat{x}\].

    :param M: number of network weights to sample
    :param N: number of predictions to sample per network weight
    :param condition: input condition to sample with
    :param device: device to run on
    :return: mean and variance of the predictive distribution
    """
    _, C, H, W = condition.shape
    mean = th.zeros([M, C, H, W])
    var = th.zeros([M, C, H, W])
    Ms = tqdm(range(M)) if progress else range(M)
    for i in Ms:
        net = partial(functional_call, models[i],
                      dict(models[i].named_parameters()))

        y = condition.repeat(N, 1, 1, 1).to(device)
        with th.no_grad():
            preds = diffusion.ddim_sample_loop(net,
                                               y.shape,
                                               model_kwargs={"y": y},
                                               device=device)
        mean[i] = preds.mean(dim=0)
        var[i] = preds.var(dim=0)
    return mean, var


if __name__ == "__main__":
    M = 10
    # Seed for reproducible results.
    rng = th.manual_seed(1)
    np.random.seed(1)

    device = "cuda" if th.cuda.is_available() else "cpu"

    dataset = ToyDataset(10000, split="train")
    dataloader = DataLoader(dataset, 64, shuffle=True, pin_memory=True)

    # Initialize network
    diffusion = create_gaussian_diffusion(steps=1000, predict_xstart=True)

    # Training loop
    models = []
    for i in tqdm(range(M)):
        primary_net = MLP([3, 8, 16, 8, 1]).to(device)
        primary_net.train()
        optimizer = th.optim.AdamW(primary_net.parameters(), 1e-2)
        prog_bar = tqdm(range(100))
        for step in prog_bar:
            for (x, y) in dataloader:
                x = x.to(device)
                y = y.to(device)
                t = th.randint(0, 1000, (len(x), ), device=device)
                net = partial(functional_call, primary_net,
                              dict(primary_net.named_parameters()))
                loss = diffusion.training_losses(
                    net, x, t, model_kwargs={"y": y})["loss"].mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            prog_bar.set_description(f"loss={loss.item():.4f}")
        models.append(primary_net)

    diffusion = create_gaussian_diffusion(steps=1000,
                                          predict_xstart=True,
                                          timestep_respacing="ddim10")
    # Testing
    eu = []
    au = []
    pred = []
    xs = th.linspace(-1.0, 1.0, 1000)
    for i in tqdm(xs):
        y = th.tensor([i]).reshape(1, 1, 1, 1)
        mean, var = get_mean_variance(M,
                                      100,
                                      models,
                                      diffusion,
                                      y,
                                      device=device)
        eu.append(mean.var())
        au.append(var.mean())
        pred.append(mean.mean())
    eu = th.vstack(eu).ravel()
    au = th.vstack(au).ravel()
    pred = th.vstack(pred).ravel()

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
    plt.title("Deep Ensemble")
    plt.savefig("toy_baseline.pdf")

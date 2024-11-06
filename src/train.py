import numpy as np
import torch as th
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import Dataset
from data.era5 import ERA5
from data.toy import ToyDataset
from guided_diffusion.script_util import create_gaussian_diffusion
from model.mlp import MLP
from model.unet import Unet
from src.hyperdm import HyperDM
from src.util import parse_train_args

if __name__ == "__main__":
    args = parse_train_args()
    print(args)

    # Seed for reproducible results.
    if not args.seed is None:
        rng = th.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = "cuda" if th.cuda.is_available() else "cpu"

    if args.dataset == Dataset.TOY:
        primary_net = MLP([3, 8, 16, 8, 1])
        dataset = ToyDataset(args.dataset_size, split="train")
    elif args.dataset == Dataset.ERA5:
        primary_net = Unet(dim=16,
                           dim_mults=(1, 2, 4, 8),
                           channels=1,
                           self_condition=True)
        dataset = ERA5(args.image_size, split="train")
        dataset = Subset(dataset, range(args.dataset_size))
    else:
        raise NotImplementedError()

    # Initialize network
    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True)
    hyperdm = HyperDM(primary_net, args.hyper_net_dims, diffusion).to(device)
    hyperdm.print_stats()
    hyperdm.train()
    optimizer = th.optim.AdamW(hyperdm.parameters(), args.lr)

    # Training loop
    dataloader = DataLoader(dataset,
                            args.batch_size,
                            shuffle=True,
                            pin_memory=True)
    prog_bar = tqdm(range(args.num_epochs))
    for step in prog_bar:
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            t = th.randint(0, args.diffusion_steps, (len(x), ), device=device)

            net = hyperdm.sample_network(device)
            loss = hyperdm.diffusion.training_losses(
                net, x, t, model_kwargs={"y": y})["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prog_bar.set_description(f"loss={loss.item():.4f}")

        th.save(hyperdm.state_dict(), args.checkpoint)

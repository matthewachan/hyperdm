from typing import List

import torch as th
from torch import nn


class MLP(th.nn.Module):

    def __init__(self, layer_channels: List[int]):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(layer_channels) - 2):
            layers.append(nn.Linear(layer_channels[i], layer_channels[i + 1]))
            layers.append(nn.ReLU())
        # Append output layer
        layers.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: a [B, 1, 1, 1] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: a [B, 1, 1, 1] Tensor of conditions.
        :return: an [B, 1, 1, 1] Tensor of outputs.
        """
        if not timesteps is None and not y is None:
            t = timesteps.reshape(x.shape)
            x = th.cat([x, t, y], dim=-1)
        return self.mlp(x)

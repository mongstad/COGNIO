"""Shared patch encoding, local recurrent interaction, learned future decoding."""

from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    seed: int = 7
    learning_rate: float = 0.001
    horizon: int = 5

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= .01:
            raise ValueError("learning rate must be finite and in [1e-6, .01]")
        if type(self.horizon) is not int or self.horizon != 5:
            raise ValueError("Prototype 1 requires a five-frame horizon")


def patchify(x):
    """B,1,128,128 -> B,256,8,8; raster patch and within-patch ordering."""
    b = x.shape[0]
    return x.reshape(b, 1, 8, 16, 8, 16).permute(0, 1, 3, 5, 2, 4).reshape(b, 256, 8, 8)


def unpatchify(x):
    b = x.shape[0]
    return x.reshape(b, 1, 16, 16, 8, 8).permute(0, 1, 4, 2, 5, 3).reshape(b, 1, 128, 128)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(256, 32, 1)
        self.recurrent = nn.Conv2d(66, 32, 3, padding=1)
        self.decoder = nn.Conv2d(32, 256, 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij")
        self.register_buffer("positions", torch.stack((x, y))[None])

    def forward(self, x, previous):
        encoded = torch.tanh(self.encoder(patchify(x)))
        joint = torch.cat((encoded, previous, self.positions.expand(x.shape[0], -1, -1, -1)), dim=1)
        hidden = .5 * previous + .5 * torch.tanh(self.recurrent(joint))
        prediction = unpatchify(torch.sigmoid(self.decoder(hidden)))
        return prediction, hidden

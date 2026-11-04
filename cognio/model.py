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
    channels: int = 32
    grid: int = 8

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= .01:
            raise ValueError("learning rate must be finite and in [1e-6, .01]")
        if type(self.horizon) is not int or self.horizon != 5:
            raise ValueError("Prototype 1 requires a five-frame horizon")
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("channels must be a positive integer; allocation is resource-checked")
        if type(self.grid) is not int or self.grid not in (8, 16, 32):
            raise ValueError("grid must be 8, 16 or 32 (divisors of the 128-pixel input)")

    @property
    def patch(self):
        return 128 // self.grid

    @property
    def hidden_shape(self):
        return (1, self.channels, self.grid, self.grid)

    @property
    def parameters(self):
        c, p = self.channels, self.patch**2
        return 18*c*c + (2*p + 20)*c + p

    @property
    def hidden_values(self):
        return self.channels * self.grid**2

    @property
    def checkpoint_bytes(self):
        # float32 weights + two Adam moments, hidden/coordinates, ZIP/pickle overhead.
        return 12*self.parameters + 4*self.hidden_values + 8*self.grid**2 + 256*1024

    @property
    def working_bytes(self):
        # Weights/gradients/Adam, rollback, candidate/serialization transients,
        # five detached contexts, activation gradients and conservative work space.
        return 40*self.parameters + 128*self.hidden_values + 512*1024**2


def patchify(x, grid=8):
    """B,1,128,128 -> B,patch²,grid,grid; raster positions and pixel ordering."""
    b = x.shape[0]
    patch = 128 // grid
    return x.reshape(b, 1, grid, patch, grid, patch).permute(0, 1, 3, 5, 2, 4).reshape(b, patch**2, grid, grid)


def unpatchify(x):
    b = x.shape[0]
    grid = x.shape[-1]
    patch = 128 // grid
    return x.reshape(b, 1, patch, patch, grid, grid).permute(0, 1, 4, 2, 5, 3).reshape(b, 1, 128, 128)


class Predictor(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c, p, g = config.channels, config.patch**2, config.grid
        self.encoder = nn.Conv2d(p, c, 1)
        self.recurrent = nn.Conv2d(2*c + 2, c, 3, padding=1)
        self.decoder = nn.Conv2d(c, p, 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g), indexing="ij")
        self.register_buffer("positions", torch.stack((x, y))[None])

    def forward(self, x, previous):
        encoded = torch.tanh(self.encoder(patchify(x, self.config.grid)))
        joint = torch.cat((encoded, previous, self.positions.expand(x.shape[0], -1, -1, -1)), dim=1)
        hidden = .5 * previous + .5 * torch.tanh(self.recurrent(joint))
        prediction = unpatchify(torch.sigmoid(self.decoder(hidden)))
        return prediction, hidden

"""Shared patch encoding, local recurrent interaction, learned future decoding."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from .inputs import CHROMA_OFFSET, components, sensory_contract


GRIDS = (8, 16, 32, 64)
TOPOLOGIES = ("standard", "toroidal")


@dataclass(frozen=True)
class Config:
    seed: int = 7
    learning_rate: float = 0.001
    horizon: int = 5
    channels: int = 32
    grid: int = 8
    input_mode: str = "y"
    topology: str = "standard"

    def __post_init__(self):
        components(self.input_mode)
        if self.topology not in TOPOLOGIES:
            raise ValueError("topology must be standard or toroidal")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32")
        if not math.isfinite(self.learning_rate) or not 1e-6 <= self.learning_rate <= .01:
            raise ValueError("learning rate must be finite and in [1e-6, .01]")
        if type(self.horizon) is not int or self.horizon != 5:
            raise ValueError("Prototype 1 requires a five-frame horizon")
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("channels must be a positive integer; allocation is resource-checked")
        if type(self.grid) is not int or self.grid not in GRIDS:
            raise ValueError("grid must be 8, 16, 32 or 64 (divisors of the 128-pixel input)")

    @property
    def patch(self):
        return 128 // self.grid

    @property
    def hidden_shape(self):
        return (1, self.channels, self.grid, self.grid)

    @property
    def input_channels(self):
        return len(components(self.input_mode))

    @property
    def input_shape(self):
        return (128, 128) if self.input_mode == "y" else (3, 128, 128)

    @property
    def sensory_contract(self):
        return sensory_contract(self.input_mode)

    @property
    def parameters(self):
        c, p = self.channels, self.input_channels*self.patch**2
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
        return (40*self.parameters + 128*self.hidden_values + 512*1024**2
                + 5*(self.input_channels-1)*2*128**2*4
                + (16*(2*self.channels+2)*(self.grid+2)**2 if self.topology == "toroidal" else 0))


def patchify(x, grid=8):
    """B,K,128,128 -> B,K*patch²,grid,grid; component then raster pixel order."""
    b, k = x.shape[:2]
    patch = 128 // grid
    return x.reshape(b, k, grid, patch, grid, patch).permute(0, 1, 3, 5, 2, 4).reshape(b, k*patch**2, grid, grid)


def unpatchify(x):
    b = x.shape[0]
    grid = x.shape[-1]
    patch = 128 // grid
    k = x.shape[1] // patch**2
    return x.reshape(b, k, patch, patch, grid, grid).permute(0, 1, 4, 2, 5, 3).reshape(b, k, 128, 128)


class Predictor(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c, p, g = config.channels, config.input_channels*config.patch**2, config.grid
        self.encoder = nn.Conv2d(p, c, 1)
        self.recurrent = nn.Conv2d(2*c + 2, c, 3, padding=1,
                                   padding_mode="circular" if config.topology == "toroidal" else "zeros")
        self.decoder = nn.Conv2d(c, p, 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g), indexing="ij")
        self.register_buffer("positions", torch.stack((x, y))[None])

    def forward(self, x, previous):
        encoded = torch.tanh(self.encoder(patchify(x, self.config.grid)))
        joint = torch.cat((encoded, previous, self.positions.expand(x.shape[0], -1, -1, -1)), dim=1)
        hidden = .5 * previous + .5 * torch.tanh(self.recurrent(joint))
        prediction = unpatchify(torch.sigmoid(self.decoder(hidden)))
        if self.config.input_mode == "ycbcr":
            prediction = torch.cat((prediction[:, :1], prediction[:, 1:] - CHROMA_OFFSET), dim=1)
        return prediction, hidden

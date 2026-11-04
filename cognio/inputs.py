"""Sensor code values and deterministic controls; no display transformations."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Frame:
    sequence: int
    timestamp: float
    values: np.ndarray


def extract_y(raw, width, height, stride):
    """YUYV = Y0 U Y1 V; retain padding semantics and own the returned memory."""
    if width <= 0 or height <= 0 or width % 2 or stride < width * 2 or len(raw) < stride * height:
        raise ValueError("short or invalid YUYV frame")
    rows = np.frombuffer(raw, dtype=np.uint8, count=stride * height).reshape(height, stride)
    return rows[:, :2 * width:2].copy()


def area_axis(a, size, axis):
    """Exact area average of piecewise-constant pixels (including partial pixels)."""
    a = np.moveaxis(a, axis, 0)
    edges = np.linspace(0, a.shape[0], size + 1)
    index = np.minimum(edges.astype(int), a.shape[0] - 1)
    prefix = np.concatenate((np.zeros_like(a[:1]), np.cumsum(a, axis=0)), axis=0)
    integral = prefix[index] + (edges - index).reshape((-1,) + (1,) * (a.ndim - 1)) * a[index]
    return np.moveaxis(np.diff(integral, axis=0) / (a.shape[0] / size), 0, axis)


def normalize(y):
    """Map the full 4:3 frame to a square; Y/255, no adaptive normalization."""
    if y.shape != (480, 640) or y.dtype != np.uint8:
        raise ValueError("expected 640x480 uint8 Y")
    return area_axis(area_axis(y.astype(np.float64) / 255, 128, 0), 128, 1).astype(np.float32)


MODES = ("static", "moving", "shape", "temporal", "zero", "no-input")


def synthetic(sequence, mode):
    """Fixed analytic controls at nominal 30 Hz; moving pattern repeats every 40 frames."""
    y, x = np.mgrid[:128, :128] / 128
    if mode == "static":
        values = .15 + .7 * ((x > .25) & (x < .65) & (y > .2) & (y < .7))
    elif mode == "moving":
        # Smooth translated grating; deliberately imperfect persistence baseline.
        phase = 2 * np.pi * sequence / 40
        values = .5 + .3 * np.sin(2 * np.pi * x - phase) * np.cos(2 * np.pi * y)
    elif mode == "shape":
        center = .5 + .25 * np.sin(2 * np.pi * sequence / 40)
        values = .1 + .8 * np.exp(-((x-center)**2 + (y-.5)**2) / .015)
    elif mode == "temporal":
        values = np.full((128, 128), .5 + .35 * np.sin(2 * np.pi * sequence / 40))
    elif mode == "zero":
        values = np.zeros((128, 128))
    elif mode == "no-input":
        return None
    else:
        raise ValueError(f"unknown synthetic mode {mode}")
    return values.astype(np.float32)

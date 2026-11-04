"""Sensor code values and deterministic controls; no display transformations."""

from dataclasses import dataclass
import numpy as np

INPUT_MODES = ("y", "ycbcr")
CHROMA_OFFSET = 128 / 255


def components(input_mode):
    if input_mode not in INPUT_MODES:
        raise ValueError("input mode must be y or ycbcr")
    return ("Y",) if input_mode == "y" else ("Y", "Cb", "Cr")


def sensory_contract(input_mode):
    names = components(input_mode)
    return dict(version=1, mode=input_mode, components=list(names),
                shape=[128, 128] if input_mode == "y" else [3, 128, 128],
                layout="HW" if input_mode == "y" else "CHW",
                native_shapes=[[480, 640]] if input_mode == "y" else [[480, 640], [480, 320], [480, 320]],
                normalization="Y/255" if input_mode == "y" else "Y/255;(Cb-128)/255;(Cr-128)/255",
                resampling="separable-area-full-frame-pair-footprints-v1",
                loss="equal-component-mean-pixel-MSE")


def component_statistics(values, input_mode):
    planes = values[None] if input_mode == "y" else values
    return {name: dict(mean=float(plane.mean()), std=float(plane.std()),
                       minimum=float(plane.min()), maximum=float(plane.max()))
            for name, plane in zip(components(input_mode), planes)}


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


def extract_ycbcr(raw, width, height, stride):
    """Byte Y0 Cb Y1 Cr: Y is H×W; Cb/Cr are H×(W/2), one per pixel pair.

    Work in bytes, independent of host word endianness; exclude row padding.
    All returned arrays own memory so the capture buffer can be requeued.
    """
    y = extract_y(raw, width, height, stride)
    rows = np.frombuffer(raw, dtype=np.uint8, count=stride * height).reshape(height, stride)
    return y, rows[:, 1:2*width:4].copy(), rows[:, 3:2*width:4].copy()


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


def normalize_ycbcr(planes):
    """CHW float32: unchanged Y/255 and signed (Cb/Cr-128)/255.

    Each chroma sample represents its two-pixel horizontal footprint. Average
    native 480×320 planes directly to 128² over the same full-frame extent as Y;
    no RGB conversion, upsampled independent measurements, or range clipping.
    """
    y, cb, cr = planes
    if any(p.shape != (480, 320) or p.dtype != np.uint8 for p in (cb, cr)):
        raise ValueError("expected native 320x480 uint8 Cb and Cr")
    chroma = [area_axis(area_axis((p.astype(np.float64)-128)/255, 128, 0), 128, 1).astype(np.float32)
              for p in (cb, cr)]
    return np.stack((normalize(y), *chroma))


MODES = ("static", "moving", "shape", "temporal", "zero", "no-input",
         "color-static", "color-moving", "color-temporal")


def synthetic(sequence, mode, input_mode="y"):
    """Fixed analytic controls at nominal 30 Hz; moving pattern repeats every 40 frames."""
    y, x = np.mgrid[:128, :128] / 128
    components(input_mode)
    cb = cr = np.zeros((128, 128))
    if mode.startswith("color-"):
        values = np.full((128, 128), .5)
        phase = 2 * np.pi * sequence / 40
        if mode == "color-static":
            cb = .3 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
            cr = .25 * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
        elif mode == "color-moving":
            cb = .3 * np.sin(2*np.pi*x-phase) * np.cos(2*np.pi*y)
            cr = .25 * np.cos(2*np.pi*x-phase) * np.sin(2*np.pi*y)
        elif mode == "color-temporal":
            cb = np.full((128, 128), .25*np.sin(phase))
            cr = np.full((128, 128), .25*np.cos(phase))
        else:
            raise ValueError(f"unknown synthetic mode {mode}")
    elif mode == "static":
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
    return (values if input_mode == "y" else np.stack((values, cb, cr))).astype(np.float32)

"""Opt-in scalar diagnostics; no camera, Qt, recording or optimizer mutations.

These routines are deliberately outside the live inference/update path. Spatial
statistics describe periodic structure, not a classifier for a failed model.
"""

import hashlib

import numpy as np
import torch

from .inputs import CHROMA_OFFSET, components


def tensor_statistics(value):
    value = value.detach().to(dtype=torch.float64)
    finite = bool(torch.isfinite(value).all())
    if not finite:
        return dict(finite=False, count=value.numel())
    return dict(finite=True, count=value.numel(), minimum=float(value.min()), maximum=float(value.max()),
                mean=float(value.mean()), rms=float(value.square().mean().sqrt()),
                l2=float(torch.linalg.vector_norm(value)),
                abs_ge_095=float((value.abs() >= .95).double().mean()))


def state_digest(value):
    """Exact recursive digest for before/after mutation checks, not anonymization."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode()+b"\0")
    visit(value)
    return digest.hexdigest()


def periodicity(plane, grid):
    """Patch-phase bias and exact Fourier lattice energy, with DC removed.

    The phase mean averages pixels sharing (row mod patch, column mod patch).
    Its variance / image variance equals the fraction of non-DC FFT energy on
    the decoder's lattice. A naturally periodic input can score highly too.
    """
    plane = np.asarray(plane, dtype=np.float64)
    if plane.shape != (128, 128) or grid not in (8, 16, 32) or not np.isfinite(plane).all():
        raise ValueError("periodicity requires finite 128x128 data and grid 8/16/32")
    patch = 128//grid
    phase = plane.reshape(grid, patch, grid, patch).mean(axis=(0, 2))
    variance = float(plane.var())
    phase_variance = float(phase.var())
    dx, dy = np.abs(np.diff(plane, axis=1)), np.abs(np.diff(plane, axis=0))
    boundary = np.arange(1,128) % patch == 0
    edge = float((dx[:, boundary].mean()+dy[boundary].mean())/2)
    interior = float((dx[:, ~boundary].mean()+dy[~boundary].mean())/2)
    spectrum = np.abs(np.fft.fft2(plane-plane.mean()))**2
    fundamental = float(spectrum[grid,0]+spectrum[-grid,0]+spectrum[0,grid]+spectrum[0,-grid])
    total = float(spectrum.sum())
    return dict(patch_pixels=patch, phase_rms=phase_variance**.5,
                lattice_energy_fraction=phase_variance/variance if variance > 1e-24 else 0.,
                fundamental_axis_energy_fraction=fundamental/total if total > 1e-16 else 0.,
                boundary_jump=edge, interior_jump=interior)


def plane_statistics(values, grid, mode):
    planes = values[None] if mode == "y" else values
    result = {}
    for index, name in enumerate(components(mode)):
        plane = np.asarray(planes[index], dtype=np.float64)
        codes = plane+(CHROMA_OFFSET if index else 0)
        result[name] = dict(minimum=float(plane.min()), maximum=float(plane.max()),
                            mean=float(plane.mean()), std=float(plane.std()),
                            sigmoid_low_share=float(np.mean(codes <= .01)),
                            sigmoid_high_share=float(np.mean(codes >= .99)),
                            histogram_10_bins=np.histogram(codes, bins=np.linspace(0,1,11))[0].tolist(),
                            periodicity=periodicity(plane, grid))
    return result


def preview_float(values):
    """Independent unclipped BT.601 reference matching the GUI's documented map.

    Diagnostic-only: never returned to the sensory/model path. Tests compare its
    rounded/clipped bytes to the actual GUI converter without importing Qt here.
    """
    y, cb, cr = np.asarray(values, dtype=np.float64)
    y, cb, cr = (255*y-16)/219, 255*cb/224, 255*cr/224
    return np.stack((y+1.402*cr, y-(.114*1.772/.587)*cb-(.299*1.402/.587)*cr, y+1.772*cb))


def prediction_statistics(values, grid, mode):
    result = dict(model=plane_statistics(values, grid, mode))
    if mode == "ycbcr":
        rgb = preview_float(values)
        result["rgb_preview"] = dict(clipped_low_share=float(np.mean(rgb < 0)),
            clipped_high_share=float(np.mean(rgb > 1)),
            black_pixel_share=float(np.mean(np.all(np.rint(255*np.clip(rgb,0,1)) == 0, axis=0))),
            clipped_mean_periodicity=periodicity(np.clip(rgb,0,1).mean(axis=0), grid))
    return result


def learned_statistics(engine):
    result = dict(parameters={name:tensor_statistics(value) for name,value in engine.model.named_parameters()},
                  adam={}, learning_rate=engine.optimizer.param_groups[0]["lr"])
    for name, parameter in engine.model.named_parameters():
        state = engine.optimizer.state.get(parameter)
        if state:
            result["adam"][name] = dict(step=float(state["step"]),
                exp_avg=tensor_statistics(state["exp_avg"]), exp_avg_sq=tensor_statistics(state["exp_avg_sq"]))
    return result


def probe_context(engine, source, previous, target=None):
    """Recompute one context for diagnosis without advancing recurrence or Adam.

    If a target has arrived, autograd.grad measures the pre-clipping gradient of
    the same one-step surrogate used by Engine._update. It never writes .grad.
    For frozen runs this is a hypothetical gradient, not an update.
    """
    activations, handles = {}, []
    try:
        for name in ("encoder", "recurrent", "decoder"):
            def capture(module, inputs, output, name=name):
                activations[name] = output
            handles.append(getattr(engine.model,name).register_forward_hook(capture))
        with torch.set_grad_enabled(target is not None):
            predicted, hidden = engine.model(engine._tensor(source), previous.detach())
            result = dict(hidden=tensor_statistics(hidden), preactivations={})
            for name, values in activations.items():
                stats = tensor_statistics(values)
                if name in ("encoder", "recurrent"):
                    derivative = 1-values.detach().tanh().square()
                    stats["tanh_derivative_le_1e_6_share"] = float((derivative <= 1e-6).float().mean())
                else:
                    derivative = values.detach().sigmoid()
                    derivative = derivative*(1-derivative)
                    stats["sigmoid_derivative_le_1e_6_share"] = float((derivative <= 1e-6).float().mean())
                result["preactivations"][name] = stats
            if target is not None:
                loss = (predicted-engine._tensor(target)).square().mean()
                gradients = torch.autograd.grad(loss, tuple(engine.model.parameters()))
                result["surrogate_loss"] = float(loss.detach())
                result["pre_clip_gradients"] = {name:tensor_statistics(gradient)
                    for (name,_),gradient in zip(engine.model.named_parameters(),gradients)}
                result["pre_clip_global_norm"] = sum(s["l2"]**2 for s in result["pre_clip_gradients"].values())**.5
            return result
    finally:
        for handle in handles:
            handle.remove()

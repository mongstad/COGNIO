"""Hardware tests are skipped explicitly when no CUDA execution device exists."""

import numpy as np
import pytest
import torch

from cognio.engine import Engine
from cognio.inputs import Frame, synthetic
from cognio.model import Config
from cognio.runtime import select_device


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")


@pytest.mark.parametrize("input_mode", ["y", "ycbcr"])
def test_cuda_observer_is_numerically_independent(input_mode):
    now = [0.]
    config = Config(channels=128, grid=16, input_mode=input_mode)
    a = Engine(config, "cuda", clock=lambda: now[0]+.001)
    b = Engine(config, "cuda", clock=lambda: now[0]+.001)
    for i in range(12):
        now[0] = i/30
        frame = Frame(i, now[0], synthetic(i, "color-moving" if input_mode == "ycbcr" else "moving", input_mode))
        a.step(frame, observe=False)
        b.step(frame, observe=True, channel_start=32)
    assert a.counts == b.counts and not a.fault and not b.fault
    assert torch.equal(a.hidden, b.hidden)
    assert all(torch.equal(value, b.model.state_dict()[key]) for key, value in a.model.state_dict().items())


@pytest.mark.parametrize("channels,grid", [(32, 8), (128, 16), (512, 16)])
@pytest.mark.parametrize("input_mode", ["y", "ycbcr"])
def test_real_cuda_optimizer_freeze_and_portable_checkpoint(tmp_path, channels, grid, input_mode):
    device, notice = select_device("cuda")
    assert "training probe passed" in notice
    now = [0.]
    config = Config(channels=channels, grid=grid, input_mode=input_mode)
    stimulus = "color-moving" if input_mode == "ycbcr" else "moving"
    gpu = Engine(config, device, clock=lambda: now[0]+.001)
    cpu = Engine(config, "cpu", clock=lambda: now[0]+.001)
    initial = gpu.model.decoder.bias.detach().clone()
    for i in range(9):
        now[0] = i/30
        frame = Frame(i, now[0], synthetic(i, stimulus, input_mode))
        a, b = gpu.step(frame), cpu.step(frame)
        assert not gpu.fault
        np.testing.assert_allclose(a["prediction"], b["prediction"], atol=3e-5, rtol=3e-4)
    assert gpu.counts["updates"] == 4
    assert not torch.equal(initial, gpu.model.decoder.bias)
    gpu.set_training(False)
    weights = {key: value.clone() for key, value in gpu.model.state_dict().items()}
    for i in range(9, 16):
        now[0] = i/30
        gpu.step(Frame(i, now[0], synthetic(i, stimulus, input_mode)), observe=False)
    assert all(torch.equal(value, gpu.model.state_dict()[key]) for key, value in weights.items())
    path = tmp_path/"gpu.pt"
    gpu.save(path)
    data = torch.load(path, weights_only=True)
    assert all(t.device.type == "cpu" for t in data["model"].values())
    restored = Engine.load(path, "cpu", clock=lambda: now[0]+.001)
    assert restored.counts == gpu.counts and not restored.training
    now[0] = 16/30
    frame = Frame(16, now[0], synthetic(16, stimulus, input_mode))
    a, b = gpu.step(frame), restored.step(frame)
    np.testing.assert_allclose(a["prediction"], b["prediction"], atol=3e-5, rtol=3e-4)

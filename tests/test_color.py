import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cognio import camera as cam
from cognio.engine import Engine
from cognio.inputs import (CHROMA_OFFSET, Frame, extract_y, extract_ycbcr, normalize,
                           normalize_ycbcr, synthetic)
from cognio.model import Config, Predictor, patchify, unpatchify
from cognio.runtime import Runtime


def test_byte_exact_native_planes_padding_and_owned_storage():
    raw = bytearray([10, 1, 20, 250, 30, 2, 40, 251, 99, 98, 97,
                     50, 3, 60, 252, 70, 4, 80, 253, 96, 95, 94])
    y, cb, cr = extract_ycbcr(raw, 4, 2, 11)
    np.testing.assert_array_equal(y, [[10, 20, 30, 40], [50, 60, 70, 80]])
    np.testing.assert_array_equal(cb, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(cr, [[250, 251], [252, 253]])
    np.testing.assert_array_equal(y, extract_y(raw, 4, 2, 11))
    assert y.shape == (2, 4) and cb.shape == cr.shape == (2, 2)
    raw[:] = bytes(len(raw))
    assert y[0, 0] == 10 and cb[0, 0] == 1 and cr[0, 0] == 250
    for width, stride, data in ((3, 8, bytes(16)), (4, 7, bytes(16)), (4, 8, bytes(15))):
        with pytest.raises(ValueError):
            extract_ycbcr(data, width, 2, stride)


@pytest.mark.parametrize("code", [0, 16, 128, 235, 255])
def test_fixed_signed_normalization_and_neutral_chroma(code):
    y = np.full((480, 640), code, np.uint8)
    cb = np.full((480, 320), code, np.uint8)
    cr = np.full_like(cb, 128)
    data = normalize_ycbcr((y, cb, cr))
    assert data.shape == (3, 128, 128) and data.dtype == np.float32
    np.testing.assert_array_equal(data[0], normalize(y))
    np.testing.assert_allclose(data[1], (code-128)/255, atol=1e-7)
    assert not data[2].any()
    e = Engine(Config(input_mode="ycbcr"), clock=lambda: .001)
    assert e.step(Frame(0, 0., data)) is not None and not e.fault


def test_chroma_area_uses_native_pair_footprints_and_partial_boundaries():
    rng = np.random.default_rng(17)
    y = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    cb = rng.integers(0, 256, (480, 320), dtype=np.uint8)
    cr = rng.integers(0, 256, (480, 320), dtype=np.uint8)
    data = normalize_ycbcr((y, cb, cr))
    # First output covers 3.75 native rows and 2.5 chroma-pair columns.
    weights = np.outer([1, 1, 1, .75], [1, 1, .5]) / (3.75*2.5)
    for index, plane in enumerate((cb, cr), 1):
        expected = np.sum(weights*(plane[:4, :3].astype(float)-128)/255)
        assert data[index, 0, 0] == pytest.approx(expected, abs=1e-7)
        # Equivalent footprint representation, not independent high-resolution samples.
        replicated = np.repeat(plane, 2, axis=1)
        np.testing.assert_allclose(data[index], normalize(replicated)-CHROMA_OFFSET, atol=1e-7)
    with pytest.raises(ValueError):
        normalize_ycbcr((y, np.repeat(cb, 2, axis=1), cr))


def test_native_color_capture_preserves_pairing_time_sequence_and_requeue():
    camera = cam.V4L2Camera.__new__(cam.V4L2Camera)
    camera.fd, camera.input_mode = 10, "ycbcr"
    row = bytes([10, 90, 20, 170])*320 + bytes([99]*8)
    camera.buffers = [row*480]
    camera.mode = dict(stride=1288)
    camera.frames = camera.driver_drops = camera.discarded = camera.bad_frames = 0
    camera.previous_sequence = 40
    requeued = []
    def ioctl(fd, code, value):
        if code == cam.DQBUF:
            return cam.Buffer(index=0, sequence=42, bytesused=1288*480, flags=0x2000,
                              timestamp=cam.Timeval(seconds=12, microseconds=345000))
        requeued.append(value.index)
        return value
    with patch.object(cam.select, "select", return_value=([10], [], [])), patch.object(cam, "ioctl", side_effect=ioctl):
        planes, stamp, sequence = camera.read()
    assert stamp == 12.345 and sequence == 42 and requeued == [0]
    assert camera.driver_drops == 1 and camera.timestamp_kind == "driver-monotonic"
    assert planes[0][0, :4].tolist() == [10, 20, 10, 20]
    assert np.all(planes[1] == 90) and np.all(planes[2] == 170)
    assert normalize_ycbcr(planes).shape == (3, 128, 128)


@pytest.mark.parametrize("grid", [8, 16, 32, 64])
def test_component_patch_order_dimensions_and_output_range(grid):
    x = torch.arange(3*128**2, dtype=torch.float32).reshape(1, 3, 128, 128)
    packed = patchify(x, grid)
    p = 128//grid
    assert packed.shape == (1, 3*p*p, grid, grid)
    assert torch.equal(unpatchify(packed), x)
    for k in range(3):
        assert torch.equal(packed[0, k*p*p:(k+1)*p*p, 2, 3].reshape(p, p),
                           x[0, k, 2*p:3*p, 3*p:4*p])
    config = Config(channels=32, grid=grid, input_mode="ycbcr")
    model = Predictor(config)
    assert sum(t.numel() for t in model.parameters()) == config.parameters
    output, hidden = model(torch.zeros_like(x), torch.zeros(config.hidden_shape))
    assert output.shape == x.shape and hidden.shape == config.hidden_shape
    assert output[:, :1].min() >= 0 and output[:, :1].max() <= 1
    assert output[:, 1:].min() >= -CHROMA_OFFSET and output[:, 1:].max() <= 1-CHROMA_OFFSET


def test_constant_y_controls_differ_only_in_color_and_score_every_component():
    now = [0.]
    engine = Engine(Config(input_mode="ycbcr"), clock=lambda: now[0]+.001)
    a = synthetic(0, "color-moving", "ycbcr")
    b = synthetic(5, "color-moving", "ycbcr")
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(synthetic(0, "color-moving"), synthetic(5, "color-moving"))
    assert not np.array_equal(a[1:], b[1:])
    initial = engine.step(Frame(0, 0., a))
    issued = initial["prediction"].copy()
    other = Engine(Config(input_mode="ycbcr"), clock=lambda: .001)
    np.testing.assert_array_equal(other.step(Frame(0, 0., a))["prediction"], issued)
    for i in range(1, 6):
        now[0] = i/30
        engine.step(Frame(i, now[0], synthetic(i, "color-moving", "ycbcr")))
    result = engine.last_result
    assert result["source"] == 0 and result["target"] == 5 and result["issued"] < result["target_time"]
    assert engine.counts["updates"] == 1
    errors = ((issued-b)**2).mean(axis=(1,2))
    baseline = ((a-b)**2).mean(axis=(1,2))
    for i, name in enumerate(("Y", "Cb", "Cr")):
        assert result["component_mse"][name] == pytest.approx(errors[i])
        assert result["component_baseline_mse"][name] == pytest.approx(baseline[i])
    assert result["mse"] == pytest.approx(errors.mean())
    assert result["baseline_mse"] == pytest.approx(baseline.mean())
    assert result["component_baseline_mse"]["Y"] == 0
    assert result["component_baseline_mse"]["Cb"] > 0 and result["component_baseline_mse"]["Cr"] > 0
    # Alter only an unseen chroma target; the already-issued prediction is immutable.
    changed = b.copy(); changed[1:] = 0
    other.step(Frame(5, 5/30, changed))
    assert other.last_result["component_mse"]["Cb"] != result["component_mse"]["Cb"]
    np.testing.assert_array_equal(initial["prediction"], issued)


def test_color_freeze_missing_and_late_targets_and_bounded_history():
    now = [0.]
    e = Engine(Config(input_mode="ycbcr"), clock=lambda: now[0]+.001)
    for i in range(12):
        now[0] = i/30; e.step(Frame(i, now[0], synthetic(i, "color-temporal", "ycbcr")))
    e.set_training(False)
    weights = copy.deepcopy(e.model.state_dict())
    optimizer = copy.deepcopy(e.optimizer.state_dict())
    hidden = e.hidden.clone()
    for i in range(12, 32):
        now[0] = i/30; e.step(Frame(i, now[0], synthetic(i, "color-temporal", "ycbcr")))
    assert all(torch.equal(v, e.model.state_dict()[k]) for k,v in weights.items())
    for key, state in optimizer["state"].items():
        assert all(torch.equal(v, e.optimizer.state_dict()["state"][key][k]) for k,v in state.items())
    assert not torch.equal(hidden, e.hidden) and len(e.pending) <= 5
    assert e.history.maxlen == e.component_history.maxlen == 300
    e.flush()
    now[0] = 32/30; e.step(Frame(32, now[0], synthetic(32, "color-static", "ycbcr")))
    now[0] = 38/30; e.step(Frame(38, now[0], synthetic(38, "color-static", "ycbcr")))
    assert e.last_result is None and e.counts["missed_targets"] == 1
    late = Engine(Config(input_mode="ycbcr"), clock=lambda: 10.)
    for i in (0,5):
        late.step(Frame(i, i/30, synthetic(i, "color-moving", "ycbcr")))
    assert late.counts["late_targets"] == 1 and late.counts["updates"] == 0


def test_grayscale_cannot_distinguish_the_constant_y_color_controls():
    now = [0.]
    a = Engine(clock=lambda: now[0]+.001)
    b = Engine(clock=lambda: now[0]+.001)
    for i in range(20):
        now[0] = i/30
        x = a.step(Frame(i, now[0], synthetic(i, "color-moving")))
        y = b.step(Frame(i, now[0], synthetic(i, "color-temporal")))
        np.testing.assert_array_equal(x["prediction"], y["prediction"])
    assert a.counts["updates"] == b.counts["updates"] == 15
    assert all(torch.equal(value,b.model.state_dict()[key]) for key,value in a.model.state_dict().items())
    assert a.last_result["component_baseline_mse"] == {"Y":0.0}


def test_color_checkpoint_exact_continuation_and_cross_mode_rejection(tmp_path):
    now = [0.]
    config = Config(channels=64, grid=16, input_mode="ycbcr")
    engine = Engine(config, clock=lambda: now[0]+.001)
    for i in range(10):
        now[0] = i/30; engine.step(Frame(i, now[0], synthetic(i, "color-moving", "ycbcr")))
    path = tmp_path/"color.pt"; engine.save(path)
    restored = Engine.load(path, clock=lambda: now[0]+.001)
    assert restored.config == config and restored.state()["sensory"] == config.sensory_contract
    for i in range(10, 20):
        now[0] = i/30; frame = Frame(i, now[0], synthetic(i, "color-moving", "ycbcr"))
        np.testing.assert_array_equal(engine.step(frame)["prediction"], restored.step(frame)["prediction"])
    assert torch.equal(engine.hidden, restored.hidden) and engine.counts == restored.counts
    for saved_mode, running_mode in (("ycbcr","y"),("y","ycbcr")):
        Engine(Config(channels=64, grid=16, input_mode=saved_mode)).save(path)
        runtime = Runtime(Engine(Config(channels=64, grid=16, input_mode=running_mode)))
        before = runtime.engine
        with pytest.raises(ValueError, match="input mode"):
            runtime._command("load", str(path))
        assert runtime.engine is before


@pytest.mark.parametrize("field", ["shape", "normalization", "components", "resampling", "loss"])
def test_checkpoint_rejects_changed_sensory_contract(tmp_path, field):
    state = Engine(Config(input_mode="ycbcr")).state()
    state["sensory"][field] = "wrong"
    path = tmp_path/"bad.pt"; torch.save(state,path)
    with pytest.raises(ValueError, match="sensory"):
        Engine.load(path)


def test_scaled_schema2_grayscale_migration(tmp_path):
    e = Engine(Config(channels=128, grid=16))
    old = e.state()
    old["schema"], old["architecture"] = 2, "patch-spatial-rnn-v2"
    old.pop("sensory"); old["config"].pop("input_mode"); old["config"].pop("topology")
    path = tmp_path/"v2.pt"; torch.save(old,path)
    loaded = Engine.load(path)
    assert loaded.config == e.config
    assert all(torch.equal(v,loaded.model.state_dict()[k]) for k,v in e.model.state_dict().items())


def test_color_range_guard_does_not_clip_negative_chroma():
    e = Engine(Config(input_mode="ycbcr"), clock=lambda: .001)
    data = synthetic(0, "color-static", "ycbcr")
    assert data[1:].min() < 0
    e.step(Frame(0, 0., data))
    np.testing.assert_array_equal(e.pending[5].source.values, data)
    for channel, bad in ((0,-.01),(1,-.51),(2,.51)):
        invalid=data.copy(); invalid[channel,0,0]=bad
        with pytest.raises(ValueError, match="range"):
            e.step(Frame(1, 1., invalid))
    with pytest.raises(ValueError, match="input mode"):
        Config(input_mode="rgb")

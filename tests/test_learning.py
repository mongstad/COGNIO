from dataclasses import replace
import copy
import numpy as np
import pytest
import torch

from cognio.engine import Engine
from cognio.inputs import Frame, extract_y, normalize, synthetic
from cognio.model import Config, patchify, unpatchify


class Stream:
    def __init__(self, engine=None):
        self.time = 0.
        self.engine = engine or Engine(clock=lambda: self.time + .001)

    def step(self, i, mode="moving"):
        self.time = i / 30
        return self.engine.step(Frame(i, self.time, synthetic(i, mode)))


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def test_patch_order_and_cross_patch_interaction():
    x = torch.arange(128**2, dtype=torch.float32).reshape(1, 1, 128, 128)
    patches = patchify(x)
    assert patches.shape == (1, 256, 8, 8)
    assert torch.equal(unpatchify(patches), x)
    assert torch.equal(patches[0, :, 2, 3].reshape(16, 16), x[0, 0, 32:48, 48:64])
    model = Engine().model
    previous = torch.zeros(1, 32, 8, 8, requires_grad=True)
    output, hidden = model(torch.zeros_like(x), previous)
    hidden[0, 0, 3, 3].backward()
    assert output.shape == (1, 1, 128, 128)
    assert previous.grad[0, :, 3, 4].abs().sum() > 0
    assert previous.grad[0, :, 7, 7].abs().sum() == 0


def test_yuyv_padding_and_fixed_area_normalization():
    raw = bytes([10, 250, 20, 251, 99, 99, 30, 252, 40, 253, 99, 99])
    assert extract_y(raw, 2, 2, 6).tolist() == [[10, 20], [30, 40]]
    for value in (0, 16, 128, 235, 255):
        result = normalize(np.full((480, 640), value, dtype=np.uint8))
        assert result.shape == (128, 128) and result.dtype == np.float32
        np.testing.assert_allclose(result, value / 255, atol=1e-7)
    y = np.zeros((480, 640), dtype=np.uint8)
    y[:, 320:] = 255
    result = normalize(y)
    assert np.all(result[:, :64] == 0) and np.all(result[:, 64:] == 1)
    with pytest.raises(ValueError):
        extract_y(b"short", 640, 480, 1280)


def test_exact_causal_pairing_and_frozen_prediction_metric():
    stream = Stream()
    initial = stream.step(0)
    original = initial["prediction"].copy()
    for i in range(1, 6):
        stream.step(i)
    result = stream.engine.last_result
    assert result["source"] == 0 and result["target"] == 5
    assert result["issued"] < result["target_time"]
    assert result["horizon_ms"] == pytest.approx(1000 / 6)
    expected = np.mean((original - synthetic(5, "moving"))**2)
    assert result["mse"] == pytest.approx(expected)
    assert stream.engine.counts["updates"] == 1
    # A future target is not used in the issuance-time prediction.
    other = Stream()
    np.testing.assert_array_equal(other.step(0)["prediction"], original)
    for i in range(1, 6):
        other.step(i, "zero")
    assert other.engine.last_result["mse"] != result["mse"]
    np.testing.assert_array_equal(initial["prediction"], original)


def test_dropped_and_late_targets_are_never_substituted():
    stream = Stream()
    stream.step(0)
    stream.step(6)
    assert stream.engine.last_result is None
    assert stream.engine.counts["missed_targets"] == 1
    stream = Stream(Engine(clock=lambda: 10.))
    for i in range(6):
        stream.step(i)
    assert stream.engine.counts["late_targets"] == 1
    assert stream.engine.counts["updates"] == 0


def test_freeze_preserves_parameters_and_optimizer_but_not_hidden():
    stream = Stream()
    for i in range(20):
        stream.step(i)
    before = stream.engine.state()
    stream.engine.set_training(False)
    for i in range(20, 40):
        stream.step(i)
    after = stream.engine.state()
    assert equal_state(before["model"], after["model"])
    assert equal_state(before["optimizer"], after["optimizer"])
    assert not torch.equal(before["hidden"], after["hidden"])
    assert before["counts"]["updates"] == after["counts"]["updates"]
    stream.engine.set_training(True)
    for i in range(40, 50):
        stream.step(i)
    assert stream.engine.counts["updates"] > before["counts"]["updates"]


def test_checkpoint_exact_continuation_and_no_raw_history(tmp_path):
    stream = Stream()
    for i in range(30):
        stream.step(i)
    path = tmp_path / "state.pt"
    stream.engine.save(path)
    restored = Engine.load(path, clock=lambda: stream.time + .001)
    assert restored.pending == stream.engine.pending == {}
    assert "input" not in stream.engine.state() and "pending" not in stream.engine.state()
    assert path.stat().st_size < 1024**2
    for i in range(30, 50):
        original = stream.step(i)
        continued = restored.step(Frame(i, stream.time, synthetic(i, "moving")))
        np.testing.assert_array_equal(original["prediction"], continued["prediction"])
    assert equal_state(stream.engine.state(), restored.state())


@pytest.mark.parametrize("field", ["hidden", "weights", "coordinates", "optimizer", "counts", "schema"])
def test_checkpoint_rejects_corruption(tmp_path, field):
    stream = Stream()
    for i in range(8):
        stream.step(i)
    state = stream.engine.state()
    if field == "hidden":
        state["hidden"][0, 0, 0, 0] = float("nan")
    elif field == "weights":
        state["model"]["encoder.weight"] = torch.zeros(1)
    elif field == "coordinates":
        state["model"]["positions"].fill_(0)
    elif field == "optimizer":
        state["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif field == "counts":
        state["counts"]["updates"] = -1
    else:
        state["schema"] = 99
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        Engine.load(path)


def test_failed_save_preserves_previous_checkpoint_and_pending(tmp_path, monkeypatch):
    stream = Stream()
    stream.step(0)
    path = tmp_path / "state.pt"
    path.write_bytes(b"previous")
    import cognio.engine as module
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError):
        stream.engine.save(path)
    assert path.read_bytes() == b"previous" and len(stream.engine.pending) == 1
    assert list(tmp_path.iterdir()) == [path]


def test_bad_update_rolls_back_and_recoverable_checkpoint(tmp_path, monkeypatch):
    stream = Stream()
    for i in range(5):
        stream.step(i)
    before = stream.engine.state()
    def broken_step():
        with torch.no_grad():
            next(stream.engine.model.parameters()).fill_(float("nan"))
    monkeypatch.setattr(stream.engine.optimizer, "step", broken_step)
    assert stream.step(5) is None
    assert stream.engine.fault and not stream.engine.training
    assert equal_state(before["model"], stream.engine.state()["model"])
    assert equal_state(before["optimizer"], stream.engine.state()["optimizer"])
    stream.engine.save(tmp_path / "recovery.pt")
    restored = Engine.load(tmp_path / "recovery.pt")
    assert not restored.fault


def test_invalid_input_and_config():
    stream = Stream()
    stream.step(0)
    with pytest.raises(ValueError):
        stream.step(0)
    with pytest.raises(ValueError):
        stream.engine.step(Frame(1, 1., np.full((128, 128), np.nan, dtype=np.float32)))
    for kwargs in ({"seed": -1}, {"learning_rate": float("nan")}, {"horizon": 4}):
        with pytest.raises(ValueError):
            Config(**kwargs)


def test_bounded_graph_free_buffers_and_controlled_learning():
    stream = Stream()
    stream.engine.set_training(False)
    for i in range(80):
        stream.step(i)
    initial = np.mean(stream.engine.history, axis=0)
    stream.engine.set_training(True)
    for i in range(80, 680):
        stream.step(i)
        assert len(stream.engine.pending) <= 5
        assert all(p.prediction.grad_fn is None and p.previous.grad_fn is None for p in stream.engine.pending.values())
    stream.engine.set_training(False)
    for i in range(680, 760):
        stream.step(i)
    final = np.mean(stream.engine.history, axis=0)
    assert final[0] < initial[0] * .5
    assert final[0] < final[1]
    assert len(stream.engine.history) <= 300 and not stream.engine.fault


def test_static_temporal_shape_zero_and_absent_controls():
    assert synthetic(0, "no-input") is None
    for mode in ("static", "temporal", "shape", "zero"):
        stream = Stream()
        for i in range(12):
            stream.step(i, mode)
        assert stream.engine.counts["updates"] == 7 and not stream.engine.fault
    assert not synthetic(1, "zero").any()
    np.testing.assert_array_equal(synthetic(0, "static"), synthetic(50, "static"))

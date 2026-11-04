import copy
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PySide6.QtWidgets import QApplication

from cognio.engine import Engine
from cognio.gui import Window, activation_tiles, signed_rgb, measurement_rgb, gray_rgb, ycbcr_rgb
from cognio.inputs import Frame, synthetic
from cognio.model import Config
from cognio.runtime import Runtime, select_device


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("application condition timed out")


def test_hidden_tiles_preserve_actual_channel_positions():
    hidden = np.arange(32 * 8 * 8).reshape(32, 8, 8)
    tiles = activation_tiles(hidden)
    for channel in range(32):
        row, col = divmod(channel, 8)
        np.testing.assert_array_equal(tiles[row*8:(row+1)*8, col*8:(col+1)*8], hidden[channel])


def test_gui_views_display_independence_and_runtime_controls(app, tmp_path):
    runtime = Runtime(Engine(), "moving", device_notice="CPU test")
    window = Window(runtime)
    window.show()
    runtime.start()
    try:
        wait_for(app, lambda: runtime.latest() is not None and runtime.latest()["counts"]["updates"] >= 8)
        window.refresh()
        snap = runtime.latest()
        # Pause only the presentation timer; sensory and learning workers still run.
        window.timer.stop()
        window.training.setChecked(False)
        wait_for(app, lambda: not runtime.latest()["training"])
        before = copy.deepcopy(runtime.engine.model.state_dict())
        hidden = runtime.engine.hidden.clone()
        count = runtime.latest()["counts"]["predictions"]
        window.display_range.setValue(.1)
        wait_for(app, lambda: runtime.latest()["counts"]["predictions"] >= count + 8)
        after = runtime.engine.model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert not torch.equal(hidden, runtime.engine.hidden)
        # Freeze the snapshot source to verify exact four-view mapping, independent of workers.
        with patch.object(runtime, "latest", return_value=snap):
            window.refresh()
            assert window.component.count() == 1 and window.component.currentText() == "Y"
            np.testing.assert_array_equal(window.input_view.rgb, gray_rgb(snap["visual"]["input"]))
            np.testing.assert_array_equal(window.prediction_view.rgb, gray_rgb(snap["visual"]["prediction"]))
            np.testing.assert_array_equal(window.error_view.rgb, gray_rgb(snap["display_result"]["error"], .25))
            np.testing.assert_array_equal(window.hidden_view.rgb, signed_rgb(activation_tiles(snap["visual"]["hidden"]), .1))
            assert "Predicted future" in window.prediction_label.text()
            assert "observed" in window.error_label.text()
            assert "measured luminance" in window.input_label.text()
            assert not np.array_equal(window.input_view.rgb, window.prediction_view.rgb)
            assert not np.array_equal(window.prediction_view.rgb, window.error_view.rgb)
        path = tmp_path / "gui.pt"
        revision = runtime.revision
        runtime.submit("save", str(path))
        wait_for(app, lambda: runtime.revision > revision)
        assert path.exists()
        runtime.submit("load", str(path))
        wait_for(app, lambda: runtime.engine.origin.startswith("RESTORED"))
        assert not runtime.engine.training
        original = runtime.engine
        runtime.submit("load", str(tmp_path / "missing.pt"))
        wait_for(app, lambda: "load failed" in runtime.notice)
        assert runtime.engine is original
        runtime.submit("reset")
        wait_for(app, lambda: runtime.engine is not original)
        assert not runtime.engine.training
        assert runtime.engine.config.seed != original.config.seed
    finally:
        window.close()
    assert not any(t.is_alive() for t in runtime.threads)


@pytest.mark.parametrize("source", ["no-input", "camera"])
def test_no_input_or_missing_camera_keeps_gui_and_controls_alive(app, source):
    runtime = Runtime(Engine(), source, "/dev/cognio-does-not-exist")
    window = Window(runtime)
    window.show()
    runtime.start()
    try:
        wait_for(app, lambda: runtime.latest() is not None)
        if source == "camera":
            wait_for(app, lambda: bool(runtime.latest()["capture_error"]))
        window.refresh()
        assert runtime.latest()["counts"]["updates"] == 0
        assert runtime.latest()["visual"] is None
        runtime.submit("training", False)
        wait_for(app, lambda: not runtime.latest()["training"])
    finally:
        window.close()


def test_cuda_fallback_is_explicit():
    with patch.object(torch.cuda, "is_available", return_value=False):
        device, notice = select_device("auto")
        assert device == "cpu" and "fallback" in notice
        with pytest.raises(ValueError, match="--compute cpu"):
            select_device("cuda")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def test_color_diagnostics_are_component_views_and_display_only(app, device):
    runtime = Runtime(Engine(Config(input_mode="ycbcr"), device=device), "color-moving")
    window = Window(runtime)
    runtime.start()
    try:
        wait_for(app, lambda: runtime.latest() is not None and runtime.latest()["counts"]["updates"] >= 5)
        runtime.submit("training", False)
        wait_for(app, lambda: not runtime.latest()["training"] and runtime.latest()["display_result"] is not None)
        before = copy.deepcopy(runtime.engine.model.state_dict())
        snap = runtime.latest()
        # Continue synchronously below, so identical delivered frames and clocks
        # can prove selector independence during learning as well as freeze.
        runtime.stop()
        with patch.object(runtime, "latest", return_value=snap):
            assert window.component.currentText() == "Combined Color (RGB preview)"
            window.refresh()
            np.testing.assert_array_equal(window.input_view.rgb, ycbcr_rgb(snap["visual"]["input"]))
            np.testing.assert_array_equal(window.prediction_view.rgb, ycbcr_rgb(snap["visual"]["prediction"]))
            np.testing.assert_array_equal(window.error_view.rgb, gray_rgb(snap["display_result"]["error"].mean(axis=0), .25))
            assert "Mean Y/Cb/Cr squared error (not RGB)" in window.error_label.text()
            assert "BT.601 limited-range assumption" in window.input_label.text()
            for component, name in enumerate(("Y","Cb","Cr")):
                window.component.setCurrentText(name)
                window.display_range.setValue(.25)
                window.refresh()
                np.testing.assert_array_equal(window.input_view.rgb, measurement_rgb(snap["visual"]["input"], component))
                np.testing.assert_array_equal(window.prediction_view.rgb, measurement_rgb(snap["visual"]["prediction"], component))
                np.testing.assert_array_equal(window.error_view.rgb, gray_rgb(snap["display_result"]["error"][component], .25))
                assert name in window.error_label.text()
                assert "Component MSE / persistence" in window.telemetry.text()
        assert all(torch.equal(value, runtime.engine.model.state_dict()[key]) for key,value in before.items())
        now = [0.]
        a, b = (Engine(Config(input_mode="ycbcr"), device=device, clock=lambda: now[0]+.001) for _ in range(2))
        runtime.engine = b
        with patch.object(runtime, "submit") as submit:
            for i in range(32):
                now[0] = i/30
                if i == 20:
                    a.set_training(False)
                    b.set_training(False)
                frame = Frame(i, now[0], synthetic(i, "color-moving", "ycbcr"))
                reference, visual = a.step(frame), b.step(frame)
                shown = dict(snap, visual=visual, result=b.last_result, display_result=b.last_result,
                             training=b.training, counts=b.counts.copy(), history=list(b.history),
                             component_history=list(b.component_history), config=b.config)
                with patch.object(runtime, "latest", return_value=shown):
                    window.component.setCurrentIndex(i % 4)
                    window.refresh()
                np.testing.assert_array_equal(reference["prediction"], visual["prediction"])
                assert list(a.history) == list(b.history)
                assert list(a.component_history) == list(b.component_history)
                if a.last_result:
                    np.testing.assert_array_equal(a.last_result["error"], b.last_result["error"])
            submit.assert_not_called()
        assert a.counts == b.counts and not a.fault and not b.fault
        assert torch.equal(a.hidden, b.hidden)
        assert all(torch.equal(value, b.model.state_dict()[key]) for key,value in a.model.state_dict().items())
        for key, state in a.optimizer.state_dict()["state"].items():
            assert all(torch.equal(value, b.optimizer.state_dict()["state"][key][name]) for name,value in state.items())
    finally:
        window.close()


def test_fault_keeps_new_input_labels_separate_from_last_prediction(app, monkeypatch):
    engine = Engine()
    forward = engine.model.forward
    def fail_after_initial_frames(*args):
        if engine.last_sequence >= 6:
            raise RuntimeError("injected numerical backend failure")
        return forward(*args)
    monkeypatch.setattr(engine.model, "forward", fail_after_initial_frames)
    runtime = Runtime(engine, "moving")
    window = Window(runtime)
    runtime.start()
    try:
        wait_for(app, lambda: runtime.latest() is not None and runtime.latest()["fault"] and
                 runtime.latest()["visual"]["input_sequence"] > runtime.latest()["visual"]["source"])
        snap = runtime.latest()
        with patch.object(runtime, "latest", return_value=snap):
            window.refresh()
        visual = snap["visual"]
        assert f"frame {visual['input_sequence']}" in window.input_label.text()
        assert f"{visual['source']} → {visual['target']}" in window.prediction_label.text()
        assert engine.counts["cleared_targets"] > 0 and not engine.pending
    finally:
        window.close()


@pytest.mark.parametrize("timestamp_kind", ["driver-monotonic", "receipt-time"])
def test_camera_disconnect_wrap_and_timestamp_guard(app, timestamp_kind):
    class FakeCamera:
        info, mode, controls, warnings = {}, {}, {}, []
        def __init__(self, path):
            self.timestamp_kind = timestamp_kind
            self.index = 0
            self.closed = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.closed = True
        def read(self):
            if self.index == 3:
                raise OSError("camera disconnected")
            sequence = (2**32 - 2 + self.index) & 0xFFFFFFFF
            self.index += 1
            time.sleep(.04)
            return np.full((480, 640), 100, np.uint8), time.monotonic(), sequence
        def stats(self):
            return dict(frames=self.index, driver_drops=0, drained_discards=0,
                        bad_frames=0, timestamp_kind=self.timestamp_kind)
    camera = FakeCamera(None)
    runtime = Runtime(Engine(), "camera")
    with patch("cognio.runtime.V4L2Camera", return_value=camera):
        runtime.start()
        try:
            wait_for(app, lambda: bool(runtime.capture_error))
            assert camera.closed
            if timestamp_kind == "driver-monotonic":
                wait_for(app, lambda: runtime.engine.last_sequence == 2**32)
                assert "disconnected" in runtime.capture_error
                assert not runtime.engine.fault
            else:
                assert "timestamps" in runtime.capture_error
                assert runtime.capture_count == 0
            runtime.submit("training", False)
            wait_for(app, lambda: not runtime.engine.training)
        finally:
            runtime.stop()

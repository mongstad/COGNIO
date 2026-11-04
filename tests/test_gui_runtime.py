import copy
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PySide6.QtWidgets import QApplication

from cognio.engine import Engine
from cognio.gui import Window, activation_tiles, signed_rgb
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

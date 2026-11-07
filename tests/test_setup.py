"""GUI lifecycle tests use actual workers and generated input; never a camera."""
import threading
import time
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from cognio.__main__ import make_parser
from cognio.engine import Engine
from cognio.model import Config
from cognio.setup import SessionWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, predicate, timeout=10):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("GUI lifecycle timed out")


def small_args(*extra):
    return make_parser().parse_args(["--gui", "--source", "moving", "--compute", "cpu", "--channels", "32",
                                     "--grid", "8", "--input-mode", "y", *extra])


def close(app, window):
    window.close()
    wait(app, lambda: window.state == "configure")
    app.processEvents()


def test_default_configuration_no_model_or_capture_and_live_scaling(app):
    with patch("cognio.setup.devices", return_value=[]), patch("cognio.setup.torch.cuda.is_available", return_value=False), \
            patch("cognio.engine.Predictor") as model, patch("cognio.runtime.Runtime.start") as start, \
            patch("cognio.session.select_device") as probe:
        window = SessionWindow(make_parser().parse_args(["--gui"]))
        window.show()
        app.processEvents()
        assert window.state == "configure" and window.runtime is None
        assert window.channels.text() == "1024" and window.grid.currentText() == "32"
        assert window.topology.currentText() == "toroidal" and window.input_mode.currentText() == "ycbcr"
        assert window.source.currentText() == "camera" and window.compute.currentText() == "cpu"
        assert not window.start_button.isEnabled() and "No unique C922" in window.capacity.text()
        window.source.setCurrentText("moving")
        for channels, grid in ((32, 8), (512, 16), (1024, 32), (1536, 32), (48, 64)):
            window.channels.setText(str(channels))
            window.grid.setCurrentText(str(grid))
            assert f"{Config(channels=channels, grid=grid, input_mode='ycbcr').parameters:,}" in window.capacity.text()
        window.channels.setText("100000000000000000000")
        assert not window.start_button.isEnabled() and "budget" in window.capacity.text()
        window.channels.setText("32")
        window.budget.setText("1")
        assert not window.start_button.isEnabled()
        window.budget.clear()
        window.channels.setText("nonsense")
        assert not window.start_button.isEnabled()
        model.assert_not_called()
        start.assert_not_called()
        probe.assert_not_called()
        window.close()


def test_start_freeze_cancel_new_session_and_stop_before_reconfigure(app, tmp_path):
    window = SessionWindow(small_args("--topology", "toroidal"))
    window.show()
    assert window.runtime is None
    window.start_button.click()
    assert window.state == "starting" and not window.form_widget.isEnabled()
    try:
        wait(app, lambda: window.state == "running" and window.runtime.engine.counts["updates"] >= 3)
        old = window.runtime
        assert old.engine.config.topology == "toroidal"
        assert not window.form_widget.isEnabled()
        window.running.training.setChecked(False)
        wait(app, lambda: old.latest() is not None and not old.latest()["training"])
        count = old.engine.counts["updates"]
        predictions = old.engine.counts["predictions"]
        wait(app, lambda: old.engine.counts["predictions"] >= predictions+3)
        assert old.engine.counts["updates"] == count
        path = tmp_path/"saved.pt"
        old.submit("save", path)
        wait(app, path.exists)
        saved = path.read_bytes()
        with patch("cognio.setup.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
            window.new_session()
        assert window.state == "running" and not old.stop_event.is_set()
        with patch("cognio.setup.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            window.new_session()
        assert window.state == "stopping"
        window.start_session()  # No overlapping run during cleanup.
        wait(app, lambda: window.state == "configure")
        assert window.runtime is None and not any(t.is_alive() for t in old.threads)
        assert path.read_bytes() == saved
        window.topology.setCurrentText("standard")
        window.start_session()
        wait(app, lambda: window.state == "running")
        assert window.runtime is not old and window.runtime.engine.config.topology == "standard"
    finally:
        close(app, window)


def test_checkpoint_inspection_locks_architecture_and_restore_starts_frozen(app, tmp_path):
    path = tmp_path/"torus.pt"
    config = Config(channels=24, grid=64, input_mode="ycbcr", topology="toroidal")
    Engine(config).save(path)
    saved = path.read_bytes()
    window = SessionWindow(small_args())
    with patch("cognio.engine.Predictor") as constructor:
        window.inspect_checkpoint(str(path))
        constructor.assert_not_called()
    assert window.selected_config() == config
    assert all(not widget.isEnabled() for widget in window.architecture_widgets)
    assert window.frozen.isChecked() and "Saved session" in window.origin.text()
    window.start_session()
    try:
        wait(app, lambda: window.state == "running" and window.runtime.engine.counts["predictions"] >= 2)
        assert window.runtime.engine.config == config and not window.runtime.engine.training
        assert window.runtime.engine.counts["updates"] == 0
        assert window.runtime.engine.origin.startswith("RESTORED")
    finally:
        close(app, window)
    assert path.read_bytes() == saved


def test_failed_restore_is_not_silently_fresh_and_changed_metadata_refused(app, tmp_path):
    missing = tmp_path/"missing.pt"
    window = SessionWindow(small_args("--checkpoint", str(missing)))
    assert not window.start_button.isEnabled()
    window.use_fresh()
    assert window.start_button.isEnabled()
    path = tmp_path/"changed.pt"
    Engine().save(path)
    window.inspect_checkpoint(path)
    Engine(Config(topology="toroidal")).save(path)
    window.start_session()
    wait(app, lambda: window.state == "configure")
    assert window.runtime is None and "differs" in window.status.text()
    window.close()


def test_startup_failure_recovers_and_close_during_start_never_acquires(app):
    window = SessionWindow(small_args())
    with patch("cognio.setup.build_runtime", side_effect=RuntimeError("allocation refused")):
        window.start_session()
        wait(app, lambda: window.state == "configure")
    assert "allocation refused" in window.status.text() and window.form_widget.isEnabled()
    from cognio.session import build_runtime
    gate = threading.Event()
    def delayed(args, **kwargs):
        assert gate.wait(5)
        return build_runtime(args, **kwargs)
    with patch("cognio.setup.build_runtime", side_effect=delayed), patch("cognio.runtime.Runtime.start") as start:
        window.start_session()
        window.close()
        assert window.closing
        gate.set()
        wait(app, lambda: window.state == "configure")
        start.assert_not_called()
    assert window.runtime is None


def test_stop_failure_prevents_second_session_and_allows_retry(app):
    window = SessionWindow(small_args())
    window.start_session()
    wait(app, lambda: window.state == "running")
    old = window.runtime
    try:
        with patch.object(old, "stop", side_effect=RuntimeError("worker still alive")):
            window.stop_session()
            wait(app, lambda: window.state == "running")
        window.start_session()
        assert window.runtime is old and "Stop failed" in window.running.status.text()
    finally:
        close(app, window)


def test_video_chooser_cancel_start_playback_and_ffmpeg_cleanup(app, tmp_path):
    import shutil
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg unavailable")
    from tools.make_video_fixture import make_video
    path = tmp_path/"test.mkv"
    make_video(path, 30)
    window = SessionWindow(small_args())
    window.source.setCurrentText("video")
    window.video_loop.setChecked(True)
    with patch("cognio.setup.QFileDialog.getOpenFileName", return_value=("", "")):
        window.start_session()
    assert window.state == "configure" and window.runtime is None
    with patch("cognio.setup.QFileDialog.getOpenFileName", return_value=(str(path), "")):
        window.start_session()
    try:
        wait(app, lambda: window.state == "running" and window.runtime.engine.counts["updates"] >= 2)
        runtime = window.runtime
        reader = runtime.video.reader
        assert reader is not None and reader.process.poll() is None
        window.running.video_pause.setChecked(True)
        wait(app, lambda: runtime.video.paused)
        window.running.video_speed.setValue(.5)
        wait(app, lambda: runtime.video.speed == .5)
        with patch("cognio.setup.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
            window.new_session()
        wait(app, lambda: window.state == "configure")
        assert reader.process.poll() is not None and not any(t.is_alive() for t in runtime.threads)
    finally:
        close(app, window)


def test_camera_not_opened_until_start_and_context_closed_before_new_session(app):
    from cognio.runtime import Runtime
    opened, released = threading.Event(), threading.Event()
    class Camera:
        info = mode = controls = {}
        warnings = []
        def __init__(self, *args, **kwargs):
            opened.set()
        def __enter__(self):
            return self
        def read(self):
            time.sleep(.01)
            return None
        def __exit__(self, *args):
            released.set()
    window = SessionWindow(small_args())
    window.source.setCurrentText("camera")
    window.camera.setCurrentText("/dev/fake-c922")
    config = window.selected_config()
    def prepare(args, **kwargs):
        return Runtime(Engine(config), "camera", "/dev/fake-c922")
    with patch("cognio.runtime.V4L2Camera", Camera), patch("cognio.setup.build_runtime", side_effect=prepare):
        assert not opened.is_set()
        window.start_session()
        wait(app, lambda: window.state == "running" and opened.is_set())
        window.stop_session()
        wait(app, lambda: window.state == "configure")
        assert released.is_set()
    window.close()

"""Visible or offscreen Qt trial; saves scalar telemetry only unless synthetic screenshot requested."""

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from cognio.engine import Engine
from cognio.model import Config
from cognio.inputs import INPUT_MODES
from cognio.gui import Window, measurement_rgb, gray_rgb
from cognio.runtime import Runtime, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("camera", "video", "moving", "static", "color-moving", "color-static"), default="moving")
    parser.add_argument("--video-file", type=Path)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--compute", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--grid", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="y")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, help="synthetic window only")
    parser.add_argument("--check-hidden-reset", action="store_true", help="exercise confirmed hidden-only reset during frozen phase")
    args = parser.parse_args()
    if args.seconds < 25 or (args.screenshot and args.source in ("camera","video")):
        parser.error("trial must last >=25 seconds; camera/video screenshots are disabled")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device, notice = select_device(args.compute)
    app = QApplication([])
    runtime = Runtime(Engine(Config(channels=args.channels, grid=args.grid, input_mode=args.input_mode), device=device), args.source,
                      device_notice=notice, video_path=args.video_file, video_loop=args.source == "video")
    window = Window(runtime)
    window.show()
    runtime.start()
    start = time.monotonic()
    samples, failures, actions, ticks = [], [], [], []
    frozen = {}
    directory = tempfile.TemporaryDirectory(prefix="cognio-gui-")
    checkpoint = Path(directory.name) / "validation.pt"
    report = dict(source=args.source, input_mode=args.input_mode, channels=args.channels, grid=args.grid, compute=notice, torch=str(torch.__version__),
                  qt_platform=app.platformName(), requested_seconds=args.seconds,
                  initial_diagnostic=window.component.currentText())

    def sample():
        now = time.monotonic()
        elapsed = now - start
        ticks.append(now)
        snap = runtime.latest()
        try:
            if snap:
                if snap["fault"] or snap["capture_error"]:
                    raise AssertionError(snap["fault"] or snap["capture_error"])
                if args.source == "video":
                    if elapsed >= 2 and "video_pause" not in actions:
                        window.video_pause.setChecked(True)
                        actions.append("video_pause")
                    if elapsed >= 2.5 and "video_pause_snapshot" not in actions:
                        assert snap["video"]["paused"]
                        frozen["paused_counts"] = snap["counts"].copy()
                        actions.append("video_pause_snapshot")
                    if elapsed >= 3.5 and "video_resume_slow" not in actions:
                        assert snap["counts"] == frozen["paused_counts"]
                        window.video_speed.setValue(.1)
                        window.video_pause.setChecked(False)
                        actions.append("video_resume_slow")
                    if elapsed >= 5.5 and "video_speed_restore" not in actions:
                        assert snap["video"]["speed"] == .1 and not snap["video"]["paused"]
                        window.video_speed.setValue(1)
                        actions.append("video_speed_restore")
                if args.input_mode == "ycbcr":
                    for at, name in ((6, "Y"), (7, "Cb"), (8, "Cr"), (9, "Combined Color (RGB preview)"),
                                     (12, "Y"), (12.5, "Cb"), (13, "Cr"), (13.5, "Combined Color (RGB preview)")):
                        action = f"view_{at:g}_{name}"
                        if elapsed >= at and action not in actions:
                            # Use one observed snapshot to check all three rendered panels.
                            with patch.object(runtime, "latest", return_value=snap):
                                window.component.setCurrentText(name)
                                window.refresh()
                            component = window.component.currentData()
                            if snap["visual"]:
                                np.testing.assert_array_equal(window.input_view.rgb, measurement_rgb(snap["visual"]["input"], component))
                                np.testing.assert_array_equal(window.prediction_view.rgb, measurement_rgb(snap["visual"]["prediction"], component))
                            else:
                                assert window.input_view.rgb is None and window.prediction_view.rgb is None
                            if snap["display_result"]:
                                error = snap["display_result"]["error"]
                                expected = error.mean(axis=0) if component is None else error[component]
                                np.testing.assert_array_equal(window.error_view.rgb, gray_rgb(expected, .25))
                            else:
                                assert window.error_view.rgb is None
                            actions.append(action)
                if elapsed >= 10 and "freeze" not in actions:
                    window.training.setChecked(False)
                    actions.append("freeze")
                if elapsed >= 11 and "freeze_snapshot" not in actions:
                    assert not snap["training"]
                    frozen["model"] = copy.deepcopy(runtime.engine.model.state_dict())
                    frozen["optimizer"] = copy.deepcopy(runtime.engine.optimizer.state_dict())
                    frozen["hidden"] = runtime.engine.hidden.clone()
                    frozen["predictions"] = snap["counts"]["predictions"]
                    frozen["updates"] = snap["counts"]["updates"]
                    actions.append("freeze_snapshot")
                if args.check_hidden_reset and elapsed >= 11.5 and "reset_hidden" not in actions:
                    engine = runtime.engine
                    original_reset = engine.reset_hidden
                    def checked_reset():
                        pending = len(engine.pending)
                        clocks = (engine.last_sequence,engine.last_timestamp,engine.last_media_time,engine.last_segment)
                        original_reset()
                        engine.reset_hidden = original_reset
                        report["hidden_reset"] = dict(zero_hidden=bool(torch.count_nonzero(engine.hidden) == 0),
                            pending_before=pending,pending_after=len(engine.pending),
                            clocks_preserved=clocks == (engine.last_sequence,engine.last_timestamp,engine.last_media_time,engine.last_segment),
                            training=engine.training)
                    engine.reset_hidden = checked_reset
                    with patch.object(QMessageBox,"question",return_value=QMessageBox.StandardButton.Yes):
                        window.reset_hidden()
                    actions.append("reset_hidden")
                if elapsed >= 14 and "verify_freeze_save" not in actions:
                    assert all(torch.equal(value, runtime.engine.model.state_dict()[key]) for key, value in frozen["model"].items())
                    for key, state in frozen["optimizer"]["state"].items():
                        assert all(torch.equal(value, runtime.engine.optimizer.state_dict()["state"][key][name])
                                   for name, value in state.items())
                    assert snap["counts"]["updates"] == frozen["updates"]
                    assert snap["counts"]["predictions"] > frozen["predictions"]
                    assert not torch.equal(frozen["hidden"], runtime.engine.hidden)
                    if args.check_hidden_reset:
                        reset = report["hidden_reset"]
                        assert reset["zero_hidden"] and reset["pending_after"] == 0 and reset["clocks_preserved"] and not reset["training"]
                    window.display_range.setValue(.35)
                    if args.channels > 32:
                        window.channel_page.setValue(2)
                    runtime.submit("save", str(checkpoint))
                    actions.append("verify_freeze_save")
                if elapsed >= 16 and "load" not in actions:
                    assert checkpoint.is_file()
                    runtime.submit("load", str(checkpoint))
                    actions.append("load")
                if elapsed >= 18 and "resume" not in actions:
                    assert snap["origin"].startswith("RESTORED") and not snap["training"]
                    assert all(torch.equal(value, runtime.engine.model.state_dict()[key]) for key, value in frozen["model"].items())
                    window.training.setChecked(True)
                    actions.append("resume")
                if elapsed >= 20 and args.screenshot and "screenshot" not in actions:
                    window.grab().save(str(args.screenshot))
                    actions.append("screenshot")
                record = {key: snap[key] for key in ("elapsed_s", "input_fps", "prediction_fps", "updates_per_s",
                          "prediction_ms", "update_ms", "rss_mib", "cpu_percent", "vram_mib", "training",
                          "peak_vram_mib", "observer_ms", "transfer_ms", "preprocess_ms", "step_ms")}
                record["display_ms"] = window.display_ms
                record["diagnostic"] = window.component.currentText()
                record["input_statistics"] = snap["input_statistics"]
                if snap.get("video"):
                    record["video"] = snap["video"]
                if snap["result"]:
                    record.update({key: snap["result"][key] for key in ("mse", "baseline_mse", "horizon_ms", "lead_ms")})
                    record.update({key: snap["result"][key] for key in ("component_mse", "component_baseline_mse")})
                if snap["visual"]:
                    record["input_age_ms"] = (now - snap["visual"]["input_time"]) * 1000
                samples.append(record)
            if elapsed >= args.seconds:
                assert snap and snap["counts"]["updates"] > frozen["updates"]
                report.update(counts=snap["counts"], capture=snap["capture"], overwritten=snap["overwritten"],
                              capture_info=snap["capture_info"], peak_rss_mib=snap["peak_rss_mib"],
                              gui_refreshes=window.refresh_count, window_exposed=window.windowHandle().isExposed(),
                              pending=len(runtime.engine.pending), history=len(runtime.engine.history))
                if args.source == "video":
                    assert snap["video"]["segment"] > 0
                    report["video"] = snap["video"]
                timer.stop()
                window.close()
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            timer.stop()
            window.close()

    timer = QTimer()
    timer.timeout.connect(sample)
    timer.start(250)
    app.exec()
    runtime.stop()
    report["actions"] = actions
    report["failures"] = failures
    report["workers_stopped"] = all(not thread.is_alive() for thread in runtime.threads)
    report["duration_s"] = time.monotonic()-start
    report["tick_max_ms"] = float(np.max(np.diff(ticks))*1000) if len(ticks) > 1 else None
    report["samples"] = samples
    # Verify every camera control we changed has been restored on normal shutdown.
    if args.source == "camera" and runtime.capture_info.get("identity"):
        from cognio import camera as cam
        fd = cam.os.open(runtime.capture_info["identity"]["path"], cam.os.O_RDWR | cam.os.O_NONBLOCK)
        try:
            report["controls_restored"] = {name: cam.ioctl(fd, cam.G_CTRL, cam.Control(cam.CONTROL_IDS[name], 0)).value == values["original"]
                                           for name, values in runtime.capture_info["controls"].items()}
            if not all(report["controls_restored"].values()):
                failures.append("camera controls were not all restored")
        finally:
            cam.os.close(fd)
    directory.cleanup()
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

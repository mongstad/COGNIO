"""Visible or offscreen Qt trial; saves scalar telemetry only unless synthetic screenshot requested."""

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from cognio.engine import Engine
from cognio.model import Config
from cognio.gui import Window
from cognio.runtime import Runtime, select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("camera", "moving", "static"), default="moving")
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--compute", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--grid", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, help="synthetic window only")
    args = parser.parse_args()
    if args.seconds < 25 or (args.screenshot and args.source == "camera"):
        parser.error("trial must last >=25 seconds; camera screenshots are disabled")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device, notice = select_device(args.compute)
    app = QApplication([])
    runtime = Runtime(Engine(Config(channels=args.channels, grid=args.grid), device=device), args.source, device_notice=notice)
    window = Window(runtime)
    window.show()
    runtime.start()
    start = time.monotonic()
    samples, failures, actions, ticks = [], [], [], []
    frozen = {}
    directory = tempfile.TemporaryDirectory(prefix="cognio-gui-")
    checkpoint = Path(directory.name) / "validation.pt"
    report = dict(source=args.source, channels=args.channels, grid=args.grid, compute=notice, torch=str(torch.__version__),
                  qt_platform=app.platformName(), requested_seconds=args.seconds)

    def sample():
        now = time.monotonic()
        elapsed = now - start
        ticks.append(now)
        snap = runtime.latest()
        try:
            if snap:
                if snap["fault"] or snap["capture_error"]:
                    raise AssertionError(snap["fault"] or snap["capture_error"])
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
                if elapsed >= 14 and "verify_freeze_save" not in actions:
                    assert all(torch.equal(value, runtime.engine.model.state_dict()[key]) for key, value in frozen["model"].items())
                    assert snap["counts"]["updates"] == frozen["updates"]
                    assert snap["counts"]["predictions"] > frozen["predictions"]
                    assert not torch.equal(frozen["hidden"], runtime.engine.hidden)
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
                if snap["result"]:
                    record.update({key: snap["result"][key] for key in ("mse", "baseline_mse", "horizon_ms", "lead_ms")})
                if snap["visual"]:
                    record["input_age_ms"] = (now - snap["visual"]["input_time"]) * 1000
                samples.append(record)
            if elapsed >= args.seconds:
                assert snap and snap["counts"]["updates"] > frozen["updates"]
                report.update(counts=snap["counts"], capture=snap["capture"], overwritten=snap["overwritten"],
                              capture_info=snap["capture_info"], peak_rss_mib=snap["peak_rss_mib"],
                              gui_refreshes=window.refresh_count, window_exposed=window.windowHandle().isExposed(),
                              pending=len(runtime.engine.pending), history=len(runtime.engine.history))
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

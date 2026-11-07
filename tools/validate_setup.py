"""Bounded visible setup/session trial; scalar-only evidence and synthetic-only screenshots."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from cognio.__main__ import make_parser
from cognio.diagnostics import state_digest
from cognio.setup import SessionWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--grid", type=int, choices=(8, 16, 32, 64), default=8)
    parser.add_argument("--topology", choices=("standard", "toroidal"), default="toroidal")
    parser.add_argument("--source", choices=("moving", "camera", "video"), default="moving")
    parser.add_argument("--video-file")
    parser.add_argument("--seconds", type=float, default=35)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, help="synthetic running window only")
    args = parser.parse_args()
    if not 30 <= args.seconds <= 180 or (args.screenshot and args.source != "moving"):
        parser.error("use 30..180 seconds; screenshots are synthetic only")
    if args.report.exists() or (args.screenshot and args.screenshot.exists()):
        parser.error("report/screenshot must be new paths")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    options = make_parser().parse_args(["--gui", "--source", args.source, "--compute", args.compute,
        "--channels", str(args.channels), "--grid", str(args.grid), "--topology", args.topology,
        "--input-mode", "ycbcr"])
    options.video_file, options.video_loop = args.video_file, args.source == "video"
    app = QApplication.instance() or QApplication([])
    window = SessionWindow(options)
    window.show()
    report = dict(config=asdict(window.selected_config()), source=args.source, compute=args.compute,
                  torch=str(torch.__version__), qt_platform=app.platformName(), actions=[], failures=[], samples=[],
                  no_runtime_before_start=window.runtime is None, initial_capacity=window.capacity.text())
    directory = tempfile.TemporaryDirectory(prefix="cognio-setup-")
    checkpoint = Path(directory.name)/"trial.pt"
    clock_start = time.monotonic()
    started = None
    ticks = []
    frozen = None
    original = None
    phase = "initial"
    captured_path = None

    def sample():
        nonlocal started, frozen, original, phase, captured_path
        now = time.monotonic()
        ticks.append(now)
        try:
            if phase == "initial" and now-clock_start >= 1:
                assert window.runtime is None and window.state == "configure"
                assert window.start_button.isEnabled(), window.capacity.text()
                window.start_button.click()
                report["actions"].append("explicit_start")
                phase = "starting"
            if phase == "starting" and window.state == "configure":
                raise AssertionError("Startup failed: " + window.status.text())
            if phase == "starting" and window.state == "running":
                original = window.runtime
                started = now
                phase = "running"
                report["compute_notice"] = original.device_notice
            if phase == "running":
                elapsed = now-started
                snap = original.latest()
                if snap:
                    assert not snap["fault"] and not snap["capture_error"], snap["fault"] or snap["capture_error"]
                    report["samples"].append({key: snap[key] for key in (
                        "elapsed_s", "input_fps", "prediction_fps", "updates_per_s", "prediction_ms", "update_ms",
                        "rss_mib", "peak_rss_mib", "vram_mib", "reserved_vram_mib", "training", "counts")})
                actions = report["actions"]
                if elapsed >= 6 and "freeze" not in actions:
                    window.running.training.setChecked(False)
                    actions.append("freeze")
                if elapsed >= 8 and "frozen_snapshot" not in actions:
                    assert not original.engine.training
                    frozen = state_digest((original.engine.model.state_dict(), original.engine.optimizer.state_dict()))
                    actions.append("frozen_snapshot")
                if elapsed >= 10 and "save" not in actions:
                    assert frozen == state_digest((original.engine.model.state_dict(), original.engine.optimizer.state_dict()))
                    assert original.engine.counts["updates"] > 0
                    original.submit("save", str(checkpoint))
                    actions.append("save")
                if elapsed >= 12 and "restore" not in actions:
                    assert checkpoint.exists()
                    original.submit("load", str(checkpoint))
                    actions.append("restore")
                if elapsed >= 14 and "resume" not in actions:
                    assert original.engine.origin.startswith("RESTORED") and not original.engine.training
                    assert frozen == state_digest((original.engine.model.state_dict(), original.engine.optimizer.state_dict()))
                    window.running.training.setChecked(True)
                    actions.append("resume")
                if elapsed >= 16 and "cancel_new_session" not in actions:
                    with patch("cognio.setup.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
                        window.new_session()
                    assert window.state == "running"
                    actions.append("cancel_new_session")
                if elapsed >= 17 and args.screenshot and "screenshot" not in actions:
                    window.grab().save(str(args.screenshot))
                    captured_path = str(args.screenshot)
                    actions.append("screenshot")
                if elapsed >= 18:
                    with patch("cognio.setup.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
                        window.new_session()
                    actions.append("confirmed_new_session")
                    phase = "stopping"
            if phase == "stopping" and window.state == "configure":
                assert not any(thread.is_alive() for thread in original.threads)
                assert window.runtime is None
                report["actions"].append("workers_stopped_before_configuration")
                window.inspect_checkpoint(str(checkpoint))
                assert all(not control.isEnabled() for control in window.architecture_widgets)
                window.start_session()
                phase = "restoring"
            if phase == "restoring" and window.state == "running":
                assert window.runtime.engine.origin.startswith("RESTORED") and not window.runtime.engine.training
                assert frozen == state_digest((window.runtime.engine.model.state_dict(), window.runtime.engine.optimizer.state_dict()))
                report["actions"].append("setup_restore_frozen")
                phase = "complete"
            if now-clock_start >= args.seconds:
                assert phase == "complete", f"trial incomplete: {phase}"
                finish()
        except Exception as exc:
            report["failures"].append(f"{type(exc).__name__}: {exc}")
            finish()

    def finish():
        timer.stop()
        report["wall_seconds"] = time.monotonic()-clock_start
        report["maximum_ui_tick_gap_ms"] = 1000*max((b-a for a,b in zip(ticks,ticks[1:])), default=0)
        report["screenshot"] = captured_path
        window.close()
        # closeEvent waits asynchronously for worker/decoder release.

    timer = QTimer()
    timer.timeout.connect(sample)
    timer.start(200)
    app.exec()
    report["final_state"] = window.state
    report["all_workers_stopped"] = window.runtime is None and (original is None or not any(t.is_alive() for t in original.threads))
    directory.cleanup()
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:report[key] for key in ("failures", "actions", "all_workers_stopped", "maximum_ui_tick_gap_ms")}))
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())

"""No Qt imports, windows, activation projection or growing telemetry history."""

from dataclasses import asdict
import json
import signal
import sys
import threading
import time
from .inputs import components


def scalar_snapshot(snapshot):
    keys = ("source", "origin", "training", "fault", "capture_error", "compute", "counts",
            "capture", "overwritten", "synthetic_skips", "input_fps", "prediction_fps",
            "updates_per_s", "prediction_ms", "update_ms", "step_ms", "transfer_ms", "preprocess_ms",
            "observer_ms", "rss_mib", "peak_rss_mib", "cpu_percent", "vram_mib", "peak_vram_mib",
            "reserved_vram_mib", "input_age_ms", "parameters", "hidden_values", "pending", "elapsed_s",
            "input_statistics")
    result = {key: snapshot[key] for key in keys}
    result["config"] = asdict(snapshot["config"])
    pairs = snapshot["history"]
    result["rolling_mse"] = sum(p[0] for p in pairs)/len(pairs) if pairs else None
    result["rolling_persistence_mse"] = sum(p[1] for p in pairs)/len(pairs) if pairs else None
    history = snapshot["component_history"]
    result["rolling_components"] = {
        name: dict(mse=sum(p[0][i] for p in history)/len(history),
                   persistence_mse=sum(p[1][i] for p in history)/len(history))
        for i, name in enumerate(components(snapshot["config"].input_mode))} if history else None
    match = snapshot["result"]
    result["matched"] = {k: v for k, v in match.items() if k != "error"} if match else None
    return result


def run(runtime, seconds=None, interval=1., save_path=None, output=sys.stdout):
    stop = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
    started = time.monotonic()
    next_report = started
    failed = False
    runtime.start()
    try:
        while not stop.wait(.02):
            now = time.monotonic()
            snapshot = runtime.latest()
            if snapshot is not None:
                failed = bool(snapshot["fault"] or snapshot["capture_error"])
                if now >= next_report or failed:
                    print(json.dumps(scalar_snapshot(snapshot), allow_nan=False), file=output, flush=True)
                    next_report = now + interval
            if failed or (seconds is not None and now-started >= seconds):
                break
    finally:
        try:
            runtime.stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if runtime.latest() is not None:
        final = scalar_snapshot(runtime.latest())
        final.update(event="run_end", counts=runtime.engine.counts.copy(),
                     workers_stopped=all(not thread.is_alive() for thread in runtime.threads))
        print(json.dumps(final, allow_nan=False), file=output, flush=True)
    if save_path:
        runtime.engine.save(save_path)
        print(json.dumps({"event": "checkpoint_saved", "path": str(save_path),
                          "counts": runtime.engine.counts}), file=output, flush=True)
    return 1 if failed else 0

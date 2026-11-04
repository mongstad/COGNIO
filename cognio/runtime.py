"""Independent capture, learning and bounded display handoff. No recording."""

from collections import deque
from dataclasses import replace
import queue
import secrets
import threading
import time

import psutil
import torch

from .camera import V4L2Camera
from .engine import Engine
from .inputs import Frame, normalize, synthetic


def select_device(requested):
    if requested == "cpu":
        return "cpu", "CPU selected"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this PyTorch installation")
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
        return "cuda", f"CUDA · {torch.cuda.get_device_name(0)}"
    except Exception as exc:
        if requested == "cuda":
            raise ValueError(f"CUDA startup failed: {exc}. Run with --compute cpu.") from exc
        return "cpu", f"CPU fallback · {exc}"


class Runtime:
    def __init__(self, engine, source="static", camera_path=None, device_notice=""):
        self.engine, self.source, self.camera_path = engine, source, camera_path
        self.device_notice = device_notice
        self.commands = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.snapshot = None
        self.threads = []
        self.capture_error = ""
        self.notice = ""
        self.capture_info = {}
        self.capture_stats = dict(frames=0, driver_drops=0, drained_discards=0, bad_frames=0,
                                  timestamp_kind="synthetic-monotonic")
        self.capture_times = deque(maxlen=120)
        self.capture_count = self.overwrites = 0
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.revision = 0

    def start(self):
        for name, target in (("cognio-capture", self._capture), ("cognio-learning", self._work)):
            thread = threading.Thread(name=name, target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self.threads):
            raise RuntimeError("worker did not stop within five seconds")

    def submit(self, action, value=None):
        self.commands.put_nowait((action, value))

    def latest(self):
        with self.lock:
            return self.snapshot

    def _publish_frame(self, frame, stats):
        with self.lock:
            self.capture_count += 1
            self.capture_times.append(time.monotonic())
            if self.frame is not None:
                self.overwrites += 1
            self.frame = frame
            self.capture_stats = stats

    def _capture(self):
        try:
            if self.source == "camera":
                with V4L2Camera(self.camera_path) as camera:
                    self.capture_info = dict(identity=camera.info, mode=camera.mode, controls=camera.controls,
                                             warnings=camera.warnings)
                    last_received = time.monotonic()
                    previous_raw = None
                    sequence = 0
                    while not self.stop_event.is_set():
                        item = camera.read()
                        if item is None:
                            if time.monotonic() - last_received > 3:
                                raise RuntimeError("No camera frame for three seconds; close and relaunch after reconnecting")
                            continue
                        y, stamp, raw_sequence = item
                        last_received = time.monotonic()
                        if camera.timestamp_kind != "driver-monotonic":
                            raise RuntimeError("Camera lacks monotonic capture timestamps; causal timing cannot be verified")
                        delta = 0 if previous_raw is None else (raw_sequence - previous_raw) & 0xFFFFFFFF
                        if previous_raw is not None and not 0 < delta < 2**31:
                            raise RuntimeError("Camera sequence restarted or moved backwards")
                        sequence = raw_sequence if previous_raw is None else sequence + delta
                        previous_raw = raw_sequence
                        self._publish_frame(Frame(sequence, stamp, normalize(y)), camera.stats())
                self.capture_info["warnings"] = list(camera.warnings)
            else:
                start = time.monotonic()
                sequence = 0
                while not self.stop_event.is_set():
                    due = start + sequence / 30
                    if self.stop_event.wait(max(0, due - time.monotonic())):
                        break
                    # Skip elapsed synthetic instants instead of generating a catch-up burst.
                    sequence = max(sequence, int((time.monotonic() - start) * 30))
                    values = synthetic(sequence, self.source)
                    if values is not None:
                        stats = dict(frames=self.capture_count + 1, driver_drops=0, drained_discards=0,
                                     bad_frames=0, timestamp_kind="synthetic-monotonic")
                        self._publish_frame(Frame(sequence, time.monotonic(), values), stats)
                    sequence += 1
        except Exception as exc:
            self.capture_error = f"{type(exc).__name__}: {exc}"

    def _command(self, action, value):
        if action == "training":
            self.engine.set_training(value)
            self.notice = ""
        elif action == "reset":
            self.engine = Engine(replace(self.engine.config, seed=secrets.randbits(32)), str(self.engine.device))
            self.engine.set_training(False)
            self.notice = "New random model, training paused. Enable training explicitly."
        elif action == "save":
            self.engine.save(value)
            self.notice = "Checkpoint saved. Pending pairs cleared; five-frame warm-up."
        elif action == "load":
            candidate = Engine.load(value, str(self.engine.device))
            candidate.set_training(False)
            self.engine = candidate
            self.notice = "Checkpoint restored, training paused. Live sensory history is unavailable."
        else:
            raise ValueError(f"unknown command: {action}")

    def _work(self):
        visual = None
        samples = deque(maxlen=120)
        last_publish = 0
        cpu = 0.0
        while not self.stop_event.is_set():
            try:
                # Drain a bounded command batch so UI requests cannot starve sensory processing.
                for _ in range(16):
                    try:
                        action, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._command(action, value)
                        if action in ("load", "reset"):
                            visual = None
                            samples.clear()
                        self.revision += 1
                    except Exception as exc:
                        self.notice = f"{action} failed; previous state retained: {exc}"
                with self.lock:
                    frame, self.frame = self.frame, None
                if frame is not None:
                    result = self.engine.step(frame)
                    if result is not None:
                        visual = result
                    elif visual is not None:
                        visual = {**visual, "input": frame.values, "input_time": frame.timestamp}
                now = time.monotonic()
                if now - last_publish >= 1 / 30:
                    counts = self.engine.counts.copy()
                    samples.append((now, counts["predictions"], counts["updates"]))
                    elapsed = max(now - samples[0][0], 1e-6)
                    rss = self.process.memory_info().rss / 1024**2
                    self.peak_rss = max(self.peak_rss, rss)
                    if int(now) != int(last_publish):
                        cpu = self.process.cpu_percent()
                    with self.lock:
                        capture_times = list(self.capture_times)
                        capture = self.capture_stats.copy()
                        overwritten = self.overwrites
                    fps = ((len(capture_times) - 1) / max(capture_times[-1] - capture_times[0], 1e-6)) if len(capture_times) > 1 else 0
                    gpu = self.engine.device.type == "cuda"
                    snap = dict(visual=visual, result=self.engine.last_result,
                                history=list(self.engine.history), counts=counts, training=self.engine.training,
                                fault=self.engine.fault, capture_error=self.capture_error, notice=self.notice,
                                origin=self.engine.origin, source=self.source, config=self.engine.config,
                                input_sequence=self.engine.last_sequence,
                                capture=capture, capture_info=copy_capture_info(self.capture_info),
                                overwritten=overwritten, revision=self.revision,
                                input_fps=fps if capture_times and now - capture_times[-1] < 3 else 0,
                                prediction_fps=(counts["predictions"] - samples[0][1]) / elapsed,
                                updates_per_s=(counts["updates"] - samples[0][2]) / elapsed,
                                prediction_ms=self.engine.prediction_ms, update_ms=self.engine.update_ms,
                                training_loss=self.engine.last_training_loss,
                                rss_mib=rss, peak_rss_mib=self.peak_rss, cpu_percent=cpu,
                                vram_mib=torch.cuda.memory_allocated() / 1024**2 if gpu else 0,
                                reserved_vram_mib=torch.cuda.memory_reserved() / 1024**2 if gpu else 0,
                                compute=self.device_notice, timestamp=now,
                                elapsed_s=now - self.started)
                    with self.lock:
                        self.snapshot = snap
                    last_publish = now
                self.stop_event.wait(.001 if frame is not None else .005)
            except Exception as exc:
                # Keep control/save/restore and capture available after a model/input fault.
                self.engine.fault = f"Runtime fault: {type(exc).__name__}: {exc}"
                self.engine.training = False
                self.engine.pending.clear()


def copy_capture_info(info):
    # Values only, never images; nested dictionaries are fixed after startup.
    return {**info, "warnings": list(info.get("warnings", []))}

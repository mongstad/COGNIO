"""Independent capture, learning and bounded display handoff. No recording."""

from collections import deque
from dataclasses import replace
import gc
import queue
import secrets
import threading
import time

import psutil
import torch

from .camera import V4L2Camera
from .engine import Engine
from .inputs import Frame, component_statistics, normalize, normalize_ycbcr, synthetic
from .model import Config


def select_device(requested, config=Config(), memory_budget_mib=None):
    if requested == "cpu":
        return "cpu", "CPU selected"
    probe = before = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in this PyTorch installation")
        # Exact requested architecture, including pairing contexts, guarded rollback,
        # backward and Adam. This probe is only called after GUI Start.
        logical = [0.]
        probe = Engine(config, "cuda", clock=lambda: logical[0]+.001,
                       memory_budget_mib=memory_budget_mib)
        before = probe.model.decoder.bias.detach().clone()
        for sequence in range(7):
            logical[0] = sequence/30
            probe.step(Frame(sequence, logical[0], synthetic(sequence, "moving", config.input_mode)), observe=False)
            if probe.fault:
                raise RuntimeError(probe.fault)
        torch.cuda.synchronize()
        if probe.counts["updates"] != 2 or torch.equal(before, probe.model.decoder.bias):
            raise RuntimeError("CUDA optimizer probe did not produce a finite parameter update")
        return "cuda", (f"CUDA exact-size training probe passed · {torch.cuda.get_device_name(0)} · "
                        f"{config.channels}×{config.grid} {config.input_mode} {config.topology} · "
                        f"torch {torch.__version__} / runtime {torch.version.cuda}")
    except Exception as exc:
        if requested == "cuda":
            raise ValueError(f"CUDA startup failed: {exc}. Run with --compute cpu.") from exc
        return "cpu", f"CPU fallback · {exc}"
    finally:
        del probe, before
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


class Runtime:
    def __init__(self, engine, source="static", camera_path=None, device_notice="", observer=True, display_hz=15,
                 video_path=None, video_speed=1., video_loop=False, video_paused=False, video_fastest=False):
        self.engine, self.source, self.camera_path = engine, source, camera_path
        self.device_notice = device_notice
        if not 1 <= display_hz <= 30:
            raise ValueError("display frequency must be in [1,30] Hz")
        self.observer, self.display_hz = observer, display_hz
        self.channel_start = 0
        self.preprocess_ms = 0.0
        self.synthetic_skips = 0
        self.input_statistics = None
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
        self.video = None
        self.video_failed = False
        self.video_segment = 0
        if source == "video":
            if not video_path:
                raise ValueError("--source video requires --video-file PATH")
            from .video import VideoPlayback
            self.video = VideoPlayback(video_path, engine.config.input_mode, self.stop_event,
                                       video_speed, video_loop, video_paused, video_fastest)

    def start(self):
        workers = [("cognio-learning", self._work)] if self.video else [("cognio-capture", self._capture), ("cognio-learning", self._work)]
        for name, target in workers:
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
                with V4L2Camera(self.camera_path, input_mode=self.engine.config.input_mode) as camera:
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
                        planes, stamp, raw_sequence = item
                        last_received = time.monotonic()
                        if camera.timestamp_kind != "driver-monotonic":
                            raise RuntimeError("Camera lacks monotonic capture timestamps; causal timing cannot be verified")
                        delta = 0 if previous_raw is None else (raw_sequence - previous_raw) & 0xFFFFFFFF
                        if previous_raw is not None and not 0 < delta < 2**31:
                            raise RuntimeError("Camera sequence restarted or moved backwards")
                        sequence = raw_sequence if previous_raw is None else sequence + delta
                        previous_raw = raw_sequence
                        started = time.perf_counter()
                        values = normalize(planes) if self.engine.config.input_mode == "y" else normalize_ycbcr(planes)
                        self.preprocess_ms = 1000 * (time.perf_counter() - started)
                        self._publish_frame(Frame(sequence, stamp, values), camera.stats())
                self.capture_info["warnings"] = list(camera.warnings)
            else:
                start = time.monotonic()
                sequence = 0
                while not self.stop_event.is_set():
                    due = start + sequence / 30
                    if self.stop_event.wait(max(0, due - time.monotonic())):
                        break
                    # Skip elapsed synthetic instants instead of generating a catch-up burst.
                    delivered = max(sequence, int((time.monotonic() - start) * 30))
                    self.synthetic_skips += delivered - sequence
                    sequence = delivered
                    started = time.perf_counter()
                    values = synthetic(sequence, self.source, self.engine.config.input_mode)
                    self.preprocess_ms = 1000 * (time.perf_counter() - started)
                    if values is not None:
                        stats = dict(frames=self.capture_count + 1, driver_drops=0, drained_discards=0,
                                     bad_frames=0, timestamp_kind="synthetic-monotonic")
                        self._publish_frame(Frame(sequence, time.monotonic(), values), stats)
                    sequence += 1
        except Exception as exc:
            self.capture_error = f"{type(exc).__name__}: {exc}"

    def _command(self, action, value):
        if action.startswith("video_"):
            if self.video is None:
                raise ValueError("Video controls require a video source")
            self.video.control(action, value)
            if action == "video_restart":
                self.engine.reset_context(self.video.segment)
                self.video_segment = self.video.segment
                self.capture_error = ""
                self.video_failed = False
            self.notice = ""
        elif action == "training":
            self.engine.set_training(value)
            self.notice = ""
        elif action == "reset_hidden":
            self.engine.reset_hidden()
            self.notice = "Transient hidden state reset; weights/Adam and training mode retained. Five-frame warm-up."
        elif action == "reset":
            self.engine = Engine(replace(self.engine.config, seed=secrets.randbits(32)), str(self.engine.device),
                                 memory_budget_mib=self.engine.memory_budget_mib)
            self.engine.set_training(False)
            self.notice = "New random model, training paused. Enable training explicitly."
        elif action == "save":
            self.engine.save(value)
            self.notice = "Checkpoint saved. Pending pairs cleared; five-frame warm-up."
        elif action == "load":
            candidate = Engine.load(value, str(self.engine.device), expected_config=self.engine.config,
                                    memory_budget_mib=self.engine.memory_budget_mib)
            candidate.set_training(False)
            self.engine = candidate
            self.notice = "Checkpoint restored, training paused. Live sensory history is unavailable."
        elif action == "channel_page":
            pages = (self.engine.config.channels + 31) // 32
            if type(value) is not int or not 0 <= value < pages:
                raise ValueError("channel page is out of range")
            self.channel_start = value * 32
        else:
            raise ValueError(f"unknown command: {action}")

    def _work(self):
        visual = None
        display_result = None
        samples = deque(maxlen=120)
        last_publish = 0
        last_observe = 0
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
                        if action in ("load", "reset", "save", "training", "video_restart", "reset_hidden"):
                            display_result = None
                        if action in ("load", "reset", "video_restart", "reset_hidden"):
                            visual = None
                            self.input_statistics = None
                            samples.clear()
                        self.revision += 1
                    except Exception as exc:
                        self.notice = f"{action} failed; previous state retained: {exc}"
                if self.video is not None and not self.video_failed:
                    try:
                        delivered = self.video.poll()
                        if self.video.segment != self.video_segment:
                            self.engine.reset_context(self.video.segment)
                            self.video_segment = self.video.segment
                            visual = display_result = None
                        self.capture_info = dict(video=self.video.info, warnings=self.video.info.get("warnings", []))
                        if delivered is not None:
                            stats = dict(frames=self.capture_count+1, driver_drops=0, drained_discards=0,
                                         bad_frames=0, timestamp_kind="video-delivery-monotonic")
                            self._publish_frame(delivered, stats)
                        if self.video.eof:
                            # No observed target exists for the tail; preserve scored metrics.
                            self.engine.counts["cleared_targets"] += len(self.engine.pending)
                            self.engine.pending.clear()
                    except InterruptedError:
                        break
                    except Exception as exc:
                        self.capture_error = f"Video input: {type(exc).__name__}: {exc}"
                        self.video_failed = True
                        self.video.close()
                with self.lock:
                    frame, self.frame = self.frame, None
                if frame is not None:
                    observe = self.observer and time.monotonic() - last_observe >= 1/self.display_hz
                    result = self.engine.step(frame, observe=observe, channel_start=self.channel_start)
                    self.input_statistics = component_statistics(frame.values, self.engine.config.input_mode)
                    if observe and result is not None:
                        visual = result
                        last_observe = time.monotonic()
                        if self.engine.last_result is not None and self.engine.last_result["error"] is not None:
                            display_result = self.engine.last_result
                    elif result is None and visual is not None:
                        visual = {**visual, "input": frame.values, "input_time": frame.timestamp,
                                  "input_sequence": frame.sequence}
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
                    snap = dict(visual=visual, result=self.engine.last_result, display_result=display_result,
                                video=self.video.status() if self.video else None,
                                history=list(self.engine.history), component_history=list(self.engine.component_history),
                                input_statistics=self.input_statistics, counts=counts, training=self.engine.training,
                                fault=self.engine.fault, capture_error=self.capture_error, notice=self.notice,
                                origin=self.engine.origin, source=self.source, config=self.engine.config,
                                input_sequence=self.engine.last_sequence,
                                input_age_ms=(now-self.engine.last_timestamp)*1000 if self.engine.last_timestamp is not None else None,
                                parameters=self.engine.config.parameters, hidden_values=self.engine.config.hidden_values,
                                pending=len(self.engine.pending), synthetic_skips=self.synthetic_skips,
                                preprocess_ms=self.preprocess_ms, step_ms=self.engine.step_ms,
                                observer_ms=self.engine.observer_ms, transfer_ms=self.engine.transfer_ms,
                                observer=self.observer, channel_start=self.channel_start,
                                capture=capture, capture_info=copy_capture_info(self.capture_info),
                                overwritten=overwritten, revision=self.revision,
                                input_fps=fps if capture_times and now - capture_times[-1] < 3 and
                                          not (self.video and (self.video.paused or self.video.eof)) else 0,
                                prediction_fps=(counts["predictions"] - samples[0][1]) / elapsed,
                                updates_per_s=(counts["updates"] - samples[0][2]) / elapsed,
                                prediction_ms=self.engine.prediction_ms, update_ms=self.engine.update_ms,
                                training_loss=self.engine.last_training_loss,
                                rss_mib=rss, peak_rss_mib=self.peak_rss, cpu_percent=cpu,
                                vram_mib=torch.cuda.memory_allocated() / 1024**2 if gpu else 0,
                                reserved_vram_mib=torch.cuda.memory_reserved() / 1024**2 if gpu else 0,
                                peak_vram_mib=torch.cuda.max_memory_allocated() / 1024**2 if gpu else 0,
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
                self.engine.counts["cleared_targets"] += len(self.engine.pending)
                self.engine.pending.clear()
        if self.video:
            self.video.close()


def copy_capture_info(info):
    # Values only, never images; nested dictionaries are fixed after startup.
    return {**info, "warnings": list(info.get("warnings", []))}

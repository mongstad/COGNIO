"""Local FFmpeg decoding and sequential playback; no model or Qt imports."""

from collections import deque
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import time

import numpy as np

from .inputs import Frame


FRAME_BYTES = 3 * 128 * 128
MAX_PIXELS = 3840 * 2160
FORMATS = {"yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p"}
UNKNOWN = {None, "unknown", "unspecified", "reserved"}
LOCAL_INPUT = ["-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,matroska,webm,avi,ogg,nut",
               "-probesize", "5000000", "-analyzeduration", "5000000"]


def normalize_video(planes, matrix, level, input_mode):
    """Direct YCbCr→YCbCr affine transform, no RGB image intermediate.

    FFmpeg only resamples code planes. Canonical model codes are BT.601 limited
    range, retaining fractional values; clip only to representable 8-bit code
    bounds, not nominal video range. Camera preprocessing is unchanged.
    """
    values = planes.astype(np.float64)
    if matrix == "bt601" and level == "tv":
        values[1:] -= 128
        values /= 255
    else:
        values[0] = (values[0]-(16 if level == "tv" else 0))/(219 if level == "tv" else 255)
        values[1:] = (values[1:]-128)/(224 if level == "tv" else 255)
        if matrix == "bt709":
            # Compose the small coefficient matrices, without constructing RGB pixels.
            kr, kb, kg = .2126, .0722, .7152
            inverse_source = np.array([[1,0,2*(1-kr)], [1,-2*kb*(1-kb)/kg,-2*kr*(1-kr)/kg], [1,2*(1-kb),0]])
            target = np.array([[.299,.587,.114], [-.299/1.772,-.587/1.772,.886/1.772],
                               [.701/1.402,-.587/1.402,-.114/1.402]])
            values = np.einsum("ij,jhw->ihw", target @ inverse_source, values)
        values[0] = np.clip((16+219*values[0])/255,0,1)
        values[1:] = np.clip(224*values[1:]/255,-128/255,127/255)
    values = values.astype(np.float32)
    return values[0].copy() if input_mode == "y" else values


def color_contract(info):
    """Resolve supported SDR YUV metadata explicitly, with reported fallbacks."""
    matrix = info.get("color_space")
    warnings = []
    if matrix in UNKNOWN:
        matrix = "smpte170m" if info["height"] <= 576 else "bt709"
        warnings.append(f"Unspecified YCbCr matrix: assuming {matrix} from stored height")
    if matrix not in ("bt709", "smpte170m", "bt470bg"):
        raise ValueError(f"Unsupported video color matrix {matrix}; use SDR BT.601/BT.709 video")
    level = info.get("color_range")
    if level in UNKNOWN:
        level = "pc" if info["pix_fmt"].startswith("yuvj") else "tv"
        warnings.append(f"Unspecified video range: assuming {level}")
    if level not in ("tv", "pc"):
        raise ValueError(f"Unsupported video range {level}")
    transfer, primaries = info.get("color_transfer"), info.get("color_primaries")
    if transfer not in UNKNOWN | {"bt709", "smpte170m", "iec61966-2-1", "gamma22", "gamma28"}:
        raise ValueError(f"Unsupported video transfer {transfer}; HDR/linear video is not supported")
    if primaries not in UNKNOWN | {"bt709", "smpte170m", "bt470bg", "bt470m"}:
        raise ValueError(f"Unsupported video primaries {primaries}; use SDR video")
    if transfer in UNKNOWN or primaries in UNKNOWN:
        warnings.append("Unspecified primaries/transfer: retained code-domain appearance, no color calibration")
    return ("bt709" if matrix == "bt709" else "bt601"), level, warnings


def probe(path, stop_event=None):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Video input must be a local regular file")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Video input requires ffmpeg and ffprobe on PATH; see docs/VIDEO.md")
    command = ["ffprobe", "-v", "error", *LOCAL_INPUT, "-select_streams", "v:0",
               "-show_entries", "stream=index,codec_name,pix_fmt,width,height,time_base,avg_frame_rate,color_range,color_space,color_transfer,color_primaries",
               "-of", "json", str(path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    output = bytearray()
    try:
        os.set_blocking(process.stdout.fileno(),False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ)
            deadline = time.monotonic()+3
            while selector.get_map():
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError("Video probe stopped")
                if time.monotonic() > deadline:
                    raise ValueError("Video metadata probe exceeded three seconds")
                for key,_ in selector.select(.05):
                    chunk = os.read(key.fd,16384)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    output.extend(chunk)
                    if len(output) > 65536:
                        raise ValueError("Video metadata probe exceeded 64 KiB output bound")
        if process.wait(timeout=1):
            raise ValueError("Cannot probe local video: " + output.decode(errors="replace")[-1500:])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        process.stdout.close()
    streams = json.loads(output).get("streams", [])
    if not streams:
        raise ValueError("File has no decodable video stream")
    info = streams[0]
    if not 0 < info.get("width", 0) * info.get("height", 0) <= MAX_PIXELS:
        raise ValueError("Video dimensions exceed the 3840×2160 pixel budget or are invalid")
    if info.get("pix_fmt") not in FORMATS:
        raise ValueError(f"Unsupported video pixel format {info.get('pix_fmt')}; require 8-bit planar YUV 420/422/444")
    matrix, level, warnings = color_contract(info)
    return path, dict(info, resolved_matrix=matrix, resolved_range=level, warnings=warnings,
                      output="128x128 planar BT.601 limited-range YCbCr; area scaling; code-domain SDR")


class VideoReader:
    """One raw frame paired with its exact FFmpeg showinfo PTS; bounded pipes.

    No FPS filter or output frame-rate conversion. Original presentation times
    must strictly increase, including VFR; missing/duplicate/backward PTS fail.
    The decoder may reorder/prefetch codec frames but cannot supply model input.
    """
    def __init__(self, path, stop_event):
        self.path, self.info = probe(path, stop_event)
        self.stop_event = stop_event
        self.raw, self.lines = bytearray(), bytearray()
        self.records, self.errors = deque(), deque(maxlen=8)
        self.time_base = None
        self.previous_pts = None
        self.index = 0
        self.selector = selectors.DefaultSelector()
        self.process = None
        matrix, level = self.info["resolved_matrix"], self.info["resolved_range"]
        filters = (f"showinfo=checksum=0,scale=128:128:flags=area+accurate_rnd+bitexact:"
                   f"in_color_matrix={matrix}:out_color_matrix={matrix}:in_range={level}:out_range={level},format=yuv444p")
        command = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-xerror",
                   "-threads", "2", "-filter_threads", "1", "-filter_buffered_frames", "8", *LOCAL_INPUT, "-max_pixels", str(MAX_PIXELS),
                   "-noautorotate", "-i", str(self.path), "-map", "0:v:0", "-an", "-sn", "-dn",
                   "-vf", filters, "-fps_mode", "passthrough", "-threads", "1",
                   "-thread_queue_size", "1", "-pix_fmt", "yuv444p", "-f", "rawvideo", "pipe:1"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            for stream, name in ((self.process.stdout, "raw"), (self.process.stderr, "log")):
                os.set_blocking(stream.fileno(), False)
                self.selector.register(stream, selectors.EVENT_READ, name)
        except Exception:
            self.close()
            raise

    def _line(self, line):
        text = line.decode(errors="replace")
        if "showinfo" not in text:
            if any(message in text.lower() for message in ("file ended prematurely", "packet corrupt", "corrupt input packet", "invalid data found")):
                raise ValueError("Video is truncated or corrupt: " + text[-500:])
            self.errors.append(text[-500:])
            return
        match = re.search(r"config in time_base: (\d+/\d+)", text)
        if match:
            self.time_base = Fraction(match[1])
        if re.search(r"\bn:\s*\d+", text):
            match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", text)
            if not match or self.time_base is None:
                raise ValueError("Video frame lacks a reliable presentation timestamp")
            if (f"fmt:{self.info['pix_fmt']} " not in text or
                    f"s:{self.info['width']}x{self.info['height']} " not in text or "i:P" not in text):
                raise ValueError("Video format/dimensions changed or interlaced video is unsupported")
            self.records.append(dict(index=int(match[1]), pts=int(match[2])*self.time_base, ready=False))
            if len(self.records) > 32:
                raise ValueError("Decoder exceeded bounded metadata queue")
        if "color_range:" in text and self.records:
            fields = dict(re.findall(r"(color_range|color_space|color_primaries|color_trc):([^\s]+)", text))
            # Reject mid-stream color changes instead of silently applying stale metadata.
            for field, key in (("color_range", "color_range"), ("color_space", "color_space"),
                               ("color_primaries", "color_primaries"), ("color_trc", "color_transfer")):
                actual, expected = fields.get(field), self.info.get(key)
                if actual not in UNKNOWN and expected not in UNKNOWN and actual != expected:
                    raise ValueError(f"Video {key} changed during decoding")
            resolved = dict(self.info, **{("color_transfer" if k == "color_trc" else k): v for k,v in fields.items()})
            if color_contract(resolved)[:2] != (self.info["resolved_matrix"], self.info["resolved_range"]):
                raise ValueError("Per-frame video color metadata conflicts with the resolved stream contract")
            self.records[-1]["ready"] = True

    def read(self, input_mode):
        deadline = time.monotonic() + 10
        while True:
            if self.stop_event.is_set():
                raise InterruptedError("Video decoding stopped")
            if len(self.raw) >= FRAME_BYTES and self.records and self.records[0]["ready"]:
                record = self.records.popleft()
                if record["index"] != self.index or (self.previous_pts is not None and record["pts"] <= self.previous_pts):
                    raise ValueError("Video presentation timestamps must strictly increase; duplicate/backward PTS unsupported")
                planes = np.frombuffer(self.raw[:FRAME_BYTES], np.uint8).reshape(3,128,128)
                del self.raw[:FRAME_BYTES]
                self.index += 1
                self.previous_pts = record["pts"]
                return record["index"], record["pts"], normalize_video(planes, self.info["resolved_matrix"], self.info["resolved_range"], input_mode)
            if not self.selector.get_map():
                code = self.process.wait(timeout=1)
                if code or self.raw or self.records or not self.index:
                    raise ValueError(f"Video decoding failed/truncated (exit {code}): " + " | ".join(self.errors))
                return None
            if time.monotonic() > deadline:
                raise ValueError("Video decoder made no complete-frame progress for ten seconds")
            for key, _ in self.selector.select(.05):
                chunk = os.read(key.fd, 16384)
                if not chunk:
                    self.selector.unregister(key.fileobj)
                elif key.data == "raw":
                    self.raw.extend(chunk)
                    if len(self.raw) > FRAME_BYTES*2:
                        raise ValueError("Decoder exceeded bounded frame buffer")
                else:
                    self.lines.extend(chunk)
                    while b"\n" in self.lines:
                        line, _, remainder = self.lines.partition(b"\n")
                        self.lines = bytearray(remainder)
                        self._line(line)
                    if len(self.lines) > 65536:
                        raise ValueError("Decoder exceeded bounded log buffer")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            for stream in (self.process.stdout, self.process.stderr):
                if stream:
                    stream.close()
        self.selector.close()


class VideoPlayback:
    """All frames, sequential learner-paced delivery; no catch-up skips/duplicates."""
    def __init__(self, path, input_mode, stop_event, speed=1., loop=False, paused=False, fastest=False, clock=time.monotonic):
        self.validate_speed(speed)
        self.path, self.input_mode, self.stop_event = path, input_mode, stop_event
        self.speed, self.loop, self.paused, self.fastest = speed, loop, paused, fastest
        self.clock = clock
        self.reader = None
        self.next_frame = None
        self.info = {}
        self.sequence = self.segment = 0
        self.media_time = self.media_index = self.position = None
        self.tick = clock()
        self.eof = False
        self.decode_wait_ms = 0.

    @staticmethod
    def validate_speed(speed):
        if not math.isfinite(speed) or not .1 <= speed <= 4:
            raise ValueError("video speed must be finite and in [0.1,4]")

    def _advance(self):
        now = self.clock()
        if self.position is not None and not self.paused:
            self.position += max(0, now-self.tick)*self.speed
        self.tick = now

    def control(self, action, value):
        self._advance()
        if action == "video_pause":
            self.paused = bool(value)
        elif action == "video_speed":
            self.validate_speed(value)
            self.speed = value
        elif action == "video_loop":
            self.loop = bool(value)
        elif action == "video_restart":
            self.restart()
        else:
            raise ValueError(f"Unknown video control {action}")

    def restart(self):
        self.close()
        self.reader = None
        self.next_frame = None
        self.position = self.media_time = self.media_index = None
        self.tick = self.clock()
        self.segment += 1
        self.eof = False

    def poll(self):
        self._advance()
        if self.paused or self.eof:
            return None
        if self.reader is None:
            self.reader = VideoReader(self.path, self.stop_event)
            self.info = self.reader.info
        if self.next_frame is None:
            started = time.perf_counter()
            self.next_frame = self.reader.read(self.input_mode)
            self.decode_wait_ms = 1000*(time.perf_counter()-started)
            self._advance()
            if self.next_frame is None:
                self.close()
                if self.loop:
                    self.restart()
                else:
                    self.eof = True
                return None
        index, pts, values = self.next_frame
        media_time = float(pts)
        if not self.fastest and self.position is not None and self.position < media_time:
            return None
        # Observation means delivery to the learner, not decode/prefetch or media PTS.
        # The caller completed the preceding Engine.step before asking for this frame.
        stamp = self.clock()
        frame = Frame(self.sequence, stamp, values, media_time, index, self.segment)
        self.sequence += 1
        self.media_index, self.media_time = index, media_time
        self.position, self.tick = media_time, stamp  # Discard backlog; never catch up by dropping.
        self.next_frame = None
        return frame

    def status(self):
        return dict(speed=self.speed, fastest=self.fastest, paused=self.paused, loop=self.loop,
                    eof=self.eof, segment=self.segment, media_index=self.media_index, media_time=self.media_time,
                    decoded_frames=self.reader.index if self.reader else 0, decode_wait_ms=self.decode_wait_ms, metadata=self.info)

    def close(self):
        if self.reader is not None:
            self.reader.close()

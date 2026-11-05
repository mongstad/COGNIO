"""Generate a small deterministic, silent FFV1 fixture; no downloaded media."""

import argparse
from pathlib import Path
import subprocess

import numpy as np


def planes(index, stimulus="moving"):
    y, x = np.mgrid[:128, :128]
    values = np.full((3,128,128), 128, dtype=np.uint8)
    values[0] = 48
    if stimulus in ("static", "moving"):
        offset = 0 if stimulus == "static" else index*3 % 96
        values[0, 40:88, offset:offset+24] = 200
    elif stimulus == "color":
        values[0] = 126
        values[1] = np.rint(128+60*np.sin(2*np.pi*(x/128-index/40))).astype(np.uint8)
        values[2] = np.rint(128+50*np.cos(2*np.pi*(y/128-index/40))).astype(np.uint8)
    else:
        raise ValueError("unknown fixture stimulus")
    return values


def make_video(path, frames=60, stimulus="moving", vfr=False):
    if not 1 <= frames <= 1200:
        raise ValueError("fixture must contain 1..1200 frames")
    filters = "settb=1/1000,setpts='if(lt(N,4),N*40,160+(N-4)*80)'" if vfr else "null"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "yuv444p", "-video_size", "128x128", "-framerate", "30",
               "-colorspace", "smpte170m", "-color_range", "tv", "-color_trc", "bt709", "-color_primaries", "bt709", "-i", "pipe:0",
               "-an", "-vf", filters, "-fps_mode", "passthrough", "-enc_time_base", "1:1000",
               "-c:v", "ffv1", "-threads", "1", "-colorspace", "smpte170m", "-color_range", "tv",
               "-color_trc", "bt709", "-color_primaries", "bt709", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for index in range(frames):
            process.stdin.write(planes(index, stimulus).tobytes())
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate(timeout=15)
        if process.returncode:
            raise RuntimeError(error.decode(errors="replace"))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stimulus", choices=("static", "moving", "color"), default="moving")
    parser.add_argument("--vfr", action="store_true")
    args = parser.parse_args()
    make_video(args.path, args.frames, args.stimulus, args.vfr)

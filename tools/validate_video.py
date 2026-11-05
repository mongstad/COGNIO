"""Bounded generated-video frozen/train/frozen controls and cadence measurements."""

import argparse
import json
import os
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import psutil
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from cognio.engine import Engine
from cognio.model import Config
from cognio.runtime import select_device
from cognio.video import VideoPlayback
from tools.make_video_fixture import make_video


def run_pass(engine, path, segment, training, speed=1., fastest=True):
    engine.set_training(training)
    engine.reset_context()
    player = VideoPlayback(path,engine.config.input_mode,threading.Event(),speed=speed,fastest=fastest)
    player.sequence = 0 if engine.last_sequence is None else engine.last_sequence+1
    player.segment = segment
    start = time.monotonic()
    baseline = engine.counts.copy()
    rss, decoder_rss = [], []
    horizons, leads = [], []
    delivery_times = []
    process = psutil.Process()
    try:
        while not player.eof:
            frame = player.poll()
            if frame is None:
                time.sleep(.001)
                continue
            delivery_times.append(frame.timestamp)
            engine.step(frame,observe=False)
            if engine.fault:
                raise RuntimeError(engine.fault)
            if engine.last_result:
                match = engine.last_result
                assert match["segment"] == segment
                assert match["target_media_index"]-match["source_media_index"] == 5
                assert match["issued"] < frame.timestamp
                horizons.append((match["media_horizon_ms"],match["horizon_ms"]))
                leads.append(match["lead_ms"])
            if len(delivery_times) % 20 == 0:
                rss.append(process.memory_info().rss/1024**2)
                try:
                    decoder_rss.append(psutil.Process(player.reader.process.pid).memory_info().rss/1024**2)
                except psutil.NoSuchProcess:
                    pass
        samples = list(engine.history)
        return dict(training=training, speed=speed, fastest=fastest, frames=len(delivery_times),
                    counts={k:engine.counts[k]-baseline[k] for k in baseline}, elapsed_s=time.monotonic()-start,
                    delivery_fps=(len(delivery_times)-1)/(delivery_times[-1]-delivery_times[0]),
                    mse=float(np.mean([s[0] for s in samples])), persistence_mse=float(np.mean([s[1] for s in samples])),
                    mean_media_wall_horizon_ms=np.mean(horizons,axis=0).tolist(), minimum_lead_ms=min(leads),
                    sampled_peak_rss_mib=max(rss,default=None), sampled_peak_decoder_rss_mib=max(decoder_rss,default=None),
                    rss_first_last_mib=[rss[0],rss[-1]] if rss else None,
                    peak_pytorch_vram_mib=torch.cuda.max_memory_allocated()/1024**2 if engine.device.type == "cuda" else 0,
                    maximum_pending_bound=engine.config.horizon)
    finally:
        player.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--channels",type=int,default=32)
    parser.add_argument("--grid",type=int,default=8)
    parser.add_argument("--report",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device, notice = select_device(args.compute)
    report = dict(compute=notice,channels=args.channels,grid=args.grid,input_mode="ycbcr",seed=7,
                  torch=str(torch.__version__),controls={},cadence={})
    with tempfile.TemporaryDirectory(prefix="cognio-video-validation-") as directory:
        root = Path(directory)
        for stimulus in ("static","moving","color"):
            short, long = root/f"{stimulus}-short.mkv", root/f"{stimulus}-long.mkv"
            make_video(short,120,stimulus)
            make_video(long,600,stimulus)
            engine = Engine(Config(channels=args.channels,grid=args.grid,input_mode="ycbcr"),device=device)
            report["controls"][stimulus] = dict(initial=run_pass(engine,short,0,False),
                training=run_pass(engine,long,1,True),frozen=run_pass(engine,short,2,False))
        paced = root/"paced.mkv"
        make_video(paced,30,"moving")
        for speed in (1.,.1):
            engine = Engine(Config(channels=args.channels,grid=args.grid,input_mode="ycbcr"),device=device)
            report["cadence"][str(speed)] = run_pass(engine,paced,0,True,speed=speed,fastest=False)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())

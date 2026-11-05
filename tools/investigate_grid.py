"""Bounded scalar-only scene-cut controls and read-only checkpoint A/B diagnosis."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import psutil
import torch

from cognio.diagnostics import (learned_statistics, plane_statistics, prediction_statistics,
                                probe_context, state_digest, tensor_statistics, periodicity)
from cognio.engine import Engine
from cognio.inputs import Frame
from cognio.model import Config
from cognio.runtime import select_device


SCENARIOS = ("steady", "smooth", "cuts", "black", "rapid", "low-light", "bright", "black-smooth")


def stimulus(index, scenario, challenge_length):
    """Same A before/after the middle third; deterministic changes only within it.

    A has a moving Gaussian on a smooth field. B has an unrelated static diagonal
    shape and different chroma. Both avoid decoder-period lattice construction.
    Smooth/cuts repeat A/B every 60 frames with equal 30-frame half-cycle duration.
    """
    y,x = np.mgrid[:128,:128]/127
    blob = np.exp(-((x-(.5+.25*np.sin(index*.055)))**2+(y-.45)**2)/.015)
    a = np.stack((.3+.3*blob+.12*x, .08*np.sin(2*np.pi*x+index*.03), .06*np.cos(2*np.pi*y)))
    b = np.stack((.75-.4*(x+y>1), -.12+.08*y, .15-.08*x))
    inside = challenge_length <= index < 2*challenge_length
    if inside:
        local = index-challenge_length
        if scenario == "smooth":
            mix = .5-.5*np.cos(2*np.pi*local/60)
            a = (1-mix)*a+mix*b
        elif scenario == "cuts" and local%60 >= 30:
            a = b
        elif scenario in ("black", "bright", "black-smooth"):
            a = np.stack((np.full_like(x,.85), np.zeros_like(x), np.zeros_like(x)))
            if scenario == "black" and 25 <= local%60 < 30:  # Five-frame black intervals, then bright.
                a.fill(0)
            elif scenario == "black-smooth" and 20 <= local%60 <= 30:
                # Ten-frame cosine dip: same integrated Y deficit per 60-frame
                # cycle as the five black frames, but gradual rather than hard.
                a[0] *= 1-np.sin(np.pi*(local%60-20)/10)**2
        elif scenario == "rapid":
            a = a if local%10 < 5 else b
        elif scenario == "low-light":
            a = a*.02
    if scenario not in SCENARIOS:
        raise ValueError("unknown scene control")
    return a.astype(np.float32)


def run_case(engine, scenario, frames, train_frames, sample_every, deadline):
    initial_counts = engine.counts.copy()
    frozen = train_frames == 0
    engine.set_training(not frozen)
    learned_before = (state_digest(engine.model.state_dict()), state_digest(engine.optimizer.state_dict()))
    records, phases = [], {}
    length = (train_frames if train_frames else frames)//3
    indices = {0,5,10,20,30,length-1,length,length+5,2*length-1,2*length,2*length+5,frames-1}
    started, peak_rss = time.monotonic(), 0
    initial = dict(hidden=tensor_statistics(engine.hidden),model_digest=learned_before[0],optimizer_digest=learned_before[1])
    processed = 0
    for index in range(frames):
        if time.monotonic() >= deadline:
            break
        if train_frames and index == train_frames:
            engine.set_training(False)
        values = stimulus(index,scenario,length)
        if engine.config.input_mode == "y":
            values = values[0].copy()
        sample = index in indices or index%sample_every == 0 or index == train_frames
        pending = engine.pending.get(index)
        # Diagnose the same old source/previous state, before this target updates
        # any weights. The target is used only in the diagnostic loss.
        probe = probe_context(engine,pending.source.values,pending.previous,values) if sample and pending else None
        stamp = time.monotonic()  # Actual delivery, after the preceding step/probe.
        visual = engine.step(Frame(index,stamp,values), observe=sample)
        if engine.fault:
            raise RuntimeError(engine.fault)
        processed += 1
        phase = ("frozen" if train_frames and index >= train_frames else
                 "before" if index < length else "challenge" if index < 2*length else "recovery")
        sums = phases.setdefault(phase,dict(pairs=0,mse_sum=0.,baseline_sum=0.,component_sum=np.zeros(engine.config.input_channels),
                    baseline_component_sum=np.zeros(engine.config.input_channels),minimum_lead_ms=None))
        match = engine.last_result
        if match and match["target"] == index:
            assert match["target"]-match["source"] == 5 and match["issued"] < stamp
            sums["pairs"] += 1
            sums["mse_sum"] += match["mse"]
            sums["baseline_sum"] += match["baseline_mse"]
            sums["component_sum"] += list(match["component_mse"].values())
            sums["baseline_component_sum"] += list(match["component_baseline_mse"].values())
            sums["minimum_lead_ms"] = min(sums["minimum_lead_ms"] or float("inf"),match["lead_ms"])
        if sample:
            record = dict(frame=index,phase=phase,updates=engine.counts["updates"]-initial_counts["updates"],
                input=plane_statistics(values,engine.config.grid,engine.config.input_mode),
                prediction=prediction_statistics(visual["prediction"],engine.config.grid,engine.config.input_mode),
                hidden=tensor_statistics(engine.hidden),probe=probe)
            if match and match["target"] == index:
                record["matched"] = {key:match[key] for key in ("source","target","mse","baseline_mse","component_mse","component_baseline_mse","lead_ms")}
                residual = engine._array(pending.prediction)-values
                record["matched"]["residual_periodicity"] = [periodicity(p,engine.config.grid)
                    for p in (residual[None] if residual.ndim == 2 else residual)]
            if index in (0,length-1,2*length-1,frames-1):
                record["learned"] = learned_statistics(engine)
            records.append(record)
            peak_rss = max(peak_rss,psutil.Process().memory_info().rss/1024**2)
    for sums in phases.values():
        count = sums["pairs"]
        for source,destination in (("mse_sum","mse"),("baseline_sum","persistence_mse"),
                                   ("component_sum","component_mse"),("baseline_component_sum","component_persistence_mse")):
            value = sums.pop(source)/count if count else None
            sums[destination] = value.tolist() if isinstance(value,np.ndarray) else value
    learned_after = (state_digest(engine.model.state_dict()), state_digest(engine.optimizer.state_dict()))
    if frozen and learned_before != learned_after:
        raise AssertionError("Frozen diagnosis mutated learned parameters or optimizer")
    return dict(config=asdict(engine.config),scenario=scenario,initial=initial,requested_frames=frames,
        processed_frames=processed,complete=processed == frames,training_frames=train_frames,
        phases=phases,counts={key:engine.counts[key]-initial_counts[key] for key in initial_counts},
        elapsed_seconds=time.monotonic()-started,sampled_peak_rss_mib=peak_rss,
        peak_pytorch_vram_mib=torch.cuda.max_memory_allocated()/1024**2 if engine.device.type == "cuda" else 0,
        frozen_learned_state_unchanged=learned_before == learned_after if frozen else None,
        final_learned=learned_statistics(engine),samples=records)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,help="read-only frozen restored/zero-hidden comparisons")
    parser.add_argument("--compute",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--channels",type=int,default=32,help="new-model channels; checkpoint uses saved configuration")
    parser.add_argument("--grids",type=int,nargs="+",choices=(8,16,32),default=[32],help="new-model grids")
    parser.add_argument("--seeds",type=int,nargs="+",default=[7],help="new-model seeds")
    parser.add_argument("--learning-rates",type=float,nargs="+",default=[.001],help="new-model learning rates; checkpoint runs are frozen")
    parser.add_argument("--scenarios",choices=SCENARIOS,nargs="+",default=list(SCENARIOS))
    parser.add_argument("--frames",type=int,default=600,help="training frames for new models; frozen frames for checkpoints")
    parser.add_argument("--evaluation-frames",type=int,default=120,help="additional frozen frames for new models")
    parser.add_argument("--sample-every",type=int,default=100)
    parser.add_argument("--max-seconds",type=float,default=180,help="whole experiment time budget, at most 1800")
    parser.add_argument("--report",type=Path,required=True,help="new scalar JSON file; existing files are never overwritten")
    args = parser.parse_args()
    if (not 30 <= args.frames <= 10000 or not 6 <= args.evaluation_frames <= 1000 or
            not 10 <= args.sample_every <= 1000 or not math.isfinite(args.max_seconds) or
            not 1 <= args.max_seconds <= 1800 or max(len(args.grids),len(args.seeds),len(args.learning_rates)) > 3):
        parser.error("invalid experiment bounds; see --help")
    if args.report.exists():
        parser.error("report already exists; choose a new path")
    expected = len(args.scenarios)*(2 if args.checkpoint else len(args.grids)*len(args.seeds)*len(args.learning_rates))
    samples_bound = (args.frames+(0 if args.checkpoint else args.evaluation_frames))//args.sample_every+16
    if expected > 32 or expected*samples_bound > 2048:
        parser.error("diagnostic report bound: at most 32 cases and 2048 sampled records; split the experiment or sample less often")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device, notice = select_device(args.compute)
    deadline = time.monotonic()+args.max_seconds
    report = dict(torch=str(torch.__version__),compute=notice,threads=2,precision="float32",cases=[],
                  clock="real monotonic sequential delivery; unpaced, not capture-rate performance",max_seconds=args.max_seconds,
                  protocol=dict(scenarios=args.scenarios,frames=args.frames,evaluation_frames=0 if args.checkpoint else args.evaluation_frames,
                                sample_every=args.sample_every,requested_cases=expected))
    before = file_digest(args.checkpoint) if args.checkpoint else None
    report["checkpoint_sha256"] = before
    try:
        if args.checkpoint:
            for scenario in args.scenarios:
                for hidden in ("restored","zero"):
                    if time.monotonic() >= deadline:
                        break
                    engine = Engine.load(args.checkpoint,device=device)
                    if "checkpoint" not in report:
                        report["checkpoint"] = dict(config=asdict(engine.config),counts=engine.counts.copy(),
                            hidden=tensor_statistics(engine.hidden),learned=learned_statistics(engine))
                    if hidden == "zero":
                        engine.reset_hidden()
                    result = run_case(engine,scenario,args.frames,0,args.sample_every,deadline)
                    result["hidden_start"] = hidden
                    report["cases"].append(result)
                    print(json.dumps(dict(scenario=scenario,hidden=hidden,complete=result["complete"],phases=result["phases"])),flush=True)
                    del engine
        else:
            for grid in args.grids:
                for seed in args.seeds:
                    for lr in args.learning_rates:
                        for scenario in args.scenarios:
                            if time.monotonic() >= deadline:
                                break
                            engine = Engine(Config(seed=seed,learning_rate=lr,channels=args.channels,grid=grid,input_mode="ycbcr"),device=device)
                            result = run_case(engine,scenario,args.frames+args.evaluation_frames,args.frames,args.sample_every,deadline)
                            report["cases"].append(result)
                            print(json.dumps(dict(grid=grid,seed=seed,lr=lr,scenario=scenario,complete=result["complete"],phases=result["phases"])),flush=True)
                            del engine
    finally:
        if args.checkpoint:
            report["checkpoint_sha256_after"] = file_digest(args.checkpoint)
            report["checkpoint_unchanged"] = report["checkpoint_sha256_after"] == before
        report["complete"] = (len(report["cases"]) == expected and all(c["complete"] for c in report["cases"])
                              and report.get("checkpoint_unchanged",True))
        with args.report.open("x") as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
            stream.write("\n")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible scalar-only learning experiment, runnable as a module."""

import argparse
import json
import time
import numpy as np
import psutil
import torch

from cognio.engine import Engine
from cognio.inputs import Frame, synthetic
from cognio.model import Config


def experiment(mode, seed, training_frames=1200):
    current = [0.]
    engine = Engine(Config(seed=seed), clock=lambda: current[0] + .001)
    phases = []
    sequence = 0
    started = time.perf_counter()
    for name, count, learning in (("initial_frozen", 240, False),
                                  ("online_training", training_frames, True),
                                  ("final_frozen", 240, False)):
        engine.set_training(learning)
        errors, baselines = [], []
        before = engine.counts.copy()
        for _ in range(count):
            current[0] = sequence / 30
            engine.step(Frame(sequence, current[0], synthetic(sequence, mode)))
            if engine.last_result and engine.last_result["target"] == sequence:
                errors.append(engine.last_result["mse"])
                baselines.append(engine.last_result["baseline_mse"])
            sequence += 1
        if engine.fault:
            raise RuntimeError(engine.fault)
        phases.append(dict(phase=name, frames=count, pairs=len(errors), updates=engine.counts["updates"]-before["updates"],
                           mean_mse=float(np.mean(errors)), persistence_mse=float(np.mean(baselines)),
                           last_120_mse=float(np.mean(errors[-120:]))))
    assert phases[-1]["updates"] == 0
    return dict(mode=mode, seed=seed, phases=phases, counts=engine.counts,
                improves_over_initial=phases[-1]["mean_mse"] < phases[0]["mean_mse"],
                beats_persistence=phases[-1]["mean_mse"] < phases[-1]["persistence_mse"],
                wall_seconds=time.perf_counter()-started,
                rss_mib=psutil.Process().memory_info().rss/1024**2,
                pending=len(engine.pending), history=len(engine.history))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-frames", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    results = []
    for mode in ("static", "moving", "shape", "temporal"):
        for seed in args.seeds:
            results.append(experiment(mode, seed, args.training_frames))
    print(json.dumps(dict(torch=str(torch.__version__), numpy=np.__version__, threads=2,
                          precision="float32", results=results), indent=2))


if __name__ == "__main__":
    main()

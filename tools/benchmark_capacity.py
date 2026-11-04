"""Bounded scalar-only capacity measurements; no camera or saved sensory history.

Unpaced cases use a virtual 30 Hz observation clock. Wall throughput is reported
separately; these cases must not be read as real-time capture performance.
"""

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import psutil
import torch

from cognio.engine import Engine
from cognio.inputs import Frame, synthetic
from cognio.model import Config, Predictor
from cognio.resources import check_budget
from cognio.runtime import select_device


def summary(values):
    return dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                maximum=float(max(values))) if values else None


def case(config, device, mode, train_frames, evaluate_frames, max_seconds, observer=False):
    logical = [0.]
    row = dict(config=asdict(config), parameters=config.parameters, hidden_values=config.hidden_values,
               device=device, stimulus=mode, observer=observer, clock="unpaced, virtual 30 Hz",
               requested_train_frames=train_frames, requested_evaluation_frames=evaluate_frames)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    engine = None
    try:
        started = time.perf_counter()
        engine = Engine(config, device, clock=lambda: logical[0]+.001)
        row["initialization_s"] = time.perf_counter()-started
        row["allocation_guard"] = engine.allocation
        started = time.perf_counter()
        deadline = started + max_seconds
        process = psutil.Process()
        cpu_start = sum(process.cpu_times()[:2])
        peak_rss = process.memory_info().rss
        phases, timings = [], {key: [] for key in ("prediction_ms", "update_ms", "step_ms", "transfer_ms", "observer_ms")}
        sequence = 0
        truncated = False
        initial_bias = engine.model.decoder.bias.detach().clone()
        for name, count, training in (("initial_frozen", evaluate_frames, False),
                                       ("training", train_frames, True),
                                       ("final_frozen", evaluate_frames, False)):
            engine.set_training(training)
            model_errors, baseline_errors = [], []
            before = engine.counts.copy()
            phase_started = time.perf_counter()
            processed = 0
            for _ in range(count):
                if time.perf_counter() >= deadline:
                    truncated = True
                    break
                logical[0] = sequence/30
                previous_updates = engine.counts["updates"]
                engine.step(Frame(sequence, logical[0], synthetic(sequence, mode)), observe=observer)
                if engine.fault:
                    raise RuntimeError(engine.fault)
                if sequence >= 10:
                    for key in timings:
                        if key != "update_ms" or engine.counts["updates"] > previous_updates:
                            timings[key].append(getattr(engine, key))
                if engine.last_result and engine.last_result["target"] == sequence:
                    model_errors.append(engine.last_result["mse"])
                    baseline_errors.append(engine.last_result["baseline_mse"])
                peak_rss = max(peak_rss, process.memory_info().rss)
                sequence += 1
                processed += 1
            phases.append(dict(phase=name, frames=processed, pairs=len(model_errors),
                               wall_s=time.perf_counter()-phase_started,
                               updates=engine.counts["updates"]-before["updates"],
                               mse=float(np.mean(model_errors)) if model_errors else None,
                               persistence_mse=float(np.mean(baseline_errors)) if baseline_errors else None))
            if truncated:
                break
        elapsed = time.perf_counter()-started
        row.update(status="time_limit" if truncated else "complete", phases=phases,
                   wall_s=elapsed, predictions_per_s=engine.counts["predictions"]/elapsed,
                   updates_per_s=engine.counts["updates"]/elapsed, counts=engine.counts.copy(),
                   timings={k: summary(v) for k,v in timings.items()},
                   cpu_percent=100*(sum(process.cpu_times()[:2])-cpu_start)/elapsed,
                   peak_rss_mib=peak_rss/1024**2,
                   decoder_bias_changed=not torch.equal(initial_bias, engine.model.decoder.bias),
                   pending=len(engine.pending), history=len(engine.history),
                   vram_mib=torch.cuda.memory_allocated()/1024**2 if device == "cuda" else 0,
                   peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else 0)
        if phases[-1]["phase"] == "final_frozen" and phases[-1]["pairs"]:
            row["beats_persistence_frozen"] = phases[-1]["mse"] < phases[-1]["persistence_mse"]
        with tempfile.TemporaryDirectory(prefix="cognio-capacity-") as directory:
            path = Path(directory)/"state.pt"
            engine.save(path)
            row["checkpoint_bytes"] = path.stat().st_size
            row["checkpoint_limit_bytes"] = config.checkpoint_bytes
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if engine is not None:
            row["counts"] = engine.counts.copy()
    finally:
        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return row


def microbench(config, device, batch, iterations=12):
    """Batch/transfer cost only; batch >1 does not change the live online contract."""
    check_budget(config, device)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed)
        model = Predictor(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    frames = np.stack([synthetic(i, "moving") for i in range(batch)])[:, None]
    hidden = torch.zeros((batch, config.channels, config.grid, config.grid), device=device)
    transfer, updates = [], []
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    for i in range(iterations+3):
        sync()
        start = time.perf_counter()
        data = torch.from_numpy(frames).to(device)
        sync()
        moved = 1000*(time.perf_counter()-start)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(data, hidden)
        prediction.square().mean().backward()
        optimizer.step()
        sync()
        trained = 1000*(time.perf_counter()-start)
        if i >= 3:
            transfer.append(moved)
            updates.append(trained)
    return dict(config=asdict(config), device=device, batch=batch, iterations=iterations,
                transfer_ms=summary(transfer), forward_backward_adam_ms=summary(updates),
                samples_per_s=batch/(np.mean(updates)/1000),
                scope="isolated zero-target MSE throughput; excludes online pairing, rollback, guards and observer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--configs", nargs="+", default=["32x8", "128x16", "256x16", "512x16", "1024x8"])
    parser.add_argument("--stimuli", nargs="+", choices=("static", "moving", "shape", "temporal"), default=["moving"])
    parser.add_argument("--train-frames", type=int, default=80)
    parser.add_argument("--evaluation-frames", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=45)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_frames, args.evaluation_frames) < 6 or not 0 < args.max_seconds <= 300:
        parser.error("use at least six frames per phase and a positive per-case limit <=300 seconds")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device, notice = select_device(args.compute)
    report = dict(torch=str(torch.__version__), cuda_runtime=torch.version.cuda,
                  cudnn=torch.backends.cudnn.version(), device=device, notice=notice,
                  threads=args.threads, precision="float32, TF32 off", results=[], microbench=[])
    for text in args.configs:
        channels, grid = map(int, text.split("x"))
        config = Config(seed=args.seed, channels=channels, grid=grid)
        for mode in args.stimuli:
            row = case(config, device, mode, args.train_frames, args.evaluation_frames, args.max_seconds, args.observer)
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps({"config": text, "stimulus": mode, "status": row["status"], "error": row.get("error")}), flush=True)
    if args.microbench:
        for config in (Config(), Config(channels=512, grid=16)):
            for batch in (1, 4):
                report["microbench"].append(microbench(config, device, batch))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()

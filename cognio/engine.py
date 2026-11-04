"""Causal prequential scoring and bounded, one-step recurrent online training."""

from collections import deque
from dataclasses import asdict, dataclass
import copy
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import psutil
import torch

from .inputs import Frame
from .model import Config, Predictor
from .resources import check_budget


@dataclass
class Pending:
    source: Frame
    issued: float
    previous: torch.Tensor
    prediction: torch.Tensor


def finite_tensor(tensor, bound=1e3):
    tensor = tensor.detach()
    return bool(torch.isfinite(tensor).all()) and float(tensor.abs().max()) <= bound


def finite_tensors(tensors, bound=None):
    """One device synchronization for all tensors, rather than one per parameter."""
    checks = []
    for tensor in tensors:
        tensor = tensor.detach()
        checks.append(torch.isfinite(tensor).all())
        if bound is not None:
            checks.append(tensor.abs().max() <= bound)
    return bool(torch.stack(checks).all()) if checks else True


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


class Engine:
    def __init__(self, config=Config(), device="cpu", clock=time.monotonic, memory_budget_mib=None):
        self.config, self.device, self.clock = config, torch.device(device), clock
        self.memory_budget_mib = memory_budget_mib
        self.allocation = check_budget(config, self.device, memory_budget_mib)
        # Only the CPU generator is used by initialization; CUDA RNG is untouched.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(config.seed)
            self.model = Predictor(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.hidden = torch.zeros(config.hidden_shape, device=self.device)
        self.pending = {}
        self.history = deque(maxlen=300)
        self.training = True
        self.fault = ""
        self.origin = f"NEW random initialization · seed {config.seed}"
        self.counts = dict(predictions=0, scores=0, updates=0, missed_targets=0, late_targets=0,
                           skipped_frames=0, cleared_targets=0)
        self.last_sequence = None
        self.last_timestamp = None
        self.last_result = None
        self.last_training_loss = None
        self.prediction_ms = self.update_ms = 0.0
        self.observer_ms = self.transfer_ms = self.step_ms = 0.0

    def set_training(self, enabled):
        if self.fault and enabled:
            raise ValueError("Model fault: reset or restore before enabling training")
        if self.training != bool(enabled):
            self.training = bool(enabled)
            # Phase comparisons never include predictions from the previous mode.
            self.flush()

    def flush(self):
        self.counts["cleared_targets"] += len(self.pending)
        self.pending.clear()
        self.history.clear()
        self.last_result = None

    def _update(self, pending, target):
        """Recompute old context using current weights; never replace the scored prediction.

        The stored previous state is detached: one recurrent step of gradients,
        no graph surviving an optimizer update, no target input to the predictor.
        """
        started = time.perf_counter()
        before_model = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.zero_grad(set_to_none=True)
            old_input = torch.from_numpy(pending.source.values).to(self.device)[None, None]
            prediction, _ = self.model(old_input, pending.previous)
            loss = (prediction - target).square().mean()
            if not finite_tensor(loss, 1):
                raise FloatingPointError("non-finite or invalid training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite gradients")
            self.optimizer.step()
            if not finite_tensors(self.model.parameters(), 1e3):
                raise FloatingPointError("invalid or excessive learned weights")
            # Adam's scalar step can reside on CPU even when moments live on CUDA.
            moments = [state[key] for state in self.optimizer.state.values() for key in ("exp_avg", "exp_avg_sq")]
            if not finite_tensors(moments):
                raise FloatingPointError("non-finite optimizer state")
            if not finite_tensors([state["step"] for state in self.optimizer.state.values()]):
                raise FloatingPointError("non-finite optimizer step")
            self.last_training_loss = float(loss.detach())
            self.counts["updates"] += 1
        except Exception:
            self.model.load_state_dict(before_model)
            self.optimizer.load_state_dict(before_optimizer)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.update_ms = 1000 * (time.perf_counter() - started)

    def step(self, frame, observe=True, channel_start=0):
        step_started = time.perf_counter()
        if (type(frame.sequence) is not int or frame.sequence < 0 or
                not math.isfinite(frame.timestamp) or frame.values.shape != (128, 128) or
                frame.values.dtype != np.float32 or not np.isfinite(frame.values).all() or
                np.min(frame.values) < 0 or np.max(frame.values) > 1):
            raise ValueError("invalid frame: expected monotonic sequence/time and finite float32 [0,1] 128x128")
        if self.last_sequence is not None and (frame.sequence <= self.last_sequence or frame.timestamp <= self.last_timestamp):
            raise ValueError("frame sequence and timestamps must strictly increase")
        if self.last_sequence is not None:
            self.counts["skipped_frames"] += frame.sequence - self.last_sequence - 1
        self.last_sequence, self.last_timestamp = frame.sequence, frame.timestamp
        if self.fault:
            return None
        # Expire absent targets; never substitute the next delivered frame.
        for sequence in list(self.pending):
            if sequence < frame.sequence:
                del self.pending[sequence]
                self.counts["missed_targets"] += 1
        pending = self.pending.pop(frame.sequence, None)
        try:
            transfer_started = time.perf_counter()
            target = torch.from_numpy(frame.values).to(self.device)[None, None]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.transfer_ms = 1000 * (time.perf_counter() - transfer_started)
            if pending is not None:
                if pending.issued >= frame.timestamp:
                    self.counts["late_targets"] += 1
                else:
                    error = (pending.prediction - target).square()
                    baseline = torch.from_numpy(pending.source.values).to(self.device)[None, None]
                    result = dict(source=pending.source.sequence, target=frame.sequence,
                                  source_time=pending.source.timestamp, target_time=frame.timestamp,
                                  issued=pending.issued, mse=float(error.mean()),
                                  baseline_mse=float((baseline - target).square().mean()),
                                  horizon_ms=1000 * (frame.timestamp - pending.source.timestamp),
                                  lead_ms=1000 * (frame.timestamp - pending.issued),
                                  error=error[0, 0].cpu().numpy().copy() if observe else None)
                    self.last_result = result
                    self.history.append((result["mse"], result["baseline_mse"]))
                    self.counts["scores"] += 1
                    if self.training:
                        self._update(pending, target)
            previous = self.hidden.detach().clone()
            started = time.perf_counter()
            with torch.no_grad():
                prediction, hidden = self.model(target, previous)
            if not finite_tensors((prediction, hidden), 1):
                raise FloatingPointError("non-finite prediction or hidden state")
            # The finite check synchronizes forward completion before issuance.
            # Observation is downstream: rendering cannot change this prediction.
            issued = self.clock()
            self.prediction_ms = 1000 * (time.perf_counter() - started)
            self.hidden = hidden.detach()
            owned = Frame(frame.sequence, frame.timestamp, frame.values.copy())
            target_sequence = frame.sequence + self.config.horizon
            self.pending[target_sequence] = Pending(owned, issued, previous, prediction.detach())
            assert len(self.pending) <= self.config.horizon
            self.counts["predictions"] += 1
            result = dict(source=frame.sequence, target=target_sequence, issued=issued,
                          input_time=frame.timestamp, input_sequence=frame.sequence)
            self.observer_ms = 0.0
            if observe:
                observer_started = time.perf_counter()
                channel_start = max(0, min(channel_start, self.config.channels - 1))
                result.update(input=owned.values, prediction=prediction[0, 0].cpu().numpy().copy(),
                              hidden=hidden[0, channel_start:channel_start+32].cpu().numpy().copy(),
                              channel_start=channel_start)
                self.observer_ms = 1000 * (time.perf_counter() - observer_started)
            self.step_ms = 1000 * (time.perf_counter() - step_started)
            return result
        except (FloatingPointError, RuntimeError) as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            self.training = False
            self.counts["cleared_targets"] += len(self.pending)
            self.pending.clear()
            return None

    def state(self):
        """No sensory inputs, targets, pending predictions or metric maps are serialized."""
        return dict(schema=2, architecture="patch-spatial-rnn-v2", config=asdict(self.config),
                    model=cpu_copy(self.model.state_dict()),
                    optimizer=cpu_copy(self.optimizer.state_dict()),
                    hidden=cpu_copy(self.hidden), counts=dict(self.counts),
                    training=self.training, torch_version=str(torch.__version__))

    def save(self, path):
        path = Path(path)
        # Atomic replacement keeps a previous checkpoint recoverable on write failure.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                temporary = stream.name
                data = self.state()
                data["counts"]["cleared_targets"] += len(self.pending)
                torch.save(data, stream)
                if stream.tell() > self.config.checkpoint_bytes:
                    raise ValueError("serialized checkpoint exceeds configuration-sized limit")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.flush()  # Both running and restored streams restart the five-frame warm-up.

    @classmethod
    def load(cls, path, device="cpu", clock=time.monotonic, expected_config=None, memory_budget_mib=None):
        path = Path(path)
        # Bound the archive before unpickling metadata; mmap avoids eager storage copies.
        maximum = int(psutil.virtual_memory().available * .25)
        if path.stat().st_size > maximum:
            raise ValueError("checkpoint exceeds available host-memory size limit")
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (len(entries) > 128 or any(e.compress_type != zipfile.ZIP_STORED for e in entries) or
                        sum(e.file_size for e in entries) > maximum):
                    raise ValueError("unsupported or oversized checkpoint archive")
        except zipfile.BadZipFile as exc:
            raise ValueError("invalid checkpoint archive") from exc
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected = {"schema", "architecture", "config", "model", "optimizer", "hidden", "counts", "training", "torch_version"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("unsupported checkpoint schema or architecture")
        legacy = data["schema"] == 1 and data["architecture"] == "patch16-spatial-rnn32-v1"
        if legacy:
            if set(data["config"]) != {"seed", "learning_rate", "horizon"}:
                raise ValueError("invalid Prototype 1 configuration")
            data["counts"] = {**data["counts"], "skipped_frames": 0, "cleared_targets": 0}
        elif data["schema"] != 2 or data["architecture"] != "patch-spatial-rnn-v2":
            raise ValueError("unsupported checkpoint schema or architecture")
        elif set(data["config"]) != {"seed", "learning_rate", "horizon", "channels", "grid"}:
            raise ValueError("incomplete architecture configuration")
        config = Config(**data["config"])
        if path.stat().st_size > config.checkpoint_bytes:
            raise ValueError("checkpoint exceeds configuration-sized limit")
        if expected_config is not None and (config.channels, config.grid) != (expected_config.channels, expected_config.grid):
            raise ValueError(f"checkpoint architecture {config.channels}×{config.grid} differs from selected "
                             f"{expected_config.channels}×{expected_config.grid}; launch a matching model explicitly")
        candidate = cls(config, device, clock, memory_budget_mib)
        expected_model = candidate.model.state_dict()
        if set(data["model"]) != set(expected_model):
            raise ValueError("model keys differ")
        for key, value in data["model"].items():
            if not torch.is_tensor(value) or value.shape != expected_model[key].shape or value.dtype != torch.float32 or not finite_tensor(value):
                raise ValueError(f"invalid model tensor {key}")
        if not torch.equal(data["model"]["positions"], expected_model["positions"].cpu()):
            raise ValueError("fixed spatial coordinates differ")
        hidden = data["hidden"]
        if not torch.is_tensor(hidden) or hidden.shape != config.hidden_shape or hidden.dtype != torch.float32 or not finite_tensor(hidden, 1):
            raise ValueError("invalid recurrent state")
        counts = data["counts"]
        if not isinstance(counts, dict) or set(counts) != set(candidate.counts) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("invalid counters")
        if counts["updates"] > counts["scores"] or counts["scores"] > counts["predictions"] or type(data["training"]) is not bool:
            raise ValueError("inconsistent progress")
        opt = data["optimizer"]
        if not isinstance(opt, dict) or set(opt) != {"state", "param_groups"} or opt["param_groups"] != candidate.optimizer.state_dict()["param_groups"]:
            raise ValueError("optimizer configuration differs")
        params = list(candidate.model.parameters())
        required = set(range(len(params))) if counts["updates"] else set()
        if set(opt["state"]) != required:
            raise ValueError("optimizer parameters differ")
        for index, state in opt["state"].items():
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam state")
            for name, tensor in state.items():
                shape = torch.Size([]) if name == "step" else params[index].shape
                if not torch.is_tensor(tensor) or tensor.shape != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("invalid Adam tensor")
            if float(state["step"]) != counts["updates"] or bool((state["exp_avg_sq"] < 0).any()):
                raise ValueError("invalid Adam counters/moments")
        candidate.model.load_state_dict(data["model"])
        candidate.optimizer.load_state_dict(opt)
        candidate.hidden = hidden.to(device, copy=True)
        candidate.counts = counts.copy()
        candidate.training = data["training"]
        candidate.origin = f"RESTORED · {path.name} · seed {candidate.config.seed}"
        return candidate

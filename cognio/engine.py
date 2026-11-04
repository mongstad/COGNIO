"""Causal prequential scoring and bounded, one-step recurrent online training."""

from collections import deque
from dataclasses import asdict, dataclass
import copy
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .inputs import Frame
from .model import Config, Predictor


@dataclass
class Pending:
    source: Frame
    issued: float
    previous: torch.Tensor
    prediction: torch.Tensor


def finite_tensor(tensor, bound=1e3):
    tensor = tensor.detach()
    return bool(torch.isfinite(tensor).all()) and float(tensor.abs().max()) <= bound


class Engine:
    def __init__(self, config=Config(), device="cpu", clock=time.monotonic):
        self.config, self.device, self.clock = config, torch.device(device), clock
        # Local initialization does not perturb the application's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = Predictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.hidden = torch.zeros((1, 32, 8, 8), device=self.device)
        self.pending = {}
        self.history = deque(maxlen=300)
        self.training = True
        self.fault = ""
        self.origin = f"NEW random initialization · seed {config.seed}"
        self.counts = dict(predictions=0, scores=0, updates=0, missed_targets=0, late_targets=0)
        self.last_sequence = None
        self.last_timestamp = None
        self.last_result = None
        self.last_training_loss = None
        self.prediction_ms = self.update_ms = 0.0

    def set_training(self, enabled):
        if self.fault and enabled:
            raise ValueError("Model fault: reset or restore before enabling training")
        if self.training != bool(enabled):
            self.training = bool(enabled)
            # Phase comparisons never include predictions from the previous mode.
            self.flush()

    def flush(self):
        self.pending.clear()
        self.history.clear()
        self.last_result = None

    def _update(self, pending, target):
        """Recompute old context using current weights; never replace the scored prediction.

        The stored previous state is detached: one recurrent step of gradients,
        no graph surviving an optimizer update, no target input to the predictor.
        """
        before_model = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        started = self.clock()
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
            if not all(finite_tensor(p) for p in self.model.parameters()):
                raise FloatingPointError("invalid or excessive learned weights")
            for state in self.optimizer.state.values():
                if not all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)):
                    raise FloatingPointError("non-finite optimizer state")
            self.last_training_loss = float(loss.detach())
            self.counts["updates"] += 1
        except Exception:
            self.model.load_state_dict(before_model)
            self.optimizer.load_state_dict(before_optimizer)
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.update_ms = 1000 * (self.clock() - started)

    def step(self, frame):
        if (type(frame.sequence) is not int or frame.sequence < 0 or
                not math.isfinite(frame.timestamp) or frame.values.shape != (128, 128) or
                frame.values.dtype != np.float32 or not np.isfinite(frame.values).all() or
                np.min(frame.values) < 0 or np.max(frame.values) > 1):
            raise ValueError("invalid frame: expected monotonic sequence/time and finite float32 [0,1] 128x128")
        if self.last_sequence is not None and (frame.sequence <= self.last_sequence or frame.timestamp <= self.last_timestamp):
            raise ValueError("frame sequence and timestamps must strictly increase")
        self.last_sequence, self.last_timestamp = frame.sequence, frame.timestamp
        if self.fault:
            return None
        # Expire absent targets; never substitute the next delivered frame.
        for sequence in list(self.pending):
            if sequence < frame.sequence:
                del self.pending[sequence]
                self.counts["missed_targets"] += 1
        target = torch.from_numpy(frame.values).to(self.device)[None, None]
        pending = self.pending.pop(frame.sequence, None)
        try:
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
                                  error=error[0, 0].cpu().numpy().copy())
                    self.last_result = result
                    self.history.append((result["mse"], result["baseline_mse"]))
                    self.counts["scores"] += 1
                    if self.training:
                        self._update(pending, target)
            previous = self.hidden.detach().clone()
            started = self.clock()
            with torch.no_grad():
                prediction, hidden = self.model(target, previous)
            if not finite_tensor(prediction, 1) or not finite_tensor(hidden, 1):
                raise FloatingPointError("non-finite prediction or hidden state")
            # CPU copies synchronize CUDA before issuance: target must be captured later.
            values = prediction[0, 0].cpu().numpy().copy()
            hidden_values = hidden[0].cpu().numpy().copy()
            issued = self.clock()
            self.prediction_ms = 1000 * (issued - started)
            self.hidden = hidden.detach()
            owned = Frame(frame.sequence, frame.timestamp, frame.values.copy())
            target_sequence = frame.sequence + self.config.horizon
            self.pending[target_sequence] = Pending(owned, issued, previous, prediction.detach())
            assert len(self.pending) <= self.config.horizon
            self.counts["predictions"] += 1
            return dict(input=owned.values, prediction=values, hidden=hidden_values,
                        source=frame.sequence, target=target_sequence, issued=issued,
                        input_time=frame.timestamp)
        except (FloatingPointError, RuntimeError) as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            self.training = False
            self.pending.clear()
            return None

    def state(self):
        """No sensory inputs, targets, pending predictions or metric maps are serialized."""
        return dict(schema=1, architecture="patch16-spatial-rnn32-v1", config=asdict(self.config),
                    model=copy.deepcopy(self.model.state_dict()),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    hidden=self.hidden.detach().clone(), counts=dict(self.counts),
                    training=self.training, torch_version=str(torch.__version__))

    def save(self, path):
        path = Path(path)
        # Atomic replacement keeps a previous checkpoint recoverable on write failure.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
                temporary = stream.name
                torch.save(self.state(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.flush()  # Both running and restored streams restart the five-frame warm-up.

    @classmethod
    def load(cls, path, device="cpu", clock=time.monotonic):
        path = Path(path)
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("checkpoint exceeds Prototype 1 size limit (16 MiB)")
        data = torch.load(path, map_location="cpu", weights_only=True)
        expected = {"schema", "architecture", "config", "model", "optimizer", "hidden", "counts", "training", "torch_version"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != 1 or data["architecture"] != "patch16-spatial-rnn32-v1":
            raise ValueError("unsupported checkpoint schema or architecture")
        candidate = cls(Config(**data["config"]), device, clock)
        expected_model = candidate.model.state_dict()
        if set(data["model"]) != set(expected_model):
            raise ValueError("model keys differ")
        for key, value in data["model"].items():
            if not torch.is_tensor(value) or value.shape != expected_model[key].shape or value.dtype != torch.float32 or not finite_tensor(value):
                raise ValueError(f"invalid model tensor {key}")
        if not torch.equal(data["model"]["positions"], expected_model["positions"].cpu()):
            raise ValueError("fixed spatial coordinates differ")
        hidden = data["hidden"]
        if not torch.is_tensor(hidden) or hidden.shape != (1, 32, 8, 8) or hidden.dtype != torch.float32 or not finite_tensor(hidden, 1):
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
        candidate.hidden = hidden.to(device)
        candidate.counts = counts.copy()
        candidate.training = data["training"]
        candidate.origin = f"RESTORED · {path.name} · seed {candidate.config.seed}"
        return candidate

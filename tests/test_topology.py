"""Boundary connectivity, immutable architecture, and retained causal learning."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cognio.diagnostics import state_digest
from cognio.engine import Engine
from cognio.inputs import Frame, synthetic
from cognio.model import Config, Predictor
from cognio.runtime import Runtime, select_device


@pytest.mark.parametrize("topology", ["standard", "toroidal"])
def test_edges_corners_and_interior_against_indexed_stencil(topology):
    config = Config(channels=1, topology=topology)
    model = Predictor(config)
    joint = torch.zeros(1, 4, 8, 8)
    joint[0, 0] = torch.arange(64).reshape(8, 8)
    # Asymmetric coefficients reveal reflection/transposition as well as wrapping.
    with torch.no_grad():
        model.recurrent.weight.zero_()
        model.recurrent.weight[0, 0] = torch.arange(1, 10).reshape(3, 3)
        model.recurrent.bias.zero_()
    expected = torch.zeros(8, 8)
    for y in range(8):
        for x in range(8):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if topology == "toroidal" or (0 <= y+dy < 8 and 0 <= x+dx < 8):
                        expected[y, x] += (3*(dy+1)+dx+2)*joint[0, 0, (y+dy)%8, (x+dx)%8]
    assert torch.equal(model.recurrent(joint)[0, 0], expected)


def test_seeded_standard_matches_accepted_forward_and_toroidal_interior():
    standard = Engine().model
    toroidal = Engine(Config(topology="toroidal")).model
    assert state_digest(standard.state_dict()) == state_digest(toroidal.state_dict())
    assert sum(p.numel() for p in standard.parameters()) == sum(p.numel() for p in toroidal.parameters()) == 35712
    x = torch.linspace(0, 1, 128**2).reshape(1, 1, 128, 128)
    previous = torch.linspace(-.5, .5, 32*8*8).reshape(1, 32, 8, 8)
    a, ah = standard(x, previous)
    b, bh = toroidal(x, previous)
    # The accepted recurrence explicitly uses zero padding. Same operation order.
    from cognio.model import patchify, unpatchify
    encoded = torch.tanh(standard.encoder(patchify(x)))
    joint = torch.cat((encoded, previous, standard.positions), 1)
    expected = .5*previous + .5*torch.tanh(torch.nn.functional.conv2d(
        joint, standard.recurrent.weight, standard.recurrent.bias, padding=1))
    assert torch.equal(ah, expected)
    assert torch.equal(a, unpatchify(torch.sigmoid(standard.decoder(expected))))
    assert torch.equal(ah[:, :, 1:-1, 1:-1], bh[:, :, 1:-1, 1:-1])
    assert torch.equal(a[:, :, 16:-16, 16:-16], b[:, :, 16:-16, 16:-16])
    assert not torch.equal(ah, bh)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("topology", ["standard", "toroidal"])
@pytest.mark.parametrize("mode", ["y", "ycbcr"])
def test_updates_freeze_and_checkpoint_continuation_at_grid64(tmp_path, device, topology, mode):
    now = [0.]
    config = Config(channels=24, grid=64, input_mode=mode, topology=topology)
    a = Engine(config, device, clock=lambda: now[0]+.001)
    initial = state_digest(a.model.state_dict())
    for i in range(12):
        now[0] = i/30
        a.step(Frame(i, now[0], synthetic(i, "moving", mode)), observe=False)
    assert not a.fault and a.counts["updates"] == a.counts["scores"] == 7
    assert a.last_result["source"] + 5 == a.last_result["target"] == 11
    assert len(a.pending) == 5 and all(p.previous.grad_fn is None for p in a.pending.values())
    assert state_digest(a.model.state_dict()) != initial
    a.set_training(False)
    digest = state_digest((a.model.state_dict(), a.optimizer.state_dict()))
    path = tmp_path/"state.pt"
    a.save(path)
    b = Engine.load(path, device, clock=lambda: now[0]+.001, expected_config=config)
    for i in range(12, 20):
        now[0] = i/30
        frame = Frame(i, now[0], synthetic(i, "moving", mode))
        a.step(frame, observe=False)
        b.step(frame, observe=True)
    assert torch.equal(a.hidden, b.hidden)
    assert digest == state_digest((a.model.state_dict(), a.optimizer.state_dict()))
    assert a.counts == b.counts and a.last_result["mse"] == b.last_result["mse"]
    runtime = Runtime(a, observer=False)
    wrong = replace(config, topology="toroidal" if topology == "standard" else "standard")
    before = state_digest((a.model.state_dict(), a.optimizer.state_dict(), a.hidden))
    with patch("cognio.engine.Predictor") as constructor:
        with pytest.raises(ValueError, match="topology"):
            Engine.load(path, device, expected_config=wrong)
        constructor.assert_not_called()
    other = tmp_path/"other.pt"
    Engine(wrong).save(other)
    with pytest.raises(ValueError, match="topology"):
        runtime._command("load", other)
    assert runtime.engine is a and before == state_digest((a.model.state_dict(), a.optimizer.state_dict(), a.hidden))


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_standard_schema_paths_and_model_free_inspection(tmp_path, schema):
    engine = Engine()
    state = engine.state()
    if schema < 4:
        state["schema"] = schema
        state["architecture"] = "patch16-spatial-rnn32-v1" if schema == 1 else f"patch-spatial-rnn-v{schema}"
        state["config"].pop("topology")
    if schema < 3:
        state.pop("sensory")
        state["config"].pop("input_mode")
    if schema == 1:
        state["config"].pop("channels")
        state["config"].pop("grid")
        state["counts"].pop("skipped_frames")
        state["counts"].pop("cleared_targets")
    path = tmp_path/f"schema{schema}.pt"
    torch.save(state, path)
    original = path.read_bytes()
    with patch("cognio.engine.Predictor") as constructor:
        metadata = Engine.inspect_checkpoint(path)
        assert metadata == dict(config=Config(), schema=schema)
        constructor.assert_not_called()
    loaded = Engine.load(path)
    assert loaded.config.topology == "standard"
    assert state_digest(loaded.model.state_dict()) == state_digest(engine.model.state_dict())
    assert path.read_bytes() == original


def test_topology_metadata_required_and_resource_refusal_preserves_file(tmp_path):
    path = tmp_path/"state.pt"
    state = Engine(Config(topology="toroidal")).state()
    state["config"].pop("topology")
    torch.save(state, path)
    with pytest.raises(ValueError, match="topology"):
        Engine.load(path)
    Engine().save(path)
    before = path.read_bytes()
    with patch("cognio.engine.Predictor") as constructor:
        with pytest.raises(ValueError, match="budget"):
            Engine.load(path, memory_budget_mib=1)
        constructor.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_probe_checks_exact_configuration_and_resource_limit():
    config = Config(channels=48, grid=64, input_mode="ycbcr", topology="toroidal")
    device, notice = select_device("cuda", config)
    assert device == "cuda" and "48×64 ycbcr toroidal" in notice
    with patch("cognio.engine.Predictor") as constructor:
        with pytest.raises(ValueError, match="budget"):
            select_device("cuda", config, memory_budget_mib=1)
        constructor.assert_not_called()


def test_headless_cli_topology_default_and_checkpoint_conflict(tmp_path):
    import json
    import subprocess
    import sys
    base = [sys.executable, "-m", "cognio", "--headless", "--compute", "cpu", "--seconds", ".3"]
    standard = subprocess.run(base, capture_output=True, text=True, timeout=20)
    assert standard.returncode == 0, standard.stderr
    config = json.loads(standard.stdout.splitlines()[-1])["config"]
    assert config == dict(seed=7, learning_rate=.001, horizon=5, channels=32, grid=8, input_mode="y", topology="standard")
    path = tmp_path/"toroidal.pt"
    toroidal = subprocess.run(base + ["--topology", "toroidal", "--save-on-exit", str(path)],
                              capture_output=True, text=True, timeout=20)
    assert toroidal.returncode == 0, toroidal.stderr
    records = [json.loads(line) for line in toroidal.stdout.splitlines()]
    assert next(record for record in reversed(records) if "config" in record)["config"]["topology"] == "toroidal"
    before = path.read_bytes()
    conflict = subprocess.run(base + ["--checkpoint", str(path), "--topology", "standard"],
                              capture_output=True, text=True, timeout=20)
    assert conflict.returncode == 2 and "checkpoint topology differs" in conflict.stderr
    assert path.read_bytes() == before

import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cognio.engine import Engine
from cognio.headless import run
from cognio.inputs import Frame, synthetic
from cognio.model import Config, Predictor, patchify, unpatchify
from cognio.resources import check_budget
from cognio.runtime import Runtime


@pytest.mark.parametrize("grid", [8, 16, 32])
def test_general_patch_order(grid):
    x = torch.arange(128**2, dtype=torch.float32).reshape(1, 1, 128, 128)
    patches = patchify(x, grid)
    p = 128//grid
    assert patches.shape == (1, p*p, grid, grid)
    assert torch.equal(unpatchify(patches), x)
    assert torch.equal(patches[0, :, 2, 3].reshape(p, p), x[0, 0, 2*p:3*p, 3*p:4*p])


@pytest.mark.parametrize("channels", [32, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("grid", [8, 16, 32])
def test_all_requested_architectures_on_meta_device(channels, grid):
    config = Config(channels=channels, grid=grid)
    # Shape/parameter validation without allocating enormous untested CPU models.
    with torch.device("meta"):
        model = Predictor(config)
        prediction, hidden = model(torch.zeros(1, 1, 128, 128), torch.zeros(config.hidden_shape))
    assert sum(p.numel() for p in model.parameters()) == config.parameters
    assert prediction.shape == (1, 1, 128, 128)
    assert hidden.shape == config.hidden_shape
    assert hidden.numel() == config.hidden_values


def test_baseline_parameterization_is_identical():
    config = Config()
    assert config.parameters == 35712 and config.hidden_values == 2048
    assert Config(channels=512, grid=16).hidden_values == 131072
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        encoder = torch.nn.Conv2d(256, 32, 1)
        recurrent = torch.nn.Conv2d(66, 32, 3, padding=1)
        decoder = torch.nn.Conv2d(32, 256, 1)
    model = Engine().model
    for name, old in (("encoder", encoder), ("recurrent", recurrent), ("decoder", decoder)):
        assert torch.equal(getattr(model, name).weight, old.weight)
        assert torch.equal(getattr(model, name).bias, old.bias)


def test_headless_observer_pages_have_identical_learning():
    now = [0.]
    config = Config(channels=64, grid=16)
    a = Engine(config, clock=lambda: now[0]+.001)
    b = Engine(config, clock=lambda: now[0]+.001)
    for i in range(15):
        now[0] = i/30
        frame = Frame(i, now[0], synthetic(i, "moving"))
        plain = a.step(frame, observe=False)
        shown = b.step(frame, observe=True, channel_start=32)
        assert "hidden" not in plain and "prediction" not in plain and "input" not in plain
        assert shown["hidden"].shape == (32, 16, 16)
        assert torch.equal(a.hidden, b.hidden)
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key])
    assert a.counts == b.counts
    assert a.last_result["error"] is None
    assert a.last_result["mse"] == b.last_result["mse"]


def test_gui_large_page_preserves_channel_positions():
    from cognio.gui import activation_tiles
    hidden = np.arange(32*16*16).reshape(32, 16, 16)
    tiles = activation_tiles(hidden)
    for index in range(32):
        row, column = divmod(index, 8)
        np.testing.assert_array_equal(tiles[row*16:(row+1)*16, column*16:(column+1)*16], hidden[index])
    partial = activation_tiles(hidden[:3])
    assert partial.shape == (16, 48)


def test_live_restore_rejects_different_dimensions(tmp_path):
    path = tmp_path/"different.pt"
    Engine(Config(channels=128, grid=16)).save(path)
    runtime = Runtime(Engine(), observer=False)
    before = runtime.engine
    with pytest.raises(ValueError, match="architecture"):
        runtime._command("load", str(path))
    assert runtime.engine is before


def test_resource_guard_precedes_allocation():
    with patch("cognio.engine.Predictor") as constructor:
        with pytest.raises(ValueError, match="budget"):
            Engine(Config(channels=100000))
        constructor.assert_not_called()
    with pytest.raises(ValueError, match="budget"):
        check_budget(Config(), "cpu", 1)
    with pytest.raises(ValueError):
        check_budget(Config(), "cpu", float("nan"))


def test_version1_migration_and_architecture_mismatch(tmp_path):
    old = Engine().state()
    old["schema"], old["architecture"] = 1, "patch16-spatial-rnn32-v1"
    old["config"].pop("channels")
    old["config"].pop("grid")
    old["counts"].pop("skipped_frames")
    old["counts"].pop("cleared_targets")
    path = tmp_path / "old.pt"
    torch.save(old, path)
    restored = Engine.load(path)
    assert restored.config == Config()
    assert restored.state()["schema"] == 2
    with pytest.raises(ValueError, match="architecture"):
        Engine.load(path, expected_config=Config(channels=128, grid=16))


def test_scaled_checkpoint_continuation_and_size_guard(tmp_path):
    now = [0.]
    engine = Engine(Config(channels=128, grid=16), clock=lambda: now[0]+.001)
    for i in range(8):
        now[0] = i/30
        engine.step(Frame(i, now[0], synthetic(i, "shape")), observe=False)
    path = tmp_path / "scaled.pt"
    engine.save(path)
    loaded = Engine.load(path, clock=lambda: now[0]+.001)
    for i in range(8, 16):
        now[0] = i/30
        frame = Frame(i, now[0], synthetic(i, "shape"))
        engine.step(frame, observe=False)
        loaded.step(frame, observe=False)
    assert all(torch.equal(value, loaded.model.state_dict()[key]) for key, value in engine.model.state_dict().items())
    assert torch.equal(engine.hidden, loaded.hidden)
    assert engine.counts == loaded.counts
    assert path.stat().st_size <= engine.config.checkpoint_bytes
    with zipfile.ZipFile(path, "a") as archive:
        prefix = archive.namelist()[0].split("/")[0]
        archive.writestr(prefix+"/padding", bytes(engine.config.checkpoint_bytes))
    with pytest.raises(ValueError, match="configuration-sized"):
        Engine.load(path)


def test_oversized_save_preserves_last_good_file(tmp_path, monkeypatch):
    engine = Engine(clock=lambda: .001)
    engine.step(Frame(0, 0., synthetic(0, "moving")), observe=False)
    path = tmp_path/"good.pt"
    path.write_bytes(b"last good checkpoint")
    monkeypatch.setattr(Config, "checkpoint_bytes", property(lambda self: 1))
    with pytest.raises(ValueError, match="configuration-sized"):
        engine.save(path)
    assert path.read_bytes() == b"last good checkpoint"
    assert len(engine.pending) == 1 and list(tmp_path.iterdir()) == [path]


def test_compressed_archive_rejected_before_tensor_loading(tmp_path):
    path = tmp_path/"compressed.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data.pkl", bytes(1024))
    with patch("cognio.engine.torch.load") as loader:
        with pytest.raises(ValueError, match="archive"):
            Engine.load(path)
        loader.assert_not_called()


def test_runtime_headless_clean_stop_save_and_scalar_only(tmp_path):
    engine = Engine(Config(channels=32, grid=16))
    runtime = Runtime(engine, "moving", observer=False)
    output = io.StringIO()
    assert run(runtime, seconds=.8, interval=.2, save_path=tmp_path/"headless.pt", output=output) == 0
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert 2 <= len(records) <= 7
    assert runtime.latest()["visual"] is None and runtime.engine.observer_ms == 0
    assert runtime.engine.counts["updates"] > 0
    assert not any(thread.is_alive() for thread in runtime.threads)
    assert Engine.load(tmp_path/"headless.pt").config == engine.config


def test_cli_headless_no_qt_import_and_sigterm(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = tmp_path/"signal.pt"
    script = """import sys
class RejectQt:
    def find_spec(self, fullname, *args):
        if fullname.startswith('PySide6'):
            raise AssertionError('headless imported Qt')
sys.meta_path.insert(0, RejectQt())
from cognio.__main__ import main
raise SystemExit(main())
"""
    log = tmp_path/"out.jsonl"
    with log.open("w") as output:
        process = subprocess.Popen([sys.executable, "-c", script, "--headless", "--source", "moving",
                                    "--compute", "cpu", "--save-on-exit", str(checkpoint)],
                                   cwd=root, stdout=output, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+20
            while log.stat().st_size == 0 and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.05)
            assert log.stat().st_size > 0
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert checkpoint.is_file()
    resumed = subprocess.run([sys.executable, "-c", script, "--headless", "--source", "moving",
                              "--compute", "cpu", "--checkpoint", str(checkpoint),
                              "--resume-training", "--seconds", ".5"], cwd=root,
                             capture_output=True, text=True, timeout=20)
    assert resumed.returncode == 0, resumed.stderr
    final = json.loads(resumed.stdout.splitlines()[-1])
    assert final["training"] and final["counts"]["updates"] > 0


def test_slow_processing_keeps_exact_targets_and_bounded_handoffs():
    engine = Engine()
    original = engine.step
    def delayed(frame, **kwargs):
        time.sleep(.19)  # Longer than the nominal horizon, target already captured.
        return original(frame, **kwargs)
    engine.step = delayed
    runtime = Runtime(engine, "moving", observer=False)
    runtime.start()
    try:
        time.sleep(1.3)
    finally:
        runtime.stop()
    assert engine.counts["updates"] == 0
    assert engine.counts["missed_targets"] + engine.counts["late_targets"] > 0
    assert engine.counts["skipped_frames"] > 0 and runtime.overwrites > 0
    assert len(engine.pending) <= 5
    assert runtime.latest()["prediction_fps"] < 10

import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from cognio.diagnostics import (periodicity, prediction_statistics, preview_float,
                                probe_context, state_digest)
from cognio.engine import Engine
from cognio.inputs import Frame, synthetic
from cognio.model import Config
from cognio.runtime import Runtime
from tools.investigate_grid import SCENARIOS, run_case, stimulus


def trained(mode="ycbcr", device="cpu"):
    engine = Engine(Config(input_mode=mode),device=device)
    for index in range(12):
        engine.step(Frame(index,time.monotonic(),synthetic(index,"moving",mode)),observe=False)
    assert engine.counts["updates"] == 7
    return engine


@pytest.mark.parametrize("mode", ["y","ycbcr"])
@pytest.mark.parametrize("device", ["cpu",pytest.param("cuda",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable"))])
def test_reset_only_transient_state_and_restarts_exact_pairing(mode,device):
    engine = trained(mode,device)
    before = engine.state()
    engine.last_media_time, engine.last_segment = 3., 2
    previous_clock = (engine.last_sequence,engine.last_timestamp,engine.last_media_time,engine.last_segment)
    engine.reset_hidden()
    after = engine.state()
    assert state_digest(before["model"]) == state_digest(after["model"])
    assert state_digest(before["optimizer"]) == state_digest(after["optimizer"])
    assert after["training"] == before["training"]
    assert torch.count_nonzero(engine.hidden) == 0
    assert not engine.pending and not engine.history and not engine.component_history and engine.last_result is None
    assert after["counts"] == {**before["counts"],"cleared_targets":before["counts"]["cleared_targets"]+5}
    assert (engine.last_sequence,engine.last_timestamp,engine.last_media_time,engine.last_segment) == previous_clock
    for i in range(12,17):
        engine.step(Frame(i,time.monotonic(),synthetic(i,"moving",mode),3+i/30,i,2))
        assert engine.last_result is None
    engine.step(Frame(17,time.monotonic(),synthetic(17,"moving",mode),3+17/30,17,2))
    assert (engine.last_result["source"],engine.last_result["target"]) == (12,17)
    assert engine.last_result["issued"] < engine.last_result["target_time"]
    engine.fault,engine.training = "preserved fault",False
    engine.reset_hidden()
    assert engine.fault == "preserved fault" and not engine.training


def test_probe_and_statistics_do_not_mutate_state_gradients_or_learning():
    a,b = trained(),trained()
    assert state_digest(a.state()) == state_digest(b.state())
    for index in range(12,20):
        target = synthetic(index,"moving","ycbcr")
        pending = a.pending[index]
        before = state_digest(a.state())
        original = pending.prediction.clone()
        result = probe_context(a,pending.source.values,pending.previous,target)
        assert result["pre_clip_global_norm"] > 0
        assert all(p.grad is None for p in a.model.parameters())
        assert all(not module._forward_hooks for module in a.model.modules())
        assert state_digest(a.state()) == before
        assert torch.equal(original,pending.prediction)
        for engine in (a,b):
            engine.step(Frame(index,time.monotonic(),target))
    assert state_digest(a.state()) == state_digest(b.state())


def test_periodicity_has_known_fourier_energy_and_zero_constant():
    y,x = np.mgrid[:128,:128]
    wave = .5+.2*np.cos(2*np.pi*x/4)
    stats = periodicity(wave,32)
    assert stats["phase_rms"] == pytest.approx(.2/np.sqrt(2))
    assert stats["lattice_energy_fraction"] == pytest.approx(1)
    assert stats["fundamental_axis_energy_fraction"] == pytest.approx(1)
    assert periodicity(np.ones((128,128)),32)["lattice_energy_fraction"] == 0
    random = np.random.default_rng(1).random((128,128))
    spectrum = np.abs(np.fft.fft2(random-random.mean()))**2
    lattice = spectrum[::32,::32].sum()-spectrum[0,0]
    assert periodicity(random,32)["lattice_energy_fraction"] == pytest.approx(lattice/spectrum.sum())
    for values in (np.zeros((4,4)),np.full((128,128),np.nan)):
        with pytest.raises(ValueError):
            periodicity(values,32)


def test_finite_decoder_bias_can_form_grid_before_display():
    engine = Engine(Config(channels=8,grid=32,input_mode="ycbcr"))
    with torch.no_grad():
        engine.model.decoder.weight.zero_()
        engine.model.decoder.bias.zero_()
        # One dark luminance subpixel repeats in every 4x4 decoded patch.
        engine.model.decoder.bias[0] = -20
    result = engine.step(Frame(0,time.monotonic(),np.zeros((3,128,128),np.float32)))
    stats = prediction_statistics(result["prediction"],32,"ycbcr")
    assert stats["model"]["Y"]["periodicity"]["lattice_energy_fraction"] == pytest.approx(1)
    assert stats["model"]["Y"]["sigmoid_low_share"] == pytest.approx(1/16)
    assert not engine.fault
    assert result["prediction"][0,0,0] < 1e-6 and result["prediction"][0,0,1] == .5


def test_diagnostic_rgb_reference_matches_actual_gui_conversion():
    from cognio.gui import ycbcr_rgb
    values = np.random.default_rng(2).random((3,128,128)).astype(np.float32)
    values[1:] -= 128/255
    reference = np.rint(255*np.clip(preview_float(values),0,1)).astype(np.uint8).transpose(1,2,0)
    np.testing.assert_array_equal(reference,ycbcr_rgb(values))


def test_scene_controls_share_prefix_recovery_and_have_bounded_transitions():
    for scenario in SCENARIOS:
        for index in (0,59,120,179):
            np.testing.assert_array_equal(stimulus(index,scenario,60),stimulus(index,"steady",60))
        for index in (60,85,89,90,119):
            values = stimulus(index,scenario,60)
            assert values.shape == (3,128,128) and values.dtype == np.float32
            assert 0 <= values[0].min() <= values[0].max() <= 1
            assert np.abs(values[1:]).max() < .5
    assert np.count_nonzero(stimulus(85,"black",60)) == 0
    assert stimulus(90,"black",60)[0].mean() == pytest.approx(.85)
    assert sum(float(stimulus(i,"black",60)[0].mean()) for i in range(60,120)) == pytest.approx(
        sum(float(stimulus(i,"black-smooth",60)[0].mean()) for i in range(60,120)),abs=1e-5)
    assert not np.array_equal(stimulus(90,"cuts",60),stimulus(90,"steady",60))


def test_bounded_runner_equal_counts_and_frozen_digests():
    for scenario in ("steady","cuts"):
        engine = Engine(Config(channels=8,grid=32,input_mode="ycbcr"))
        result = run_case(engine,scenario,96,90,100,time.monotonic()+30)
        assert result["complete"] and result["counts"]["updates"] == 85
        assert result["counts"]["scores"] == 86 and result["counts"]["late_targets"] == 0
        assert result["phases"]["frozen"]["pairs"] == 1
    engine = Engine()
    result = run_case(engine,"steady",30,0,100,time.monotonic()+30)
    assert result["frozen_learned_state_unchanged"] and result["counts"]["updates"] == 0
    short = run_case(Engine(),"steady",30,0,100,time.monotonic()-1)
    assert not short["complete"] and short["processed_frames"] == 0


def test_hidden_reset_gui_confirmation_and_worker_command(monkeypatch):
    from PySide6.QtWidgets import QApplication,QMessageBox
    from cognio.gui import Window
    app = QApplication.instance() or QApplication([])
    engine = trained()
    runtime = Runtime(engine,"no-input")
    window = Window(runtime)
    before = state_digest(engine.hidden)
    monkeypatch.setattr(QMessageBox,"question",lambda *a: QMessageBox.StandardButton.No)
    window.reset_hidden()
    assert runtime.commands.empty() and state_digest(engine.hidden) == before
    monkeypatch.setattr(QMessageBox,"question",lambda *a: QMessageBox.StandardButton.Yes)
    window.reset_hidden()
    assert runtime.commands.qsize() == 1
    runtime.start()
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        app.processEvents()
        snap = runtime.latest()
        if snap and snap["revision"] == 1:
            break
        time.sleep(.01)
    window.close()
    assert torch.count_nonzero(engine.hidden) == 0
    assert snap["visual"] is None and snap["display_result"] is None
    assert "weights/Adam" in snap["notice"]
    assert not any(t.is_alive() for t in runtime.threads)


def test_checkpoint_diagnostic_cli_preserves_file_and_refuses_overwrite(tmp_path):
    checkpoint = tmp_path/"owner.pt"
    trained().save(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = tmp_path/"report.json"
    command = [sys.executable,"-m","tools.investigate_grid","--checkpoint",str(checkpoint),"--frames","30",
               "--scenarios","steady","--max-seconds","30","--report",str(report)]
    result = subprocess.run(command,capture_output=True,text=True,timeout=40)
    assert result.returncode == 0,result.stderr
    data = json.loads(report.read_text())
    assert data["checkpoint_unchanged"] and data["complete"]
    assert [c["hidden_start"] for c in data["cases"]] == ["restored","zero"]
    assert all(c["frozen_learned_state_unchanged"] for c in data["cases"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    rejected = subprocess.run([*command[:-1],str(checkpoint)],capture_output=True,text=True,timeout=10)
    assert rejected.returncode == 2 and "already exists" in rejected.stderr
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_reset_hidden_cli_opt_in_and_default_restore(tmp_path):
    checkpoint = tmp_path/"original.pt"
    engine = trained()
    engine.save(checkpoint)
    assert torch.equal(Engine.load(checkpoint).hidden,engine.hidden)
    for reset in (False,True):
        destination = tmp_path/f"after-{reset}.pt"
        command = [sys.executable,"-m","cognio","--source","no-input","--compute","cpu","--checkpoint",str(checkpoint),
                   "--seconds","0.1","--save-on-exit",str(destination)]
        if reset:
            command.append("--reset-hidden")
        result = subprocess.run(command,capture_output=True,text=True,timeout=15)
        assert result.returncode == 0,result.stderr
        restored = Engine.load(destination)
        assert state_digest(restored.model.state_dict()) == state_digest(engine.model.state_dict())
        assert state_digest(restored.optimizer.state_dict()) == state_digest(engine.optimizer.state_dict())
        assert torch.equal(restored.hidden,torch.zeros_like(engine.hidden) if reset else engine.hidden)
        assert not restored.training

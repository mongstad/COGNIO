import copy
from fractions import Fraction
import io
import json
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from cognio.engine import Engine
from cognio.headless import run, video_command
from cognio.inputs import Frame, synthetic
from cognio.model import Config
from cognio.runtime import Runtime
from cognio.video import VideoPlayback, VideoReader, color_contract, probe
from tools.make_video_fixture import make_video, planes


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg/ffprobe unavailable")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("videos")
    for stimulus in ("static", "moving", "color"):
        make_video(root/f"{stimulus}.mkv", 18, stimulus)
    make_video(root/"vfr.mkv", 12, "moving", vfr=True)
    return root


@pytest.mark.parametrize("stimulus", ["static", "moving", "color"])
@pytest.mark.parametrize("mode", ["y", "ycbcr"])
def test_decoded_frame_order_and_exact_native_plane_values(clips, stimulus, mode):
    reader = VideoReader(clips/f"{stimulus}.mkv", threading.Event())
    try:
        for index in range(18):
            n, pts, values = reader.read(mode)
            expected = planes(index, stimulus).astype(np.float32)
            expected[1:] -= 128
            expected /= 255
            assert n == index and abs(float(pts)-index/30) <= .0005
            np.testing.assert_array_equal(values, expected[0] if mode == "y" else expected)
        assert reader.read(mode) is None
    finally:
        reader.close()
    assert reader.process.poll() is not None


@pytest.mark.parametrize("matrix,level", [("smpte170m","tv"), ("bt709","tv"), ("smpte170m","pc"), ("bt709","pc")])
@pytest.mark.parametrize("pixel_format", ["yuv420p","yuv422p","yuv444p"])
def test_video_color_matrix_and_range_conversion(tmp_path, matrix, level, pixel_format):
    # Independent forward equations for neutral gray and R/G/B primaries.
    rgb = np.array([[.5,.5,.5], [1,0,0], [0,1,0], [0,0,1]])
    kr, kb = (.2126,.0722) if matrix == "bt709" else (.299,.114)
    y = rgb @ np.array([kr,1-kr-kb,kb])
    cb, cr = (rgb[:,2]-y)/(2*(1-kb)), (rgb[:,0]-y)/(2*(1-kr))
    samples = np.stack(((16+219*y) if level == "tv" else 255*y,
                        128+(224 if level == "tv" else 255)*cb,
                        128+(224 if level == "tv" else 255)*cr))
    native = np.repeat(np.repeat(np.clip(np.rint(samples),0,255).astype(np.uint8)[:,None,:],128,axis=1),32,axis=2)
    path = tmp_path/"primaries.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "yuv444p", "-video_size", "128x128",
                    "-colorspace", matrix, "-color_range", level, "-color_trc", "bt709", "-color_primaries", "bt709",
                    "-i", "pipe:0", "-frames:v", "1", "-c:v", "ffv1", "-threads", "1", "-colorspace", matrix,
                    "-color_range", level, "-color_trc", "bt709", "-color_primaries", "bt709", "-pix_fmt",pixel_format,str(path)],
                   input=native.tobytes(), capture_output=True, check=True, timeout=10)
    reader = VideoReader(path, threading.Event())
    try:
        _,_,values = reader.read("ycbcr")
        from cognio.gui import ycbcr_rgb
        reconstructed = ycbcr_rgb(values)[64, [16,48,80,112]]
        np.testing.assert_allclose(reconstructed, np.rint(rgb*255), atol=4, rtol=0)
    finally:
        reader.close()


def test_vfr_exact_pairing_media_and_wall_horizons(clips):
    player = VideoPlayback(clips/"vfr.mkv", "ycbcr", threading.Event(), fastest=True)
    engine = Engine(Config(input_mode="ycbcr"))
    observed, predictions = [], []
    try:
        while not player.eof:
            frame = player.poll()
            if frame is None:
                continue
            observed.append(frame)
            predictions.append(engine.step(frame)["prediction"])
            assert len(engine.pending) <= 5
            if frame.sequence >= 5:
                result = engine.last_result
                old = observed[-6]
                assert (result["source"], result["target"]) == (frame.sequence-5,frame.sequence)
                assert result["issued"] < frame.timestamp
                assert result["media_horizon_ms"] == pytest.approx(1000*(frame.media_time-old.media_time))
                assert result["baseline_mse"] == pytest.approx(float(np.mean((old.values-frame.values)**2)))
                assert result["mse"] == pytest.approx(float(np.mean((predictions[-6]-frame.values)**2)))
        np.testing.assert_allclose([f.media_time for f in observed], [0,.04,.08,.12,.16,.24,.32,.4,.48,.56,.64,.72])
        assert engine.counts["scores"] == 7 and engine.counts["late_targets"] == 0
    finally:
        player.close()


@pytest.mark.parametrize("speed", [1., .1])
def test_paced_speed_pause_resume_and_speed_change_with_clock(clips, speed):
    clock = [10.]
    player = VideoPlayback(clips/"vfr.mkv", "y", threading.Event(), speed=speed, clock=lambda: clock[0])
    try:
        first = player.poll()
        assert first.media_index == 0
        clock[0] += .039/speed
        assert player.poll() is None
        player.control("video_pause", True)
        sequence = player.sequence
        clock[0] += 100
        assert player.poll() is None and player.sequence == sequence
        player.control("video_pause", False)
        player.control("video_speed", .1)
        clock[0] += .009
        assert player.poll() is None
        clock[0] += .002
        second = player.poll()
        assert second.media_index == 1 and second.sequence == first.sequence+1
        # Falling far behind retains the next frame; no catch-up skipping/duplicates.
        clock[0] += 10
        assert player.poll().media_index == 2
        assert player.poll() is None
    finally:
        player.close()


def test_repeat_and_restart_do_not_pair_across_passes(clips):
    player = VideoPlayback(clips/"moving.mkv", "y", threading.Event(), loop=True, fastest=True)
    engine = Engine()
    pairs = []
    try:
        while player.sequence < 41:
            frame = player.poll()
            if frame is None:
                continue
            result = engine.step(frame)
            if frame.media_index == 0 and frame.segment:
                assert engine.last_result is None and len(engine.pending) == 1
                # Zero prior recurrent state at a boundary; weights and Adam retained.
                assert torch.count_nonzero(engine.pending[frame.sequence+5].previous) == 0
            if engine.last_result:
                match = engine.last_result
                pairs.append((match["segment"],match["source_media_index"],match["target_media_index"]))
                assert match["target_media_index"]-match["source_media_index"] == 5
        assert engine.counts["scores"] == 26 and engine.counts["cleared_targets"] == 10
        player.control("video_pause", True)
        player.control("video_restart", None)
        assert player.poll() is None
        player.control("video_pause", False)
        frame = player.poll()
        assert frame.media_index == 0 and frame.sequence == 41 and frame.segment == 3
        engine.step(frame)
        assert engine.last_result is None
    finally:
        player.close()


@pytest.mark.parametrize("mode", ["y", "ycbcr"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_headless_eof_freeze_and_cross_source_checkpoint(clips, tmp_path, mode, device):
    engine = Engine(Config(input_mode=mode), device=device)
    engine.set_training(False)
    before = copy.deepcopy(engine.model.state_dict())
    runtime = Runtime(engine, "video", observer=False, video_path=clips/"color.mkv", video_fastest=True)
    output = io.StringIO()
    checkpoint = tmp_path/"video.pt"
    assert run(runtime, seconds=10, interval=.1, save_path=checkpoint, output=output) == 0
    final = [json.loads(line) for line in output.getvalue().splitlines() if '"run_end"' in line][0]
    assert final["video"]["eof"] and final["workers_stopped"]
    assert engine.counts["predictions"] == 18 and engine.counts["scores"] == 13 and engine.counts["updates"] == 0
    assert engine.counts["late_targets"] == 0 and engine.counts["skipped_frames"] == 0 and runtime.overwrites == 0
    assert all(torch.equal(value, engine.model.state_dict()[key]) for key,value in before.items())
    assert runtime.latest()["visual"] is None
    loaded = Engine.load(checkpoint, device=device)
    assert loaded.config == engine.config
    assert "video" not in torch.load(checkpoint, weights_only=True)
    loaded.step(Frame(0,time.monotonic(),synthetic(0,"moving",mode)))
    assert not loaded.fault  # A matching camera/synthetic architecture can continue it.
    back = Runtime(loaded, "video", observer=False, video_path=clips/"static.mkv", video_fastest=True)
    loaded.set_training(True)
    # New Runtime video session starts at sequence zero; checkpoint reload resets timeline.
    back.engine = Engine.load(checkpoint, device=device)
    back.engine.set_training(True)
    assert run(back, seconds=10, interval=1, output=io.StringIO()) == 0
    assert back.engine.counts["updates"] == 13


def test_missing_corrupt_unsupported_and_color_metadata_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path/"missing.mp4")
    with pytest.raises(ValueError, match="regular file"):
        probe(tmp_path)
    bad = tmp_path/"bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="probe"):
        probe(bad)
    info = dict(height=720, pix_fmt="yuv420p")
    assert color_contract(info)[:2] == ("bt709","tv")
    assert color_contract(dict(info,height=480,pix_fmt="yuvj420p"))[:2] == ("bt601","pc")
    for field,value in (("color_transfer","smpte2084"),("color_space","bt2020nc"),("color_primaries","bt2020")):
        with pytest.raises(ValueError,match="Unsupported"):
            color_contract(dict(info,**{field:value}))
    unsupported = tmp_path/"ten-bit.mkv"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=black:size=32x32:duration=0.1",
                    "-vf","format=yuv420p10le","-c:v","ffv1","-threads","1",str(unsupported)],
                   check=True,capture_output=True,timeout=10)
    with pytest.raises(ValueError,match="pixel format"):
        probe(unsupported)


def test_h264_mp4_with_b_frames_and_native_420(clips,tmp_path):
    target = tmp_path/"reordered.mp4"
    subprocess.run(["ffmpeg","-v","error","-i",str(clips/"moving.mkv"),"-c:v","libx264","-threads","1",
                    "-bf","3","-pix_fmt","yuv420p","-crf","18",str(target)],check=True,capture_output=True,timeout=15)
    runtime = Runtime(Engine(),"video",observer=False,video_path=target,video_fastest=True)
    assert run(runtime, seconds=10, output=io.StringIO()) == 0
    assert runtime.engine.counts["predictions"] == 18 and runtime.engine.counts["scores"] == 13
    assert runtime.engine.last_result["media_horizon_ms"] == pytest.approx(166.667,abs=1)


def test_bad_timestamps_rejected_without_fictitious_frames(clips):
    reader = VideoReader(clips/"moving.mkv", threading.Event())
    try:
        reader._line(b"[showinfo] config in time_base: 1/1000")
        with pytest.raises(ValueError,match="timestamp"):
            reader._line(b"[showinfo] n: 0 pts: NOPTS pts_time:NOPTS")
        reader.read("y")
        reader.previous_pts = Fraction(1000)
        with pytest.raises(ValueError,match="strictly increase"):
            reader.read("y")
    finally:
        reader.close()


def test_stopped_decoder_and_invalid_controls(clips):
    event = threading.Event()
    reader = VideoReader(clips/"moving.mkv", event)
    event.set()
    try:
        with pytest.raises(InterruptedError):
            reader.read("y")
    finally:
        reader.close()
    assert reader.process.poll() is not None
    for line, expected in (("pause",("video_pause",True)), ("resume",("video_pause",False)),
                           ("restart",("video_restart",None)), ("speed 0.1",("video_speed",.1)),
                           ("loop on",("video_loop",True)), ("train off",("training",False))):
        assert video_command(line) == expected
    for line in ("speed nan", "speed 0", "seek 2", "loop maybe"):
        with pytest.raises(ValueError):
            video_command(line)


def test_truncated_container_is_not_silent_successful_eof(clips,tmp_path):
    data = (clips/"color.mkv").read_bytes()
    path = tmp_path/"truncated.mkv"
    path.write_bytes(data[:len(data)//2])
    reader = None
    try:
        with pytest.raises((ValueError,json.JSONDecodeError),match="truncated|probe|corrupt|Expecting"):
            reader = VideoReader(path,threading.Event())
            while reader.read("y") is not None:
                pass
    finally:
        if reader:
            reader.close()


@pytest.mark.parametrize("speed", [1., .1])
def test_actual_runtime_pacing_and_horizon(clips, speed):
    runtime = Runtime(Engine(), "video", observer=False, video_path=clips/"moving.mkv", video_speed=speed)
    before = time.monotonic()
    assert run(runtime, seconds=12, interval=1, output=io.StringIO()) == 0
    elapsed = time.monotonic()-before
    assert runtime.latest()["video"]["eof"]
    assert elapsed >= .566/speed
    assert elapsed < .566/speed+3
    match = runtime.engine.last_result
    assert match["media_horizon_ms"] == pytest.approx(167,abs=1)
    assert match["horizon_ms"] >= 160/speed
    assert match["horizon_ms"] < 167/speed+200
    assert runtime.engine.counts["predictions"] == 18 and runtime.engine.counts["scores"] == 13


def test_video_observer_does_not_change_learned_state(clips):
    engines = []
    for observe in (False, True):
        engine = Engine(Config(input_mode="ycbcr"))
        runtime = Runtime(engine, "video", observer=observe, video_path=clips/"color.mkv", video_fastest=True)
        assert run(runtime, seconds=10, interval=1, output=io.StringIO()) == 0
        engines.append(engine)
    a,b = engines
    assert a.counts == b.counts and list(a.history) == list(b.history)
    assert torch.equal(a.hidden,b.hidden)
    assert all(torch.equal(v,b.model.state_dict()[k]) for k,v in a.model.state_dict().items())


def test_restore_retains_hidden_state_within_a_later_video_pass(tmp_path):
    now = [0.]
    engine = Engine(clock=lambda: now[0]+.001)
    for i in range(8):
        now[0] = i/30
        engine.step(Frame(i,now[0],synthetic(i,"moving"),i/30,i,3))
    checkpoint = tmp_path/"later-pass.pt"
    engine.save(checkpoint)
    restored = Engine.load(checkpoint,clock=lambda: now[0]+.001)
    now[0] = 8/30
    frame = Frame(8,now[0],synthetic(8,"moving"),8/30,8,3)
    np.testing.assert_array_equal(engine.step(frame)["prediction"],restored.step(frame)["prediction"])
    assert torch.equal(engine.hidden,restored.hidden)


def test_video_cli_stdin_controls_without_qt_and_invalid_file(clips, tmp_path):
    script = """import sys
class RejectQt:
    def find_spec(self, fullname, *args):
        if fullname.startswith('PySide6'):
            raise AssertionError('headless imported Qt')
sys.meta_path.insert(0, RejectQt())
from cognio.__main__ import main
raise SystemExit(main())
"""
    command = [sys.executable,"-c",script,"--source","video","--compute","cpu","--video-file",str(clips/"moving.mkv"),
               "--video-controls","--video-paused","--video-fastest","--seconds","10"]
    result = subprocess.run(command, input="speed nan\nspeed 0.1\ntrain off\nloop off\nrestart\nresume\n",
                            capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records[0]["event"] == "control_error"
    final = records[-1]
    assert final["event"] == "run_end" and final["video"]["eof"] and final["workers_stopped"]
    assert final["video"]["speed"] == .1 and final["counts"]["scores"] == 13 and final["counts"]["updates"] == 0
    assert not final["training"] and final["video"]["segment"] == 1
    bad = subprocess.run([sys.executable,"-c",script,"--source","video","--compute","cpu","--video-file",str(tmp_path/"absent.mp4")],
                         capture_output=True,text=True,timeout=15)
    assert bad.returncode == 1
    assert "Video input" in json.loads(bad.stdout.splitlines()[-1])["capture_error"]


def test_video_gui_pause_freeze_restart_and_eof(clips):
    from PySide6.QtWidgets import QApplication
    from cognio.gui import Window
    app = QApplication.instance() or QApplication([])
    runtime = Runtime(Engine(Config(input_mode="ycbcr")), "video", video_path=clips/"color.mkv", video_loop=True)
    window = Window(runtime)
    window.show()
    runtime.start()
    def wait(predicate):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError("video GUI condition timed out")
    try:
        wait(lambda: runtime.latest() and runtime.latest()["counts"]["scores"] >= 5)
        window.video_pause.setChecked(True)
        wait(lambda: runtime.latest()["video"]["paused"])
        counts = runtime.engine.counts.copy()
        hidden = runtime.engine.hidden.clone()
        deadline = time.monotonic()+.2
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert runtime.engine.counts == counts and torch.equal(hidden,runtime.engine.hidden)
        window.training.setChecked(False)
        wait(lambda: not runtime.latest()["training"])
        before = copy.deepcopy(runtime.engine.model.state_dict())
        window.video_loop.setChecked(False)
        window.video_speed.setValue(.1)
        window.send("video_restart")
        wait(lambda: runtime.latest()["video"]["segment"] == 1)
        assert not runtime.engine.pending and torch.count_nonzero(runtime.engine.hidden) == 0
        window.video_speed.setValue(4)
        window.video_pause.setChecked(False)
        wait(lambda: runtime.latest()["video"]["eof"])
        window.refresh()
        assert "EOF" in window.video_label.text()
        assert "Media horizon" in window.error_label.text()
        assert "+5 video frames" in window.prediction_label.text()
        assert all(torch.equal(v,runtime.engine.model.state_dict()[k]) for k,v in before.items())
    finally:
        window.close()
    assert all(not thread.is_alive() for thread in runtime.threads)

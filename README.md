# COGNIO

COGNIO Prototype 1 learns online from local luminance measurements, starting
with random weights. It predicts the frame **five camera sequence numbers in
the future**, compares that earlier prediction with the real observation and
a copy-current-frame baseline, and displays its actual recurrent activations.

The desktop window has a primary 32-channel internal-state canvas and three
separate diagnostic views: current input, future prediction, and matched
squared prediction error. It is a small learning experiment, not an object
recognizer or evidence of consciousness or biological equivalence. It contains
no physical field solver and is separate from SENTIO.

## Setup and run on Fedora 44

Python 3.11+ is required. The exercised environment is Python 3.14.10 with
CPU PyTorch; CUDA is optional and has not been validated for this prototype.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m cognio --source moving --compute cpu
```

Installation downloads libraries; the running application has no network or
upload path. Default launch uses a synthetic static scene and enables training.
No camera is activated unless `--source camera` is passed. Close any other
application using the camera, then run as the desktop user:

```bash
python -m cognio --list-devices
python -m cognio --source camera --compute cpu
# Select explicitly if multiple cameras are present:
python -m cognio --source camera --device /dev/v4l/by-id/YOUR-CAMERA-video-index0
```

Discovery prefers persistent `/dev/v4l/by-id` paths and requires a unique C922
capture node. `/dev/video0` is only an explicit, temporary choice. Access errors
appear in the window; close and relaunch after freeing/reconnecting the device.
The Codex sandbox may hide video devices: use a normal desktop terminal.

`--compute auto` probes CUDA and visibly falls back to CPU. `--compute cuda`
fails with a CPU launch instruction if unavailable. No CUDA toolkit or GPU is
needed for the documented CPU installation. Defaults are two PyTorch CPU
threads, seed 7, Adam learning rate 0.001, and float32. Use `--help` for launch
options; `--seed` and `--learning-rate` affect new models only.

## Controls and interpretation

- **Train online:** off freezes weights and Adam state. Camera capture,
  prediction, scoring, and recurrent activations continue. Transitions clear
  pending pairs and error history so phases are not mixed.
- **Display ±:** fixed symmetric activation color range only. Blue is negative,
  dark is zero, amber is positive. It changes no model values or inputs. All 32
  channels appear in row-major 4×8 tiles; each preserves its 8×8 positions.
- **Reset to new random model:** confirms discarding unsaved learning, picks a
  fresh seed, clears progress, and pauses training.
- **Save checkpoint / Restore checkpoint:** explicit local `.pt` files with
  model/config, Adam state, recurrent state, and counters; no raw input images
  or pending targets. Restore begins frozen. Successful save clears pending
  pairs; overwriting an existing file requires an explicit save.

Every launch clearly says **NEW random initialization** or **RESTORED**. There
is no automatic save, automatic resume, or image recording. Save explicitly
before exit if you want to keep learning:

```bash
python -m cognio --source camera --checkpoint /path/to/cognio-state.pt
```

Checkpoints can contain learned scene information, including hidden activations;
they are not an anonymization mechanism. A restored run cannot replay unrecorded
observations. Five new camera frames are needed for initial pairing, and
missed/late targets are skipped rather than relabeled.

The GUI reports input/prediction/update rates, inference and update time,
input/target age, effective horizon, unseen-frame MSE versus persistence,
training loss separately, drops, RAM, CPU, and PyTorch VRAM. The error map is
**the earlier prediction versus its matched target**. Prediction and error
panels can refer to different frame numbers; read their labels.

## Validation and owner trial

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m tools.validate_learning > /tmp/cognio-learning.json
# Visible camera trial; exercises freeze/save/restore and stores scalars only:
python -m tools.validate_gui --source camera --seconds 90 --report /tmp/cognio-live.json
```

For manual review, follow the [owner protocol](docs/OWNER_TRIAL.md). Synthetic
controls are `static`, `moving` (translating grating), `shape` (translating
Gaussian blob), `temporal` (brightness variation), `zero` (valid black frames),
and `no-input` (no observations).

See [architecture and numerical contract](docs/ARCHITECTURE.md) and
[measured validation and limitations](docs/VALIDATION.md). Repeated synthetic
gratings demonstrated improvement over persistence; this did **not** generalize
to every control tested. Attractive activation patterns are not evidence of
useful prediction or durable memory.

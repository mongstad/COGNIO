# COGNIO

COGNIO Prototype 1 learns online from local luminance measurements, starting
with random weights. It predicts the frame **five camera sequence numbers in
the future**, compares that earlier prediction with the real observation and
a copy-current-frame baseline, and displays its actual recurrent activations.

The optional desktop window has a primary internal-state canvas (32 channels
per page) and three separate diagnostic views: current input, future prediction, and matched
squared prediction error. It is a small learning experiment, not an object
recognizer or evidence of consciousness or biological equivalence. It contains
no physical field solver and is separate from SENTIO.

## Setup and run on Fedora 44

Python 3.11+ is required. The reference CPU environment uses Python 3.14.10.
Headless operation does not import or require Qt. The tested CPU installation
below also includes the optional GUI and test dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m cognio --gui --source moving --compute cpu
```

Installation downloads libraries; the running application has no network or
upload path. Default launch is **headless**, uses a synthetic static scene, and
enables training. Pass `--gui` explicitly to open the observer window.
No camera is activated unless `--source camera` is passed. Close any other
application using the camera, then run as the desktop user:

```bash
python -m cognio --list-devices
python -m cognio --gui --source camera --compute cpu
# Select explicitly if multiple cameras are present:
python -m cognio --gui --source camera --device /dev/v4l/by-id/YOUR-CAMERA-video-index0
```

Discovery prefers persistent `/dev/v4l/by-id` paths and requires a unique C922
capture node. `/dev/video0` is only an explicit, temporary choice. Access errors
appear in the window; close and relaunch after freeing/reconnecting the device.
The Codex sandbox may hide video devices: use a normal desktop terminal.

`--compute auto` probes actual CUDA forward/backward/Adam execution and reports
CPU fallback when unavailable. `--compute cuda`
fails with a CPU launch instruction if unavailable. No CUDA toolkit or GPU is
needed for the documented CPU installation. Defaults are two PyTorch CPU
threads, seed 7, Adam learning rate 0.001, and float32. Use `--help` for launch
options; `--seed` and `--learning-rate` affect new models only. Baseline dimensions
remain 32 channels and an 8×8 grid. Scaling measurements and CPU/CUDA installation
details are in [the capacity guide](docs/CAPACITY.md).

## Larger networks and headless runs

```bash
# 512 channels × 16×16 positions = 131,072 recurrent values.
# Use a CUDA-enabled environment for --compute cuda (see capacity guide).
python -m cognio --headless --source moving --compute cuda --channels 512 --grid 16 \
  --seconds 30 --save-on-exit /tmp/cognio-512.pt
python -m cognio --gui --source camera --compute cuda --channels 512 --grid 16
# Restore saved dimensions, initially frozen:
python -m cognio --headless --source camera --compute cpu --checkpoint /tmp/cognio-512.pt \
  --seconds 30
```

Supported grids are 8, 16 and 32; channel counts are positive integers checked
against available-memory estimates before allocation. `--memory-budget-mib`
can impose a smaller working allowance. Larger dimensions mean different
learned architectures; restoring mismatched dimensions is rejected. Original
Prototype 1 checkpoints migrate at 32×8 without resizing weights.

Headless runs emit bounded periodic scalar JSON telemetry (`--telemetry-interval`,
default one second), stop on Ctrl-C/SIGTERM or `--seconds`, and release capture.
`--resume-training` explicitly enables learning after a checkpoint restore;
otherwise restored runs start frozen. `--save-on-exit` is an explicit atomic
save and can replace the specified file.
Without it, no checkpoint is written. Headless observation never transfers or
renders activations. GUI observation is separately paced (`--display-hz`, default
15); use **Channel page** for at most 32 measured channels at once.

## Controls and interpretation

- **Train online:** off freezes weights and Adam state. Camera capture,
  prediction, scoring, and recurrent activations continue. Transitions clear
  pending pairs and error history so phases are not mixed.
- **Display ±:** fixed symmetric activation color range only. Blue is negative,
  dark is zero, amber is positive. It changes no model values or inputs. All 32
  channels on the selected page appear in row-major tiles; each preserves its
  configured spatial grid. Unused cells of a partial page are blank.
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
python -m cognio --gui --source camera --checkpoint /path/to/cognio-state.pt
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

See [architecture and numerical contract](docs/ARCHITECTURE.md),
[baseline validation](docs/VALIDATION.md), and
[capacity measurements and limitations](docs/CAPACITY_VALIDATION.md). Repeated synthetic
gratings demonstrated improvement over persistence; this did **not** generalize
to every control tested. Attractive activation patterns are not evidence of
useful prediction or durable memory.

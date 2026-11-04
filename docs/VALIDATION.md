# Issue #2 validation — 2026-11-04

Baseline: accepted `main` 09d5568 (documentation PR #1 merged). Dedicated
branch `codex/issue-2-prototype-1`. No prior application implementation existed.

## Environment and commands actually exercised

Fedora 44 KDE/Wayland; Ryzen 7 5800X host; Logitech C922 through `uvcvideo`.
Python 3.14.10, PyTorch **2.15.0+cpu**, NumPy 2.4.3, PySide6 6.11.2,
psutil 7.2.2, pytest 9.2.0. NVIDIA driver 615.78.08 detected RTX 3070 Ti
8 GiB (driver-reported CUDA capability 13.4). This CPU PyTorch installation
cannot run CUDA; auto fallback and explicit-CUDA failure behavior were tested.
No GPU training, CUDA numerical equivalence, or GPU speedup is claimed.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python -m tools.validate_learning > /tmp/cognio-learning.json
QT_QPA_PLATFORM=offscreen .venv/bin/python -m tools.validate_gui \
  --source moving --seconds 25 --report /tmp/cognio-gui-synthetic.json \
  --screenshot /tmp/cognio-synthetic.png
.venv/bin/python -m tools.validate_gui --source camera --seconds 90 \
  --compute auto --report /tmp/cognio-c922-live.json
.venv/bin/python -m compileall -q cognio tools tests
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.venv/bin/cognio --help
git diff --check
```

The camera command ran outside the sandbox on the visible Wayland desktop.
An initial attempt failed with EBUSY because SENTIO held the camera. The owner
closed SENTIO, then the retry succeeded. The busy-camera test now also verifies
that format failure occurs before changing camera controls. No SENTIO process
was terminated by the agent, and no SENTIO source was changed.

## Automated checks

30 tests passed after the camera contention and UI status fixes. Coverage:

- V4L2 ctypes ABI against compiled installed Linux headers; strict format,
  stride, dimensions and rate negotiation; unique camera selection; drained
  buffers, gaps, invalid frames, requeue and failed-initialization cleanup;
  mocked disconnect cleanup, sequence wrap and missing-timestamp rejection.
- Fixed Y extraction/normalization, padding, resampling positions; exact patch
  inverse/order, network dimensions and neighboring-patch interaction.
- Exact five-frame pairing and issuance-before-target checks; immutable
  original prediction scoring, altered future targets, missing/late rejection.
- Parameter updates; frozen weights/Adam with changing hidden state;
  bounded graph-free contexts; finite static/moving/temporal/shape/zero runs.
- Checkpoint continuation with bitwise-identical future CPU outputs/updates,
  rejection of corrupted tensors/config/counters/moments, failed-write
  preservation, rollback/recovery after an injected invalid optimizer update.
- Qt four-view mapping, channel tiling, display independence, freeze/reset/
  save/restore, failed restore preserving live state, absent camera/no-input
  responsiveness, worker shutdown and explicit CPU fallback.

The test invocation disables unrelated globally installed pytest plugins;
the earlier default invocation passed but emitted an external anyio plugin
deprecation warning. A tensor-to-scalar warning found during initial testing
was removed by detaching diagnostic tensors explicitly.

## Controlled repeated-stimulus experiment

Each mode/seed: 240 frames initially frozen, 1200 training, 240 finally frozen,
at synthetic nominal 30 Hz, 128², horizon 5, Adam lr 0.001, two CPU threads,
float32. Each frozen phase has 235 exact matched pairs; training has 1195
updates. The stimulus keeps advancing during freeze. Evaluation is on later
observations of the **same periodic control**, not an unfamiliar-scene holdout.

| Control | Seed | Initial frozen MSE | Final frozen MSE | Persistence MSE | Beats persistence? |
| --- | ---: | ---: | ---: | ---: | --- |
| Static shape | 7 | 0.123489 | 0.000959 | 0 | No |
| Static shape | 11 | 0.123343 | 0.001035 | 0 | No |
| Translating grating | 7 | 0.023489 | 0.000893 | 0.013180 | Yes |
| Translating grating | 11 | 0.023418 | 0.008188 | 0.013180 | Yes |
| Translating blob | 7 | 0.145938 | 0.013660 | 0.012035 | No |
| Translating blob | 11 | 0.146191 | 0.013660 | 0.012035 | No |
| Varying brightness | 7 | 0.063098 | 0.063559 | 0.035198 | No |
| Varying brightness | 11 | 0.063062 | 0.062768 | 0.035198 | No |

The grating demonstrates learned predictive improvement over persistence for
both seeds, with substantial seed dependence. Static prediction error decreases
but cannot beat exact static persistence. The blob does not beat persistence;
brightness learning is ineffective here, and seed 7 slightly worsens. These
results do not support general predictive competence or durable memory.

An initial experiment script incorrectly asserted that every moving/temporal
control must beat persistence and stopped on the blob. It was changed to emit
the measured comparisons, including failures, instead of treating that research
expectation as a universal pass condition. The separate automated positive
grating test still requires reduced unseen error and improvement over persistence.
Complete small scalar results: [learning.json](evidence/learning.json).

## Actual desktop application and C922 trial

The 25-second offscreen synthetic trial exercised the real Qt window and
workers, freeze/save/restore/resume, and display range. 750 captured synthetic
frames, no dropped/missed/late targets, no failure, all workers stopped. Its
rendered screenshot was inspected for four separate readable views; no screenshot
is committed. [Synthetic summary](evidence/synthetic-gui-summary.json).

The **90.29-second visible Wayland C922 trial** negotiated YUYV 640×480 @30,
stride 1280, driver-monotonic timestamps; received 2693 camera frames. Qt
reported the window exposed and delivered 2819 GUI refreshes. A 250 ms probe
timer's maximum interval was 255.4 ms. Scripted controls worked during capture.
This verifies rendering/event-loop responsiveness, not physical mouse feel or
a human's subjective view of the private scene.

| Measurement, active training after 25 s | Mean | 95th percentile |
| --- | ---: | ---: |
| Input FPS | 30.004 | 30.027 |
| Predictions/s | 29.995 | 30.142 |
| Updates/s | 29.995 | 30.142 |
| Forward prediction time | 0.337 ms | 0.439 ms |
| Training update time | 2.098 ms | 3.469 ms |
| Effective five-frame horizon | 167.113 ms | 168.006 ms |
| Remaining predictive lead | 124.161 ms | 128.612 ms |
| Input age at sampled display snapshot | 73.944 ms | 94.129 ms |
| Process CPU (100%=one core) | 70.83% | 73.2% |
| Resident RAM | 530.35 MiB | 534.28 MiB |

Peak RSS was 541.18 MiB; PyTorch VRAM was zero (CPU execution). No driver,
drained, invalid or handoff drops, missed targets, or late targets occurred.
Pending contexts remained five and history capped at 300. Total restored-model
counters ended at 2633 predictions, 2613 scores and 2443 updates; loading the
saved checkpoint intentionally rewound counters, while capture totals continued.

Freeze preserved weights while inputs/predictions/activations continued. Save,
restore (frozen), and resume passed. Shutdown stopped both workers and restored
all six changed camera controls; no control warning remained. No camera image
or video was saved. [Scalar live summary](evidence/c922-live-summary.json).

Sampled live MSE averaged 0.002680 versus persistence 0.000307. These are
250 ms telemetry samples during training, not a controlled frozen benchmark.
The live scene was uncontrolled and persistence was better on average; there
is **no claim of useful learned real-scene prediction** from this trial.

## Remaining owner verification and limits

Run [OWNER_TRIAL.md](OWNER_TRIAL.md) for deliberate static/moving scenes,
viewpoint changes, unfamiliar/familiar frozen comparisons, physical interaction,
and checkpoint use across a real application restart. Physical unplug/replug,
multiple attached C922s, long-duration memory behavior beyond this short trial,
different display scaling, CUDA and other package versions were not tested.

The one-step recurrent surrogate, short context, no replay, shared patch
decoder and seed sensitivity limit learning. A save/load boundary discards
pending sensory pairs; restored hidden state spans an unobserved live gap.
Forced process termination cannot guarantee camera-control restoration. No
recognition, consciousness, biological equivalence or durable-memory conclusion
follows from these measurements or the activation visualization.

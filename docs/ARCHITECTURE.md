# Prototype 1 numerical and runtime contract

## Separation and input

`camera.py` owns V4L2 acquisition/cleanup; `inputs.py` owns normalization and
synthetic streams; `model.py` is the neural network; `engine.py` owns pairing,
learning and checkpoints; `runtime.py` separates capture/learning threads;
`gui.py` only presents snapshots and sends explicit commands. `setup.py` owns
configuration/session lifecycle; `session.py` constructs a run only after explicit
GUI Start (or a headless launch). [Setup/topology contract](SETUP_TOPOLOGY.md)
defines the new GUI defaults and measured capacity limits.

`video.py` owns optional local FFmpeg decoding, source color conversion and
sequential playback. It supplies real monotonic delivery timestamps plus separate
media PTS/index/segment metadata. Unlike live capture, video delivery happens on
the learner worker after the preceding step, preserving all frames even in
fastest mode. New passes clear temporal contexts; the +5 target, issuance guard,
optimizer and model remain unchanged. [VIDEO.md](VIDEO.md) defines the precise
pacing, VFR, pause, EOF, resource and source-color contracts.

The V4L2 adapter, area-integration method and camera ABI tests were adapted
from the same owner's [SENTIO at 4bf7a98](https://github.com/mongstad/SENTIO/tree/4bf7a98c6ed9441c6cda9667d821aff7bc78776f).
COGNIO imports no SENTIO package, wave solver or learned state. Its layout
follows the large-primary-canvas/separate-diagnostics convention.

Require progressive YUYV 640×480 at negotiated 30 FPS. Extract Y0/Y1 without
RGB conversion, respecting row padding. Normalize fixed code values by 255,
then area-average to 128×128. The full 4:3 frame maps to a square, deliberately
distorting aspect ratio; no crop, adaptive contrast, gamma inversion or claim
of calibrated radiance. YUYV is processed camera video, not raw Bayer data.

Issue #5 adds opt-in `ycbcr` measurements while retaining that Y path exactly.
Extract native Cb/Cr at 480 rows × 320 columns; each sample represents one
horizontal two-pixel footprint. Independently area-average those native planes
to 128² over the same full-frame extent. Normalize (code−128)/255 in float64
before resampling and cast to float32, with no clipping or RGB conversion.
`Frame.values` is HW for Y or CHW in Y,Cb,Cr order for color. [COLOR.md](COLOR.md)
defines geometry, ranges and decoder/loss semantics.

Request manual exposure 156×100 µs, disable dynamic exposure-rate adjustment
and autofocus/automatic white balance, and request 4500 K white balance.
Save original controls and restore in reverse order on exit/failure. Unsupported
controls produce warnings; actual rates are measured. Negotiate format before
changing controls so a busy camera fails without changing another app's controls.

Use V4L2 sequence numbers (unwrapped at 32 bits) and driver monotonic timestamps.
Missing monotonic timestamps cause explicit capture failure because receipt
time cannot establish target age. Missing/disconnected cameras leave the GUI
and checkpoint controls alive; three seconds without a frame stops capture.
Reconnect requires relaunch. Normal exit closes handles and MMAP buffers.

## Architecture and learning

The following dimensions describe the accepted 32×8 reference. Issue #4
generalizes the same operations to configurable channels `C` and grid `G`
(8, 16, 32 or 64), with patch width `P=128/G`. Encoder/decoder channels are `P²`;
the recurrent convolution has `2C+2` inputs and `C` outputs. Learned parameters
are `18C² + (2P²+20)C + P²`; hidden values are `CG²`. Initialization order and
the default architecture are unchanged. No learning objective or recurrent
update rule changes with size. [Capacity guide](CAPACITY.md) documents bounds.

For color, replace `P²` with `3P²` in encoder/decoder dimensions and the parameter
formula. Patch vectors order component first, then within-patch row/column.
Y uses sigmoid output; Cb/Cr use sigmoid minus 128/255, retaining signed
measurements. The recurrent state and update rule are shared and unchanged.

PyTorch seeded random initialization; default seed 7. **35,712 learned
parameters**, no pretrained weights/classes/labels/external data. Fixed
coordinates are not learned parameters.

1. Split `1×1×128×128` into 16×16 non-overlapping patches in raster order:
   `1×256×8×8`. A shared 1×1 convolution (256→32) and tanh encode each patch.
   Within-patch channels preserve pixel row/column order.
2. Concatenate encoding, previous `1×32×8×8` recurrent state, and fixed x/y
   coordinates in [-1,1]. A padded 3×3 convolution (66→32) and tanh interact
   across adjacent patches.
3. `h_new = 0.5*h_previous + 0.5*tanh(convolution(...))`.
4. Shared 1×1 convolution (32→256), sigmoid, and inverse patch arrangement
   decode the future `1×1×128×128` luminance prediction.

This is a small leaky spatial RNN, not a ConvGRU or full backpropagation through
time. Hidden state starts at zero and stays in [-1,1] for valid inputs. Zero
padding distinguishes boundaries in Standard; coordinates retain absolute position.
Opt-in Toroidal circularly pads only the recurrent 3×3 convolution, with the same
parameter shapes/counts and fixed coordinates. Topology is immutable and distinct
in checkpoint identity; camera images themselves are not periodic. Patch
boundaries do not define objects. Missing frames receive no fabricated updates.

For source sequence `s`, store the prediction for **exactly s+5**. Complete
inference (including CUDA-to-CPU synchronization when used) before recording
issuance time. Accept target `s+5` only if `issued_time < target_capture_time`.
Expire absent target sequences and reject already-captured targets, with separate
counters. Horizon is `target_capture_time - source_capture_time`; predictive
lead is `target_capture_time - issued_time`. Five frames is only nominally
166.7 ms, not a fixed wall-time interval.

Score the immutable original prediction against the observed target using MSE.
Score the source frame as persistence using identical pairing. A bounded
300-pair history reports both. Training loss is separately labeled.

Color loss/scoring is the mean squared error over all 3×128² values, equivalently
one third of the three per-component pixel MSEs. Native chroma subsampling does
not implicitly halve either chroma weight. Per-component prediction/persistence
errors have a separate bounded 300-pair scalar history, cleared at the same
phase boundaries. This is an explicit equal-component objective in normalized
code units, not perceptual or radiometric color error.

Training recomputes a forward pass from the **stored source frame and detached
previous hidden state** using current weights, minimizing MSE against the
now-known target. This one-step surrogate avoids retaining autograd graphs
across optimizer updates. The target is never an input to that recomputation;
its output never replaces the originally scored prediction. The current
observation can then inform learning and the next causal future prediction.

Adam: default lr 0.001, betas (0.9,0.999), epsilon 1e-8, no weight decay,
global gradient norm clip 1. One update per accepted pair when training is
enabled; batch 1, float32, no autocast, dropout, replay or augmentation. CLI
enables deterministic PyTorch algorithms. CPU reproducibility was exercised;
cross-version/device bitwise equivalence is not promised.

## Bounds, faults and checkpoints

At most five pending contexts (source, prediction, detached previous state):
approximately 680 KiB payload at 32×8, scaling as documented in the capacity
guide, with no autograd graph. Request three kernel
buffers; accept only 2–8 with checked sizes (maximum 4 MiB each). Capture drains
only that finite queue. Capture→learner holds one newest frame; overwrites,
kernel gaps, invalid and drained frames have separate counters. Learner→GUI
holds one newest snapshot; commands are bounded at 16. Error history is 300
pairs; rate histories are 120 samples. Framework memory is larger than payload
bounds and is displayed separately.

Inputs must be finite float32 [0,1] 128², with strictly increasing sequence/time.
In color mode, require 3×128² with Y in [0,1] and Cb/Cr in
[-128/255,127/255]. Range violations are rejected, never silently clamped.
Check loss, gradients, weights, Adam moments, predictions and hidden state.
Invalid arithmetic or weights above 1000 absolute stop learning visibly; a
failed update restores pre-update weights and Adam state. Capture continues.
Reset/load recovers; explicit save can preserve the last valid weights. No
automatic disk write occurs on fault.

Schema 4 checkpoints store complete dimensions/config, weights, Adam, hidden state,
counters, training flag and framework version. Initialization is the only
random operation, so seed plus complete weights suffices without advancing
random-generator state. Load uses `weights_only=True`, limits file size to
an available-host-memory archive limit plus a configuration-sized storage limit,
and validates shapes/dtypes/finiteness, fixed coordinates, Adam
configuration/moments and counters in a candidate engine. Swap only on success.
Supported files are checkpoints produced by this application. Schema 4 also
stores and validates the exact sensory mode, component order, input/target
shape, native geometry, normalization, resampling and loss contract. Original
schema 1 checkpoints migrate explicitly at grayscale 32×8; schema 2 retains its
saved grayscale dimensions. Schemas 1–3 explicitly restore as Standard. Schema 4
requires topology; cross-topology restores are rejected without mutation. GUI restore requires the current mode/dimensions;
CLI restore uses saved choices unless conflicting ones were explicitly requested.
Serialized tensors are on CPU, regardless of the
execution device. Limits and archive validation are detailed in CAPACITY.md.

Save uses a temporary file, fsync and atomic replacement. No raw source/target
images, pending predictions, error maps or sensory history are serialized.
Successful save flushes pending pairs/history in the running engine so restored
and running continuations share a clean pairing boundary. Hidden state remains.
Restoration freezes training unless CLI `--resume-training` explicitly enables
it. A live restart has an unreproducible gap
in observations; retained hidden state does not reconstruct it.

The explicit `reset_hidden()` diagnostic clears only hidden recurrence and
pending prediction/error histories. It retains learned weights, Adam, training
and fault status, progress (apart from counting cleared targets), and all input
sequence/media clocks. CLI `--reset-hidden` requires a checkpoint; the GUI has a
separate confirmation. Ordinary restore retains the saved hidden state. Video
pass reset uses the same operation plus its existing media-boundary handling.
Opt-in scalar diagnostics live in `diagnostics.py` and `tools.investigate_grid`;
they are not invoked by normal learning or rendering. See
[GRID_DIAGNOSTICS.md](GRID_DIAGNOSTICS.md) for measurement definitions and limits.

## Display contract

The primary canvas tiles at most 32 actual hidden channels per selected page
in row-major order, with each configured spatial map intact. This is a fixed reshape, not
PCA, a decoded camera image or an artistic projection. Signed colors use a
fixed symmetric user range and report clipping. Display values never feed
back to the engine.

Input/prediction use fixed grayscale [0,1]. Squared error uses black=0,
white≥0.25; clipping is display-only. Source/target IDs, actual horizon and
target age stay visible even if processing stalls. Faults and capture errors
remain visible alongside the last successful state.
Color diagnostics default to Combined Color: a display-only BT.601 limited-range
RGB reconstruction of aligned input/prediction planes. Combined error is the
per-pixel mean of the three already scored squared component errors in model
units, with the same black=0 / white≥0.25 scale. The conversion never enters
learning; [RGB_PREVIEW.md](RGB_PREVIEW.md) defines its assumptions and clipping.
Individual diagnostics select Y, Cb or Cr in those same three panels. Cb/Cr use a
fixed signed blue/amber map at ±128/255, explicitly labeled as display colors,
not RGB measurements; their error maps remain squared numeric error. Selection
does not enter capture, preprocessing, recurrence or training. The primary
hidden-state canvas and its independent display range are unchanged.

Process CPU% uses 100%=one core; RSS includes Qt/PyTorch overhead. GPU readings
are PyTorch allocated/reserved memory, not whole-device utilization. Capture,
prediction and update rates are independent; a responsive GUI does not imply
capture-rate learning. Headless runs do not import Qt or create activation
snapshots. Observation copies happen after causal issuance, at a separate
bounded cadence, and only for the selected page. Identical delivered frames
produce identical model updates with/without observation in CPU/CUDA tests; overloaded host
scheduling can still change which live frames reach the bounded handoff.

API references: [V4L2 MMAP](https://docs.kernel.org/userspace-api/media/v4l/mmap.html)
and [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).

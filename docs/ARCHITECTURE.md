# Prototype 1 numerical and runtime contract

## Separation and input

`camera.py` owns V4L2 acquisition/cleanup; `inputs.py` owns normalization and
synthetic streams; `model.py` is the neural network; `engine.py` owns pairing,
learning and checkpoints; `runtime.py` separates capture/learning threads;
`gui.py` only presents snapshots and sends explicit commands.

The V4L2 adapter, area-integration method and camera ABI tests were adapted
from the same owner's [SENTIO at 4bf7a98](https://github.com/mongstad/SENTIO/tree/4bf7a98c6ed9441c6cda9667d821aff7bc78776f).
COGNIO imports no SENTIO package, wave solver or learned state. Its layout
follows the large-primary-canvas/separate-diagnostics convention.

Require progressive YUYV 640×480 at negotiated 30 FPS. Extract Y0/Y1 without
RGB conversion, respecting row padding. Normalize fixed code values by 255,
then area-average to 128×128. The full 4:3 frame maps to a square, deliberately
distorting aspect ratio; no crop, adaptive contrast, gamma inversion or claim
of calibrated radiance. YUYV is processed camera video, not raw Bayer data.

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
padding distinguishes boundaries; coordinates retain absolute position. Patch
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
approximately 680 KiB payload, with no autograd graph. Request three kernel
buffers; accept only 2–8 with checked sizes (maximum 4 MiB each). Capture drains
only that finite queue. Capture→learner holds one newest frame; overwrites,
kernel gaps, invalid and drained frames have separate counters. Learner→GUI
holds one newest snapshot; commands are bounded at 16. Error history is 300
pairs; rate histories are 120 samples. Framework memory is larger than payload
bounds and is displayed separately.

Inputs must be finite float32 [0,1] 128², with strictly increasing sequence/time.
Check loss, gradients, weights, Adam moments, predictions and hidden state.
Invalid arithmetic or weights above 1000 absolute stop learning visibly; a
failed update restores pre-update weights and Adam state. Capture continues.
Reset/load recovers; explicit save can preserve the last valid weights. No
automatic disk write occurs on fault.

Schema 1 checkpoints store architecture/config, weights, Adam, hidden state,
counters, training flag and framework version. Initialization is the only
random operation, so seed plus complete weights suffices without advancing
random-generator state. Load uses `weights_only=True`, limits file size to
16 MiB, and validates shapes/dtypes/finiteness, fixed coordinates, Adam
configuration/moments and counters in a candidate engine. Swap only on success.
Supported files are checkpoints produced by this application.

Save uses a temporary file, fsync and atomic replacement. No raw source/target
images, pending predictions, error maps or sensory history are serialized.
Successful save flushes pending pairs/history in the running engine so restored
and running continuations share a clean pairing boundary. Hidden state remains.
GUI/CLI restoration freezes training. A live restart has an unreproducible gap
in observations; retained hidden state does not reconstruct it.

## Display contract

The primary canvas tiles all actual hidden activations: channels 0–31 in
row-major 4×8 order, each 8×8 spatial map intact. This is a fixed reshape, not
PCA, a decoded camera image or an artistic projection. Signed colors use a
fixed symmetric user range and report clipping. Display values never feed
back to the engine.

Input/prediction use fixed grayscale [0,1]. Squared error uses black=0,
white≥0.25; clipping is display-only. Source/target IDs, actual horizon and
target age stay visible even if processing stalls. Faults and capture errors
remain visible alongside the last successful state.

Process CPU% uses 100%=one core; RSS includes Qt/PyTorch overhead. GPU readings
are PyTorch allocated/reserved memory, not whole-device utilization. Capture,
prediction and update rates are independent; a responsive GUI does not imply
capture-rate learning.

API references: [V4L2 MMAP](https://docs.kernel.org/userspace-api/media/v4l/mmap.html)
and [PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).

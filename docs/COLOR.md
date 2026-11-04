# Issue #5 — native YUYV color measurements

`--input-mode y` is the unchanged grayscale baseline and default for new models.
`--input-mode ycbcr` learns and predicts brightness plus native Cb/Cr code
measurements. It integrates with configurable channels/grid, CUDA, headless
execution and the optional GUI. The sensory mode is fixed for a model's lifetime;
GUI Reset starts new random weights in that same mode and architecture.

## Measurement and target contract

The accepted capture format remains progressive YUYV 640×480 at 30 FPS.
The [Linux V4L2 packing specification](https://docs.kernel.org/userspace-api/media/v4l/pixfmt-packed-yuv.html#subsampling-1)
defines byte order **Y0 Cb Y1 Cr**, with one Cb and one Cr sample for each
horizontal pixel pair. Extraction reads bytes, not native-endian words; respects
row stride/padding; rejects incomplete/odd-width input; and copies planes before
the MMAP buffer is requeued. Timestamp, sequence, controls and cleanup paths
are retained. Format metadata includes the negotiated YCbCr encoding enum.

| Component | Native array (rows × columns) | Model plane | Fixed normalization |
| --- | --- | --- | --- |
| Y | 480 × 640 | 128 × 128 | `code / 255`, range [0,1] |
| Cb (U) | 480 × 320 | 128 × 128 | `(code − 128) / 255`, range [−128/255,127/255] |
| Cr (V) | 480 × 320 | 128 × 128 | `(code − 128) / 255`, same range |

Cb/Cr code 128 is exactly neutral zero. Normalization occurs in float64 before
the existing separable area integration, followed by float32 conversion. There
is no unsigned subtraction, adaptive normalization, gamma/range expansion,
sign-destroying clamp or RGB conversion in the input/target path. Values outside
the model contract fail validation. Driver colorimetry is recorded rather than
silently reinterpreted as calibrated radiance or a different transfer function.

Each chroma value is represented as constant over its two-Y-pixel horizontal
footprint. Its native 320-column plane is directly area-averaged across the
same full-frame extent as Y. A target column spans five Y pixels or 2.5 chroma
footprints; fractional overlaps are weighted. This equals explicitly repeating
each chroma value twice and then averaging, but does not allocate or claim new
independent chroma samples. Each output row averages 3.75 native rows. Full-frame
4:3-to-square mapping and area filtering match the existing Y baseline.

YUYV is already processed camera video. Y is luma (the existing UI calls it
luminance), Cb/Cr are color differences; none is Bayer RAW, a photon count or a
full wavelength spectrum. No RGB diagnostic conversion is implemented.

Model/target layout is `HW` for Y and `CHW` in **Y,Cb,Cr** order for color,
batched internally as `B×K×128×128`. With grid `G`, patch width `P=128/G`, and
`K=1` or `3`, patch vectors have `KP²` entries: component first, then pixel
row/column. Spatial patch positions, fixed coordinates and cross-patch
recurrence are unchanged. Decoder Y output uses sigmoid; Cb/Cr use sigmoid
minus 128/255. Both signed chroma and positive Y are predicted in their target
units. Hidden state remains the same measured leaky spatial RNN.

Learned parameters are `18C² + (2KP²+20)C + KP²`:

| Recurrent configuration | Y parameters | YCbCr parameters | Hidden scalars |
| --- | ---: | ---: | ---: |
| 32 × 8×8 | 35,712 | 68,992 | 2,048 |
| 512 × 16×16 | 4,794,432 | 4,925,632 | 131,072 |

The five-frame causal target is unchanged. Scoring uses the immutable prediction
issued before the matched target's capture; training subsequently recomputes
the old source and detached prior state, as documented in ARCHITECTURE.md.
Targets are never relabeled when frames are dropped or predictions are late.

Color training/scoring uses:

`MSE = (MSE_Y + MSE_Cb + MSE_Cr) / 3`

Each component MSE averages 128² pixels in normalized code units; equal weights
are explicit, not an accidental consequence of native Y having twice as many
samples. A code difference of one has magnitude 1/255 in every component.
Persistence copies every corresponding source component and uses identical
pairing and weights. Per-component errors and persistence are reported alongside
the aggregate. Color aggregate and Y-only MSE have different objectives and
must not be compared as if they measured the same target. Static persistence
can be exact even when a trained neural predictor has residual error.

## Bounds, checkpoints and diagnostics

At most five source/prediction/prior-state contexts remain, with no retained
autograd graph; capture has one latest-frame slot. Color adds 1.25 MiB of sensory
context payload over grayscale, plus its larger encoder/decoder. The resource
estimate and configuration-sized checkpoint ceiling account for these changes.
Adam settings, clipping, finite guards and rollback are unchanged. Aggregate and
per-component scalar histories are independently bounded to 300 matched pairs
and cleared together at phase/checkpoint boundaries.

Schema **3**, architecture `patch-spatial-rnn-v3`, stores complete config including
`input_mode`, CPU model/Adam/recurrent tensors and counters. Its `sensory` metadata
records input/target shape, layout, native plane geometry, component order,
normalization, resampling and loss contract; all fields are checked on restore.
Schema 1 migrates to grayscale 32×8. Schema 2 migrates to grayscale with its
original dimensions. No weights are silently converted between modes.
GUI restore requires the current mode/dimensions; CLI uses saved choices unless
an explicitly conflicting argument is supplied. Restore starts frozen unless
`--resume-training` is explicit. Atomic saves preserve the previous file on failure.

No raw planes, pending camera frames, error maps or metric history enter a
checkpoint. Learned tensors can still contain scene information. Camera images
are not recorded by default, and no data is transmitted by the running app.

The primary visualization remains actual recurrent state. In color mode,
**Diagnostic component** chooses Y, Cb or Cr for input, prediction and matched
error panels. Y remains grayscale; chroma uses a fixed signed blue/amber map at
±128/255, explicitly labeled as display colors rather than sensor RGB. Error
maps are squared errors with the existing display ceiling 0.25. Component/page/
display-range choices affect presentation only. Headless runs import no Qt,
copy no activations for display and emit scalar per-component errors plus
input min/max/mean/spatial standard deviation.

## Reproducible commands

Use the existing environments or [documented setup](CAPACITY.md). Close other
camera users before a live run. Headless is the default; `--gui` opens the observer.

```bash
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --input-mode ycbcr --channels 512 --grid 16
.venv-cuda/bin/python -m cognio --headless --source camera --compute cuda \
  --input-mode ycbcr --channels 512 --grid 16 --seconds 15 \
  --save-on-exit /tmp/cognio-color.pt
.venv-cuda/bin/python -m cognio --headless --source color-moving --compute cuda \
  --checkpoint /tmp/cognio-color.pt --resume-training --seconds 2
# This conflicting restore must fail:
.venv-cuda/bin/python -m cognio --headless --compute cpu \
  --checkpoint /tmp/cognio-color.pt --input-mode y --seconds 1
```

Controls `color-static`, `color-moving` and `color-temporal` use constant Y=0.5.
Static chroma has smooth spatial structure; moving chroma translates a smooth
pattern; temporal chroma uses sine/cosine phase at a 40-frame period. These are
analytic model-plane controls, not claims of captured camera data. Grayscale
receives identical constant Y across them; ordinary existing synthetic controls
receive neutral Cb/Cr when selected in color mode.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
.venv/bin/python -m tools.benchmark_capacity --compute cpu --configs 32x8 \
  --input-mode ycbcr --stimuli color-static color-moving color-temporal \
  --train-frames 600 --evaluation-frames 120 --max-seconds 90 \
  --output /tmp/cognio-color-cpu.json
.venv-cuda/bin/python -m tools.benchmark_capacity --compute cuda \
  --configs 32x8 512x16 --input-mode ycbcr \
  --stimuli color-static color-moving color-temporal \
  --train-frames 600 --evaluation-frames 120 --max-seconds 90 \
  --output /tmp/cognio-color-cuda.json
QT_QPA_PLATFORM=wayland .venv-cuda/bin/python -m tools.validate_gui \
  --source camera --input-mode ycbcr --channels 512 --grid 16 --compute cuda \
  --seconds 30 --report /tmp/cognio-color-live.json
```

## Measured validation — 2026-11-04

Accepted reference: `6341e666486e5abd93a33821fd8fa1f05be8471c` (merged PR #6).
The existing assigned issue-5 branch was reused. Environments: Fedora 44,
Ryzen 7 5800X, RTX 3070 Ti 8 GiB; CPU PyTorch 2.15.0+cpu and CUDA
2.14.1+cu130 / runtime 13.0 / cuDNN 9.24.0. Settings remain seed 7, float32,
Adam lr 0.001, two CPU threads, deterministic algorithms, TF32/autotuning off.

Final complete suites: **CPU 88 passed / 8 CUDA skips (19.45 s)**;
**actual CUDA 96 passed (22.73 s)**. Compilation and whitespace checks passed.

Byte-exact fixtures cover Y0/Cb/Y1/Cr order, even pixel pairs, odd row padding,
owned buffers, native dimensions, signed extremes/neutral values and fractional
area weights. Tests also cover all three patch grids, per-component scoring,
future-target independence, dropped/late targets, freeze of weights/Adam,
observer independence, mode/normalization corruption rejection, schema-1/2/3
compatibility, headless Qt exclusion/SIGTERM/save/resume, and component-view
mapping. Actual CUDA optimizer and CPU-portable checkpoint tests cover both
modes at 32×8, 128×16 and 512×16.

Direct comparison with checkpoints produced by accepted main before editing
confirmed **bitwise-identical Y preprocessing and 20 continued predictions**
at both 32×8 and 128×16 (schema 2). The original issue-2 schema-1 checkpoint
also retained its 20 bitwise-identical continued predictions. These checks used
the same CPU environment, not an assumed cross-device equivalence.
The end-to-end grayscale control test also produced identical predictions and
learned weights for moving-color versus temporal-color streams with identical Y.

Bounded synthetic runs used 120 initial frozen, 600 training and 120 final
frozen frames: 595 Adam updates and 115 matched pairs per evaluation. Every case
completed without numeric faults. Their virtual 30 Hz clock is an unpaced
compute/learning control; actual wall throughput is separate from camera timing.
The final CUDA frozen errors were:

| Configuration / control | Y MSE | Cb MSE | Cr MSE | Aggregate | Persistence aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32×8 static | 0.000092 | 0.000183 | 0.000170 | 0.000148 | 0 |
| 32×8 moving | 0.000083 | 0.000266 | 0.000218 | 0.000189 | 0.007444 |
| 32×8 temporal | 0.000475 | 0.000864 | 0.000619 | 0.000653 | 0.012204 |
| 512×16 static | 0.000002 | 0.000032 | 0.000018 | 0.000017 | 0 |
| 512×16 moving | 0.000026 | 0.000304 | 0.000416 | 0.000248 | 0.007444 |
| 512×16 temporal | 0.000001 | 0.054675 | 0.092191 | 0.048956 | 0.012204 |

Moving persistence Cb/Cr MSE was 0.013180 / 0.009153; temporal persistence was
0.017595 / 0.019016. Thus both sizes learned useful chroma prediction on this
repeated moving control; **512×16 failed to beat persistence on changing temporal
color**. Its errors remained finite. No tuning or objective change was made to
conceal this limitation. The CPU 32×8 final aggregate errors were 0.000148 static,
0.000189 moving and 0.000651 temporal. These one-seed controls demonstrate
learning from chroma that is absent from Y-only input; they do not establish
object/color naming, broad generalization, or a universal benefit from size.

The 30.16-second **visible Wayland C922 color GUI** trial at 512×16 passed
freeze, Cb/Cr selection, paging, save/restore/resume and clean shutdown. All six
changed camera controls were restored. Steady training samples after 22 seconds
averaged **30.00 capture / 29.99 prediction / 29.87 updates per second**.
Mean preprocessing / forward / update / whole-engine-step costs were
6.21 / 0.67 / 6.72 / 8.32 ms; display refresh averaged 1.95 ms. Sampled peak RSS
was 2262 MiB and PyTorch peak allocated VRAM 228.4 MiB, including the validator's
freeze snapshots and restore. Displayed input age averaged 109 ms, matched
horizon 166.1 ms. The 250 ms validation timer's largest interval was 263.5 ms.

The live trial recorded four handoff overwrites, no driver/drained/bad frames,
and final restored model counters of two missed and zero late targets. Model
counters rewind on restore while capture counts remain per process. Cb mean
varied from 0.00900 to 0.01236 and Cr from −0.01838 to −0.01299 across sampled
frames; mean spatial standard deviations were 0.01150 and 0.01769. This confirms
non-neutral, changing measured color input. The uncontrolled live scene did
**not** beat persistence in any component and was not a frozen learning study.

A separate **15-second headless C922 color** run made 437 predictions and 431
updates, with one missed target/one skipped frame and no late targets or faults.
Post-startup sampled rates averaged 30.00 capture / 29.71 prediction / 29.58
updates per second; the final rolling window was 30.02 / 30.02 / 30.02. Peak
sampled RSS was 2029 MiB and PyTorch peak allocation 171.6 MiB. Observer cost
was zero. Both workers stopped, the explicit 59,642,517-byte checkpoint saved,
and the installed CLI restored its saved mode and resumed updates for two seconds
on `color-moving`. An **eight-second Y-only C922** regression also stopped
cleanly: 227 predictions / 221 updates, one missed/one skipped, no late targets;
final rolling rates were approximately 30/s.

An additional **25-second offscreen synthetic color GUI** trial passed the same
controls. Its screenshot was visually inspected: four distinct views, signed Cb
units/colors, component selection, per-component metrics and controls fit at
1320×940. Only synthetic imagery was captured, outside the repository. A GUI
control test checks all three components against their exact snapshot arrays.

Full scalar evidence, regression counts and additional headless/synthetic GUI
checks are recorded under [evidence/issue-5](evidence/issue-5/). No camera imagery,
raw video or checkpoint is committed. PyTorch memory excludes the CUDA context
and other consumers; sampled maxima do not prove leak freedom. No long-duration
soak, physical unplug/replug, color calibration, or controlled real-object/
similar-luma scene study was performed. CPU 512×16 color live performance was
not tested. Recurrent training may diverge across backends despite deterministic
execution within a fixed environment.

For owner review, follow [OWNER_TRIAL.md](OWNER_TRIAL.md) in both modes. Check
per-component frozen errors against persistence, component selection, explicit
save/restore and rejection of a grayscale checkpoint into the color model.
Use the constant-Y synthetic controls to isolate chroma; use real scenes to
assess practical responsiveness without treating colored activation maps as
evidence of recognition.

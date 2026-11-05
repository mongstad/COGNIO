# Issue #8 — combined color diagnostic preview

In `--input-mode ycbcr`, **Diagnostic view** defaults to **Combined Color (RGB
preview)**. The current input and future prediction panels reconstruct color
from their own aligned Y, Cb and Cr planes. Their frame IDs retain their original
meaning; the error panel refers to an earlier prediction and its observed +5
target, not necessarily the prediction currently shown above it.

Y, Cb and Cr selections remain available. Y uses the original fixed grayscale
map; Cb/Cr use signed blue/amber display colors. A `y`-only model still has the
original grayscale panels and no color selector. The primary neural-state
canvas, channel paging and Display ± retain their original behavior.

## Fixed measurement-to-display contract

The conversion lives only in `gui.py`; no acquisition, preprocessing, model,
optimizer, checkpoint, target, timing or metric code changes. Learning remains
in the existing native processed measurement units, not RGB.

The preview assumes **BT.601 Y′CbCr with limited-range code values**. This is
consistent with the C922's reported encoding 1 (601), colorspace 8 (sRGB), and
quantization 0 (default). V4L2 specifies limited-range Y′CbCr as the default for
sRGB webcams: see [Linux colorspace documentation](https://docs.kernel.org/userspace-api/media/v4l/colorspaces-details.html#colorspace-srgb-v4l2-colorspace-srgb).
The UI labels this explicitly as an assumption. This fixed preview does not
auto-select matrices/ranges for other devices, profiles or synthetic sources.
Synthetic controls use the same display assumption; they remain analytic model
measurements, not calibrated color targets.

The [local video source](VIDEO.md) explicitly canonicalizes supported decoded
YCbCr to BT.601 limited-range model codes before learning. Its metadata-aware
preprocessing is separate from this fixed display conversion.

For stored model planes `y = Ycode/255`, `cb = (Cbcode−128)/255`,
`cr = (Crcode−128)/255`, first recover limited-range signal values:

```text
Y′ = (255*y − 16) / 219
Cb = 255*cb / 224
Cr = 255*cr / 224
R′ = Y′ + 1.402*Cr
G′ = Y′ − (0.114*1.772/0.587)*Cb − (0.299*1.402/0.587)*Cr
B′ = Y′ + 1.772*Cb
```

Chroma is already centered at zero; subtracting 128 again would be wrong.
Dividing the training data by 255 did **not** declare full-range video. Limited
Y codes 16/235 become display black/white; code 128 chroma is neutral. Thus Y-only
diagnostics and combined neutral-gray previews deliberately use different luma
display scaling. The BT.601 coefficients act on nonlinear R′G′B′; the preview
applies no additional gamma, linear-light transform, white balance, adaptive
contrast, gamut mapping or display-profile calibration.

All conversion arithmetic uses float64 on a bounded 128² snapshot. Only the
final RGB is clipped to [0,1], multiplied by 255, rounded to nearest (ties to
even) and converted to contiguous `uint8` H×W×3 RGB bytes for Qt. No source
array is modified. Finite predictions outside nominal video range saturate
safely; a non-finite input/result pixel displays magenta. The engine's separate
numerical fault handling is unchanged. Out-of-gamut clipping can hide component
differences; use the numeric component diagnostics to inspect them.

The preview represents the existing **128×128 area-averaged model planes**.
Native YUYV has one Cb/Cr pair per two horizontal Y samples, and the preview
does not restore lost resolution, add chroma measurements or reverse filtering.
It is not raw Bayer, spectral information, physically calibrated color, or
evidence of recognition. Camera settings, reported colorimetry, transfer
functions, resampling and the monitor can all affect appearance.

## Combined error

For each pixel the engine already supplies three squared errors against the
correct observed target. The combined panel displays:

```text
E = ((Ypred−Ytarget)² + (Cbpred−Cbtarget)² + (Crpred−Crtarget)²) / 3
```

These are the original normalized code units. The fixed grayscale heatmap is
black at zero, white at or above 0.25, with clipping only in presentation. It is
neither RGB reconstruction nor error measured in clipped RGB space. Its spatial
mean agrees with the aggregate MSE up to reduction-rounding differences. The
plot, per-component metrics and persistence baseline are independent of view
selection. Changing the selector only renders the latest snapshots; it sends
no command to the learner.

## Reproducible owner trial

From the repository, close any other camera user and run:

```bash
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --input-mode ycbcr --channels 512 --grid 16
# Synthetic color preview without activating the camera:
.venv/bin/python -m cognio --gui --source color-moving --compute cpu --input-mode ycbcr
# Grayscale regression:
.venv/bin/python -m cognio --gui --source moving --compute cpu --input-mode y
```

1. Verify Combined Color is initially selected and both input/prediction panels
   show color previews. An untrained prediction need not resemble its input.
   Check that the error label says mean Y/Cb/Cr squared error, not RGB.
2. Cycle Y → Cb → Cr → Combined Color while training. Check corresponding labels,
   advancing frame IDs and unchanged meaning of scalar MSE/persistence.
3. Freeze training; wait for updates/s to settle to zero. Cycle all views again.
   Total updates must remain fixed while camera, predictions and scoring continue.
   Channel page and Display ± still operate only on measured hidden activations.
4. Resume, optionally save/restore explicitly, and close normally. Check grayscale
   separately. No camera image capture is needed. See [OWNER_TRIAL.md](OWNER_TRIAL.md)
   for the longer controlled-scene learning protocol.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
QT_QPA_PLATFORM=wayland .venv-cuda/bin/python -m tools.validate_gui \
  --source camera --compute cuda --input-mode ycbcr --channels 512 --grid 16 \
  --seconds 30 --report /tmp/cognio-issue8-c922.json
QT_QPA_PLATFORM=offscreen .venv-cuda/bin/python -m tools.validate_gui \
  --source color-moving --compute cuda --input-mode ycbcr --channels 512 --grid 16 \
  --seconds 25 --report /tmp/cognio-issue8-synthetic.json \
  --screenshot /tmp/cognio-issue8-synthetic.png
```

The scripted trial checks all four diagnostic selections both while training
and frozen, exact snapshot-to-panel mapping, unchanged frozen weights/Adam,
continued hidden evolution/prediction, page/range controls, save/restore/resume
and clean worker shutdown. Camera screenshots are disabled. Human judgment of
scene colors and responsiveness remains part of owner review.

## Performed validation — 2026-11-05

Accepted main was `dcbbeb37841e11621525a5aab4cb6955e1ab413f` (merged PR #7).
The already assigned issue-8 branch was reused. No dependencies were added.
The diff confirms acquisition, normalization, engine, model, runtime, headless
entry point and checkpoint code are unchanged.

- Accepted-base CPU suite: **88 passed, 8 CUDA skips** (20.34 s).
- Final CPU suite: **95 passed, 9 CUDA skips** (21.83 s).
- Final actual RTX 3070 Ti CUDA suite: **104 passed** (26.80 s).
- Compilation and `git diff --check`: passed. Final suites ran concurrently;
  their durations are not CPU/GPU performance comparisons.

Fixtures check neutral gray, rounded limited-range red/green/blue code samples
(within one display code), exact fractional primary/mixed-color reconstruction,
component order, noncontiguous CHW input, HWC RGB output, clipping, invalid
pixels, and no source mutation. GUI checks compare every diagnostic panel with
its corresponding snapshot and preserve the grayscale mapping. Paired engines
on both CPU and CUDA receive 32 identical frames, with training then freeze;
cycling every diagnostic produces **bitwise-identical predictions, scored error
maps, weights, hidden state and Adam state**, and identical aggregate/component
histories and counters. No learner command is sent by the selector. Existing
checkpoint, headless and future-target causality tests passed in the full suites.

The **30.29-second visible Wayland C922** trial at 512 channels × 16² passed
all four views while learning and frozen, frozen weight/Adam checks, continued
prediction/hidden evolution, paging/range controls, save/restore/resume and
shutdown. Camera metadata was YUYV 640×480 at 30 FPS, colorspace 8, encoding 1,
quantization 0, transfer 1. All six changed controls were restored; both workers
stopped, with no failures or camera warnings. Window exposure and panel contents
were checked programmatically; camera imagery was not captured for visual review.

After 22 seconds, 33 steady samples averaged:

| Measurement | Result |
| --- | ---: |
| Capture / predictions / updates per second | 30.00 / 30.02 / 29.86 |
| Forward / optimizer update / engine step | 0.68 / 6.78 / 8.49 ms |
| Preprocessing / observer copy / display refresh | 6.21 / 0.052 / 2.90 ms |
| Displayed input age / matched prediction horizon | 107.2 / 166.4 ms |
| Peak sampled RSS / PyTorch allocated VRAM | 2260.7 / 228.4 MiB |

The 250 ms validation timer's largest interval was 257.5 ms; the window had
960 refreshes including selector-triggered refreshes. There were four handoff
overwrites, zero driver/drained/invalid frames, two missed and zero late targets.
Final model counters were 831 predictions, 809 scores and 640 updates; restore
rewinds model counters while capture remains per process (894 frames). This
interactive trial includes deliberate freezing and checkpoint operations; whole
run update counts are not a steady-throughput measurement. Memory peaks include
the validator's frozen-state copies and checkpoint restoration.

A **25.06-second offscreen synthetic CUDA** trial at the same dimensions passed
the same controls. Its 1320×940 synthetic-only screenshot was visually inspected:
combined input and prediction are distinct color panels, the error is explicitly
labeled as mean component error, and the primary hidden-state canvas, controls
and telemetry fit. No scene screenshot, checkpoint or temporary log is committed.
Compact scalar evidence is in [evidence/issue-8/validation.json](evidence/issue-8/validation.json).

These checks establish display mapping and numerical separation for identical
delivered frames, not calibrated sensor RGB or better learning. An overloaded
host can still change live frame delivery. No long soak, physical disconnect,
controlled color-target study, monitor calibration or alternative-camera matrix/
range validation was performed. CPU and CUDA use the existing different PyTorch
builds; cross-backend bitwise equivalence is not claimed. RSS maxima are sampled;
PyTorch memory excludes the CUDA context and other consumers. Manual scene-color
judgment and the longer owner trial remain owner verification.

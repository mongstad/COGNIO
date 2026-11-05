# Local video input — issue #9

COGNIO can observe a local video sequentially with the existing model, optimizer,
five-frame target and persistence baseline. This is processed recorded imagery,
not live camera measurements, native C922 YUYV, raw Bayer or spectral data.
There are no labels, audio processing, downloads, network ingestion or automatic
recordings. Playing an educational video does not establish language, object
or conceptual understanding.

## Backend and launch

Video mode requires **FFmpeg and ffprobe on PATH**, tested with Fedora's FFmpeg
8.1.3. The backend needs the `showinfo`, `scale` and `format` filters and the
file's codec decoder. No Python dependency was added; camera and synthetic modes
do not invoke FFmpeg. A Fedora FFmpeg build with the required codecs is needed;
missing tools/codecs and invalid files produce visible errors rather than a
synthetic fallback. No driver or CUDA installation changes are required.

Supported containers are local MP4/MOV, MKV/WebM, AVI, Ogg and NUT. Support is
also subject to the installed decoder. This first implementation accepts
progressive **8-bit planar YUV 4:2:0, 4:2:2 or 4:4:4**, up to 3840×2160 pixels,
with SDR BT.601/BT.709 matrix metadata. RGB, high-bit-depth/HDR, interlaced,
changing-resolution/format/colorimetry and unreliable timestamp streams fail
explicitly. Stored raster orientation is used (`-noautorotate`); display rotation
metadata is not applied. Audio/subtitles/data streams are ignored. Network
protocols and playlist demuxers are excluded; a regular local file is required.

From the repository, using the existing CPU/CUDA environments:

```bash
.venv-cuda/bin/python -m cognio --gui --source video --video-file /path/to/clip.mp4 \
  --input-mode ycbcr --compute cuda --channels 512 --grid 16 --video-speed 0.1 --video-loop
# Omit the file only in GUI mode to open a native file chooser:
.venv/bin/python -m cognio --gui --source video --compute cpu --input-mode ycbcr
# Headless: stop automatically at EOF; fastest sequential processing:
.venv/bin/python -m cognio --headless --source video --video-file /path/to/clip.mkv \
  --compute cpu --video-fastest --save-on-exit /tmp/cognio-video.pt
# Resume the learned architecture/mode on another clip; explicitly enable training:
.venv-cuda/bin/python -m cognio --headless --source video --video-file /path/to/another.mp4 \
  --compute cuda --checkpoint /tmp/cognio-video.pt --resume-training --video-speed 1
```

`--input-mode y` retains a single model plane. `ycbcr` observes all three canonical
planes. Existing configuration, CPU/CUDA selection, training freeze, checkpoints
and optional observation remain available. A file is not embedded in a checkpoint.

## Playback controls and timing

The GUI provides Pause video, Playback × (0.1–4), Repeat video and Restart video.
These are independent of Train online. Freezing training retains sensory delivery,
prediction and scoring; pausing video supplies **no new observations**, recurrent
updates, predictions or training pairs. A paused displayed image is not repeatedly
fed as a new frame. Restart returns to the beginning and retains the pause state.
There is no arbitrary seek or start/end trimming in this implementation.

Headless users can opt into bounded newline-delimited stdin controls:

```bash
.venv/bin/python -m cognio --headless --source video --video-file /path/to/clip.mp4 \
  --compute cpu --video-controls --video-paused
# Type one command per line in that terminal:
# resume
# pause
# speed 0.1
# loop on
# restart
# train off
# train on
```

`loop off` is also accepted. Ctrl-C/SIGTERM or `--seconds` stops the run. EOF ends
headless runs normally unless repeat is enabled; the GUI remains open for restart,
inspection and explicit saving. Invalid stdin commands report an error without
changing playback; the command buffer is capped at 1024 bytes. `--video-paused`
requires GUI controls or explicit `--video-controls`. No stdin is read by default.

The decoder emits every presentation frame once, with its original ordered PTS
and stream time base, using `-fps_mode passthrough`. It neither duplicates frames
to a nominal FPS nor skips frames to catch up. Equal images at distinct genuine
presentation timestamps are valid observations. Missing, duplicate or backward
PTS and mid-stream format changes fail instead of inventing timing. VFR positive
intervals determine the paced gaps; there is no assumed constant media cadence.

At 1×, successive delivered frames are separated by at least their media-time
interval; at 0.1×, by at least ten times that interval. Pauses stop the pacing
clock. Speed changes rescale the remaining media interval. Processing or decoding
that overruns a deadline delays playback; it does not drop or duplicate frames,
and excess elapsed time is discarded rather than causing catch-up bursts. Thus
requested speed is an upper rate, not a guaranteed real-time playback rate.

`--video-fastest` removes only the pacing delay. Decoder delivery and learning
run sequentially on one worker, with no video capture thread overwriting frames.
A new observation is timestamped with the actual monotonic wall clock **after
the previous model step has completed**. Codec prefetch/reordering can happen in
the bounded decoder, but those frames cannot reach the model early. The engine's
real `issued_time < target_observation_time` guard is retained in paced and fastest
modes; there is no fabricated future clock or disabled causality check. Fixed
frames/configuration produce the same learned trajectory with/without display
or pacing in a fixed backend; wall timing and cross-backend results can differ.

Sequence numbers increase globally across repeats/restarts. Each pass has a
separate segment and media-frame index. At a new pass, pending predictions, error
history and recurrent hidden state are cleared; weights, Adam state, training
flag and progress counters remain. There are no cross-pass targets. The first
five frames of each pass warm up pairing. At final EOF, unresolved tail predictions
are counted as cleared, while scored errors remain visible. Saved counters rewind
on restore as before; playback position is not restored.

Every accepted prediction still targets exactly `s+5`. The error panel/headless
record separately reports **media horizon** and **wall horizon/lead**. Five frames
of 30 FPS content span roughly 167 ms of media even when slow replay takes about
1.67 seconds or fast execution only a few milliseconds. Existing `horizon_ms`,
`source_time`, `target_time` and `issued` remain wall-clock quantities. Added
`media_horizon_ms`, source/target media indices and segment describe content time.
VFR horizons use the actual two matched PTS values. Input FPS is actual delivery
rate, not the container's nominal/average FPS. The GUI labels processed video,
pass, media position, pause/EOF and requested pacing.

## Video-to-model measurement contract

FFmpeg decodes the first video stream and area-resamples the full stored raster
to 128×128 planar YCbCr without an RGB image intermediary. Original YUV chroma
subsampling is recorded; resampling to three aligned planes does not create new
independent chroma measurements or imply the file originated as camera 4:2:2.
This backend's area filter produces 8-bit resampled codes; unlike the camera's
float64 area integration, that stage has an additional 8-bit rounding step.
The existing full-frame-to-square spatial mapping is preserved.

Matrix/range resolution is explicit. Tagged BT.709 or BT.601 (SMPTE170M/BT470BG)
and limited/full range are honored. Missing matrix metadata falls back to BT.601
for stored height ≤576, otherwise BT.709. Missing range falls back to limited,
except `yuvj*` formats imply full. Fallbacks appear in metadata and GUI warnings.
Per-frame metadata must remain compatible with the resolved contract. Primaries
and transfer tags are recorded and unsupported HDR/wide-gamut tags rejected.
Missing tags are reported; there is no gamma linearization, primary-gamut
conversion, white-balance adjustment or color calibration.

After resampling in source code units, `normalize_video` performs an explicit
**direct YCbCr-to-YCbCr affine transform**, composing two 3×3 coefficient matrices
without allocating RGB pixels. Source limited range expands Y using `(Y−16)/219`
and chroma using `(C−128)/224`; full range uses `Y/255` and `(C−128)/255`. BT.709
uses luma coefficients (0.2126, 0.7152, 0.0722), BT.601 (0.299, 0.587, 0.114).
The destination is BT.601 limited-range codes, then the unchanged model units:
`Ycode/255`, `(Cbcode−128)/255`, `(Crcode−128)/255`. Fractional transformed values
are retained as float32; only the final representable model-code bounds are
clipped, not the narrower nominal limited range. Native BT.601 limited codes
take the direct normalization path. Grayscale takes the same canonical Y plane.
The RGB diagnostic can therefore keep its fixed BT.601 limited-range assumption.

This canonicalization is deterministic preprocessing, not a learned encoder or
an RGB training target. Camera normalization is unchanged. Different transfer
functions/primaries, compression, clipping and resampling still limit physical
color equivalence across sources. Checkpoint compatibility means matching model
shape and code-unit objective, not interchangeable physical camera/video scenes.

Schema-3 checkpoints remain compatible across camera, synthetic and video sources
with matching architecture/input mode. They store learned state, not file paths,
media position, raw frames, decoder buffers or pending predictions. The historical
`sensory.native_shapes` field describes the baseline camera adapter; it is not
provenance or a claim about the file's native geometry. Restore retains the saved
hidden state initially, as before; an explicit video restart/repeat clears it.

## Bounds and cleanup

The application holds at most one preprocessed upcoming video frame plus the
engine's existing five contexts. The raw pipe accumulator is checked against two
128² three-byte frames; stderr lines/output are capped at 64 KiB, diagnostic text
at eight short lines, and frame metadata at 32 records. FFmpeg is constrained to
two decoder threads, one filter thread, eight buffered filter frames, an output
queue of one packet and the 4K pixel budget. Codec reference frames and framework
overhead add memory, but no whole-file cache or growing sensory history is built.

The metadata probe is capped at 3 seconds/64 KiB output with bounded probing;
decode has a ten-second complete-frame timeout and interruptible waits. Both
subprocesses are reaped on errors, stop and EOF. Corrupt/truncated-container
diagnostics are treated as failure even when FFmpeg returns success for a partial
file. A GUI input error leaves checkpoint controls available. Reported application
RSS excludes the decoder subprocess; validation measures both separately.

## Validation and owner trial

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
.venv/bin/python -m tools.make_video_fixture /tmp/cognio-color.mkv --stimulus color --frames 120
.venv/bin/python -m tools.validate_video --compute cpu --report /tmp/cognio-video-cpu.json
.venv-cuda/bin/python -m tools.validate_video --compute cuda --channels 512 --grid 16 \
  --report /tmp/cognio-video-cuda.json
QT_QPA_PLATFORM=wayland .venv-cuda/bin/python -m tools.validate_gui --source video \
  --video-file /tmp/cognio-color.mkv --compute cuda --input-mode ycbcr --channels 512 --grid 16 \
  --seconds 30 --report /tmp/cognio-video-gui.json
```

For an owner-supplied file, run the first GUI command above with its local path.
Compare 1× and 0.1× rates; inspect both horizons, frame indices, color diagnostics
and warnings. Pause must stop all observations; Train online off must leave
playback/scoring active. Repeat/restart must show a new pass and five-frame
warm-up. Test EOF with repeat off, save/restore explicitly, and close normally.
Use a known SDR clip first; no copyrighted video or frames need to be committed.
Generated clips and files are temporary; no owner video is bundled.

Backend references: [FFmpeg showinfo and scale filters](https://ffmpeg.org/ffmpeg-filters.html),
[frame passthrough and decoder options](https://ffmpeg.org/ffmpeg.html),
and [protocol restrictions](https://ffmpeg.org/ffmpeg-protocols.html).

## Performed checks — 2026-11-05

Base: accepted main `c84b8e1acb3c90f1b851fe0fbde1d73e53e8fe24`, merged PR #10.
The existing assigned issue-9 branch was reused. Python 3.14.10, FFmpeg 8.1.3,
CPU PyTorch 2.15.0+cpu, CUDA PyTorch 2.14.1+cu130 on RTX 3070 Ti. Numerical
controls used seed 7, float32, two CPU threads, deterministic algorithms and
TF32/autotuning off. No dependency or driver changes were made.

Accepted-base CPU suite: **95 passed, 9 skips**. Final suites: **CPU 130 passed,
11 CUDA skips (47.50 s)**; **actual CUDA 141 passed (53.15 s)**. Suites ran
concurrently; durations are not CPU/GPU performance comparisons. Compilation,
whitespace and local documentation-link checks passed.

Tests exercise exact decoded FFV1 frame order/values in both sensory modes,
known neutral/primary colors under all four BT.601/709 limited/full-range
combinations at 420/422/444 sampling, H.264 MP4 with B frames, actual VFR PTS,
exact +5 error/persistence pairing, missing/invalid timestamps, pause/speed
changes, 1×/0.1× wall pacing, overload without skipping, repeat/restart isolation,
EOF, missing/corrupt/truncated/10-bit files, shutdown, headless stdin controls
without Qt, GUI controls, observer independence and CPU/CUDA checkpoint/freeze
behavior. A restored checkpoint continues exactly within a later video pass.
The existing camera, synthetic, capacity, checkpoint and future-leakage suites
remain green. A generated-only 1320×940 video GUI screenshot was inspected;
playback controls, separate horizons and four views fit without overlap.

The generated learning protocol decoded 120 initial frozen, 600 training and
120 final frozen frames per stimulus. Each training phase performed 595 updates;
each evaluation had 115 matched pairs. Each phase began with zero temporal
context; learned parameters/Adam carried forward. Frozen evaluation uses a new
pass of repeated content, not an unfamiliar-scene holdout. Final errors were:

| Configuration | Stimulus | Initial frozen MSE | Final frozen MSE | Persistence MSE |
| --- | --- | ---: | ---: | ---: |
| CPU 32×8 | Static shape | 0.032631 | 0.000284 | 0 |
| CPU 32×8 | Translating shape | 0.032664 | 0.007804 | 0.011224 |
| CPU 32×8 | Changing color | 0.016515 | 0.000465 | 0.009164 |
| CUDA 512×16 | Static shape | 0.032296 | 0.001272 | 0 |
| CUDA 512×16 | Translating shape | 0.032267 | 0.007814 | 0.011224 |
| CUDA 512×16 | Changing color | 0.015876 | 0.001699 | 0.009164 |

Both configurations beat persistence on these repeated moving/color controls;
static persistence remains exact and stronger. Larger capacity is not uniformly
better. No numeric faults or cross-pass pairs occurred. The 600-frame training
phases' first/last sampled RSS changed by less than 1 MiB; application peaks
across phases were about 387 MiB CPU and 2016 MiB CUDA, with decoder peaks about
67 MiB. CUDA peak allocated memory was 171.6 MiB. These short sampled results
support the bounded design but do not prove long-term leak freedom or 4K capacity.

Separate 30-frame paced decoder/learner controls measured **29.53/29.55 FPS** at
1× (CPU/CUDA), and **2.995/2.995 FPS** at 0.1×. Mean media horizon remained
166.68 ms; mean wall horizon was 169.29/169.18 ms at 1× and 1669.08/1669.10 ms at
0.1×. Every scored target followed its prediction's real wall-clock issuance.
These measurements use the sequential validation driver; GUI/runtime scheduling
adds overhead and requested speed is an upper rate.

The final **31.10-second visible Wayland video GUI** at 512×16 passed pause/resume,
speed change to 0.1× and back, repeat (seven passes reached), all diagnostic
views, training freeze, checkpoint restore/resume and shutdown. It delivered
735 frames, with zero handoff overwrites, skipped or late targets. Restored model
counters ended at 679 predictions, 629 scores and 487 updates; 45 contexts were
cleared at intended phase/pass/checkpoint boundaries. Peak sampled application
RSS was 2253 MiB and PyTorch allocation 228.4 MiB, including validator copies.
After 22 s, 33 samples averaged 27.74 delivered frames / 27.74 predictions /
26.45 updates per second (including repeat warm-ups); forward/update/display
costs averaged 0.60/6.23/2.11 ms and displayed input age 59.7 ms.
No errors or color fallback warnings occurred for the fully tagged fixture.

A separate **30.26-second visible C922 color GUI regression** passed all controls,
restored all six changed camera settings and stopped both workers. After 22 s,
33 samples averaged **30.00 capture / 29.99 predictions / 29.85 updates per second**.
It recorded one handoff overwrite, one missed target, zero late/driver-dropped/
invalid frames; peak sampled RSS was 2268 MiB and PyTorch allocation 228.4 MiB.
The live window and control behavior were checked programmatically; no camera
imagery was captured for review. All validation windows/processes were closed.

Development checks caught and corrected three relevant pitfalls: fixture input
color metadata had to be explicit to prevent encoder-side conversion; truncated
MKV could otherwise be accepted as EOF despite FFmpeg returning success; and
later-pass checkpoint restoration must not clear saved hidden state. The GUI
validator also needed to allow intentionally empty error views at loop warm-up.
All final regressions and the final video GUI trial passed after corrections.

Scalar records are in [evidence/issue-9](evidence/issue-9/). No owner-supplied local
file path was provided; pointing the webcam at a video exercised the camera
path, not direct file decoding. Testing an owner file with the command/protocol
above remains explicit owner verification. No long soak, physical disconnect,
color-target calibration, actual 4K playback, exhaustive codecs or cross-version
bitwise reproducibility study was performed. Audio, arbitrary seeking and HDR
remain outside this implementation. No private/copyrighted media, frame captures,
checkpoints or temporary logs are committed.

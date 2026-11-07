# Experiment setup and recurrent topology — issue #14

GUI launches now open configuration, including launches with explicit CLI flags.
No model, optimizer, camera stream, decoder or training update exists before
**Start**. Device discovery only queries capture identity. CUDA availability and
free-memory queries do not test learning; the exact training probe runs after
Start. There is no GUI direct-start flag. Headless launch behavior remains
unchanged: synthetic static input, Standard 32×8, Y, training enabled.

```bash
# Configuration screen; edit choices, then explicitly press Start.
.venv-cuda/bin/python -m cognio --gui
# Prefill a small experiment; it still waits for Start.
.venv/bin/python -m cognio --gui --compute cpu --source moving \
  --channels 32 --grid 8 --input-mode y --topology standard
# Existing scripts remain Standard unless explicitly changed.
.venv-cuda/bin/python -m cognio --headless --source moving --compute cuda \
  --channels 128 --grid 64 --input-mode ycbcr --topology toroidal --seconds 30
```

## Configuration and capacity

Fresh GUI defaults are Toroidal, 1,024 channels, 32×32 positions, YCbCr, live
camera, seed 7, Adam learning rate 0.001 and the unchanged +5-frame objective.
CUDA is selected when detectable, otherwise CPU. Before starting CUDA, the
application checks the selected configuration's budget and exercises that exact
model through two causal updates, including contexts, backward, Adam, rollback
copies and finite guards. Probe weights are discarded; the session is freshly
initialized from the displayed seed. Probe failures leave the setup open with
an actionable error; CPU can be selected explicitly. There is no silent size
reduction or GUI CUDA-to-CPU fallback.

Channels are a positive integer text entry, with no arbitrary 512/1,024 ceiling.
Grids 8/16/32/64 are supported (16/8/4/2 input pixels per patch). The 64 grid
has real patch-order, forward/backward and checkpoint tests. Other grids are
rejected. The form recalculates parameter count, recurrent values and available
working budget on edits. Start checks current memory again; estimates can change
as other processes allocate memory. It never selects the largest model for you.

For channels C, grid G, input components K and P=128/G:

- Parameters: `18*C*C + (2*K*P*P + 20)*C + K*P*P`.
- Hidden values: `C*G*G`.
- Standard working allowance (bytes): `40*parameters + 128*hidden_values +
  512*1024**2 + 5*(K-1)*2*128**2*4`.
- Toroidal adds `16*(2*C+2)*(G+2)**2` bytes for padded joint tensors and their
  backward/transient copies. Learned parameter shapes and counts are unchanged.

The additional allowance is applied to CPU RAM and, for CUDA, GPU VRAM.
The GUI also displays approximate total process RAM as current RSS plus the
allowance; framework/context mappings are not learned model storage. The safe limit is
half available host RAM and at most 60% of free VRAM, further reduced by an
optional user MiB limit. It covers weights, gradients, Adam, rollback and restore
transients, five contexts, activations and workspace. It is a conservative
preflight estimate, not an exact peak or a guarantee against changes in external
memory pressure or backend workspace. Startup exceptions return to configuration;
runtime numerical/update faults remain visible and retain the recoverable model.
Large configurations may run much slower than capture, with honest skipped-frame
and target counters. A budget fit never promises 30 updates/s.

## Architectural meaning and saved sessions

Standard uses the identical seeded modules and zero-padded 3×3 recurrent
convolution from accepted main. Its default forward output is bitwise compared
against that recurrence. Toroidal uses PyTorch circular padding on both spatial
axes of that convolution only. Encoder, decoder, component/patch order, fixed
absolute x/y coordinates, leaky hidden update, loss, optimizer and +5 pairing are
unchanged. Coordinates themselves are not periodic: this is a wrapped connection
rule, not a translation-equivariant representation or a geometric 3D torus.
Camera images are not periodic; unrelated edges can interact artificially.

Topology is immutable for a session. Equal dimensions and seed initialize the
same parameter values/shapes/counts, but different connectivity makes the
architectures distinct. Checkpoint schema 4 (`patch-spatial-rnn-v4`) requires
`config.topology` and the sensory contract. Missing/unknown topology fails.
Schemas 1, 2 and 3 migrate explicitly to Standard; old files are not rewritten.
Schema 4 Standard also restores normally. Cross-topology live restore is rejected
before candidate construction, leaving current weights/state untouched. There is
no weight conversion or cross-topology migration option.

**Restore Checkpoint…** reads bounded, restricted, memory-mapped checkpoint
metadata and shows its saved architecture before **Restore & Start**. Architecture,
seed and saved learning rate are locked; full tensor/Adam validation happens when
starting. If the configuration changes on disk after inspection, startup refuses
it. CLI `--checkpoint` prefills the same screen; explicitly conflicting dimensions,
mode or topology are errors. Restore starts frozen unless the owner explicitly
unchecks the freeze option or supplied `--resume-training`. `--reset-hidden`
remains an explicit checkpoint-only option.

## Session lifecycle and owner trial

1. Launch `--gui`. Verify configuration is visible, channels are editable both
   below and above 1,024, grid 64 is selectable, and estimates change. With no
   unique C922, the screen explains how to refresh/select a camera or change
   source. No camera controls are written during discovery.
2. For a short synthetic run, select moving, CPU/available CUDA and 32×8.
   Press **Start**. Architecture/preprocessing fields disappear from the running
   view and are locked. The four measured views, diagnostic component selector,
   hidden-channel pages and performance/drop statistics remain available.
3. Freeze training: updates stop while input, predictions and hidden state
   continue. Save to a new local checkpoint. Live restore accepts only matching
   dimensions, input mode and topology. Try an incompatible topology file and
   confirm the existing session remains intact.
4. Select **Stop / New Session…**. Cancel first: the existing session continues.
   Select again and confirm discarding in-memory learning. Save beforehand if
   needed. The setup returns only after capture/learner threads and FFmpeg stop;
   existing checkpoint files are unchanged. Choose another topology and Start,
   or inspect a checkpoint and restore its immutable architecture.
5. Select local video with no path and Start: a chooser opens; cancel leaves
   configuration untouched. Choose a clip, verify pause/speed/repeat, then New
   Session and confirm its decoder exits. Close the application to stop/exit.
6. For a physical C922, close other camera consumers and run the command below.
   Check the camera indicator stays off until Start and turns off after New
   Session/Exit. Confirm the four views, freeze/resume, checkpoint controls,
   negotiated YUYV input, measured FPS, drops and responsiveness on your scene.

```bash
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --channels 1024 --grid 32 --input-mode ycbcr --topology toroidal
# Optional persistent camera path: --device /dev/v4l/by-id/YOUR-CAMERA-video-index0
```

No automatic checkpoint overwrite, media recording or network transmission was
added. A fresh session uses the displayed seed (reproducible initialization);
the existing running random-reset control still confirms and chooses a new seed.
Closing exits without auto-saving, as before. Checkpoints may contain learned
scene information and are not anonymized.

## Reproducible validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
.venv-cuda/bin/python -m tools.benchmark_capacity --compute cuda \
  --configs 32x8 512x16 1024x32 1536x32 128x64 --input-mode ycbcr \
  --topologies standard toroidal --stimuli moving shape \
  --train-frames 80 --evaluation-frames 40 --max-seconds 40 \
  --output /tmp/cognio-topology-capacity.json
.venv-cuda/bin/python -m tools.validate_setup --compute cuda --channels 1024 \
  --grid 32 --source moving --report /tmp/cognio-setup-gui.json
# Physical camera; report is scalars only, no screenshots/media.
.venv-cuda/bin/python -m tools.validate_setup --compute cuda --channels 1024 \
  --grid 32 --source camera --report /tmp/cognio-setup-c922.json
```

Capacity/comparison cases use seed 7, float32, two CPU threads, deterministic
algorithms, TF32 disabled and a virtual 30 Hz source clock. Each includes 40
initial frozen, 80 training and 40 final frozen frames. Predictions precede their
exact targets; frozen metrics compare with persistence on later frames of the
same repeated control. Moving is a deliberately spatially periodic grating;
shape is a localized translating Gaussian with nonperiodic edges. These generated
patterns are controls, not evidence of natural-scene generalization. Wall rates
are separate from virtual sequence time. Sampled process RSS includes framework
and prior allocator retention; PyTorch peak VRAM excludes CUDA context and other
applications. Time-limit cases are incomplete, explicitly labeled as such.

Measured results and unavailable hardware checks are recorded below. No claim of superior toroidal learning is implied by connectivity
or attractive activation maps.

## Measured results (2026-11-07)

Hardware: Ryzen 7 5800X, RTX 3070 Ti 8 GiB; PyTorch 2.14.1+cu130.
Scalar evidence: [CUDA matrix](evidence/issue-14/capacity-cuda.json),
[CPU matrix](evidence/issue-14/capacity-cpu.json),
[visible CUDA GUI trial](evidence/issue-14/setup-gui-cuda.json).

All 20 CUDA cases completed, with finite updates at every requested size. The
following periodic-control measurements are **Standard / Toroidal**, YCbCr.
Training-phase rates divide updates by training-phase wall time (including
five warm-up frames); whole-run rates in JSON also include frozen phases.

| Channels × grid | Working allowance MiB | Peak PyTorch VRAM MiB | Training updates/s | Final frozen MSE | Persistence MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 × 8×8 | 516 / 516 | 8 / 8 | 189.4 / 166.8 | 0.010276 / 0.009098 | 0.004393 |
| 512 × 16×16 | 717 / 722 | 172 / 174 | 123.6 / 132.6 | 0.002272 / 0.001779 | 0.004393 |
| 1024 × 32×32 | 1366 / 1402 | 877 / 879 | 43.2 / 38.7 | 0.007497 / 0.004632 | 0.004393 |
| 1536 × 32×32 | 2332 / 2386 | 1908 / 1912 | 21.6 / 17.6 | 0.004208 / 0.004455 | 0.004393 |
| 128 × 64×64 | 589 / 606 | 43 / 43 | 176.4 / 156.6 | 0.006583 / 0.006563 | 0.004393 |

At 512×16 both topologies beat persistence on the periodic control. Most other
short-run configurations did not; Toroidal 1,536×32 was worse than Standard on
that control. Every nonperiodic moving-Gaussian final frozen MSE exceeded its
persistence baseline (0.003670). This single-seed, short repeated-control study
does not show general superiority of either topology, durable learning, or
natural-camera generalization. Apparent channel saturation remains possible
with this unchanged recurrent model and optimizer.

GPU peak tensor allocations were below the working allowance in this matrix.
Sampled process RSS was about 1,789–1,874 MiB, including CUDA/framework mappings
and retained allocator memory; this is larger than small-model working allowances.
The updated setup separately labels additional working allowance and approximate
total host RSS. Framework first-use mapping and external pressure mean neither
estimate is an exact peak guarantee.

The bounded CPU matrix used the same configurations/seed and stimuli but shorter
10 frozen / 16 training / 10 frozen phases with 30 seconds per case:

```bash
.venv-cuda/bin/python -m tools.benchmark_capacity --compute cpu \
  --configs 32x8 512x16 1024x32 1536x32 128x64 --input-mode ycbcr \
  --topologies standard toroidal --stimuli moving shape \
  --train-frames 16 --evaluation-frames 10 --max-seconds 30 \
  --output /tmp/cognio-topology-cpu.json
```

- 32×8: 215.72–223.20 training updates/s; sampled RSS 755–757 MiB. All four cases completed.
- 512×16: 8.44–8.82 training updates/s; sampled RSS 894–911 MiB. All four cases completed.
- 1024×32: 0.80–0.82 training updates/s; sampled RSS 1052–1066 MiB. All four cases completed.
- 1536×32: 0.35–0.36 training updates/s; sampled RSS 1374–1400 MiB. All four cases hit the time limit after 10 valid updates; no final frozen evaluation.
- 128×64: 13.82–15.92 training updates/s; sampled RSS 837–837 MiB. All four cases completed.

The limit is checked between complete steps, so a final slow CPU step can cross
30 seconds. The four 1,536×32 CPU cases are incomplete evaluations, not failures
of finite allocation/update. CPU/GPU error values from unequal training lengths
must not be compared as a device-accuracy experiment.

The visible Wayland GUI trial ran 40 seconds at Toroidal 1,024×32, YCbCr, CUDA
with paced synthetic moving input. It verified no runtime before Start, real
training, freeze, exact weight/Adam preservation, save, live restore, resume,
cancelled/confirmed New Session, joined workers, setup checkpoint inspection and
frozen restoration. All workers stopped on exit. Active samples averaged
30.00 input FPS and 29.92 prediction FPS, 3.28 ms prediction and 21.28 ms update
latency. Rolling update-rate samples averaged 19.96/s (maximum 30.00/s); the
rolling window includes freeze transitions. Sampled peak RSS was 2,295 MiB;
steady tensor allocation about 242 MiB, reserved GPU memory 1,046 MiB. The
maximum 200 ms UI sampling gap was 308 ms. One source frame was skipped and one
exact target missed; zero late targets were reported. These are measured
synthetic session results, not C922 performance.

A direct visible launch inside the sandbox aborted before validation; the
same trial ran successfully with desktop/Wayland access. Synthetic screenshots
were inspected locally and were not committed. No private camera imagery or
checkpoint was committed. **The C922 live trial was unavailable**: both identity
enumeration and device-node checks found no camera (`/dev/video*` and
`/dev/v4l/by-id` absent), even with host access. Use the exact camera commands
above after connecting the C922. Physical camera indicator/control cleanup,
unplug/replug, long-duration training, and human mouse interaction remain owner
checks. Automated fake-camera lifecycle tests verify the context is opened only
after Start and released before configuration returns.

The separate 35-second visible Wayland video trial used the generated silent
FFV1 color fixture at Toroidal 32×8, YCbCr, CUDA. Its startup, freeze/save/restore,
New Session and frozen setup restore all passed; workers/decoder stopped. The
maximum 200 ms UI tick gap was 219 ms.
[Video trial evidence](evidence/issue-14/setup-video-gui.json). Reproduce with:

```bash
.venv/bin/python -m tools.make_video_fixture /tmp/cognio-issue14.mkv --frames 90 --stimulus color
.venv-cuda/bin/python -m tools.validate_setup --compute cuda --channels 32 --grid 8 \
  --source video --video-file /tmp/cognio-issue14.mkv --seconds 35 \
  --report /tmp/cognio-setup-video.json
```

Final full regressions: CPU-only environment **170 passed, 18 CUDA skips**
(69.82 s); GPU-enabled environment **188 passed** (79.92 s). After a final
presentation adjustment to keep capacity/Start visible in a fixed footer, all
**8 setup lifecycle tests passed again** (4.47 s), and the default Wayland
configuration screen was visually inspected. Instrumentation confirmed zero
Predictor constructions and zero Runtime starts before Start. The only failed
regression during development was a new CLI test incorrectly reading the final
checkpoint-save event as telemetry; it was corrected to select a telemetry
record, then both full suites passed. `git diff --check` and CLI help passed.

Final review also checked the opt-in grid diagnostics at the new 64×64 size.
Its two-pixel patch fundamental lies at the Fourier Nyquist bin, so the positive
and negative frequencies must be counted only once. That diagnostic was corrected
without changing the learning path or earlier grid results. After this correction
and the footer layout change, the focused GPU-enabled diagnostics/setup suites
passed **22 tests** (15.11 s).

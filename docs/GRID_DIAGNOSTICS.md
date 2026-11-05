# Prediction-grid investigation — issue #11

The owner-supplied affected checkpoint reproduces a strong four-pixel grid in
numerical Y predictions under frozen training and generated inputs. Clearing
hidden state does not provide a lasting recovery: the learned dynamics rapidly
return to saturation and periodic output. The artifact precedes display-only RGB
conversion. The original onset in the owner's long camera run remains unknown.

This change provides an explicit transient-state reset and reproducible scalar
diagnostics. It does not change architecture, learning rate, objective, Adam,
activation functions, camera preprocessing, RGB reconstruction or normal restore
behavior. It does not claim to repair the affected learned weights.

## Preserve and compare checkpoints

Keep known-good and affected checkpoints as separate originals. Checkpoints can
encode learned scene information. The tool below only reads the supplied file,
checks SHA-256 before/after, and writes scalar JSON to a **new** destination.
It refuses to overwrite any existing report path, including a checkpoint.
It neither saves modified checkpoints nor records camera frames or predictions.
The report contains learned-state statistics, not anonymized scientific evidence;
review it before sharing. No owner images or checkpoint files belong in Git.

```bash
# Same weights and Adam, frozen; compare saved versus zero hidden state.
.venv-cuda/bin/python -m tools.investigate_grid \
  --checkpoint /path/to/affected.pt --compute cuda \
  --scenarios steady cuts black --frames 180 --sample-every 100 \
  --max-seconds 180 --report /tmp/cognio-affected-ab.json
# Repeat with a known-good checkpoint and a different NEW report path.
# CPU is supported: use --compute cpu (and the available Python environment).
```

Each A/B case loads the same checkpoint anew, freezes learning, and supplies the
same deterministic sequence. A retains saved hidden state; B explicitly zeros it.
Both keep weights and optimizer identical, verified by exact recursive digests.
Later samples in B show warm-up behavior without a third, differently trained
model. No checkpoint adaptation is performed by this command. Saved dimensions,
mode, seed, optimizer and learning rate govern checkpoint cases; new-model
`--channels`, `--grids`, `--seeds` and `--learning-rates` do not override them.

For a live owner check, launch the same saved model twice, initially frozen:

```bash
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --checkpoint /path/to/affected.pt
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --checkpoint /path/to/affected.pt --reset-hidden
```

Run these sequentially, closing the first window before the second. The GUI's
**Reset transient hidden state only…** asks for explicit confirmation and clears
hidden recurrence plus pending predictions/error history. Learned weights,
Adam, training/fault status, sequence clocks and counters are retained, except
cleared pending targets are counted. Pairing needs five new observations. The
control does not clear a numerical fault or silently pause/resume training.
For a frozen comparison keep Train online off. Live scenes cannot guarantee
identical frames across launches; the offline A/B tool supplies that control.

Inspect Y as well as Combined Color. Note whether a change persists after 10–30
frames, rather than judging only the first prediction after reset. A temporary
visual improvement is not restored learning. Ordinary checkpoint load still
retains hidden state; video repeat/restart retains its existing context reset.

## Controlled scene sequences

```bash
.venv-cuda/bin/python -m tools.investigate_grid --compute cuda \
  --channels 1024 --grids 32 --seeds 7 --learning-rates 0.001 0.0001 \
  --scenarios steady smooth cuts black rapid low-light \
  --frames 900 --evaluation-frames 120 --sample-every 150 \
  --max-seconds 480 --report /tmp/cognio-new-model-controls.json

.venv/bin/python -m tools.investigate_grid --compute cpu \
  --channels 32 --grids 16 32 --seeds 7 11 --learning-rates 0.001 \
  --scenarios steady smooth cuts black rapid low-light \
  --frames 900 --evaluation-frames 120 --sample-every 150 \
  --max-seconds 240 --report /tmp/cognio-small-controls.json

# Isolate hard black intervals from bright exposure and smooth dimming;
# repeat at another seed, with identical starting weights within each seed.
.venv-cuda/bin/python -m tools.investigate_grid --compute cuda \
  --channels 1024 --grids 32 --seeds 7 11 --learning-rates 0.001 \
  --scenarios steady bright black black-smooth low-light \
  --frames 900 --evaluation-frames 120 --sample-every 150 \
  --max-seconds 300 --report /tmp/cognio-transition-controls.json
```

For 900 training frames, the first 300 are scene A (a moving Gaussian with smooth
spatial luminance/chroma). Only the next 300 depend on condition; the following
300 return to A, followed by 120 frozen A frames. There are 895 training updates
and 115 frozen scored pairs, with identical input/update counts per complete case.
Training is continuous across scene cuts; pending targets are not reset there.
Changing to frozen evaluation clears pending pairs as in the application.

| Condition | Middle-third stimulus |
| --- | --- |
| `steady` | Continue A, no scene cut |
| `smooth` | Cosine blend between A and unrelated static scene B, 60-frame cycle |
| `cuts` | Hard A/B alternation, 30 frames each |
| `black` | Bright Y=0.85, neutral chroma, five zero-Y frames per 60-frame cycle |
| `rapid` | Hard A/B alternation every five frames |
| `low-light` | A multiplied by 0.02 |
| `bright` | Constant Y=0.85, neutral chroma |
| `black-smooth` | Bright field with ten-frame cosine dips, same integrated Y deficit as `black` |

These are controlled code-domain measurements, not camera recordings or a
physical lighting simulation. New-model comparisons share seed/initial weights;
different grids have different architectures. The lower learning rate is an
experimental comparison, not a new default or a validated recovery treatment.
Evaluation is later generated input from repeated scenes, not unfamiliar-scene
generalization. Zero-Y is a valid input code; nominal video black can differ.

The driver delivers sequential observations with actual monotonic wall timestamps
after the previous step completes. The engine's `issued < target` guard and exact
`s+5` pairing remain active. These are accelerated diagnostic runs, not real-time
camera benchmarks; diagnostic probes add substantial overhead. Frame index sets
the stimulus, and wall time measures execution. CLI deterministic algorithms,
two CPU threads, float32, fixed seeds and disabled TF32/autotuning match project
conventions. Cross-device/version bitwise identity is not promised.

Every invocation has a whole-run time budget (maximum 1800 seconds), up to 32
cases and 2048 sampled scalar records. Frame counts are bounded. Incomplete runs
are reported with `complete=false` and exit status 2; never interpret them as
complete comparisons. The deadline is checked between frames/cases; a current
probe, load, final summary or serialization can extend wall time beyond it.
One model and bounded five-context engine history are
active at a time. No whole sequence or sensory cache is allocated. Scalar
reports can be sizeable; use summaries for committed evidence.

## What the measurements mean

- Prediction and persistence MSE use the same immutable +5 prediction/target
  pairs. Aggregate and component errors are reported before, during, after the
  challenge and in frozen evaluation. Pair lead must remain positive.
- Each output component has min/max, mean/std, ten-bin histograms and shares
  within 0.01 of sigmoid endpoints (Cb/Cr offset undone for this measurement).
- Patch phase RMS is the standard deviation of the mean prediction at each
  `(row mod patch, column mod patch)` location. Patch width is 4 at grid=32 and
  8 at grid=16. Phase variance / image variance is the exact non-DC Fourier
  energy fraction on that lattice. Fundamental axis energy and boundary versus
  interior jumps provide additional spatial checks. Residuals are also measured.
  A high fraction alone can reflect a nearly constant image or a genuinely
  periodic scene; compare absolute RMS, input and residual metrics together.
- Hidden magnitude ≥0.95 measures saturation, not a failure verdict. Encoder/
  recurrent preactivations report the fraction with float32 tanh derivative
  ≤1e-6. Decoder logits similarly report nearly zero sigmoid derivatives.
- Gradient probes recompute the same old-source/detached-state surrogate before
  the target's update. `autograd.grad` measures pre-clipping norms without
  writing parameter `.grad`, changing hidden state, or stepping Adam. In frozen
  cases these are hypothetical gradients, not performed learning. Weights,
  Adam moments and steps are summarized at phase boundaries and at the end.
- An independent NumPy reference measures unclipped RGB reconstruction and
  clipping shares. Its clipped/rounded output is tested against the actual GUI
  converter. The diagnostic module imports no Qt; no RGB enters learning.

All diagnostics are opt-in and outside the live learning/rendering path. Ordinary
GUI selectors and display range retain their display-only behavior.

## Findings and limitations

The supplied checkpoint passed existing schema/shape/finiteness/Adam validation.
It has 18,993,200 parameters, 1,048,576 hidden values and 1,693,202 updates, at
seed 246733648, learning rate 0.001, YCbCr, 1024×32. Saved hidden saturation was
99.9775%; recurrent weights ranged from −120.4144 to 27.6110. Finite/bounded
hidden values therefore do not establish a useful, unsaturated learner.

In the frozen CUDA steady-input comparison, saved-state predictions already had
four-pixel Y structure. At frame 179, Y phase RMS was 0.19535 and lattice energy
fraction 0.89752; 56.25% of Y pixels were below 0.01. RGB clipping made 39.11% of
pixels fully black, but the numerical Y pattern existed first. Recurrent tanh
derivatives were ≤1e-6 at 99.9869% of probed positions; 18.75% of all decoder
sigmoid derivatives were ≤1e-6. These measurements are consistent with severe
recurrent and partial decoder saturation.

Zero-state comparison had no Y phase structure on the first prediction, but by
frame 10 phase RMS was 0.22070, lattice fraction 0.90714 and hidden saturation
99.7503%. By frame 179, its phase RMS was 0.19756 and hidden saturation 99.9126%.
Late steady-input MSE was 0.029948 (saved) versus 0.029947 (zero), against
persistence 0.00013095. Cuts and black intervals were not needed to reproduce
the already affected checkpoint. Weights/optimizer digests stayed identical in
every frozen comparison, and the original checkpoint hash remained unchanged.

This rules out RGB conversion alone, active Adam updates during diagnosis, and
saved hidden state alone as sufficient explanations. The learned weights drive
the system back toward saturated periodic output. It does not prove when those
weights became problematic, which historical observation caused it, or whether
a different past failure had the same mechanism. The owner supplied no known-good
checkpoint for a matched historical comparison.

CPU frozen comparisons using the same CUDA-build PyTorch environment also
reproduced strong periodicity with saved and zero recurrence. At frame 20,
Y phase RMS was 0.20229 / 0.19653 (saved / zero), compared with CUDA
0.20287 / 0.19658. This is agreement in the failure behavior, not bitwise equality.

### Fresh-model experiments

All 46 fresh-model cases completed: 24 small-model cases (32 channels, grids
16/32, seeds 7/11, six conditions), 12 owner-size cases (1024×32, seed 7, six
conditions, two learning rates), and 10 follow-up cases (1024×32, seeds 7/11,
five conditions). Each performed exactly 895 updates and 1010 scores including
115 frozen pairs; none had late/missed/skipped targets or numerical faults.
Some conditions deliberately repeat prior cases to verify the follow-up.

At 1024×32 and lr=0.001, both seeds developed persistent uniform near-white
predictions after bright exposure, with or without later hard/smooth black dips.
Low-light exposure produced uniform near-black predictions. For seed 7 these
extremes were already present at frame 305, **before the first black interval
at frame 325**. They remained through 300 recovery training frames and 120 frozen
frames. The steady A control did not develop those extremes. Thus the initial
change in stimulus/continued learning can reproduce uniform collapse here;
the specific brief-black-interval hypothesis is not required by this evidence.

| 1024×32 condition | Frozen MSE, seed 7 | Frozen MSE, seed 11 | Final Y behavior at lr=0.001 |
| --- | ---: | ---: | --- |
| Steady A | 0.001374 | 0.001283 | No endpoint collapse |
| Bright | 0.132280 | 0.132168 | 100% of Y ≥0.99 |
| Five-frame black intervals | 0.133388 | 0.132105 | 100% of Y ≥0.99 |
| Matched smooth black dips | 0.133412 | 0.132295 | 100% of Y ≥0.99 |
| Low light | 0.048499 | 0.048448 | 100% of Y ≤0.01 |

All frozen persistence MSEs in this table are 0.00012697. The owner checkpoint's
**partial periodic** dark output is distinct from these **uniform** synthetic
collapses; a shared historical cause is not established. In the seed-7 low-light
case, probed decoder logits reached approximately −17.7 during the challenge;
bright/black cases reached about +16.9. Near-zero sigmoid derivatives and very
small gradients can leave a finite but poorly recovering decoder. The non-cut
1024×32 control also had ~99.98% hidden saturation at the end without endpoint
collapse, so hidden saturation alone must not be used as a failure detector.

The experimental lr=0.0001 seed-7 comparisons avoided endpoint collapse in all
six conditions. Frozen MSE for black intervals was 0.001377 (versus 0.133388)
and for low light 0.001060 (versus 0.048499); the no-cut control was 0.000719.
This is evidence of learning-rate sensitivity in this bounded protocol. It is
not a demonstrated repair of the owner's weights, a universal safe rate, or a
reason to silently change the accepted learning rule. All 24 small-model cases
also avoided Y endpoint collapse; final Y phase RMS ranged about 0.0011–0.0082.
Some nearly constant outputs had high *relative* lattice fractions but very
small absolute structure, confirming why both metrics are necessary.

### Performed regression and live checks — 2026-11-05

Base was accepted main `3f0fb26e7c8623a77b8adf2fec3b7ed14ad08194` (PR #12).
The existing issue-11 branch was reused. Python 3.14.10, CPU PyTorch 2.15.0+cpu,
CUDA PyTorch 2.14.1+cu130, RTX 3070 Ti. No dependencies/drivers were changed.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python -m tools.validate_gui \
  --source color-moving --compute cpu --input-mode ycbcr --channels 32 --grid 32 \
  --seconds 25 --check-hidden-reset --report /tmp/cognio-reset-synthetic.json
QT_QPA_PLATFORM=wayland .venv-cuda/bin/python -m tools.validate_gui \
  --source camera --compute cuda --input-mode ycbcr --channels 1024 --grid 32 \
  --seconds 30 --check-hidden-reset --report /tmp/cognio-reset-c922.json
```

Accepted-base CPU suite: **130 passed, 11 skips**. Final CPU: **141 passed,
13 CUDA skips (57.23 s)**. Actual CUDA: **154 passed (64.26 s)**. Suites ran
concurrently; durations are not device speed comparisons. Tests cover hidden-only
reset preserving weights/Adam/mode/clocks, exact post-reset pairing, unchanged
default restore, fault retention, CLI and confirmed/cancelled GUI operation,
diagnostic noninterference with gradients and subsequent learning, analytical
Fourier/phase controls, independent RGB equivalence, bounded scene runs, frozen
checkpoint hashing and overwrite refusal. Existing camera/color/grayscale/video,
GPU, capacity, checkpoint, headless and causality regressions remained green.

The synthetic GUI trial passed; its generated-only 1320×940 layout was visually
reviewed. The new control and four views fit without overlap. That image was not
committed. The **30.29-second visible C922 Wayland trial** used a new 1024×32
YCbCr model and two CPU threads. It verified hidden reset to exactly zero, five
pending targets cleared, unchanged clocks and frozen weights/Adam, then save,
restore and resumed training. It produced 960 GUI refreshes with no failures.

After 22 s, 33 samples averaged **30.00 capture / 29.99 predictions / 29.61 updates
per second**; forward/update/display costs were 3.25/19.18/3.40 ms, input age
127.1 ms. Peak sampled RSS was 2436 MiB and PyTorch allocation 1340.9 MiB,
including validator copies. There were 15 handoff overwrites over the run;
restored counters ended with two skipped/missed targets and zero late targets.
Counters rewind on checkpoint restore, while capture/overwrite counters do not.
There were zero driver drops, drained discards or invalid frames. All six changed
camera settings were restored, workers stopped and the window closed. The live
GUI was checked programmatically; no camera imagery was recorded for review.

Compilation, whitespace and local documentation-link checks passed. Compact
scalar evidence is in [evidence/issue-11](evidence/issue-11/): checkpoint A/B
comparisons, scene-control tables with selected onset/gradient/Adam statistics,
and regression/live-trial metadata. Full temporary scalar reports can be
regenerated with the commands above. No raw frames, prediction images/arrays, owner paths,
private checkpoints or camera identifiers are included in the committed reports.

### Remaining questions and proposed follow-up

The present evidence identifies saturated recurrence/decoder output as the
proximate behavior and shows why hidden-only reset cannot repair this checkpoint.
There is no proven small correction that both recovers these weights and
preserves accepted learning behavior. The PR therefore delivers diagnostics,
not an unverified model/optimizer change.

Owner verification should compare a known-good checkpoint with this affected one
using the same frozen A/B command, then inspect the live Y and RGB views without
overwriting either original. The historical onset still needs an earlier
checkpoint or reproducible scene sequence. A separate owner-approved follow-up
could compare limited learning-rate/update interventions from identical states
immediately before the demonstrated transition, including frozen and fresh-Adam
ablations. It should measure no-cut performance and recovery as well as clipping,
and keep objective/architecture changes separate. No such intervention is applied
automatically here, and no new issue is created by this work.

No million-update reproduction, multi-day soak, known-good historical comparison,
eight-thread replication, unfamiliar-scene generalization study or validated
weight-recovery procedure was performed. Short controls cannot establish that
every long-running configuration will remain stable.

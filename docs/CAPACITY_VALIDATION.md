# Issue #4 validation — 2026-11-04

**512 channels × 16×16 on CUDA is the recommended initial larger configuration.**
It has 4,794,432 parameters and 131,072 recurrent scalars (64× the baseline
state), and sustained approximately 30 predictions and learning updates/s in
the short C922 trials. This is a compute-capacity recommendation. Larger models
did not consistently improve learning: changing-brightness prediction became
worse at the unchanged learning rate and objective.

Evidence is scalar-only JSON under [evidence/issue-4](evidence/issue-4/).
Reports from issue #2 remain historical and unchanged. No captured images,
checkpoints, raw camera logs or environment files are committed.

## Environment and method

Fedora 44 KDE, kernel 7.2.15-200.fc44.x86_64, Ryzen 7 5800X, 48 GiB installed
RAM, RTX 3070 Ti 8 GiB, driver 615.78.08. Python 3.14.10; two PyTorch CPU threads,
seed 7, float32, deterministic algorithms, TF32 off, cuDNN autotuning off,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, Adam lr 0.001, batch one, horizon five frames.

The existing `.venv` remained PyTorch 2.15.0+cpu. A separate `.venv-cuda` was
created with `--system-site-packages`, then installed the CUDA wheel and test
dependencies locally. Matched CPU/CUDA benchmarks both used **2.14.1+cu130**,
CUDA runtime **13.0**, cuDNN **9.24.0**. The loaded PyTorch `libcudart.so.13`
reported runtime version 13000; the driver's API version was 13040. These are
different measurements, not evidence that PyTorch used runtime 13.4.
The system driver and original CPU environment were not changed.
[CAPACITY.md](CAPACITY.md) provides fresh-environment setup; dependency pins
describe the tested direct packages, not a complete transitive lockfile.

Unpaced synthetic cases use a virtual 30 Hz sequence/timestamp clock, with
actual synchronized wall timings measured separately. They test causal pairing
and compute throughput but do not establish camera-rate operation. Each matrix
case has 40 initial frozen, 80 training and 40 final frozen frames, with a
45-second wall limit. Complete cases make 75 Adam updates and score 35 targets
in each frozen phase. First ten forward samples are excluded from timing;
cold first-Adam cost is included. [Capacity method](CAPACITY.md#benchmark-method)
defines timing boundaries and memory accounting.

## Capacity and overhead

Mean forward/update times in milliseconds; measured checkpoint sizes in MiB.
CPU and CUDA columns use the same build, input stream and numerical settings.

| Channels × grid | CPU forward / update | CUDA forward / update | CUDA peak allocated MiB | Checkpoint MiB |
| --- | ---: | ---: | ---: | ---: |
| 32 × 8×8 | 0.291 / 2.083 | 0.442 / 5.138 | 3.0 | 0.425 |
| 128 × 16×16 | 1.316 / 7.059 | 0.439 / 3.748 | 12.9 | 3.727 |
| 256 × 16×16 | 3.851 / 23.603 | 0.422 / 3.700 | 45.9 | 14.194 |
| **512 × 16×16** | **12.503 / 88.214** | **0.553 / 5.468** | **167.1** | **55.378** |
| 1024 × 8×8 | 34.168 / 446.365 | 1.138 / 11.809 | 648.4 | 222.496 |
| 1024 × 16×16 | 63.441 / 552.622† | 1.437 / 13.542 | 651.5 | 218.745 |
| 2048 × 8×8 | not run | 4.695 / 42.400 | 2561.9 | 877.0 |
| 512 × 32×32 | not run | 0.882 / 6.377 | 239.4 | 56.320 |

† CPU 1024×16 reached its deadline at 45.29 s, after 112 frames / 67 updates.
It did **not** reach final frozen evaluation. All other listed cases completed
without numeric faults. Full precision, parameter/state counts, timing p95/max,
phase durations, rates, counters and memory are in
[CPU](evidence/issue-4/capacity-cpu.json) and
[CUDA](evidence/issue-4/capacity-cuda.json) reports.

The 32×8 CUDA mean includes an 86 ms cold first training update; its update
p95 was 5.40 ms. CPU is a sensible baseline at this small size. At 512×16,
whole-run unpaced throughput was 18.31 predictions / 8.58 updates per second
on CPU and 265.56 / 124.48 on CUDA. These aggregate rates include frozen
phases; they are neither training-only rates nor camera rates.

Sampled process peak RSS was 866–1125 MiB for the matched-build CPU matrix and
1984–2052 MiB for CUDA. PyTorch allocated VRAM excludes the CUDA context and
other applications. RSS includes framework mappings, so it can exceed the model
working allowance. Estimates guard allocations rather than promise total RSS.

The separate, seeded batch microbenchmark used 12 iterations after three warm-ups:

| Configuration / batch | CPU transfer / forward-backward-Adam ms | CUDA transfer / forward-backward-Adam ms |
| --- | ---: | ---: |
| 32×8 / 1 | 0.005 / 0.864 | 0.041 / 0.867 |
| 32×8 / 4 | 0.005 / 1.489 | 0.057 / 0.859 |
| 512×16 / 1 | 0.015 / 60.844 | 0.100 / 2.141 |
| 512×16 / 4 | 0.015 / 149.811 | 0.095 / 3.681 |

It excludes online pairing, rollback copies and finite guards; this explains
its lower update cost. Batch four is an isolated throughput probe, not a live
application feature. Microbenchmarks were rerun after explicitly seeding their
initialization; these final values replace the earlier unseeded measurements.

2048×8 fit safely but its 42.4 ms update alone exceeds a 33.3 ms frame budget.
512×32 fit and ran unpaced; it was not tested with the live camera. Neither
2048×32 nor CPU 2048/32-grid upper probes were executed: the smaller CPU
1024×16 already exhausted its time limit, and the CUDA upper probes were
sufficient for a bounded first scaling comparison. Shape-only meta-device
tests cover all requested combinations without claiming physical execution.

## Frozen prediction quality and numerical limits

Longer bounded controls used 120 initial frozen, 600 training, and 120 final
frozen frames, with a 180-second limit per case. Every case completed, with
595 actual updates and 115 matched pairs per evaluation. Values below are
**final frozen previously unseen target MSE**, not recomputed training loss:

| Repeated control | CPU 32×8 | CPU 512×16 | CUDA 32×8 | CUDA 512×16 | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 0.008382 | 0.000003 | 0.008382 | 0.000003 | 0 |
| Moving grating | 0.004368 | 0.004469 | 0.006218 | 0.003716 | 0.013180 |
| Changing brightness | 0.063958 | 0.285231 | 0.063949 | 0.276347 | 0.034487 |

The static model error decreased from about 0.123, but exact static persistence
is unbeatable. Both sizes beat persistence on the grating. CUDA 512×16 improved
on its baseline there; CPU 512×16 did not. Changing brightness at 512×16 became
substantially worse than its initial frozen MSE of about 0.06378, despite finite
weights, gradients and losses. No learning-rate tuning or objective change was
introduced to hide that result. Larger capacity alone is not a learning result.

Early CPU/CUDA predictions agree within the regression tolerances, but small
rounding differences can accumulate through online recurrence and Adam. These
single-seed results are not cross-device bitwise equivalence, statistical
superiority, unfamiliar-scene generalization, or evidence of recognition.
Full phase errors and measured costs are in
[learning-cpu.json](evidence/issue-4/learning-cpu.json) and
[learning-cuda.json](evidence/issue-4/learning-cuda.json).

## Actual application and C922 trials

The real C922 negotiated YUYV 640×480 at 30 FPS with monotonic capture timestamps.
No camera image was recorded. These short uncontrolled-scene trials establish
execution and timing, not real-world learning improvement.

| 512×16 trial | Duration | Capture / predictions / updates per second | Sampled peak RSS / PyTorch peak VRAM MiB |
| --- | ---: | ---: | ---: |
| CUDA headless | 15 s | 30.00 / 29.66 / 29.46 | 2019 / 167.1 |
| CUDA visible Wayland GUI | 30.33 s | 30.00 / 29.99 / 29.89 | 2251 / 222.5 |
| CPU headless, original 2.15 CPU build | 12.06 s | 30.01 / 15.08 / 6.26 | 558 / 0 |

CUDA rates are means of steady sampled rate windows (headless after 4 s; GUI
after 22 s, training enabled). CPU rates are the final rolling window; this
CPU trial is not a same-build timing comparison with CUDA.

- Headless CUDA made 436 predictions / 429 updates, with two skipped source
  frames and two missed targets; no late targets. Observer cost was exactly
  zero. It stopped both workers and explicitly saved a roughly 55 MiB CPU-tensor
  checkpoint. The installed `cognio` entry point restored it and resumed
  learning in a separate two-second synthetic run.
- Visible CUDA GUI passed freeze, advancing state during freeze, page two,
  display range, explicit save/restore, resume and cleanup. It recorded 944 GUI
  refreshes and no asserted failures. All six changed camera controls were
  restored. Three handoff overwrites occurred around checkpoint operations;
  final restored counters show one missed target and zero late targets.
  Restoring rewinds model counters, while capture counts remain per process.
- GUI steady mean forward/update/preprocessing/whole-engine-step times were
  0.63 / 6.50 / 3.35 / 7.90 ms. Display refresh averaged 1.12 ms; observer-copy
  samples averaged 0.037 ms including steps without observation. Displayed
  input age averaged 95 ms; measured matched horizon averaged 167.4 ms.
  The 250 ms validation timer's maximum interval was 253.9 ms. This is a
  responsiveness check, not measured mouse-to-screen latency.
- CPU headless made 177 predictions / 72 updates, with 170 skipped source
  frames, 91 missed and 11 late targets. Capture remained near 30 FPS.
  It exited cleanly without faults and demonstrated honest overload accounting.
- An additional 25.20 s offscreen CUDA synthetic GUI trial passed the same
  controls. Its screenshot was visually inspected: four separate labeled
  views, channels 32–63 of 512, correct state/parameter counts and visible
  controls. Only synthetic imagery was captured, outside the repository.

Summaries: [live GUI](evidence/issue-4/c922-gui.json),
[headless CUDA](evidence/issue-4/c922-headless-cuda.json),
[headless CPU](evidence/issue-4/c922-headless-cpu.json),
[synthetic GUI](evidence/issue-4/synthetic-gui.json).
There was no long memory soak, physical unplug/replug test, or manual repeated-
object/unfamiliar-scene trial. Peak memory is sampled, not a leak-freedom proof.

## Regression and reproduction

- CPU environment: **64 passed, 4 CUDA skips**, 11.26 s.
- CUDA environment with hardware access: **68 passed**, 13.23 s.
- Tests include exact patch ordering and meta-device dimensions/counts;
  preserved baseline initialization; actual CUDA forward/backward/Adam;
  early CPU/CUDA agreement; CPU and CUDA observer independence; freeze;
  slow paced processing; scalar-only headless SIGTERM/save/resume with Qt
  imports forbidden; portable, mismatched, corrupt, oversized and legacy
  checkpoints; failed-save preservation; GUI separation and fault labels.
- An additional schema-1 checkpoint was produced using archived accepted
  `main` at `3e216be`, after 30 moving frames. Its next 20 CPU predictions
  after restore were bitwise identical to the original implementation's
  continuation, ending at 40 updates. It also loaded in the CUDA-build CPU
  environment. This goes beyond manufacturing a legacy-shaped test fixture.
- Offline editable package installation and installed console entry point passed.

Commands used for the principal reproducible checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-cuda/bin/python -m pytest -q
.venv-cuda/bin/python -m tools.benchmark_capacity --compute cpu \
  --configs 32x8 128x16 256x16 512x16 1024x8 1024x16 \
  --microbench --output /tmp/cognio-capacity-cpu.json
.venv-cuda/bin/python -m tools.benchmark_capacity --compute cuda \
  --configs 32x8 128x16 256x16 512x16 1024x8 1024x16 2048x8 512x32 \
  --microbench --output /tmp/cognio-capacity-cuda.json
for device in cpu cuda; do
  .venv-cuda/bin/python -m tools.benchmark_capacity --compute "$device" \
    --configs 32x8 512x16 --stimuli static moving temporal \
    --train-frames 600 --evaluation-frames 120 --max-seconds 180 \
    --output "/tmp/cognio-learning-$device.json"
done
.venv-cuda/bin/python -m tools.validate_gui --source camera --compute cuda \
  --channels 512 --grid 16 --seconds 30 --report /tmp/cognio-c922-gui.json
.venv-cuda/bin/python -m cognio --headless --source camera --compute cuda \
  --channels 512 --grid 16 --seconds 15 --save-on-exit /tmp/cognio-large.pt
.venv-cuda/bin/cognio --headless --source moving --compute cuda \
  --checkpoint /tmp/cognio-large.pt --resume-training --seconds 2
QT_QPA_PLATFORM=offscreen .venv-cuda/bin/python -m tools.validate_gui \
  --source moving --compute cuda --channels 512 --grid 16 --seconds 25 \
  --report /tmp/cognio-synthetic-gui.json --screenshot /tmp/cognio-synthetic.png
```

Run GPU and camera commands from a desktop terminal with hardware access and
close other camera users. Capacity reports were collected before the final
documentation commit; final full suites checked the completed source. The
microbenchmark portion was separately rerun with the final seeded function;
the commands above reproduce both portions with the corrected implementation.

For owner review, start the larger visible observer:

```bash
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --channels 512 --grid 16
```

Follow [OWNER_TRIAL.md](OWNER_TRIAL.md), including freeze, page selection,
checkpoint restore, repeated motion and an unfamiliar scene. Compare matched
frozen MSE with persistence before claiming improvement. Use 32×8 CPU for the
baseline. 128×16 CPU is a promising smaller option from unpaced timings, but
its live camera cadence was not verified here. Keep 1024/2048 and 32-grid
choices experimental until separately validated with the intended live scene
and longer duration. No long-running training job was left running.

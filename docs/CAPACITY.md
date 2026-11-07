# Issue #4: configurable capacity, headless execution and CUDA

The accepted 32-channel, 8×8 model remains the headless default and numerical reference.
Issue #14 adds an explicit [GUI setup workflow and topology choice](SETUP_TOPOLOGY.md),
with different initial GUI selections and support for grid 64.
The objective, 128² luminance input, five-camera-frame horizon, MSE/Adam update,
one-step detached recurrence, persistence comparison and privacy contract remain
unchanged. Increasing capacity is an experiment, not evidence of better learning.
Issue #5 adds a separate [native color measurement mode](COLOR.md). The counts
and table below describe `--input-mode y`; color uses three sensory components
in the encoder/decoder while preserving the same recurrent dimensions.

## Reproduce the environments

The original `.venv` uses PyTorch 2.15.0+cpu. A separate `.venv-cuda` was tested
with **PyTorch 2.14.1+cu130, CUDA runtime 13.0, cuDNN 9.24.0**, Python 3.14.10,
NumPy 2.4.3, PySide6 6.11.2 and psutil 7.2.2. GPU: RTX 3070 Ti 8 GiB, NVIDIA
driver 615.78.08, Fedora 44. Same-build CPU/CUDA comparisons use `.venv-cuda`
with an explicit `--compute cpu` or `--compute cuda`; CPU and CUDA do not need
separate installs for that comparison.

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install -r requirements-cuda-tested.txt
.venv-cuda/bin/python -m pip install --no-deps -e .
```

The CUDA wheels bundle their runtime libraries; no system driver or system
CUDA installation was modified. The CPU environment was preserved. The original
checkpoint format also loaded under the CUDA-build environment. The runtime
does no downloads or network transmission. Qt is an optional package extra;
headless mode neither imports it nor creates activation images.

Before selecting CUDA, the application checks the exact requested configuration and
runs two guarded causal updates, including backward and Adam, then discards the
probe weights. This occurs only after Start for GUI launches. Explicit CUDA
failure produces an error with a CPU launch instruction; only `auto` can fall
back. The chosen device/build/runtime appears in telemetry. Real full-model
CUDA training is separately covered by tests and measurements.

## Dimensions and safeguards

For channels `C`, grid width `G`, patch width `P=128/G`:

- Learned parameters `N = 18C² + (2P²+20)C + P²`.
- Hidden-state scalars `H = CG²`.
- Shared encoder: `P² → C`; spatial recurrence: `(2C+2) → C`, kernel 3×3;
  shared decoder: `C → P²`. Fixed x/y coordinates and exact patch ordering remain.

| Channels × grid | Patch pixels | Parameters | Hidden scalars |
| --- | --- | ---: | ---: |
| 32 × 8×8 | 16×16 | 35,712 | 2,048 |
| 128 × 16×16 | 8×8 | 313,920 | 32,768 |
| 256 × 16×16 | 8×8 | 1,217,600 | 65,536 |
| 512 × 16×16 | 8×8 | 4,794,432 | 131,072 |
| 1024 × 8×8 | 16×16 | 19,419,392 | 65,536 |
| 1024 × 16×16 | 8×8 | 19,025,984 | 262,144 |
| 2048 × 8×8 | 16×16 | 76,587,264 | 131,072 |
| 512 × 32×32 | 4×4 | 4,745,232 | 524,288 |

Channels are positive integers, not restricted to these benchmark choices.
Grids 8, 16, 32 and 64 divide the fixed input exactly. Configuration changes are
launch-time choices, not silent resizing of a running or restored learned model.
GUI Reset changes the seed and clears learning at the same dimensions.

Before constructing a model, estimate working bytes as **40N + 128H + 512 MiB**.
Color adds 1.25 MiB for the extra two components in five pending input/prediction
contexts, and uses its own increased parameter count `N`. Toroidal adds
`16*(2C+2)*(G+2)²` bytes for padded joint/gradient transients (issue #14).
This includes weights, gradients, Adam moments, rollback copies, candidate/
serialization transients, detached contexts, activations and a workspace reserve.
Require that estimate to fit half of currently available host memory and, on
CUDA, 60% of currently free GPU memory. `--memory-budget-mib` can further lower
the allowance. These conservative checks are not allocation guarantees; another
process, backend workspace or fragmentation can still cause an explicit fault.
The estimate does not impose a fixed channel ceiling or reserve that memory.
It is a model working allowance; framework/shared-library mappings and other
process allocations can make measured RSS substantially larger.

The online batch remains one. At most five pending source/prediction/previous-
state contexts are retained, each roughly `2×128²×4 + H×4` bytes. Capture has one
newest-frame slot. Expire only exact missing target IDs, reject targets captured
before prediction issuance, and count source skips, handoff overwrites, expired
targets, late targets and contexts cleared by mode/checkpoint boundaries.
No model-speed compensation invents frames or changes the five-frame target.

## Optional observer and headless control

```bash
.venv-cuda/bin/python -m cognio --headless --source camera --compute cuda \
  --channels 512 --grid 16 --seconds 30 --save-on-exit /tmp/cognio-large.pt
.venv-cuda/bin/python -m cognio --gui --source camera --compute cuda \
  --channels 512 --grid 16 --display-hz 15
.venv-cuda/bin/python -m cognio --headless --source camera --compute cuda \
  --checkpoint /tmp/cognio-large.pt --seconds 30
```

Default launch is headless. `--seconds` bounds the run; Ctrl-C/SIGTERM also stop
workers and release the camera. Faults stop a headless run with nonzero status;
an explicitly supplied save destination can preserve recoverable model state.
Periodic scalar JSON has no images or activation arrays and no growing in-memory
log. A final record identifies clean worker shutdown. Restores default to frozen;
pass `--resume-training` with `--checkpoint` for explicit headless continuation.
GUI checkpoint controls
remain available when capture/model processing fails.

The GUI observes a page of at most 32 real channels, with full spatial resolution
and fixed tile ordering; partial pages pad unused tiles with blank cells. Page,
color range and `--display-hz` do not change the model. Headless mode performs no
activation copy. The observer is downstream of causal prediction issuance and
has a bounded cadence; error maps and input/prediction views retain their actual
source/target IDs and age. Headless and observer engines produce identical updates
on identical delivered frames in CPU and CUDA regression tests. Under overload, scheduling can change
which live frames are delivered; the counters expose that sampling difference.

## Checkpoint storage

Store complete dimensions, weights, Adam moments, recurrent state, seed/settings,
counters and framework metadata. All serialized tensors are on CPU. Current
schema 3 also records the sensory contract. Original schema 1 is migrated only
as grayscale 32×8; schema 2 retains its grayscale dimensions. A GUI restore
requires current mode/dimensions; CLI restore uses saved choices and rejects
explicitly conflicting mode/channel/grid arguments. No weight interpolation occurs.

The old 16 MiB cap is replaced by **12N + 4H + 8G² + 256 KiB**: float32 model
weights plus two Adam moments, hidden state, positions, and serialization
metadata. The 512×16 ceiling is about 55.62 MiB; 2048×8 is about 877.22 MiB.
Before reading metadata, the archive must fit 25% of available host RAM, use
uncompressed ZIP entries, have at most 128 entries (this fixed architecture has
only six learned tensors), and fit the same uncompressed-byte bound. Load uses
restricted `weights_only=True` and memory mapping; configuration/shape/dtype,
finite values, fixed positions, optimizer and counters are checked before swap.
The model allocation check also applies during restore.

Writes use temporary files, fsync and atomic replacement. A failed/oversized save
leaves the previous file intact. Successful save clears pending pairs in both
the saved counters and live engine. No raw inputs, targets, predictions, error
maps or recording history are serialized. A checkpoint can still contain learned
scene information. Restoring live input starts frozen across an unobserved gap.

## Benchmark method

`tools.benchmark_capacity` runs bounded cases with seed 7, two CPU threads,
float32, TF32 disabled, deterministic algorithms, and no cudnn autotuning. Its
unpaced stream has a **virtual 30 Hz observation clock**: targets are supplied
only after predictions, but wall throughput is reported separately. It cannot
establish real-camera speed or a 167 ms wall-clock horizon; paced camera and
synthetic trials provide that evidence.

The short capacity matrix uses 40 initial frozen frames, 80 training frames,
40 final frozen frames, at most 45 seconds per case. A time-limited case is
explicitly incomplete. Evaluation compares earlier predictions against the same
matched future targets and persistence. These are later samples of repeated
controls, not unfamiliar-scene generalization. Longer bounded comparisons use
the same settings with more training frames, reported separately.

Reported forward/update/transfer/observer timings include synchronizations.
Update time includes rollback copies, recomputation, backward, clipping, Adam
and finite guards. Whole-step time also includes matching/scoring. CPU RSS and
PyTorch allocated/peak GPU bytes are measured; CUDA context and other applications
consume additional device memory not shown by PyTorch's allocator. GPU workspace
and cold first-use costs are included where encountered. RSS maxima are maxima
of sampled process readings, not instantaneous OS high-water measurements.
Phase durations and
overall rates distinguish training from frozen evaluation.

The separate batch-1/batch-4 microbenchmark measures transfer and isolated
forward/backward/Adam cost with a zero-target MSE. It excludes the online pairing,
rollback, guards and observer. Batch-4 throughput is not claimed as a capability
of the batch-1 live learner and does not change its learning contract.

See [issue #4 measured results](CAPACITY_VALIDATION.md) for commands, outcomes,
failures, recommended starting configurations and untested limits.

References: [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html),
[serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html),
and [installation](https://pytorch.org/get-started/locally/).

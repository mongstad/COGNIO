# AGENTS.md — COGNIO

Repository-wide instructions for Astra/Codex and other coding agents.

## Mission

COGNIO explores **continuous, online, self-supervised learning from real-world sensory measurements**. The initial instrument is a Logitech C922 webcam. A small network begins with randomly initialized weights, receives an ongoing stream of spatially arranged luminance measurements, adjusts its learned parameters from prediction errors, and exposes its **actual internal neural activity** for direct observation.

This is distinct from [SENTIO](https://github.com/mongstad/SENTIO), which investigates camera-driven scalar-wave fields. COGNIO is a separate learning experiment. Reuse proven components selectively, but do not merge the projects or import SENTIO's wave solver as the neural network.

The goal is to observe and measure changes in internal representations over time, **not** to assert intelligence, consciousness, object understanding, or biological equivalence.

## Ownership and issue workflow

- The repository owner controls merges, release publication, tags, and significant changes to scope.
- Work on **one explicitly assigned issue at a time**. Read the issue, this file, current README, and the relevant code before implementing.
- Verify the issue number, its current GitHub status, the current branch, and the latest accepted `main`. **Never infer the active issue from an old prompt or current branch alone.** If a prompt includes a placeholder like `#XX`, resolve the explicit current instruction rather than guessing.
- Create a dedicated branch, implement within issue scope, push, and open one PR for owner review. Do not merge or force-push shared branches without authorization.
- Do not invent cross-issue dependencies; only block work on a dependency when it is technically necessary. Keep unrelated issues and branches untouched.
- Make routine engineering choices independently; escalate actual blockers, research-goal changes, or risky architectural departures.
- PR handoff must describe what changed, how to run it, tests **actually performed**, live hardware checks, limitations, and items still requiring owner verification.

## Prototype 1 boundaries

- **Camera:** Logitech C922 using Linux V4L2; baseline measured YUYV 640×480 at 30 FPS, on Fedora 44. Enumerate/select the camera robustly; `/dev/video0` is not a guaranteed permanent path.
- **Model input:** begin with 128×128 luminance values; preserve spatial and temporal relationships. A candidate starting design is 16×16 patches, shared visual encoding, a small spatial/recurrent hidden state, and next-frame or short-horizon patch prediction. Document exact design and any departures.
- **Initialization:** random learned weights; no pretrained visual encoders, embeddings, language models, labeled data, pretrained recognition, or hardcoded object classes.
- **Learning:** ongoing self-supervised prediction from current observations to later *real* camera observations. Predictions must be generated causally before the target is seen; no leakage from the future.
- **Training data:** only a short, bounded in-memory rolling buffer needed for prediction/learning. Do not persist raw camera images, video datasets, or private scenes by default.
- **Checkpoints:** explicitly save and restore model weights, optimizer state when necessary, network architecture/configuration, recurrent state if appropriate, and progress/counters. No unbounded sensory history or raw video.
- **Visualization:** show four clearly labeled views: (1) actual camera input, (2) predicted future frame, (3) prediction-error map aligned with the appropriate observed target, and **(4) real internal neural activation/state visualization as the primary view**. Never substitute a generated artistic image for actual hidden activations.
- Provide live training on/off (with sensory processing continuing), input/learning-rate telemetry, model performance and camera FPS; allow owner-controlled reset and checkpoint operations.
- Structural plasticity (grow/prune connections), microphone/audio, 3D field dynamics, multi-camera input, and biologically literal claims are **future work**, not prototype 1.

## Scientific and technical rigor

- The prediction target must be explicit: pixel/patch values and prediction horizon(s), with timestamps or reliable frame indices. A suggested initial horizon is five camera frames (about 167 ms nominally at 30 FPS); do not assume the delivery cadence is perfect.
- Prevent leakage of current or future targets into the prediction being scored. Compare against a simple persistence baseline (copy the latest frame). Report errors over time on previously unseen frames, not merely training loss.
- A patch-based model must preserve spatial positions and allow cross-patch interaction; never claim patch boundaries automatically correspond to objects.
- Visualized internal state must be directly traceable to measured activations or a **fixed, documented projection** of them. Clearly distinguish internal-state visuals, predictions, input, and display-only contrast/color controls.
- Measure learning behavior with controlled repeated stimulus and holdout/frozen-training periods; do not call an attractive heatmap evidence of learning.
- Define and document optimizer, loss, update cadence, precision, random seeds, model size, rolling-buffer bounds, and whether frames are sampled/dropped. The model may train slower than camera capture; never hide that fact.
- Guard against NaN/Inf loss, invalid gradients, exploding weights, memory leaks, and unsafe unbounded allocations. Make failure visible and preserve a recoverable checkpoint.
- If training changes when display settings change, treat this as a bug. Display-only controls cannot alter data, model state, or learning.
- Keep privacy local: no unsolicited network transmission, cloud training, telemetry upload, or hidden recording. Explain camera activation and ensure device cleanup on exit.
- Prefer a small, testable implementation on Fedora Linux 44, Ryzen 7 5800X, 48 GiB RAM, and NVIDIA RTX 3070 Ti 8 GB. GPU support is useful but must not be assumed until verified; provide a usable failure mode if CUDA is unavailable.

## Testing and owner validation

- Test tensor dimensions and patch ordering; frame acquisition/format and normalization; timestamps/buffer alignment; no-future-leakage; prediction-target correctness; loss/gradient updates; train freeze; model state consistency; save/restore; and GUI view separation.
- Include deterministic synthetic motion and static-scene controls before using the actual webcam. Include a simple persistence prediction baseline.
- Test the **real C922 live GUI** when hardware access permits. Report measured capture FPS, training updates/s, prediction throughput/latency, memory/VRAM use, and visual responsiveness. If hardware access is blocked, say so explicitly and provide an exact owner-run command; do not pretend synthetic tests were a live demonstration.
- A runnable application and concise owner protocol are required for the first implementation issue; avoid endless research-only cycles.
- Preserve scientific ambiguity: repeated patterns, lower prediction error, and recurrent activations do not by themselves prove object recognition or consciousness.
- Record commands, settings, results, failures, and limitations in small, reproducible reports. Do not commit captured private imagery, large checkpoints, temporary logs, or caches without explicit owner authorization.

## Completion

One scoped branch, one PR, reproducible commands, tests and real-world observations where available. Leave the PR **unmerged** and ready for owner review.

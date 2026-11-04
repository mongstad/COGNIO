# Owner trial

Run from the repository with `.venv` activated, close SENTIO or other camera
users, and start `python -m cognio --gui --source camera --compute cpu`. Expect a
**NEW random initialization** unless `--checkpoint` was supplied. Camera
activity is explicit; the application stores no raw scene images or video.

1. **Static familiar scene, 60 seconds:** keep viewpoint and display range
   fixed. Confirm four labeled views, ~five-frame measured horizon, advancing
   frame IDs, updates, and signed hidden-channel changes. Note model MSE and
   persistence MSE; persistence is usually strong for a static scene. Lower
   training loss alone is insufficient.
2. **Repeated motion, 60 seconds:** move an item periodically along the same
   path. Record input/prediction/update rates, errors, drop counts, horizon and
   input age. Compare model and persistence on the same matched targets.
3. **Freeze training, 30 seconds:** uncheck Train online. After the rolling
   rate window settles, updates/s must be zero and total updates must stop.
   Input, predictions, scoring, and recurrent activations should continue.
   Changing Display ± must only recolor the hidden canvas.
4. **Viewpoint/unfamiliar scene while frozen, 30 seconds:** change viewpoint
   or introduce a different arrangement, then revisit the familiar scene.
   Compare errors and activation response. This checks behavior under change;
   it does not prove recognition, retention, or generalization.
5. **Resume, save, restore:** resume training, save explicitly, then restore
   that file. Confirm RESTORED, saved progress counters, frozen training,
   and fresh five-frame pairing warm-up. Resume explicitly. Close and
   relaunch with `--checkpoint /path/to/file.pt`; no frames are replayed.
6. **Reset and exit:** save wanted state first. Confirm the reset dialog, new
   seed, cleared progress and paused training. Close normally and verify the
   camera is released for another application.

Use the same scene/motion, seed, learning rate, lighting, camera settings and
durations for comparison; repeat trials. Call it predictive improvement only
when matched **previously unseen** target MSE is lower than persistence during
a frozen evaluation, with comparable drops and horizons. The repeated synthetic
grating is a positive control, not a real-world generalization benchmark.

The scripted visible trial automates freeze/resume/save/load and checks worker
shutdown, timing and camera-control restoration. It does not move physical
objects, test human interaction feel, or replace the manual scene trial:

```bash
python -m tools.validate_gui --source camera --seconds 90 --compute cpu \
  --report /tmp/cognio-owner-live.json
```

Reports contain scalar measurements/settings only. Checkpoints remain explicit.
After a disconnect, save wanted model state, close, reconnect and relaunch.
No silent synthetic fallback is used.

For issue #4, repeat with `--channels 512 --grid 16 --compute cuda` in the tested
CUDA environment. Change **Channel page** as well as display range. Confirm
the configuration/parameter/state counts, independently measured rates and
drop accounting. Run the same source with `--headless --seconds 30`; inspect
scalar telemetry and explicit checkpoint save/restore without a Qt window.
See [capacity measurements and limits](CAPACITY.md) before choosing a size.

For issue #5, explicitly create a new color model with `--input-mode ycbcr`.
Use the Diagnostic component selector to inspect Y, Cb and Cr input, future
prediction and matched error; the signed chroma display colors are not RGB.
Check all three component MSE/persistence values, not just the aggregate. While
training is frozen, change colored items/lighting and compare Cb/Cr responses
with Y; ordinary camera scenes cannot guarantee constant luminance. Save and
restore within this mode, then verify loading a grayscale checkpoint is rejected
without changing the active model. Use the constant-Y `color-moving` synthetic
source for an exact control. [COLOR.md](COLOR.md) records performed checks and
the remaining physical-scene validation.

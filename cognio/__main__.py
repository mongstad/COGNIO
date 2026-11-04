import argparse
import json
import math
import os
import sys

from .inputs import INPUT_MODES, MODES


def main():
    parser = argparse.ArgumentParser(description="COGNIO: local, randomly initialized online predictive learning")
    parser.add_argument("--source", choices=("camera", *MODES), default="static",
                        help="default static synthetic control; camera is explicit")
    parser.add_argument("--device", help="V4L2 path, preferably /dev/v4l/by-id/...")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--compute", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--checkpoint", help="explicitly restore this checkpoint; begins with training frozen")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--resume-training", action="store_true", help="explicitly enable training after --checkpoint restore")
    parser.add_argument("--channels", type=int, help="new model channels (default 32); restore must match")
    parser.add_argument("--grid", type=int, choices=(8, 16, 32), help="spatial state grid (default 8)")
    parser.add_argument("--input-mode", choices=INPUT_MODES, help="new model sensory mode (default y); restore must match")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="open the optional four-view observer")
    mode.add_argument("--headless", action="store_true", help="scalar telemetry only (default)")
    parser.add_argument("--seconds", type=float, help="headless duration; otherwise stop with Ctrl-C/SIGTERM")
    parser.add_argument("--telemetry-interval", type=float, default=1.)
    parser.add_argument("--save-on-exit", help="explicit headless checkpoint destination (atomic replacement)")
    parser.add_argument("--display-hz", type=float, default=15.)
    parser.add_argument("--memory-budget-mib", type=float, help="optional lower working-memory allowance")
    args = parser.parse_args()
    if args.list_devices:
        from .camera import devices
        print(json.dumps(devices(), indent=2))
        return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from .engine import Engine
    from .model import Config
    from .runtime import Runtime, select_device
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
            raise ValueError("seconds must be positive and finite")
        if not math.isfinite(args.telemetry_interval) or args.telemetry_interval < .1:
            raise ValueError("telemetry interval must be finite and at least 0.1 seconds")
        if args.gui and (args.seconds is not None or args.save_on_exit):
            raise ValueError("--seconds and --save-on-exit are headless options; use GUI checkpoint controls")
        if args.resume_training and (not args.checkpoint or args.frozen):
            raise ValueError("--resume-training requires --checkpoint and cannot be combined with --frozen")
        device, notice = select_device(args.compute)
        if args.checkpoint:
            engine = Engine.load(args.checkpoint, device, memory_budget_mib=args.memory_budget_mib)
            if args.input_mode is not None and args.input_mode != engine.config.input_mode:
                raise ValueError("checkpoint input mode differs from the explicitly requested --input-mode")
            if ((args.channels is not None and args.channels != engine.config.channels) or
                    (args.grid is not None and args.grid != engine.config.grid)):
                raise ValueError("checkpoint dimensions differ from the explicitly requested architecture")
        else:
            engine = Engine(Config(args.seed, args.learning_rate, channels=args.channels if args.channels is not None else 32,
                                   grid=args.grid if args.grid is not None else 8,
                                   input_mode=args.input_mode or "y"), device,
                            memory_budget_mib=args.memory_budget_mib)
        if args.resume_training:
            engine.set_training(True)
        elif args.frozen or args.checkpoint:
            engine.set_training(False)
        runtime = Runtime(engine, args.source, args.device, notice, observer=args.gui, display_hz=args.display_hz)
    except Exception as exc:
        parser.error(str(exc))
    if not args.gui:
        from .headless import run
        return run(runtime, args.seconds, args.telemetry_interval, args.save_on_exit)
    from PySide6.QtWidgets import QApplication
    from .gui import Window
    app = QApplication(sys.argv[:1])
    window = Window(runtime)
    window.show()
    runtime.start()
    try:
        return app.exec()
    finally:
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())

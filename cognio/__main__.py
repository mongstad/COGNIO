import argparse
import json
import sys

from .inputs import MODES


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
    args = parser.parse_args()
    if args.list_devices:
        from .camera import devices
        print(json.dumps(devices(), indent=2))
        return 0
    import torch
    from PySide6.QtWidgets import QApplication
    from .engine import Engine
    from .gui import Window
    from .model import Config
    from .runtime import Runtime, select_device
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        device, notice = select_device(args.compute)
        engine = Engine.load(args.checkpoint, device) if args.checkpoint else Engine(Config(args.seed, args.learning_rate), device)
        if args.frozen or args.checkpoint:
            engine.set_training(False)
    except Exception as exc:
        parser.error(str(exc))
    app = QApplication(sys.argv[:1])
    runtime = Runtime(engine, args.source, args.device, notice)
    window = Window(runtime)
    window.show()
    runtime.start()
    try:
        return app.exec()
    finally:
        runtime.stop()


if __name__ == "__main__":
    raise SystemExit(main())

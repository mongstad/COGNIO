import argparse
import json
import math
import os
import sys

from .inputs import INPUT_MODES, MODES


def make_parser():
    parser = argparse.ArgumentParser(description="COGNIO: local, randomly initialized online predictive learning")
    parser.add_argument("--source", choices=("camera", "video", *MODES), default=None,
                        help="headless default: static; GUI default selection: camera (starts only on Start)")
    parser.add_argument("--device", help="V4L2 path, preferably /dev/v4l/by-id/...")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--video-file", help="local video path (requires ffmpeg/ffprobe)")
    parser.add_argument("--video-speed", type=float, default=1., help="paced video speed 0.1..4 (default 1)")
    parser.add_argument("--video-loop", action="store_true", help="repeat video; clear temporal context between passes")
    parser.add_argument("--video-paused", action="store_true", help="begin video paused; resume with GUI or headless control")
    parser.add_argument("--video-fastest", action="store_true", help="sequential unpaced video; retain real wall-clock causality")
    parser.add_argument("--video-controls", action="store_true", help="headless commands on stdin: pause/resume/restart/speed X/loop on|off/train on|off")
    parser.add_argument("--compute", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--checkpoint", help="explicitly restore this checkpoint; begins with training frozen")
    parser.add_argument("--reset-hidden", action="store_true",
                        help="with --checkpoint: clear transient recurrent state only; retain learned weights/Adam")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--resume-training", action="store_true", help="explicitly enable training after --checkpoint restore")
    parser.add_argument("--channels", type=int, help="new model channels (headless 32; GUI 1024); restore must match")
    parser.add_argument("--grid", type=int, choices=(8, 16, 32, 64), help="spatial state grid (headless 8; GUI 32)")
    parser.add_argument("--input-mode", choices=INPUT_MODES, help="sensory mode (headless y; GUI ycbcr); restore must match")
    parser.add_argument("--topology", choices=("standard", "toroidal"),
                        help="headless default: standard; GUI initial selection: toroidal")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="open the optional four-view observer")
    mode.add_argument("--headless", action="store_true", help="scalar telemetry only (default)")
    parser.add_argument("--seconds", type=float, help="headless duration; otherwise stop with Ctrl-C/SIGTERM")
    parser.add_argument("--telemetry-interval", type=float, default=1.)
    parser.add_argument("--save-on-exit", help="explicit headless checkpoint destination (atomic replacement)")
    parser.add_argument("--display-hz", type=float, default=15.)
    parser.add_argument("--memory-budget-mib", type=float, help="optional lower working-memory allowance")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.source = args.source or ("camera" if args.gui else "static")
    if args.list_devices:
        from .camera import devices
        print(json.dumps(devices(), indent=2))
        return 0
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if args.source != "video" and (args.video_file or args.video_speed != 1 or args.video_loop or args.video_paused or args.video_fastest or args.video_controls):
            raise ValueError("video options require --source video")
        if args.gui and args.video_controls:
            raise ValueError("--video-controls is headless only; use the GUI playback controls")
        if args.video_paused and not args.gui and not args.video_controls:
            raise ValueError("--video-paused in headless mode requires --video-controls to resume")
        if args.source == "video" and not args.gui and not args.video_file:
            raise ValueError("--source video requires --video-file PATH")
        if args.source == "video":
            from .video import VideoPlayback
            VideoPlayback.validate_speed(args.video_speed)
        if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
            raise ValueError("seconds must be positive and finite")
        if not math.isfinite(args.telemetry_interval) or args.telemetry_interval < .1:
            raise ValueError("telemetry interval must be finite and at least 0.1 seconds")
        if args.gui and (args.seconds is not None or args.save_on_exit):
            raise ValueError("--seconds and --save-on-exit are headless options; use GUI checkpoint controls")
        if args.resume_training and (not args.checkpoint or args.frozen):
            raise ValueError("--resume-training requires --checkpoint and cannot be combined with --frozen")
        if args.reset_hidden and not args.checkpoint:
            raise ValueError("--reset-hidden requires --checkpoint; new models already start with zero hidden state")
        if not args.gui:
            from .session import build_runtime
            runtime = build_runtime(args)
    except Exception as exc:
        parser.error(str(exc))
    if not args.gui:
        from .headless import run
        return run(runtime, args.seconds, args.telemetry_interval, args.save_on_exit,
                   control_input=sys.stdin if args.video_controls else None)
    from PySide6.QtWidgets import QApplication
    from .setup import SessionWindow
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = SessionWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

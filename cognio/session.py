"""Session construction shared by CLI and the explicit GUI Start action."""

from .engine import Engine
from .model import Config
from .runtime import Runtime, select_device


def requested_config(args):
    if args.checkpoint:
        config = Engine.inspect_checkpoint(args.checkpoint)["config"]
        for name in ("channels", "grid", "input_mode", "topology"):
            value = getattr(args, name)
            if value is not None and value != getattr(config, name):
                label = "input mode" if name == "input_mode" else name
                raise ValueError(f"checkpoint {label} differs from the explicitly requested --{name.replace('_', '-')}")
        return config
    return Config(seed=args.seed, learning_rate=args.learning_rate,
                  channels=args.channels if args.channels is not None else 32,
                  grid=args.grid if args.grid is not None else 8,
                  input_mode=args.input_mode or "y", topology=args.topology or "standard")


def build_runtime(args, expected_config=None):
    """No capture until Runtime.start; failures leave checkpoint files untouched."""
    config = requested_config(args)
    if expected_config is not None and config != expected_config:
        raise ValueError("checkpoint configuration changed since inspection; select it again")
    if args.source == "camera":
        # Query identity only; no format/control writes until the capture worker starts.
        import os
        from .camera import choose_device, identity
        camera_path = choose_device(args.device)
        fd = os.open(camera_path, os.O_RDWR | os.O_NONBLOCK)
        try:
            identity(fd, camera_path)
        finally:
            os.close(fd)
    else:
        camera_path = args.device
    device, notice = select_device(args.compute, config, args.memory_budget_mib)
    if args.checkpoint:
        engine = Engine.load(args.checkpoint, device, expected_config=config,
                             memory_budget_mib=args.memory_budget_mib)
    else:
        engine = Engine(config, device, memory_budget_mib=args.memory_budget_mib)
    if args.resume_training:
        engine.set_training(True)
    elif args.frozen or args.checkpoint:
        engine.set_training(False)
    if args.reset_hidden:
        engine.reset_hidden()
        engine.origin += " · transient hidden state explicitly reset; learned weights/Adam retained"
    return Runtime(engine, args.source, camera_path, notice, observer=args.gui, display_hz=args.display_hz,
                   video_path=args.video_file, video_speed=args.video_speed, video_loop=args.video_loop,
                   video_paused=args.video_paused, video_fastest=args.video_fastest)

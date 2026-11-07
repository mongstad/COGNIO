"""Experiment setup and session lifecycle. No model or stream exists before Start."""

import copy
from dataclasses import asdict
import gc
import queue
import threading

import torch
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .camera import devices
from .engine import Engine
from .gui import Window
from .inputs import MODES
from .model import Config, GRIDS, TOPOLOGIES
from .resources import check_budget
from .session import build_runtime, requested_config


def selector(values, selected):
    box = QComboBox()
    box.addItems([str(v) for v in values])
    box.setCurrentText(str(selected))
    return box


class SessionWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = copy.copy(args)
        self.runtime = self.running = None
        self.checkpoint_config = None
        self.checkpoint_path = None
        self.state = "configure"
        self.closing = False
        self.worker = None
        self.results = queue.Queue(maxsize=1)
        self.setWindowTitle("COGNIO · configure experiment")
        self.resize(960, 900)
        self.setStyleSheet("QWidget { background: #101720; color: #dbe5ef; font-size: 13px; }"
                          "QPushButton, QComboBox, QLineEdit, QDoubleSpinBox { background: #273649; padding: 3px; }"
                          "QLabel { background: transparent; } QPushButton:disabled { color: #7b8795; }")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.form_widget = QWidget()
        self.scroll.setWidget(self.form_widget)
        self.configuration = QWidget()
        setup_layout = QVBoxLayout(self.configuration)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.addWidget(self.scroll, 1)
        self.stack.addWidget(self.configuration)
        layout = QVBoxLayout(self.form_widget)
        title = QLabel("COGNIO / Configure experiment")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        layout.addWidget(title)
        self.origin = QLabel("Fresh session · random weights will be created on Start.")
        self.origin.setWordWrap(True)
        layout.addWidget(self.origin)
        layout.addWidget(QLabel("No model, acquisition or learning runs here. Settings lock after Start."))
        form = self.form = QFormLayout()
        layout.addLayout(form)
        self.topology = selector(TOPOLOGIES, args.topology or "toroidal")
        # Editable text avoids an artificial scientific size ceiling from QSpinBox.
        self.channels = QLineEdit(str(args.channels if args.channels is not None else 1024))
        self.grid = selector(GRIDS, args.grid if args.grid is not None else 32)
        self.input_mode = selector(("y", "ycbcr"), args.input_mode or "ycbcr")
        self.source = selector(("camera", "video", *MODES), args.source or "camera")
        self.compute = selector(("cuda", "cpu"), "cuda" if args.compute == "auto" and torch.cuda.is_available()
                                else "cpu" if args.compute == "auto" else args.compute)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(.000001, .01)
        self.learning_rate.setSingleStep(.0001)
        self.learning_rate.setValue(args.learning_rate)
        self.seed = QLineEdit(str(args.seed))
        self.budget = QLineEdit("" if args.memory_budget_mib is None else str(args.memory_budget_mib))
        self.budget.setPlaceholderText("Automatic conservative budget; optional lower MiB limit")
        self.architecture_widgets = (self.topology, self.channels, self.grid, self.input_mode, self.seed, self.learning_rate)
        for label, control in (("Recurrent topology", self.topology), ("Recurrent channels", self.channels),
                ("Spatial grid (per axis)", self.grid), ("Input measurements", self.input_mode),
                ("Prediction horizon", QLabel("5 observed frames · fixed objective")),
                ("Source", self.source), ("Compute", self.compute),
                ("Adam learning rate", self.learning_rate), ("Random seed (uint32)", self.seed),
                ("Working-memory limit (MiB)", self.budget)):
            form.addRow(label, control)
        self.camera = QComboBox()
        self.camera.setEditable(True)
        self.camera.addItem("")
        self.camera.setToolTip("Blank: discover unique C922. Or select/type a V4L2 capture path.")
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.camera, 1)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh_devices)
        camera_row.addWidget(refresh)
        form.addRow("Camera device", camera_row)
        self.camera_notice = QLabel()
        self.camera_notice.setWordWrap(True)
        form.addRow(self.camera_notice)
        self.video_path = QLineEdit(args.video_file or "")
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_path, 1)
        self.video_choose = QPushButton("Choose video…")
        self.video_choose.clicked.connect(self.choose_video)
        video_row.addWidget(self.video_choose)
        form.addRow("Local video file", video_row)
        self.video_speed = QDoubleSpinBox()
        self.video_speed.setRange(.1, 4)
        self.video_speed.setSingleStep(.1)
        self.video_speed.setValue(args.video_speed)
        self.video_loop = QCheckBox("Repeat")
        self.video_loop.setChecked(args.video_loop)
        self.video_paused = QCheckBox("Start paused")
        self.video_paused.setChecked(args.video_paused)
        self.video_fastest = QCheckBox("Fastest sequential delivery")
        self.video_fastest.setChecked(args.video_fastest)
        playback_row = QHBoxLayout()
        for control in (self.video_speed, self.video_loop, self.video_paused, self.video_fastest):
            playback_row.addWidget(control)
        form.addRow("Video playback ×", playback_row)
        self.video_rows = (video_row, playback_row)
        self.frozen = QCheckBox("Start with learning frozen (sensory processing continues)")
        self.frozen.setChecked(args.frozen or bool(args.checkpoint and not args.resume_training))
        layout.addWidget(self.frozen)
        self.reset_hidden = QCheckBox("On restore: explicitly clear transient hidden state only")
        self.reset_hidden.setChecked(args.reset_hidden)
        self.reset_hidden.setEnabled(False)
        layout.addWidget(self.reset_hidden)
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        setup_layout.addWidget(footer)
        self.capacity = QLabel()
        self.capacity.setWordWrap(True)
        self.capacity.setStyleSheet("background: #1a2836; padding: 12px;")
        footer_layout.addWidget(self.capacity)
        note = QLabel("Toroidal wraps the recurrent grid's top/bottom and left/right edges. Camera images are not "
                      "periodic: unrelated edges may interact artificially. Fixed x/y coordinates and sensory patches "
                      "remain unchanged. This is an experimental 2D connection rule.")
        note.setWordWrap(True)
        footer_layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #f3b665;")
        footer_layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.restore_button = QPushButton("Restore Checkpoint…")
        self.restore_button.clicked.connect(self.choose_checkpoint)
        self.fresh_button = QPushButton("Use fresh random model")
        self.fresh_button.clicked.connect(self.use_fresh)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_session)
        for button in (self.restore_button, self.fresh_button, self.start_button):
            buttons.addWidget(button)
        footer_layout.addLayout(buttons)
        footer_layout.addWidget(QLabel("Local only · no automatic save or media recording · close to exit"))
        for control in (self.topology, self.grid, self.input_mode, self.source, self.compute, self.camera):
            control.currentTextChanged.connect(self.update_estimate)
        for control in (self.channels, self.seed, self.budget, self.video_path):
            control.textChanged.connect(self.update_estimate)
        self.learning_rate.valueChanged.connect(self.update_estimate)
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.poll_worker)
        self.poll.start(50)
        self.refresh_devices()
        if args.device:
            self.camera.setCurrentText(args.device)
        if args.checkpoint:
            try:
                config = requested_config(args)  # Honor explicit CLI architecture conflicts.
                self.inspect_checkpoint(args.checkpoint, config)
            except Exception as exc:
                self.status.setText(str(exc))
                # Do not turn a failed restore into an accidental fresh session.
                self.checkpoint_path = args.checkpoint
        self.update_estimate()

    def refresh_devices(self):
        selected = self.camera.currentText()
        self.camera.blockSignals(True)
        self.camera.clear()
        self.camera.addItem("")
        found = devices()  # QUERYCAP only; no acquisition or controls changed.
        for device in found:
            self.camera.addItem(device["path"])
        self.camera.setCurrentText(selected)
        self.camera.blockSignals(False)
        self.camera_available = sum("c922" in device["card"].lower() for device in found) == 1
        self.camera_notice.setText(" · ".join(device["card"] for device in found) or
                                   "No capture devices found. Connect a C922, refresh, or choose video/synthetic input.")
        self.update_estimate()

    def selected_config(self):
        if self.checkpoint_path:
            if self.checkpoint_config is None:
                raise ValueError("Checkpoint inspection failed. Select a valid checkpoint or explicitly choose a fresh model.")
            return self.checkpoint_config
        return Config(seed=int(self.seed.text()), learning_rate=self.learning_rate.value(),
                      channels=int(self.channels.text()), grid=int(self.grid.currentText()),
                      input_mode=self.input_mode.currentText(), topology=self.topology.currentText())

    def memory_limit(self):
        return float(self.budget.text()) if self.budget.text().strip() else None

    def update_estimate(self, *_):
        if not hasattr(self, "start_button") or self.state != "configure":
            return
        video = self.source.currentText() == "video"
        for row in self.video_rows:
            self.form.setRowVisible(row, video)
        for control in (self.video_path, self.video_choose, self.video_speed, self.video_loop, self.video_paused, self.video_fastest):
            control.setEnabled(video)
        self.camera.setEnabled(self.source.currentText() == "camera")
        summary = ""
        try:
            config = self.selected_config()
            summary = (f"{config.parameters:,} learned parameters · {config.hidden_values:,} recurrent values\n"
                       f"Additional working allowance: {config.working_bytes/1024**2:,.0f} MiB "
                       "CPU RAM" + (" and GPU VRAM" if self.compute.currentText() == "cuda" else "") + "\n")
            if self.compute.currentText() == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA unavailable in this environment. Select CPU or use the CUDA installation.")
            allocation = check_budget(config, self.compute.currentText(), self.memory_limit())
            summary += f"Approximate total process RAM: {allocation['estimated_host_total_mib']:,.0f} MiB (current RSS + allowance)\n"
            summary += f"Current safe budget: {allocation['budget_mib']:,.0f} MiB · host available: {allocation['host_available_mib']:,.0f} MiB"
            if allocation["gpu_free_mib"] is not None:
                summary += f" · GPU free: {allocation['gpu_free_mib']:,.0f} MiB"
            summary += ("\nPreflight budget fits. Exact CUDA forward/backward/Adam check runs on Start."
                        if self.compute.currentText() == "cuda" else "\nCPU selected; allocation is checked again on Start.")
            summary += " Estimates are conservative allowances, not peak-memory or speed guarantees."
            if self.source.currentText() == "camera" and not self.camera.currentText() and not self.camera_available:
                raise ValueError("No unique C922 available. Refresh/select a device or choose another source.")
            self.capacity.setText(summary)
            self.start_button.setEnabled(True)
        except Exception as exc:
            self.capacity.setText(summary + "\nCannot start: " + str(exc))
            self.start_button.setEnabled(False)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose local visual video (audio ignored)")
        if path:
            self.video_path.setText(path)

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Inspect checkpoint before restoring", "", "COGNIO checkpoint (*.pt)")
        if path:
            try:
                self.inspect_checkpoint(path)
            except Exception as exc:
                self.status.setText(f"Checkpoint inspection failed: {exc}")

    def inspect_checkpoint(self, path, config=None):
        metadata = Engine.inspect_checkpoint(path)
        config = config or metadata["config"]
        self.checkpoint_path, self.checkpoint_config = path, config
        for key in ("topology", "grid", "input_mode"):
            getattr(self, key).setCurrentText(str(getattr(config, key)))
        self.channels.setText(str(config.channels))
        self.seed.setText(str(config.seed))
        self.learning_rate.setValue(config.learning_rate)
        for control in self.architecture_widgets:
            control.setEnabled(False)
        self.frozen.setChecked(not self.args.resume_training)
        self.reset_hidden.setEnabled(True)
        self.origin.setText(f"Saved session · {path}\nSchema {metadata['schema']} · {config.topology} · "
                            f"{config.channels} channels × {config.grid}×{config.grid} · {config.input_mode}\n"
                            "Architecture is locked. Full tensor/optimizer validation occurs on Restore & Start.")
        self.start_button.setText("Restore & Start")
        self.status.clear()
        self.update_estimate()

    def use_fresh(self):
        self.checkpoint_path = self.checkpoint_config = None
        for control in self.architecture_widgets:
            control.setEnabled(True)
        self.reset_hidden.setChecked(False)
        self.reset_hidden.setEnabled(False)
        self.frozen.setChecked(False)
        self.origin.setText("Fresh session · new random weights with the displayed seed will be created on Start.")
        self.start_button.setText("Start")
        self.status.clear()
        self.update_estimate()

    def start_session(self):
        if self.state != "configure":
            return
        self.update_estimate()
        if not self.start_button.isEnabled():
            return
        if self.source.currentText() == "video" and not self.video_path.text():
            self.choose_video()
            if not self.video_path.text():
                return
        config = self.selected_config()
        args = copy.copy(self.args)
        for name, value in asdict(config).items():
            setattr(args, name, value)
        args.gui = True
        args.source, args.compute = self.source.currentText(), self.compute.currentText()
        args.device = self.camera.currentText() or None
        args.checkpoint = self.checkpoint_path
        args.memory_budget_mib = self.memory_limit()
        args.frozen = self.frozen.isChecked()
        args.resume_training = bool(args.checkpoint and not args.frozen)
        args.reset_hidden = bool(args.checkpoint and self.reset_hidden.isChecked())
        args.video_file = self.video_path.text() or None
        args.video_speed = self.video_speed.value()
        args.video_loop, args.video_paused = self.video_loop.isChecked(), self.video_paused.isChecked()
        args.video_fastest = self.video_fastest.isChecked()
        self.state = "starting"
        self.configuration.setEnabled(False)
        self.status.setText("Starting: checking the exact selected allocation and preparing the session…")
        self.launch_worker(lambda: build_runtime(args, expected_config=config))

    def launch_worker(self, work):
        def execute():
            try:
                self.results.put((work(), None))
            except Exception as exc:
                # Do not retain an allocation failure's traceback/tensors in the UI.
                self.results.put((None, f"{type(exc).__name__}: {exc}"))
        self.worker = threading.Thread(target=execute, name="cognio-session", daemon=True)
        self.worker.start()

    def poll_worker(self):
        if self.worker is None or self.worker.is_alive():
            return
        self.worker.join()
        self.worker = None
        result, error = self.results.get_nowait()
        if error:
            self.status.setText(error)
            self.configuration.setEnabled(True)
            if self.state == "stopping":
                # Keep ownership of a possibly live worker; never construct a second run.
                self.state = "running"
                self.running.setEnabled(True)
                self.running.timer.start(33)
                self.running.status.setText("Stop failed; retry New Session/Exit. " + error)
                self.closing = False
                return
            self.state = "configure"
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            self.update_estimate()
        elif self.state == "starting":
            self.runtime = result
            if self.closing:
                self.runtime = None  # Never started: no camera/decoder resources to stop.
                self.state = "configure"
            else:
                self.running = Window(result, new_session=self.new_session)
                self.stack.addWidget(self.running)
                self.stack.setCurrentWidget(self.running)
                self.resize(1320, 940)
                self.state = "running"
                self.setWindowTitle("COGNIO · running experiment")
                result.start()
        elif self.state == "stopping":
            self.stack.setCurrentWidget(self.configuration)
            self.stack.removeWidget(self.running)
            self.running.runtime = None
            self.running.deleteLater()
            self.running = self.runtime = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            self.state = "configure"
            self.configuration.setEnabled(True)
            self.status.setText("Previous session fully stopped. Configure and Start a new session.")
            self.setWindowTitle("COGNIO · configure experiment")
            self.update_estimate()
        if self.closing:
            self.close()

    def new_session(self):
        if self.state != "running":
            return
        answer = QMessageBox.question(self, "New session",
            "Stop the current session and discard its in-memory learning? Save a checkpoint first if you want to keep it. "
            "Existing checkpoint files will not be changed.")
        if answer == QMessageBox.StandardButton.Yes:
            self.stop_session()

    def stop_session(self):
        self.state = "stopping"
        self.running.timer.stop()
        self.running.setEnabled(False)
        self.running.status.setText("Stopping workers and releasing input resources…")
        self.launch_worker(self.runtime.stop)

    def closeEvent(self, event):
        if self.state == "configure":
            self.poll.stop()
            event.accept()
        else:
            self.closing = True
            if self.state == "running":
                self.stop_session()
            event.ignore()

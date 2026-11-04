"""Qt-only presentation of immutable measured snapshots; no model mutations."""

import queue
import time
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)


def activation_tiles(hidden):
    """All 32 real channels, row-major 4x8 tiles, each preserving its 8x8 positions."""
    if hidden.shape != (32, 8, 8):
        raise ValueError("expected 32 channels of 8x8 hidden activations")
    return hidden.reshape(4, 8, 8, 8).transpose(0, 2, 1, 3).reshape(32, 64)


def signed_rgb(values, display_range):
    scaled = np.clip(values / display_range, -1, 1)
    positive = np.maximum(scaled, 0)[..., None]
    negative = np.maximum(-scaled, 0)[..., None]
    return (20 + positive * np.array([235, 120, 30]) + negative * np.array([40, 135, 235])).astype(np.uint8)


def gray_rgb(values, maximum=1):
    gray = (255 * np.clip(values / maximum, 0, 1)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


class ImageView(QLabel):
    def __init__(self, minimum):
        super().__init__("Waiting for measurements")
        self.setMinimumSize(*minimum)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb = None

    def display(self, rgb):
        self.rgb = np.ascontiguousarray(rgb)
        self.redraw()

    def clear_image(self):
        self.rgb = None
        self.clear()
        self.setText("Waiting for measurements")

    def redraw(self):
        if self.rgb is not None:
            a = self.rgb
            image = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_RGB888).copy()
            self.setPixmap(QPixmap.fromImage(image).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.FastTransformation))

    def resizeEvent(self, event):
        self.redraw()
        super().resizeEvent(event)


class ErrorPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.history = []
        self.setMinimumHeight(130)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151e2a"))
        painter.setPen(QColor("#dbe5ef"))
        painter.drawText(10, 22, "Previously unseen targets · MSE   /   cyan: model   amber: persistence")
        if not self.history:
            return
        high = max(max(pair) for pair in self.history) or 1e-6
        painter.drawText(10, 42, f"0 … {high:.5f} · last {len(self.history)} matched targets in this phase")
        for column, color in ((0, "#68e1db"), (1, "#f3b665")):
            painter.setPen(QPen(QColor(color), 2))
            points = [QPointF(10 + i / max(1, len(self.history)-1) * (self.width()-20),
                              self.height()-10 - pair[column]/high * (self.height()-62))
                      for i, pair in enumerate(self.history)]
            for a, b in zip(points, points[1:]):
                painter.drawLine(a, b)


class Window(QMainWindow):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.setWindowTitle("COGNIO · online predictive learning · Prototype 1")
        self.resize(1320, 940)
        self.setStyleSheet("QWidget { background: #101720; color: #dbe5ef; font-size: 13px; }"
                           "QPushButton, QDoubleSpinBox { background: #273649; padding: 6px; }"
                           "QLabel { background: transparent; }")
        root = QWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        main = QVBoxLayout()
        side = QVBoxLayout()
        row.addLayout(main, 4)
        row.addLayout(side, 2)
        title = QLabel("COGNIO / 4 · actual recurrent neural state")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main.addWidget(title)
        self.state_label = QLabel("32 channels · channel 0–31 in row-major tiles · each tile = 8×8 spatial positions")
        self.state_label.setWordWrap(True)
        main.addWidget(self.state_label)
        self.hidden_view = ImageView((480, 240))
        main.addWidget(self.hidden_view, 1)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        main.addWidget(self.legend)
        controls = QHBoxLayout()
        self.training = QCheckBox("Train online (capture and state continue when off)")
        self.training.setChecked(runtime.engine.training)
        self.training.toggled.connect(lambda enabled: self.send("training", enabled))
        controls.addWidget(self.training)
        self.display_range = QDoubleSpinBox()
        self.display_range.setRange(.01, 1)
        self.display_range.setDecimals(2)
        self.display_range.setSingleStep(.05)
        self.display_range.setValue(1)
        controls.addWidget(QLabel("Display ±"))
        controls.addWidget(self.display_range)
        main.addLayout(controls)
        buttons = QHBoxLayout()
        for label, action in (("Reset to new random model…", self.reset), ("Save checkpoint…", self.save),
                              ("Restore checkpoint…", self.restore)):
            button = QPushButton(label)
            button.clicked.connect(action)
            buttons.addWidget(button)
        main.addLayout(buttons)
        self.origin = QLabel()
        self.origin.setWordWrap(True)
        main.addWidget(self.origin)
        self.plot = ErrorPlot()
        main.addWidget(self.plot)
        self.telemetry = QLabel()
        self.telemetry.setWordWrap(True)
        self.telemetry.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main.addWidget(self.telemetry)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #f3b665;")
        main.addWidget(self.status)
        privacy = QLabel("Local only · camera activated only with --source camera · no image recording\n"
                         "Patterns and lower prediction error do not establish recognition or consciousness.")
        privacy.setWordWrap(True)
        main.addWidget(privacy)
        self.input_label, self.input_view = self.panel(side, "1 / Current measured luminance")
        self.prediction_label, self.prediction_view = self.panel(side, "2 / Predicted future luminance")
        self.error_label, self.error_view = self.panel(side, "3 / Matched squared prediction error")
        self.refresh_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    @staticmethod
    def panel(layout, title):
        label = QLabel(title)
        label.setWordWrap(True)
        view = ImageView((180, 180))
        layout.addWidget(label)
        layout.addWidget(view, 1)
        return label, view

    def send(self, action, value=None):
        try:
            self.runtime.submit(action, value)
        except queue.Full:
            self.status.setText("Control queue full; retry when processing catches up.")

    def reset(self):
        if QMessageBox.question(self, "New random model", "Discard unsaved learning and initialize a new random model?") == QMessageBox.StandardButton.Yes:
            self.send("reset")

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save model and optimizer (no images)", "cognio-state.pt", "COGNIO checkpoint (*.pt)")
        if path:
            self.send("save", path)

    def restore(self):
        path, _ = QFileDialog.getOpenFileName(self, "Restore model; training will pause", "", "COGNIO checkpoint (*.pt)")
        if path:
            self.send("load", path)

    def refresh(self):
        snap = self.runtime.latest()
        if snap is None:
            return
        self.refresh_count += 1
        self.training.blockSignals(True)
        self.training.setChecked(snap["training"])
        self.training.blockSignals(False)
        self.origin.setText(f"{snap['origin']}\n{snap['compute']} · source: {snap['source']}")
        visual, result = snap["visual"], snap["result"]
        now = time.monotonic()
        if visual is not None:
            limit = self.display_range.value()
            self.hidden_view.display(signed_rgb(activation_tiles(visual["hidden"]), limit))
            self.legend.setText(f"Blue negative / dark zero / amber positive · fixed display ±{limit:.2f}"
                                f" · {100*np.mean(np.abs(visual['hidden']) > limit):.1f}% clipped\n"
                                "All tiles are measured activations, with no learned or adaptive display projection.")
            self.input_view.display(gray_rgb(visual["input"]))
            self.prediction_view.display(gray_rgb(visual["prediction"]))
            self.input_label.setText(f"1 / Current measured luminance · frame {snap['input_sequence']}\n"
                                     f"Y/255 · age {1000*(now-visual['input_time']):.0f} ms")
            self.prediction_label.setText(f"2 / Predicted future luminance · {visual['source']} → {visual['target']}\n"
                                          f"Issued {1000*(now-visual['issued']):.0f} ms ago · horizon +5 camera frames")
        else:
            for view in (self.hidden_view, self.input_view, self.prediction_view):
                view.clear_image()
        if result is not None:
            self.error_view.display(gray_rgb(result["error"], .25))
            self.error_label.setText(f"3 / Squared error · {result['source']} → observed {result['target']}\n"
                                     f"Horizon {result['horizon_ms']:.1f} ms · target age {1000*(now-result['target_time']):.0f} ms"
                                     " · black 0 / white ≥0.25")
        else:
            self.error_view.clear_image()
            self.error_label.setText("3 / Matched squared error · waiting for a causal +5 target")
        self.plot.history = snap["history"]
        self.plot.update()
        means = np.mean(snap["history"], axis=0) if snap["history"] else (float("nan"), float("nan"))
        counts, capture = snap["counts"], snap["capture"]
        loss = snap["training_loss"]
        loss_text = f"{loss:.6f}" if loss is not None else "—"
        self.telemetry.setText(
            f"Input {snap['input_fps']:.1f} FPS · predictions {snap['prediction_fps']:.1f}/s · updates {snap['updates_per_s']:.1f}/s\n"
            f"Rolling unseen MSE {means[0]:.6f} · persistence {means[1]:.6f} · pairs {len(snap['history'])}\n"
            f"Adam lr {snap['config'].learning_rate:g} · latest training loss {loss_text}\n"
            f"Prediction {snap['prediction_ms']:.2f} ms · update {snap['update_ms']:.2f} ms · total updates {counts['updates']}\n"
            f"Drops: driver {capture['driver_drops']}, drained {capture['drained_discards']}, invalid {capture['bad_frames']}, "
            f"handoff {snap['overwritten']} · missed targets {counts['missed_targets']} · late targets {counts['late_targets']}\n"
            f"RSS {snap['rss_mib']:.0f} MiB (peak {snap['peak_rss_mib']:.0f}) · CPU {snap['cpu_percent']:.0f}% (100% = one core) "
            f"· PyTorch VRAM {snap['vram_mib']:.1f} / reserved {snap['reserved_vram_mib']:.1f} MiB")
        self.status.setText("\n".join(v for v in (snap["fault"], snap["capture_error"], snap["notice"],
                                                  *snap["capture_info"].get("warnings", [])) if v)
                            or ("Learning online" if snap["training"] else "Training frozen · sensory processing continues"))

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.runtime.stop()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Shutdown incomplete", str(exc))
            event.ignore()
            self.timer.start(33)
            return
        event.accept()

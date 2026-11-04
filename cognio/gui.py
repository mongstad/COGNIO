"""Qt-only presentation of immutable measured snapshots; no model mutations."""

import queue
import time
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget, QSpinBox, QComboBox)
from .inputs import CHROMA_OFFSET, components


def activation_tiles(hidden):
    """At most 32 measured channels; raster tiles with the full spatial grid intact."""
    if hidden.ndim != 3 or not 1 <= hidden.shape[0] <= 32 or hidden.shape[1] != hidden.shape[2]:
        raise ValueError("expected a page of 1..32 square hidden channels")
    count, grid, _ = hidden.shape
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    padded = np.zeros((rows*columns, grid, grid), dtype=hidden.dtype)
    padded[:count] = hidden
    return padded.reshape(rows, columns, grid, grid).transpose(0, 2, 1, 3).reshape(rows*grid, columns*grid)


def signed_rgb(values, display_range):
    scaled = np.clip(values / display_range, -1, 1)
    positive = np.maximum(scaled, 0)[..., None]
    negative = np.maximum(-scaled, 0)[..., None]
    return (20 + positive * np.array([235, 120, 30]) + negative * np.array([40, 135, 235])).astype(np.uint8)


def gray_rgb(values, maximum=1):
    gray = (255 * np.clip(values / maximum, 0, 1)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def component_plane(values, component):
    return values if values.ndim == 2 else values[component]


def measurement_rgb(values, component):
    if component is None:
        return ycbcr_rgb(values)
    plane = component_plane(values, component)
    return gray_rgb(plane) if component == 0 else signed_rgb(plane, CHROMA_OFFSET)


def ycbcr_rgb(values):
    """Display-only BT.601 limited-range reconstruction from CHW model units.

    Undo Y/255 and (Cb/Cr-128)/255, then expand nominal Y codes 16..235
    and chroma excursions ±112. This is a fixed C922 preview assumption,
    not calibration or a change to sensory normalization. Clip RGB only;
    invalid/non-finite pixels are magenta. Return owned, contiguous HWC bytes.
    """
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("RGB preview requires CHW Y, Cb, Cr measurements")
    # Float64 avoids overflow even for extreme finite float32 predictions.
    with np.errstate(over="ignore", invalid="ignore"):
        y, cb, cr = np.asarray(values, dtype=np.float64)
        y = (255*y - 16) / 219
        cb, cr = 255*cb / 224, 255*cr / 224
        rgb = np.stack((y + 1.402*cr,
                        y - (0.114*1.772/0.587)*cb - (0.299*1.402/0.587)*cr,
                        y + 1.772*cb), axis=-1)
    valid = np.isfinite(values).all(axis=0) & np.isfinite(rgb).all(axis=-1)
    rgb = np.rint(255*np.clip(np.nan_to_num(rgb), 0, 1)).astype(np.uint8)
    rgb[~valid] = (255, 0, 255)
    return np.ascontiguousarray(rgb)


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
        self.state_label = QLabel("Measured hidden channels · waiting for observer snapshot")
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
        self.channel_page = QSpinBox()
        self.channel_page.setRange(1, (runtime.engine.config.channels + 31)//32)
        self.channel_page.valueChanged.connect(lambda value: self.send("channel_page", value-1))
        controls.addWidget(QLabel("Channel page"))
        controls.addWidget(self.channel_page)
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
        self.component = QComboBox()
        if runtime.engine.config.input_mode == "ycbcr":
            self.component.addItem("Combined Color (RGB preview)", None)
            diagnostic_label = QLabel("Diagnostic view · display only")
            diagnostic_label.setWordWrap(True)
            side.addWidget(diagnostic_label)
            side.addWidget(self.component)
        for index, name in enumerate(components(runtime.engine.config.input_mode)):
            self.component.addItem(name, index)
        self.component.setToolTip(
            "Combined Color assumes BT.601 limited range on the aligned 128×128 Y/Cb/Cr planes. "
            "RGB is clipped for display; invalid pixels are magenta. No sensor color calibration. "
            "Cb/Cr alone use signed display colors. Selection never changes learning.")
        self.input_label, self.input_view = self.panel(side, "1 / Current measured luminance")
        self.prediction_label, self.prediction_view = self.panel(side, "2 / Predicted future luminance")
        self.error_label, self.error_view = self.panel(side, "3 / Matched squared prediction error")
        self.refresh_count = 0
        self.display_ms = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)
        self.component.currentIndexChanged.connect(self.refresh)

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
        started = time.perf_counter()
        snap = self.runtime.latest()
        if snap is None:
            return
        self.refresh_count += 1
        self.training.blockSignals(True)
        self.training.setChecked(snap["training"])
        self.training.blockSignals(False)
        self.origin.setText(f"{snap['origin']}\n{snap['compute']} · source: {snap['source']} · input: {snap['config'].input_mode}")
        visual, result = snap["visual"], snap.get("display_result", snap["result"])
        component = self.component.currentData()
        name = self.component.currentText()
        combined = component is None
        measurement = "color · RGB preview" if combined else ("luminance" if name == "Y" else f"{name} color difference")
        units = ("BT.601 limited-range assumption · display only" if combined else
                 "Y/255" if name == "Y" else f"({name}−128)/255 · blue negative / amber positive · display only")
        now = time.monotonic()
        if visual is not None:
            limit = self.display_range.value()
            first = visual.get("channel_start", 0)
            last = first + len(visual["hidden"]) - 1
            config = snap["config"]
            self.state_label.setText(f"Channels {first}–{last} of {config.channels} · each tile {config.grid}×{config.grid} · "
                                     f"{config.hidden_values:,} actual state values · {config.parameters:,} learned parameters")
            self.hidden_view.display(signed_rgb(activation_tiles(visual["hidden"]), limit))
            self.legend.setText(f"Blue negative / dark zero / amber positive · fixed display ±{limit:.2f}"
                                f" · {100*np.mean(np.abs(visual['hidden']) > limit):.1f}% clipped\n"
                                "All tiles are measured activations, with no learned or adaptive display projection.")
            self.input_view.display(measurement_rgb(visual["input"], component))
            self.prediction_view.display(measurement_rgb(visual["prediction"], component))
            self.input_label.setText(f"1 / Current measured {measurement} snapshot · frame {visual['input_sequence']}\n"
                                     f"{units} · age {1000*(now-visual['input_time']):.0f} ms")
            self.prediction_label.setText(f"2 / Predicted future {measurement} · {visual['source']} → {visual['target']}\n"
                                          f"Issued {1000*(now-visual['issued']):.0f} ms ago · horizon +5 camera frames")
        else:
            for view in (self.hidden_view, self.input_view, self.prediction_view):
                view.clear_image()
        if result is not None and result["error"] is not None:
            error = result["error"].mean(axis=0) if combined else component_plane(result["error"], component)
            error_name = "Mean Y/Cb/Cr squared error (not RGB)" if combined else f"{name} squared error"
            self.error_view.display(gray_rgb(error, .25))
            self.error_label.setText(f"3 / {error_name} · {result['source']} → observed {result['target']}\n"
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
        component_text = ""
        if snap["config"].input_mode == "ycbcr" and snap["component_history"]:
            per = np.mean(snap["component_history"], axis=0)
            component_text = "Component MSE / persistence: " + " · ".join(
                f"{name} {per[0,i]:.5f}/{per[1,i]:.5f}" for i, name in enumerate(components("ycbcr"))) + "\n"
        self.telemetry.setText(
            f"Input {snap['input_fps']:.1f} FPS · predictions {snap['prediction_fps']:.1f}/s · updates {snap['updates_per_s']:.1f}/s\n"
            f"Rolling unseen MSE {means[0]:.6f} · persistence {means[1]:.6f} · pairs {len(snap['history'])}\n"
            f"{component_text}"
            f"Adam lr {snap['config'].learning_rate:g} · latest training loss {loss_text}\n"
            f"Prediction {snap['prediction_ms']:.2f} ms · update {snap['update_ms']:.2f} ms · total updates {counts['updates']}\n"
            f"Observer copy {snap['observer_ms']:.2f} ms · display {self.display_ms:.2f} ms · step {snap['step_ms']:.2f} ms\n"
            f"Drops: driver {capture['driver_drops']}, drained {capture['drained_discards']}, invalid {capture['bad_frames']}, "
            f"handoff {snap['overwritten']} · missed targets {counts['missed_targets']} · late targets {counts['late_targets']}\n"
            f"RSS {snap['rss_mib']:.0f} MiB (peak {snap['peak_rss_mib']:.0f}) · CPU {snap['cpu_percent']:.0f}% (100% = one core) "
            f"· PyTorch VRAM {snap['vram_mib']:.1f} / reserved {snap['reserved_vram_mib']:.1f} MiB")
        self.status.setText("\n".join(v for v in (snap["fault"], snap["capture_error"], snap["notice"],
                                                  *snap["capture_info"].get("warnings", [])) if v)
                            or ("Learning online" if snap["training"] else "Training frozen · sensory processing continues"))
        self.display_ms = (time.perf_counter() - started)*1000

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

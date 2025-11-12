#!/usr/bin/env python3
"""Minimal PyQt6 front-end for main.py Rydberg-EIT simulation."""
from __future__ import annotations

import sys
from pathlib import Path


import matplotlib
import numpy as np
matplotlib.use("Agg")

import re

from PyQt6.QtCore import QProcess, Qt, QTimer, QEvent
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from visualize_cell import render_scene


REPO_ROOT = Path(__file__).resolve().parent


class SimulationGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Rydberg EIT Simulator")

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.isotope = QComboBox()
        self.isotope.addItems(["Rb87", "Rb85"])
        self.rf_frequency = QLineEdit("2377")
        self.rf_amplitudes = QLineEdit("0 0.01 0.05 0.1")
        self.probe_power = QLineEdit("1e-4")
        self.control_power = QLineEdit("10e-3")
        self.probe_waist = QLineEdit("100e-6")
        self.control_waist = QLineEdit("100e-6")
        self.cell_length = QLineEdit("0.10")
        self.cell_cross = QLineEdit("0.02")
        self.temperature = QLineEdit("300")
        self.detuning_span = QLineEdit("300")
        self.detuning_points = QLineEdit("401")
        self.output_file = QLineEdit("eit_rf_gui.png")
        self.override_pressure = QCheckBox("Override pressure")
        self.pressure_torr = QLineEdit("")
        self.pressure_torr.setEnabled(False)
        self.override_pressure.toggled.connect(self._toggle_pressure_field)
        self.probe_label_text = ""
        self.control_label_text = ""
        self.auto_n = QCheckBox("Auto select n")
        self.auto_n.setChecked(True)
        
        self.n_value = QLineEdit("50")
        self.np_value = QLineEdit("51")
        self.n_value.setEnabled(False)
        self.np_value.setEnabled(False)
        self.auto_n.toggled.connect(self._toggle_n_fields)
        self.normalize = QCheckBox("Normalize baseline")
        self.baseline_amp = QLineEdit("1000")
        self.timing = QCheckBox("Show timing")
        self.no_show = QCheckBox("Skip plot window")
        self.no_show.setChecked(True)
        self.fit_peaks = QCheckBox("Fit peaks")
        self.auto_rotate = QCheckBox("Auto-rotate visualization")
        self.auto_rotate.setChecked(True)
        self.auto_rotate.stateChanged.connect(self._toggle_auto_rotate)

        forms = QVBoxLayout()

        rf_group = self._make_rows([
            ("Isotope", self.isotope),
            ("RF frequency (MHz)", self.rf_frequency),
            ("RF amplitudes (V/cm)", self.rf_amplitudes),
            ("Temperature (K)", self.temperature),
            ("Cell length (m)", self.cell_length, "Cell cross (m)", self.cell_cross),
            ("Probe span (MHz)", self.detuning_span),
            ("Probe points", self.detuning_points),
        ])
        beam_form = self._make_rows([
            ("Probe power (W)", self.probe_power, "Probe waist (m)", self.probe_waist),
            ("Control power (W)", self.control_power, "Control waist (m)", self.control_waist),
        ])
        state_layout = QHBoxLayout()
        state_label = QLabel("Level select")
        state_label.setMinimumWidth(140)
        state_layout.addWidget(state_label)
        state_layout.addWidget(self.auto_n)
        state_layout.addSpacing(10)
        state_layout.addWidget(QLabel("n"))
        state_layout.addWidget(self.n_value)
        state_layout.addWidget(QLabel("n_p"))
        state_layout.addWidget(self.np_value)
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(self.override_pressure)
        pressure_label = QLabel("Custom pressure (Torr)")
        pressure_label.setMinimumWidth(140)
        pressure_layout.addWidget(pressure_label)
        pressure_layout.addWidget(self.pressure_torr)

        misc_group = self._make_rows([
            ("Output figure", self.output_file),
            ("Baseline RF (V/cm)", self.baseline_amp),
        ])
        toggles_layout = QHBoxLayout()
        toggles_layout.addWidget(self.normalize)
        toggles_layout.addWidget(self.timing)
        toggles_layout.addWidget(self.no_show)
        toggles_layout.addWidget(self.fit_peaks)

        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Extra CLI args (e.g., --doppler-method uniform --doppler-width 3.0)")

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)

        forms.addLayout(rf_group)
        forms.addLayout(beam_form)
        forms.addLayout(state_layout)
        forms.addLayout(pressure_layout)
        forms.addLayout(misc_group)
        forms.addLayout(toggles_layout)
        forms.addWidget(self.extra_args)
        forms.addSpacing(10)
        forms.addWidget(self.run_button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.preview_label = QLabel("Preview not available.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(600, 450)

        self.visual_fig = Figure(figsize=(3, 3))
        self.visual_ax = self.visual_fig.add_subplot(111, projection="3d")
        self.visual_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.visual_canvas = FigureCanvas(self.visual_fig)
        self.visual_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.visual_canvas.installEventFilter(self)
        self.rotation_angle = 0.0
        self.rotation_speed = 1.0
        self.visual_timer = QTimer(self)
        self.visual_timer.timeout.connect(self._spin_visualization)
        if self.auto_rotate.isChecked():
            self.visual_timer.start(100)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._reset_visual_view)
        self._update_visualization()

        log_layout = QVBoxLayout()
        log_row = QHBoxLayout()
        log_row.setSpacing(10)
        log_row.addWidget(self.output, 1)
        visual_column = QVBoxLayout()
        visual_column.addWidget(self.visual_canvas)
        button_row = QHBoxLayout()
        button_row.addWidget(self.auto_rotate)
        button_row.addWidget(self.reset_view_btn)
        visual_column.addLayout(button_row)
        log_row.addLayout(visual_column, 1)
        log_layout.addLayout(log_row)
        log_layout.addWidget(self.preview_label)

        wrapper = QHBoxLayout()
        wrapper.addLayout(forms)
        wrapper.addLayout(log_layout)
        self.setLayout(wrapper)
        for widget in [self.cell_length, self.cell_cross,
                       self.probe_waist, self.control_waist]:
            widget.editingFinished.connect(self._update_visualization)

    def _make_rows(self, rows):
        layout = QVBoxLayout()
        for row in rows:
            row_layout = QHBoxLayout()
            for idx in range(0, len(row), 2):
                label = QLabel(row[idx])
                label.setMinimumWidth(140)
                row_layout.addWidget(label)
                row_layout.addWidget(row[idx + 1])
            layout.addLayout(row_layout)
        return layout

    # ------------------------------------------------------------------ helpers
    def _read_stdout(self) -> None:
        data = self.process.readAllStandardOutput().data().decode()
        self.output.append(data)
        self._parse_wavelengths(data)

    def _read_stderr(self) -> None:
        data = self.process.readAllStandardError().data().decode()
        self.output.append(f"<span style='color:#d00;'>{data}</span>")

    def _process_finished(self) -> None:
        self.run_button.setEnabled(True)
        code = self.process.exitCode()
        if code == 0:
            self.output.append("<b>Simulation finished successfully.</b>")
            self._update_preview()
            self._update_visualization()
        else:
            self.output.append(f"<b style='color:#d00;'>Simulation failed (code {code}).</b>")

    # ------------------------------------------------------------------ actions
    def run_simulation(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Running", "Simulation already in progress.")
            return

        python_exec = sys.executable
        cmd = [
            python_exec,
            str(REPO_ROOT / "main.py"),
            "--isotope",
            self.isotope.currentText(),
            "--cell-length",
            self.cell_length.text().strip(),
            "--rf-frequency",
            self.rf_frequency.text().strip(),
            "--probe-power",
            self.probe_power.text().strip(),
            "--control-power",
            self.control_power.text().strip(),
            "--probe-waist",
            self.probe_waist.text().strip(),
            "--control-waist",
            self.control_waist.text().strip(),
            "--temperature",
            self.temperature.text().strip(),
            "--probe-span",
            self.detuning_span.text().strip(),
            "--probe-points",
            self.detuning_points.text().strip(),
            "--output",
            self.output_file.text().strip(),
        ]
        if self.override_pressure.isChecked() and self.pressure_torr.text().strip():
            cmd.extend(["--pressure-torr", self.pressure_torr.text().strip()])

        if self.auto_n.isChecked():
            cmd.append("--auto-n")
        else:
            cmd.extend(["--n", self.n_value.text().strip()])
            if self.np_value.text().strip():
                cmd.extend(["--np", self.np_value.text().strip()])
        if self.normalize.isChecked():
            cmd.append("--normalize-baseline")
        if self.baseline_amp.text().strip():
                cmd.extend(["--baseline-rf-amplitude", self.baseline_amp.text().strip()])
        if self.no_show.isChecked():
            cmd.append("--no-show")
        if self.timing.isChecked():
            cmd.append("--timing")
        if self.fit_peaks.isChecked():
            cmd.append("--fit-peaks")
        extra = self.extra_args.text().strip()
        if extra:
            cmd.extend(extra.split())

        amps = [a for a in self.rf_amplitudes.text().split() if a]
        if amps:
            cmd.append("--rf-amplitudes")
            cmd.extend(amps)

        self.output.clear()
        self.output.append("Running: " + " ".join(cmd))
        self.run_button.setEnabled(False)
        self.process.start(cmd[0], cmd[1:])

    def _update_visualization(self) -> None:
        try:
            cell_length = float(self.cell_length.text())
            cell_cross = float(self.cell_cross.text())
            probe_waist = float(self.probe_waist.text())
            control_waist = float(self.control_waist.text())
        except ValueError:
            return

        render_scene(self.visual_ax, cell_length, cell_cross,
                     probe_waist, control_waist, resolution=120, zoom=1.0,
                     probe_label=self.probe_label_text,
                     control_label=self.control_label_text)
        self._reset_visual_view()
        if self.auto_rotate.isChecked():
            self.visual_timer.start(100)

    def _reset_visual_view(self) -> None:
        self.visual_ax.view_init(elev=0, azim=70,roll=70)
        self.visual_ax.set_proj_type('persp')
        self.visual_ax.dist = 10
        self.visual_canvas.draw_idle()
        if self.auto_rotate.isChecked():
            self.visual_timer.start(100)
        else:
            self.visual_timer.start(100)
            self.auto_rotate.toggle()

    def _toggle_n_fields(self, checked: bool) -> None:
        self.n_value.setEnabled(not checked)
        self.np_value.setEnabled(not checked)

    def _toggle_pressure_field(self, checked: bool) -> None:
        self.pressure_torr.setEnabled(checked)

    def _update_preview(self) -> None:
        path = self.output_file.text().strip()
        if not path:
            self.preview_label.setText("No output file specified.")
            return
        full_path = (REPO_ROOT / path).resolve()
        if not full_path.exists():
            self.preview_label.setText(f"No file at {full_path}.")
            return
        pixmap = QPixmap(str(full_path))
        if pixmap.isNull():
            self.preview_label.setText("Failed to load image.")
        else:
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def _parse_wavelengths(self, text: str) -> None:
        probe_match = re.search(r"Probe transition:.*?~([\d\.]+)\s+nm", text)
        control_match = re.search(r"Coupling transition:.*?~([\d\.]+)\s+nm", text)
        updated = False
        if probe_match:
            value = float(probe_match.group(1))
            self.probe_label_text = f"λ ≈ {value:.2f} nm"
            updated = True
        if control_match:
            value = float(control_match.group(1))
            self.control_label_text = f"λ ≈ {value:.2f} nm"
            updated = True
        if updated:
            self._update_visualization()

    def _spin_visualization(self) -> None:
        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
        azim = (self.rotation_angle) % 360
        elev = 100 * np.sin(np.radians(self.rotation_angle / 2))
        self.visual_ax.view_init(elev=elev, azim=azim, roll=70)
        self.visual_canvas.draw_idle()

    def eventFilter(self, obj, event):
        if obj is self.visual_canvas and event.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
        ):
            self.visual_timer.stop()
            if self.auto_rotate.isChecked():
                self.auto_rotate.setChecked(False)
        return super().eventFilter(obj, event)

    def _toggle_auto_rotate(self, state):
        if state == Qt.CheckState.Checked.value:
            self.visual_timer.start(100)
        else:
            self.visual_timer.stop()

def main() -> None:
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.resize(1600, 900)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

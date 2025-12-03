#!/usr/bin/env python3
"""Minimal PyQt6 front-end for main.py Rydberg-EIT simulation."""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from collections import OrderedDict
import shutil


import matplotlib
import numpy as np
matplotlib.use("Agg")

import re
import subprocess

from PyQt6.QtCore import QProcess, Qt, QTimer, QEvent
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QGroupBox,
    QFileDialog,
    QTabWidget,
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
        self.base_defaults = self._load_config(base_only=True)
        self.defaults = self._load_config()
        self.save_order = list(self.base_defaults.keys() or self.defaults.keys())
        self.isotope = QComboBox()
        self.isotope.addItems(["Rb87", "Rb85"])
        self.isotope.setCurrentText(self.defaults.get("isotope", "Rb87"))
        self.rf_frequency = QLineEdit(self.defaults.get("rf_frequency", "2377"))
        self.rf_amplitudes = QLineEdit(self.defaults.get("rf_amplitudes", "0 0.01 0.05 0.1"))
        self.probe_power = QLineEdit(self.defaults.get("probe_power", "1e-4"))
        self.control_power = QLineEdit(self.defaults.get("control_power", "10e-3"))
        self.probe_waist = QLineEdit(self.defaults.get("probe_waist", "100e-6"))
        self.control_waist = QLineEdit(self.defaults.get("control_waist", "100e-6"))
        self.cell_length = QLineEdit(self.defaults.get("cell_length", "0.10"))
        self.cell_cross = QLineEdit(self.defaults.get("cell_cross", "0.02"))
        self.temperature = QLineEdit(self.defaults.get("temperature", "300"))
        self.detuning_span = QLineEdit(self.defaults.get("detuning_span", "300"))
        self.detuning_points = QLineEdit(self.defaults.get("detuning_points", "401"))
        self.output_file = QLineEdit(self.defaults.get("output_file", "eit_rf_gui.png"))
        self.control_detuning = QLineEdit(self.defaults.get("control_detuning", "0.0"))
        self.transit_rate = QLineEdit(self.defaults.get("transit_rate", "0.15"))
        self.probe_linewidth = QLineEdit(self.defaults.get("probe_linewidth", "0.01"))
        self.control_linewidth = QLineEdit(self.defaults.get("control_linewidth", "0.01"))
        self.override_pressure = QCheckBox("Override pressure")
        self.override_pressure.setChecked(bool(self.defaults.get("override_pressure", False)))
        self.pressure_torr = QLineEdit("")
        self.pressure_torr.setEnabled(False)
        self.override_pressure.toggled.connect(self._toggle_pressure_field)
        self.enable_sweep_plot = QCheckBox("Generate sweep plot")
        self.enable_sweep_plot.setChecked(bool(self.defaults.get("enable_sweep_plot",
                                                                 self.defaults.get("sweep_plot", False))))
        self.sweep_output = QLineEdit(self.defaults.get("sweep_output", "eit_rf_sweep.png"))
        self.sweep_output.setEnabled(self.enable_sweep_plot.isChecked())
        self.enable_sweep_plot.toggled.connect(self._toggle_sweep_field)
        self.sweep_points = QLineEdit(self.defaults.get("sweep_points", "20"))
        self.sweep_points.setEnabled(self.enable_sweep_plot.isChecked())
        self.probe_label_text = ""
        self.control_label_text = ""
        self.auto_n = QCheckBox("Auto select n")
        self.auto_n.setChecked(bool(self.defaults.get("auto_n", True)))
        self.auto_n_only = QCheckBox("Auto-n only (skip simulation)")
        self.auto_n_only.setChecked(bool(self.defaults.get("auto_n_only", False)))
        
        self.n_value = QLineEdit("50")
        self.np_value = QLineEdit("51")
        self.n_value.setEnabled(False)
        self.np_value.setEnabled(False)
        self.auto_n.toggled.connect(self._toggle_n_fields)
        self.normalize = QCheckBox("Normalize baseline")
        self.normalize.setChecked(bool(self.defaults.get("normalize_baseline", False)))
        self.baseline_amp = QLineEdit(self.defaults.get("baseline_amp", "1000"))
        self.timing = QCheckBox("Show timing")
        self.timing.setChecked(bool(self.defaults.get("timing", False)))
        self.no_show = QCheckBox("Skip plot window")
        self.no_show.setChecked(bool(self.defaults.get("no_show", True)))
        self.fit_peaks = QCheckBox("Fit peaks")
        self.fit_peaks.setChecked(bool(self.defaults.get("fit_peaks", False)))
        self.fit_profile = QComboBox()
        self.fit_profile.addItem("Gaussian", "gaussian")
        self.fit_profile.addItem("Lorentzian", "lorentzian")
        fit_profile_default = self.defaults.get("fit_profile", "gaussian")
        idx = self.fit_profile.findData(fit_profile_default)
        if idx >= 0:
            self.fit_profile.setCurrentIndex(idx)
        self.auto_rotate = QCheckBox("Auto-rotate visualization")
        self.auto_rotate.setChecked(bool(self.defaults.get("auto_rotate", True)))
        self.auto_rotate.stateChanged.connect(self._toggle_auto_rotate)
        self.debug_timing = QCheckBox("Show timing")
        self.debug_timing.setChecked(bool(self.defaults.get("debug_timing", False)))
        self.debug_verbose = QCheckBox("Verbose (decoder/backend)")
        self.debug_verbose.setChecked(bool(self.defaults.get("verbose", False)))
        self.show_levels = QCheckBox("Show level diagram")
        self.show_levels.setChecked(bool(self.defaults.get("show_levels", False)))

        # Decoder inputs
        self.dec_n = QLineEdit("50")
        self.dec_np = QLineEdit("")
        self.dec_split = QLineEdit("10.0")
        self.dec_offset = QLineEdit("0.0")
        self.dec_df_correction = QLineEdit("1.58")
        self.decoder_output = QTextEdit()
        self.decoder_output.setReadOnly(True)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Extra CLI args (e.g., --doppler-method uniform --doppler-width 3.0)")
        if self.defaults.get("extra_args"):
            self.extra_args.setText(self.defaults["extra_args"])
        self.cli_help_box = QTextEdit()
        self.cli_help_box.setReadOnly(True)
        self.cli_help_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cli_help_box.setText("Loading CLI options...")
        self.summary_box = QGroupBox("Simulation summary")
        self.summary_box.setMinimumWidth(300)
        summary_layout = QFormLayout()
        self.summary_fields = {}
        for key in ("Selected n", "Selected n_p", "Probe transition", "Coupling transition",
                    "RF transition", "RF detuning", "Last peak ratio", "Last peak offset"):
            label = QLabel("—")
            summary_layout.addRow(key + ":", label)
            self.summary_fields[key] = label
        self.summary_box.setLayout(summary_layout)

        # Tabs setup
        self.tabs = QTabWidget()

        # Simulation tab
        sim_tab = QWidget()
        sim_layout = QVBoxLayout()
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
        sim_layout.addLayout(self._make_rows([
            ("RF frequency (MHz)", self.rf_frequency),
            ("RF amplitudes (V/cm)", self.rf_amplitudes),
            ("Probe span (MHz)", self.detuning_span),
            ("Probe points", self.detuning_points),
        ]))
        sim_layout.addWidget(self.auto_n_only)
        sim_layout.addLayout(state_layout)
        sim_layout.addStretch()
        sim_tab.setLayout(sim_layout)
        self.tabs.addTab(sim_tab, "Simulation")

        # Cell tab
        cell_tab = QWidget()
        cell_layout = QVBoxLayout()
        pressure_layout = QHBoxLayout()
        pressure_layout.addWidget(self.override_pressure)
        pressure_label = QLabel("Custom pressure (Torr)")
        pressure_label.setMinimumWidth(140)
        pressure_layout.addWidget(pressure_label)
        pressure_layout.addWidget(self.pressure_torr)
        cell_layout.addLayout(self._make_rows([
            ("Isotope", self.isotope),
            ("Temperature (K)", self.temperature),
            ("Cell length (m)", self.cell_length, "Cell cross (m)", self.cell_cross),
            ("Transit rate (MHz)", self.transit_rate),
        ]))
        cell_layout.addLayout(pressure_layout)
        cell_layout.addStretch()
        cell_tab.setLayout(cell_layout)
        self.tabs.addTab(cell_tab, "Cell")

        # Lasers tab
        laser_tab = QWidget()
        laser_layout = QVBoxLayout()
        laser_layout.addLayout(self._make_rows([
            ("Probe power (W)", self.probe_power, "Probe waist (m)", self.probe_waist),
            ("Control power (W)", self.control_power, "Control waist (m)", self.control_waist),
            ("Control detuning (MHz)", self.control_detuning),
            ("Probe linewidth (MHz)", self.probe_linewidth, "Control linewidth (MHz)", self.control_linewidth),
        ]))
        laser_layout.addStretch()
        laser_tab.setLayout(laser_layout)
        self.tabs.addTab(laser_tab, "Lasers")

        # Plot/Fit tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addLayout(self._make_rows([
            ("Output figure", self.output_file),
            ("Baseline RF (V/cm)", self.baseline_amp),
        ]))
        plot_toggles = QHBoxLayout()
        plot_toggles.addWidget(self.normalize)
        plot_toggles.addWidget(self.no_show)
        plot_toggles.addWidget(self.fit_peaks)
        plot_toggles.addWidget(self.show_levels)
        plot_layout.addLayout(plot_toggles)
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Fit profile"))
        profile_layout.addWidget(self.fit_profile)
        plot_layout.addLayout(profile_layout)
        sweep_layout = QHBoxLayout()
        sweep_layout.addWidget(self.enable_sweep_plot)
        sweep_label = QLabel("Sweep figure")
        sweep_label.setMinimumWidth(140)
        sweep_layout.addWidget(sweep_label)
        sweep_layout.addWidget(self.sweep_output)
        sweep_layout.addWidget(QLabel("Points"))
        sweep_layout.addWidget(self.sweep_points)
        plot_layout.addLayout(sweep_layout)
        plot_layout.addStretch()
        plot_tab.setLayout(plot_layout)
        self.tabs.addTab(plot_tab, "Plot/Fit")

        # CLI tab
        cli_tab = QWidget()
        cli_layout = QVBoxLayout()
        cli_layout.addWidget(QLabel("Extra CLI args"))
        cli_layout.addWidget(self.extra_args)
        cli_layout.addWidget(QLabel("All CLI options (from main.py --help)"))
        cli_layout.addWidget(self.cli_help_box, 1)
        cli_layout.addWidget(QLabel("Simulation log"))
        cli_layout.addWidget(self.output)
        cli_tab.setLayout(cli_layout)
        self.tabs.addTab(cli_tab, "CLI")

        # Decoder tab
        decoder_tab = QWidget()
        decoder_layout = QVBoxLayout()
        decoder_layout.addLayout(self._make_rows([
            ("n", self.dec_n, "n_p", self.dec_np),
            ("Measured splitting (MHz)", self.dec_split),
            ("Peak offset (MHz)", self.dec_offset),
            ("Df correction", self.dec_df_correction),
        ]))
        self.decode_button = QPushButton("Run Decoder")
        self.decode_button.clicked.connect(self._run_decoder)
        decoder_layout.addWidget(self.decode_button)
        decoder_layout.addWidget(QLabel("Decoder output"))
        decoder_layout.addWidget(self.decoder_output)
        decoder_tab.setLayout(decoder_layout)
        self.tabs.addTab(decoder_tab, "Decoder")

        # Debug tab
        debug_tab = QWidget()
        debug_layout = QVBoxLayout()
        debug_layout.addWidget(self.debug_timing)
        debug_layout.addWidget(self.debug_verbose)
        debug_layout.addStretch()
        debug_tab.setLayout(debug_layout)
        self.tabs.addTab(debug_tab, "Debug")

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_custom_config)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: green;")
        self.decoder_status_label = QLabel("Decoder: Idle")
        self.decoder_status_label.setStyleSheet("color: green;")

        self.preview_label = QLabel("Preview not available.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(600, 450)
        self.generated_plots: list[str] = []
        self.preview_index = 0
        self.prev_plot_btn = QPushButton("◀ Back")
        self.next_plot_btn = QPushButton("Next ▶")
        self.save_plot_btn = QPushButton("Save plot as")
        self.prev_plot_btn.clicked.connect(self._show_previous_plot)
        self.next_plot_btn.clicked.connect(self._show_next_plot)
        self.save_plot_btn.clicked.connect(self._save_current_plot)
        self.prev_plot_btn.setEnabled(False)
        self.next_plot_btn.setEnabled(False)

        self.visual_fig = Figure(figsize=(4, 4))
        self.visual_ax = self.visual_fig.add_subplot(111, projection="3d")
        self.visual_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.visual_canvas = FigureCanvas(self.visual_fig)
        self.visual_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self.visual_canvas.installEventFilter(self)
        self.level_fig = Figure(figsize=(3, 2))
        self.level_ax = self.level_fig.add_subplot(111)
        self.level_canvas = FigureCanvas(self.level_fig)
        self.level_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        self.level_canvas.setFixedHeight(180)
        self.rotation_angle = 0.0
        self.rotation_speed = 1.0
        self.visual_timer = QTimer(self)
        self.visual_timer.timeout.connect(self._spin_visualization)
        if self.auto_rotate.isChecked():
            self.visual_timer.start(100)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._reset_visual_view)
        self._update_visualization()
        self.backend_json_path: str | None = None
        self.backend_data = None

        # Right column layout: summary + level diagram (left), visualization (right), preview spanning both
        right_top = QHBoxLayout()
        right_top.setSpacing(10)
        summary_col = QVBoxLayout()
        summary_col.addWidget(self.summary_box)
        summary_col.addWidget(self.level_canvas)
        right_top.addLayout(summary_col)
        visual_col = QVBoxLayout()
        visual_col.addWidget(self.visual_canvas)
        button_row = QHBoxLayout()
        button_row.addWidget(self.auto_rotate)
        button_row.addWidget(self.reset_view_btn)
        visual_col.addLayout(button_row)
        right_top.addLayout(visual_col, 1)

        self.preview_container = QVBoxLayout()
        self.preview_container.addWidget(self.preview_label)

        right_layout = QVBoxLayout()
        right_layout.addLayout(right_top)
        right_layout.addLayout(self.preview_container)

        self.preview_buttons_layout = QHBoxLayout()
        self.preview_buttons_layout.addWidget(self.save_plot_btn)
        self.preview_buttons_layout.addWidget(self.prev_plot_btn)
        self.preview_buttons_layout.addWidget(self.next_plot_btn)

        content_row = QHBoxLayout()
        content_row.addWidget(self.tabs, 1)
        content_row.addLayout(right_layout, 1)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.save_config_btn)
        bottom_row.addWidget(self.run_button)
        bottom_row.addSpacing(10)
        bottom_row.addWidget(self.status_label)
        bottom_row.addSpacing(10)
        bottom_row.addWidget(self.decoder_status_label)
        bottom_row.addStretch()
        bottom_row.addLayout(self.preview_buttons_layout)

        main_layout = QVBoxLayout()
        main_layout.addLayout(content_row, 1)
        main_layout.addLayout(bottom_row)
        self.setLayout(main_layout)
        for widget in [self.cell_length, self.cell_cross,
                       self.probe_waist, self.control_waist]:
            widget.editingFinished.connect(self._update_visualization)
        self.show_levels.stateChanged.connect(self._update_visualization)
        self.show_levels.stateChanged.connect(lambda state: self.level_canvas.setVisible(bool(state)))
        self._set_preview_paths([])
        self.cli_help_box.setText(self._load_cli_help())
        self._draw_level_diagram(None)
        self.level_canvas.setVisible(self.show_levels.isChecked())

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

    def _cleanup_backend_file(self):
        if self.backend_json_path:
            try:
                Path(self.backend_json_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.backend_json_path = None

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def _load_backend_results(self) -> bool:
        path = self.backend_json_path
        if not path:
            return False
        self.backend_json_path = None
        backend_path = Path(path)
        if not backend_path.exists():
            self.output.append(f"<b>Backend JSON not found:</b> {backend_path}")
            QMessageBox.warning(self, "Backend error", f"Backend JSON not found:\n{backend_path}")
            self._set_status("Status: Warning (no data)", "red")
            return False
        try:
            with backend_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            self.output.append(f"<b>Failed to parse backend JSON:</b> {exc}")
            QMessageBox.warning(self, "Backend error", f"Failed to parse backend JSON:\n{exc}")
            try:
                backend_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._set_status("Status: Warning (parse error)", "red")
            return False
        try:
            backend_path.unlink(missing_ok=True)
        except Exception:
            pass
        self.backend_data = data
        self._apply_backend_results(data)
        return True

    def _apply_backend_results(self, data: dict) -> None:
        probe_lambda = data.get("probe_lambda_nm")
        control_lambda = data.get("control_lambda_nm")
        if isinstance(probe_lambda, (int, float)):
            self.probe_label_text = f"λ ≈ {probe_lambda:.2f} nm"
        if isinstance(control_lambda, (int, float)):
            self.control_label_text = f"λ ≈ {control_lambda:.2f} nm"
        plots = data.get("plots", {})
        plot_paths = []
        for key in ("transmission", "sweep"):
            path = plots.get(key)
            if path:
                plot_paths.append(path)
        if plot_paths:
            self._set_preview_paths(plot_paths)
        else:
            self._set_preview_paths([])
        self._update_summary_fields(data)
        if data.get("auto_n_only"):
            self.output.append("<i>Auto-n only mode: simulation skipped.</i>")
        logs = data.get("logs")
        if logs:
            self.output.append("<b>Backend logs:</b>")
            for line in logs:
                self.output.append(str(line))
        self._draw_level_diagram(data)

    def _update_summary_fields(self, data: dict) -> None:
        entries = {
            "Selected n": data.get("selected_n"),
            "Selected n_p": data.get("selected_np"),
        }
        probe_freq = data.get("probe_freq_hz")
        if probe_freq:
            probe_lambda = data.get("probe_lambda_nm")
            entries["Probe transition"] = f"{probe_freq/1e12:.6f} THz (~{probe_lambda:.2f} nm)" if probe_lambda else f"{probe_freq/1e12:.6f} THz"
        else:
            entries["Probe transition"] = None
        control_freq = data.get("control_freq_hz")
        if control_freq:
            control_lambda = data.get("control_lambda_nm")
            entries["Coupling transition"] = f"{control_freq/1e12:.6f} THz (~{control_lambda:.2f} nm)" if control_lambda else f"{control_freq/1e12:.6f} THz"
        else:
            entries["Coupling transition"] = None
        rf_res = data.get("rf_res_hz")
        if rf_res:
            entries["RF transition"] = f"{rf_res/1e9:.3f} GHz"
        else:
            entries["RF transition"] = None
        rf_det = data.get("rf_detuning_mhz")
        if rf_det is not None:
            entries["RF detuning"] = f"{rf_det:+.3f} MHz"
        else:
            entries["RF detuning"] = None
        entries["Last peak ratio"] = data.get("amplitudes", [{}])[-1].get("peak_ratio") if data.get("amplitudes") else None
        entries["Last peak offset"] = data.get("amplitudes", [{}])[-1].get("peak_center_mhz") if data.get("amplitudes") else None

        for key, label in self.summary_fields.items():
            value = entries.get(key)
            label.setText(str(value) if value is not None else "—")

    def _gather_config(self) -> dict:
        entries = {
            "isotope": self.isotope.currentText(),
            "rf_frequency": self.rf_frequency.text().strip(),
            "rf_amplitudes": self.rf_amplitudes.text().strip(),
            "probe_power": self.probe_power.text().strip(),
            "control_power": self.control_power.text().strip(),
            "probe_waist": self.probe_waist.text().strip(),
            "control_waist": self.control_waist.text().strip(),
            "cell_length": self.cell_length.text().strip(),
            "cell_cross": self.cell_cross.text().strip(),
            "temperature": self.temperature.text().strip(),
            "detuning_span": self.detuning_span.text().strip(),
            "detuning_points": self.detuning_points.text().strip(),
            "output_file": self.output_file.text().strip(),
            "control_detuning": self.control_detuning.text().strip(),
            "transit_rate": self.transit_rate.text().strip(),
            "probe_linewidth": self.probe_linewidth.text().strip(),
            "control_linewidth": self.control_linewidth.text().strip(),
            "baseline_amp": self.baseline_amp.text().strip(),
            "sweep_output": self.sweep_output.text().strip(),
            "sweep_points": self.sweep_points.text().strip(),
            "auto_n": self.auto_n.isChecked(),
            "auto_n_only": self.auto_n_only.isChecked(),
            "normalize_baseline": self.normalize.isChecked(),
            "timing": self.timing.isChecked(),
            "no_show": self.no_show.isChecked(),
            "fit_peaks": self.fit_peaks.isChecked(),
            "enable_sweep_plot": self.enable_sweep_plot.isChecked(),
            "sweep_plot": self.enable_sweep_plot.isChecked(),
            "auto_rotate": self.auto_rotate.isChecked(),
            "override_pressure": self.override_pressure.isChecked(),
            "fit_profile": self.fit_profile.currentData(),
            "extra_args": self.extra_args.text().strip(),
            "debug_timing": self.debug_timing.isChecked(),
        }
        ordered = OrderedDict()
        for key in self.save_order:
            if key in entries:
                ordered[key] = entries.pop(key)
        for k, v in entries.items():
            ordered[k] = v
        return ordered
    def _load_cli_help(self) -> str:
        try:
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "main.py"), "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            help_text = result.stdout
            text = help_text if help_text else "Failed to load --help output."
            marker = "options:"
            if marker in text:
                text = text.split(marker, 1)[1]
                text = text.lstrip("\n")
            return text.strip("\n")
        except Exception as exc:
            return f"Failed to load CLI options: {exc}"

    def _load_config(self, base_only: bool = False) -> dict:
        custom_path = REPO_ROOT / "customGUIconfig.json"
        cfg_path = REPO_ROOT / "defaultGUIconfig.json"
        path = cfg_path if base_only or not custom_path.exists() else custom_path
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

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
            self._set_status("Status: Finished", "green")
            loaded = self._load_backend_results()
            if not loaded:
                self._update_preview()
            else:
                self._refresh_preview_label()
            self._update_visualization()
        else:
            self.output.append(f"<b style='color:#d00;'>Simulation failed (code {code}).</b>")
            self._cleanup_backend_file()
            self._set_status("Status: Failed (see CLI tab for more info)", "red")

    # ------------------------------------------------------------------ actions
    def run_simulation(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Running", "Simulation already in progress.")
            return

        if not self._validate_inputs():
            return

        self._cleanup_backend_file()
        self.backend_data = None
        self._set_status("Status: Running…", "orange")
        self.preview_label.setText("Running simulation…")
        self.preview_label.setStyleSheet("color: orange;")
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
            "--control-detuning",
            self.control_detuning.text().strip(),
            "--temperature",
            self.temperature.text().strip(),
            "--transit-rate",
            self.transit_rate.text().strip(),
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
        if self.debug_timing.isChecked():
            cmd.append("--timing")
        if self.debug_verbose.isChecked():
            cmd.append("--verbose")
        if self.fit_peaks.isChecked():
            cmd.append("--fit-peaks")
            cmd.extend(["--fit-profile", self.fit_profile.currentData()])
        if self.auto_n_only.isChecked():
            cmd.append("--auto-n-only")
        if self.enable_sweep_plot.isChecked():
            cmd.append("--sweep-plot")
            if self.sweep_output.text().strip():
                cmd.extend(["--sweep-output", self.sweep_output.text().strip()])
            if self.sweep_points.text().strip():
                cmd.extend(["--sweep-points", self.sweep_points.text().strip()])
        if self.probe_linewidth.text().strip():
            cmd.extend(["--probe-linewidth", self.probe_linewidth.text().strip()])
        if self.control_linewidth.text().strip():
            cmd.extend(["--control-linewidth", self.control_linewidth.text().strip()])
        extra = self.extra_args.text().strip()
        if extra:
            cmd.extend(extra.split())

        amps = [a for a in self.rf_amplitudes.text().split() if a]
        if amps:
            cmd.append("--rf-amplitudes")
            cmd.extend(amps)

        backend_tmp = tempfile.NamedTemporaryFile(prefix="eit_backend_", suffix=".json", delete=False)
        backend_tmp.close()
        self.backend_json_path = backend_tmp.name
        cmd.extend(["--backend-json", self.backend_json_path])

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
            self.visual_timer.start()

    def _reset_visual_view(self) -> None:
        self.visual_ax.view_init(elev=0, azim=70,roll=70)
        self.visual_ax.set_proj_type('persp')
        self.visual_ax.dist = 10
        self.visual_canvas.draw_idle()
        #if self.auto_rotate.isChecked():
            #self.visual_timer.start(100)
            #self.auto_rotate.toggle()
        #else:
            #self.visual_timer.start(100)
            #self.auto_rotate.toggle()

    def _toggle_n_fields(self, checked: bool) -> None:
        self.n_value.setEnabled(not checked)
        self.np_value.setEnabled(not checked)

    def _toggle_pressure_field(self, checked: bool) -> None:
        self.pressure_torr.setEnabled(checked)

    def _toggle_sweep_field(self, checked: bool) -> None:
        self.sweep_output.setEnabled(checked)
        self.sweep_points.setEnabled(checked)

    def _validate_inputs(self) -> bool:
        # Required numeric: rf_frequency
        try:
            float(self.rf_frequency.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Input error", "RF frequency must be a number.")
            self._set_status("Status: Invalid input", "red")
            return False
        amps = [a for a in self.rf_amplitudes.text().split() if a.strip()]
        if not amps:
            QMessageBox.warning(self, "Input error", "Please provide at least one RF amplitude.")
            self._set_status("Status: Invalid input", "red")
            return False
        try:
            [float(a) for a in amps]
        except ValueError:
            QMessageBox.warning(self, "Input error", "RF amplitudes must be numeric.")
            self._set_status("Status: Invalid input", "red")
            return False
        return True

    def _set_preview_paths(self, paths: list[str]) -> None:
        self.generated_plots = paths
        self.preview_index = 0
        self._refresh_preview_label()

    def _refresh_preview_label(self) -> None:
        total = len(self.generated_plots)
        self.prev_plot_btn.setEnabled(total > 1 and self.preview_index > 0)
        self.next_plot_btn.setEnabled(total > 1 and self.preview_index < total - 1)
        if total == 0:
            self.preview_label.setText("No plot images available.")
            return
        path = Path(self.generated_plots[self.preview_index])
        if not path.exists():
            self.preview_label.setText(f"No file at {path}.")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.preview_label.setText("Failed to load image.")
        else:
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def _update_preview(self) -> None:
        paths: list[str] = []
        main_path = self.output_file.text().strip()
        if main_path:
            full = (REPO_ROOT / main_path).resolve()
            if full.exists():
                paths.append(str(full))
        if self.enable_sweep_plot.isChecked():
            sweep_path = self.sweep_output.text().strip()
            if sweep_path:
                full = (REPO_ROOT / sweep_path).resolve()
                if full.exists():
                    paths.append(str(full))
        self._set_preview_paths(paths)

    def _show_previous_plot(self) -> None:
        if self.preview_index > 0:
            self.preview_index -= 1
            self._refresh_preview_label()

    def _show_next_plot(self) -> None:
        if self.preview_index + 1 < len(self.generated_plots):
            self.preview_index += 1
            self._refresh_preview_label()

    def _rebuild_preview_container(self) -> None:
        if not hasattr(self, "preview_container"):
            return
        while self.preview_container.count():
            item = self.preview_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.preview_container.addWidget(self.preview_label)

    def _draw_level_diagram(self, data: dict | None) -> None:
        ax = self.level_ax
        ax.clear()
        ax.set_axis_off()
        ax.set_ylim(-0.5, 3.0)
        ax.set_xlim(-0.5, 2.5)

        # Energies (arbitrary units; order: 5S < 5P < nP < nD)
        y_g, y_e, y_rp, y_r = 0.0, 1.0, 2.0, 2.3
        ax.hlines([y_g, y_e, y_r, y_rp], 0, 1, colors=["gray", "gray", "#5a1f1f", "#2c3f7a"], linewidth=2)
        n_val = data.get("selected_n") if data else None
        np_val = data.get("selected_np") if data else None
        n_label = f"{n_val}D5/2" if n_val else "nD5/2"
        np_label = f"{np_val}P3/2" if np_val else "(n+1)P3/2"
        ax.text(-0.05, y_g, "5S1/2", va="center", ha="right")
        ax.text(-0.05, y_e, "5P3/2", va="center", ha="right")
        ax.text(-0.05, y_r, n_label, va="center", ha="right")
        ax.text(-0.05, y_rp, np_label, va="center", ha="right")

        if data:
            probe_lbl = data.get("probe_lambda_nm")
            control_lbl = data.get("control_lambda_nm")
            rf_det = data.get("rf_detuning_mhz")
            ax.annotate("", xy=(0.5, y_e), xytext=(0.5, y_g),
                        arrowprops=dict(arrowstyle="->", color="red"))
            ax.text(0.55, (y_e + y_g) / 2, f"λp ~ {probe_lbl:.1f} nm" if probe_lbl else "λp",
                    va="center", ha="left")
            ax.annotate("", xy=(0.3, y_r), xytext=(0.3, y_e),
                        arrowprops=dict(arrowstyle="->", color="blue"))
            ax.text(0.35, (y_r + y_e) / 2, f"λc ~ {control_lbl:.1f} nm" if control_lbl else "λc",
                    va="center", ha="left")
            ax.annotate("", xy=(0.8, y_rp), xytext=(0.8, y_r),
                        arrowprops=dict(arrowstyle="->", color="orange"))
            rf_res_hz = data.get("rf_res_hz") if data else None
            label_rf = None
            if rf_res_hz:
                label_rf = f"RF ≈ {rf_res_hz/1e9:.3f} GHz"
            elif rf_det is not None:
                label_rf = f"ΔRF = {rf_det:+.2f} MHz"
            if label_rf:
                ax.text(0.9, (y_r + y_rp) / 2, label_rf,
                        va="center", ha="left")
        self.level_canvas.draw_idle()

    def _save_current_plot(self) -> None:
        if not self.generated_plots:
            QMessageBox.information(self, "Save plot", "No plot available to save.")
            return
        src = Path(self.generated_plots[self.preview_index])
        if not src.exists():
            QMessageBox.warning(self, "Save plot", f"Source plot not found:\n{src}")
            return
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot as",
            str(src.name),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
        )
        if not dest_path:
            return
        try:
            shutil.copy(src, dest_path)
            QMessageBox.information(self, "Save plot", f"Saved plot to:\n{dest_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save plot", f"Failed to save plot:\n{exc}")

    def _run_decoder(self) -> None:
        self.decoder_output.setPlainText("Decoder running…")
        self.decoder_status_label.setText("Decoder: Running…")
        self.decoder_status_label.setStyleSheet("color: orange;")
        QApplication.processEvents()
        self.decode_button.setEnabled(False)
        python_exec = sys.executable
        cmd = [
            python_exec,
            str(REPO_ROOT / "decoder.py"),
            "--isotope", self.isotope.currentText(),
            "--n", self.dec_n.text().strip(),
            "--measured-splitting", self.dec_split.text().strip(),
            "--peak-offset", self.dec_offset.text().strip(),
            "--df-correction", self.dec_df_correction.text().strip(),
        ]
        if self.dec_np.text().strip():
            cmd.extend(["--np", self.dec_np.text().strip()])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Expect JSON payload on stdout; parse and present succinctly.
            data = json.loads(result.stdout)
            lines = []
            e_field = data.get("e_field_v_cm_est")
            det = data.get("inferred_rf_detuning_mhz")
            if e_field is not None:
                lines.append(f"E-field ≈ {e_field:.3f} V/cm")
            if det is not None:
                lines.append(f"RF detuning ≈ {det:.3f} MHz")
            if not lines:
                lines.append("No decoder output available.")
            self.decoder_output.setPlainText("\n".join(lines))
        except subprocess.CalledProcessError as exc:
            self.decoder_output.setPlainText(exc.stdout + "\n" + exc.stderr)
            QMessageBox.warning(self, "Decoder failed", f"Decoder error:\n{exc.stderr}")
            self.decoder_status_label.setText("Decoder: Failed")
            self.decoder_status_label.setStyleSheet("color: red;")
        except Exception as exc:
            self.decoder_output.setPlainText(f"Failed to parse decoder output: {exc}")
            QMessageBox.warning(self, "Decoder failed", f"Failed to parse decoder output:\n{exc}")
            self.decoder_status_label.setText("Decoder: Failed")
            self.decoder_status_label.setStyleSheet("color: red;")
        finally:
            self.decode_button.setEnabled(True)
            self.decoder_status_label.setText("Decoder: Idle")
            self.decoder_status_label.setStyleSheet("color: green;")

    def _save_custom_config(self) -> None:
        cfg = self.base_defaults.copy()
        cfg.update(self._gather_config())
        path = REPO_ROOT / "customGUIconfig.json"
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
            QMessageBox.information(self, "Config saved", f"Saved current settings to:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save config:\n{exc}")

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
        elev = 100 * np.sin(np.radians(self.rotation_angle / 2)) % 360
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
            self.visual_timer.start()
        else:
            self.visual_timer.stop()

def main() -> None:
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.resize(1900, 1000)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

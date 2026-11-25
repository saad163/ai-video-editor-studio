"""
ui/views/settings_view.py
=========================
Settings & AI Configuration tab for AI Video Editor Studio.
Allows users to configure:
  - Google Gemini API Key
  - Whisper Model Size (tiny, small, medium, large-v3)
  - Compute Device Mode (auto, cpu, cuda)
  - Storage & Cache Directories
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFrame, QSpinBox, QFileDialog, QMessageBox, QScrollArea, QListView
)
from PySide6.QtCore import Qt

from utils.config import get_config, reload_config, PROJECT_ROOT
from core.ffmpeg_wrapper import _nvenc_available


from PySide6.QtGui import QDesktopServices, QColor, QPalette


def _make_combo_view() -> QListView:
    """Create a styled QListView popup for combo boxes with high-contrast text."""
    v = QListView()
    pal = v.palette()
    pal.setColor(QPalette.Text, QColor("#F4F4F5"))
    pal.setColor(QPalette.WindowText, QColor("#F4F4F5"))
    pal.setColor(QPalette.Base, QColor("#1A1A1F"))
    pal.setColor(QPalette.Window, QColor("#1A1A1F"))
    pal.setColor(QPalette.Highlight, QColor("#2A201A"))
    pal.setColor(QPalette.HighlightedText, QColor("#FB923C"))
    v.setPalette(pal)
    v.setStyleSheet(
        "QListView { background-color: #1A1A1F; color: #F4F4F5; border: 1px solid #2C2C35; border-radius: 8px; selection-background-color: #2A201A; selection-color: #FB923C; } "
        "QListView::item { color: #F4F4F5; background-color: #1A1A1F; min-height: 28px; padding: 6px 12px; } "
        "QListView::item:hover, QListView::item:selected { background-color: #2A201A; color: #FB923C; }"
    )
    return v


class SettingsView(QWidget):
    """Settings & API configuration panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(18)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(18)

        # ── Header ─────────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(3)

        title_lbl = QLabel("Settings & AI Configuration")
        title_lbl.setObjectName("viewTitle")
        subtitle_lbl = QLabel("Configure local hardware parameters, model sizes, and cloud API keys")
        subtitle_lbl.setObjectName("viewSubtitle")

        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        cfg = get_config()

        # ── 1. AI API Configuration Card ───────────────────────────────────────
        api_card = QFrame()
        api_card.setObjectName("cardFrame")
        api_layout = QVBoxLayout(api_card)
        api_layout.setContentsMargins(20, 20, 20, 20)
        api_layout.setSpacing(12)

        sec1_title = QLabel("Google Gemini API")
        sec1_title.setObjectName("sectionTitle")
        api_layout.addWidget(sec1_title)

        key_lbl = QLabel("Gemini API Key (used for semantic scene context matching in Template 2)")
        key_lbl.setObjectName("inputLabel")
        api_layout.addWidget(key_lbl)

        key_row = QHBoxLayout()
        self.edit_gemini_key = QLineEdit()
        self.edit_gemini_key.setEchoMode(QLineEdit.Password)
        self.edit_gemini_key.setText(cfg.gemini_api_key)
        self.edit_gemini_key.setPlaceholderText("AIzaSy...")

        btn_toggle_key = QPushButton("Show")
        btn_toggle_key.setObjectName("secondaryButton")
        btn_toggle_key.clicked.connect(self._toggle_key_visibility)

        key_row.addWidget(self.edit_gemini_key)
        key_row.addWidget(btn_toggle_key)
        api_layout.addLayout(key_row)

        layout.addWidget(api_card)

        # ── 2. Local AI & Hardware Card ────────────────────────────────────────
        hw_card = QFrame()
        hw_card.setObjectName("cardFrame")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setContentsMargins(20, 20, 20, 20)
        hw_layout.setSpacing(14)

        sec2_title = QLabel("Hardware & Model Configuration")
        sec2_title.setObjectName("sectionTitle")
        hw_layout.addWidget(sec2_title)

        hw_grid = QHBoxLayout()
        hw_grid.setSpacing(16)

        # Whisper Model Size
        model_box = QVBoxLayout()
        model_box.setSpacing(4)
        model_lbl = QLabel("Whisper Model")
        model_lbl.setObjectName("inputLabel")
        self.combo_model = QComboBox()
        self.combo_model.setView(_make_combo_view())
        self.combo_model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.combo_model.setCurrentText(cfg.whisper_model_size)
        model_box.addWidget(model_lbl)
        model_box.addWidget(self.combo_model)
        hw_grid.addLayout(model_box)

        # Compute Device
        dev_box = QVBoxLayout()
        dev_box.setSpacing(4)
        dev_lbl = QLabel("Compute Device")
        dev_lbl.setObjectName("inputLabel")
        self.combo_device = QComboBox()
        self.combo_device.setView(_make_combo_view())
        self.combo_device.addItems(["auto (Auto-Detect CUDA)", "cpu (Force CPU int8)", "cuda (Force NVIDIA GPU)"])
        if cfg.compute_device == "cpu":
            self.combo_device.setCurrentIndex(1)
        elif cfg.compute_device == "cuda":
            self.combo_device.setCurrentIndex(2)
        else:
            self.combo_device.setCurrentIndex(0)
        dev_box.addWidget(dev_lbl)
        dev_box.addWidget(self.combo_device)
        hw_grid.addLayout(dev_box)

        hw_layout.addLayout(hw_grid)

        # Hardware diagnostics text
        has_cuda = _nvenc_available()
        diag_lbl = QLabel(
            f"Hardware Status: {'NVIDIA NVENC active (RTX hardware acceleration)' if has_cuda else 'Intel Iris Xe / CPU Mode (fallback enabled)'}"
        )
        diag_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {'#0d9488' if has_cuda else '#d97706'};")
        hw_layout.addWidget(diag_lbl)

        layout.addWidget(hw_card)

        # ── 3. Directories Card ────────────────────────────────────────────────
        dir_card = QFrame()
        dir_card.setObjectName("cardFrame")
        dir_layout = QVBoxLayout(dir_card)
        dir_layout.setContentsMargins(20, 20, 20, 20)
        dir_layout.setSpacing(14)

        sec3_title = QLabel("Storage Directories")
        sec3_title.setObjectName("sectionTitle")
        dir_layout.addWidget(sec3_title)

        # Output folder
        out_lbl = QLabel("Default Output Directory")
        out_lbl.setObjectName("inputLabel")
        out_row = QHBoxLayout()
        self.edit_output_dir = QLineEdit(str(cfg.output_dir))
        btn_browse_out = QPushButton("Browse")
        btn_browse_out.setObjectName("secondaryButton")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.edit_output_dir)
        out_row.addWidget(btn_browse_out)
        dir_layout.addWidget(out_lbl)
        dir_layout.addLayout(out_row)

        # Temp / Scratch folder
        temp_lbl = QLabel("Temporary Cache Directory")
        temp_lbl.setObjectName("inputLabel")
        temp_row = QHBoxLayout()
        self.edit_temp_dir = QLineEdit(str(cfg.temp_dir))
        btn_browse_temp = QPushButton("Browse")
        btn_browse_temp.setObjectName("secondaryButton")
        btn_browse_temp.clicked.connect(self._browse_temp_dir)
        temp_row.addWidget(self.edit_temp_dir)
        temp_row.addWidget(btn_browse_temp)
        dir_layout.addWidget(temp_lbl)
        dir_layout.addLayout(temp_row)

        layout.addWidget(dir_card)

        # ── Save Button ────────────────────────────────────────────────────────
        save_btn = QPushButton("💾  Save Settings to .env")
        save_btn.setObjectName("primaryButton")
        save_btn.setFixedHeight(42)
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _toggle_key_visibility(self) -> None:
        sender = self.sender()
        if self.edit_gemini_key.echoMode() == QLineEdit.Password:
            self.edit_gemini_key.setEchoMode(QLineEdit.Normal)
            if sender:
                sender.setText("Hide")
        else:
            self.edit_gemini_key.setEchoMode(QLineEdit.Password)
            if sender:
                sender.setText("Show")

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.edit_output_dir.text())
        if d:
            self.edit_output_dir.setText(d)

    def _browse_temp_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Temp Directory", self.edit_temp_dir.text())
        if d:
            self.edit_temp_dir.setText(d)

    def _save_settings(self) -> None:
        env_path = PROJECT_ROOT / ".env"
        gemini_key = self.edit_gemini_key.text().strip()
        model_size = self.combo_model.currentText().strip()

        dev_text = self.combo_device.currentText()
        if "cpu" in dev_text:
            device = "cpu"
        elif "cuda" in dev_text:
            device = "cuda"
        else:
            device = "auto"

        out_d = self.edit_output_dir.text().strip()
        temp_d = self.edit_temp_dir.text().strip()

        lines = [
            f"GEMINI_API_KEY={gemini_key}",
            f"WHISPER_MODEL_SIZE={model_size}",
            f"COMPUTE_DEVICE={device}",
            f"OUTPUT_DIR={out_d}",
            f"TEMP_DIR={temp_d}",
            "LOG_LEVEL=INFO",
            "MAX_TEMP_SIZE_GB=5\n",
        ]
        try:
            env_path.write_text("\n".join(lines), encoding="utf-8")
            reload_config()
            QMessageBox.information(self, "Settings Saved", "Configuration saved to .env successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Settings", f"Could not write to .env: {e}")


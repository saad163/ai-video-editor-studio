"""
ui/views/bulk_view.py
=====================
Bulk Generation Tab for AI Video Editor Studio.
Pro Studio Dark-Slate & Midnight Indigo Design System:
  - Deep obsidian workspace background (#0B0F19)
  - Rich slate cards (#131B2E) with crisp borders (#1E293B)
  - Unified batch typography controls & live cinema preview
  - Dark studio table with pill-shaped status badges
  - Non-blocking sequential execution via BulkQueueWorker
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QProgressBar,
    QFrame, QScrollArea, QCheckBox, QMessageBox, QAbstractItemView,
    QComboBox, QSlider, QColorDialog, QListView
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QPalette

from utils.config import get_config
from core.ffmpeg_wrapper import probe_media
from ui.workers.bulk_worker import BulkQueueWorker, BulkJob


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


# Button styles for small controls
BTN_INACTIVE_STYLE = """
    QPushButton {
        background-color: #222228;
        color: #A1A1AA;
        border: 1px solid #32323D;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
        padding: 0px;
        text-align: center;
    }
    QPushButton:hover {
        background-color: #2A2A34;
        border-color: #F97316;
        color: #F4F4F5;
    }
"""

BTN_ACTIVE_STYLE = """
    QPushButton {
        background-color: #2A201A;
        color: #FB923C;
        border: 1.5px solid #EA580C;
        border-radius: 6px;
        font-weight: 800;
        font-size: 11px;
        padding: 0px;
        text-align: center;
    }
"""


class BulkView(QWidget):
    """Queue-based multi-video generator view with unified batch typography."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[BulkQueueWorker] = None
        self._is_paused = False

        # Global typography settings
        self._font_color = "#FFFF00"          # Yellow default
        self._highlight_color = "#FFFF00"
        self._stroke_color = "#000000"
        self._stroke_thickness = 3.5
        self._bg_style = "semi_transparent_black"
        self._has_bg_box = True
        self._bg_color = "#000000"
        self._bg_opacity = 0.5
        self._font_size = 58
        self._font_family = "Arial"
        self._resolution = "2k"
        self._preset_name = "Viral Shorts (Yellow + Box)"

        self._setup_ui()

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Scroll Area for responsive scaling ─────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background-color: #121214;")

        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(container)
        content_layout.setContentsMargins(36, 24, 36, 28)
        content_layout.setSpacing(16)

        # ── 1. Header ──────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_lbl = QLabel("Bulk Video Generation")
        title_lbl.setObjectName("viewTitle")
        sub_lbl = QLabel("Queue multiple voiceover tracks, apply unified typography presets, and batch-render 2K masters")
        sub_lbl.setObjectName("viewSubtitle")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        header_row.addLayout(title_box)

        header_row.addStretch()

        self.queue_badge = QLabel("0 Videos Queued")
        self.queue_badge.setObjectName("creditBadge")
        header_row.addWidget(self.queue_badge)

        content_layout.addLayout(header_row)

        # ── 2. Top Folder Selectors Card ───────────────────────────────────────
        cfg = get_config()
        card_folders = QFrame()
        card_folders.setObjectName("cardFrame")
        folders_layout = QVBoxLayout(card_folders)
        folders_layout.setContentsMargins(18, 16, 18, 16)
        folders_layout.setSpacing(14)

        # Folder 1: Audio Source Directory
        audio_in_box = QVBoxLayout()
        audio_in_box.setSpacing(4)
        audio_in_lbl = QLabel("Bulk Audio Directory (Voiceovers Folder)")
        audio_in_lbl.setObjectName("fieldLabel")
        audio_in_row = QHBoxLayout()
        audio_in_row.setSpacing(8)

        self.edit_audio_dir = QLineEdit()
        self.edit_audio_dir.setPlaceholderText("Select directory with audio files (.mp3, .wav, .m4a)...")

        btn_browse_audio_dir = QPushButton("Browse")
        btn_browse_audio_dir.setObjectName("secondaryButton")
        btn_browse_audio_dir.setFixedWidth(84)
        btn_browse_audio_dir.setCursor(Qt.PointingHandCursor)
        btn_browse_audio_dir.clicked.connect(self._browse_audio_directory)

        btn_scan_audio = QPushButton("Scan Directory")
        btn_scan_audio.setObjectName("tealButton")
        btn_scan_audio.setFixedWidth(130)
        btn_scan_audio.setCursor(Qt.PointingHandCursor)
        btn_scan_audio.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366F1, stop:1 #8B5CF6);
                color: #FFFFFF;
                border: 1px solid #7C3AED;
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
                padding: 6px 14px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4F46E5, stop:1 #7C3AED); }
        """)
        btn_scan_audio.clicked.connect(self._scan_current_directory)

        btn_open_audio_dir = QPushButton("Open")
        btn_open_audio_dir.setObjectName("toolButton")
        btn_open_audio_dir.setFixedWidth(68)
        btn_open_audio_dir.setCursor(Qt.PointingHandCursor)
        btn_open_audio_dir.clicked.connect(lambda: self._open_in_explorer(self.edit_audio_dir.text()))

        audio_in_row.addWidget(self.edit_audio_dir)
        audio_in_row.addWidget(btn_browse_audio_dir)
        audio_in_row.addWidget(btn_scan_audio)
        audio_in_row.addWidget(btn_open_audio_dir)

        audio_hint = QLabel("Select a folder containing voiceover audios. The scanner auto-pairs with matching character folders.")
        audio_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        audio_in_box.addWidget(audio_in_lbl)
        audio_in_box.addLayout(audio_in_row)
        audio_in_box.addWidget(audio_hint)
        folders_layout.addLayout(audio_in_box)

        # Folder 2: Default Footage Library
        footage_box = QVBoxLayout()
        footage_box.setSpacing(4)
        footage_lbl = QLabel("Footage Library / Visual Assets Folder")
        footage_lbl.setObjectName("fieldLabel")
        footage_row = QHBoxLayout()
        footage_row.setSpacing(8)

        self.edit_footage_dir = QLineEdit()
        self.edit_footage_dir.setPlaceholderText("Select folder containing visual assets or character subfolders...")
        sample_footage = cfg.project_root / "assets" / "sample_footage"
        if sample_footage.exists():
            self.edit_footage_dir.setText(str(sample_footage))

        btn_browse_footage = QPushButton("Browse")
        btn_browse_footage.setObjectName("secondaryButton")
        btn_browse_footage.setFixedWidth(84)
        btn_browse_footage.setCursor(Qt.PointingHandCursor)
        btn_browse_footage.clicked.connect(self._browse_footage_directory)

        btn_open_footage = QPushButton("Open")
        btn_open_footage.setObjectName("toolButton")
        btn_open_footage.setFixedWidth(68)
        btn_open_footage.setCursor(Qt.PointingHandCursor)
        btn_open_footage.clicked.connect(lambda: self._open_in_explorer(self.edit_footage_dir.text()))

        footage_row.addWidget(self.edit_footage_dir)
        footage_row.addWidget(btn_browse_footage)
        footage_row.addWidget(btn_open_footage)

        footage_hint = QLabel("Default media directory used when an audio file doesn't have an exact matching subfolder.")
        footage_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        footage_box.addWidget(footage_lbl)
        footage_box.addLayout(footage_row)
        footage_box.addWidget(footage_hint)
        folders_layout.addLayout(footage_box)

        # Folder 3: Output Destination
        out_box = QVBoxLayout()
        out_box.setSpacing(4)
        out_lbl = QLabel("Output Destination Folder")
        out_lbl.setObjectName("fieldLabel")
        out_row = QHBoxLayout()
        out_row.setSpacing(8)

        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setPlaceholderText("Select destination directory for finished MP4s...")
        self.edit_output_dir.setText(str(cfg.output_dir))

        btn_browse_out = QPushButton("Browse")
        btn_browse_out.setObjectName("secondaryButton")
        btn_browse_out.setFixedWidth(84)
        btn_browse_out.setCursor(Qt.PointingHandCursor)
        btn_browse_out.clicked.connect(self._browse_output_directory)

        btn_open_out = QPushButton("Open")
        btn_open_out.setObjectName("toolButton")
        btn_open_out.setFixedWidth(68)
        btn_open_out.setCursor(Qt.PointingHandCursor)
        btn_open_out.clicked.connect(lambda: self._open_in_explorer(self.edit_output_dir.text()))

        out_row.addWidget(self.edit_output_dir)
        out_row.addWidget(btn_browse_out)
        out_row.addWidget(btn_open_out)

        out_hint = QLabel("Completed 2K/1080p videos and generated .srt / .txt chapters are saved here.")
        out_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        out_box.addWidget(out_lbl)
        out_box.addLayout(out_row)
        out_box.addWidget(out_hint)
        folders_layout.addLayout(out_box)

        content_layout.addWidget(card_folders)

        # ── 3. Global Batch Typography & Output Settings Card ─────────────────
        self.style_frame = QFrame()
        self.style_frame.setObjectName("cardFrame")
        style_layout = QVBoxLayout(self.style_frame)
        style_layout.setContentsMargins(18, 16, 18, 16)
        style_layout.setSpacing(14)

        # Header Toggle Button
        self.btn_toggle_style = QPushButton("▲  Global Typography & Output Settings For Batch")
        self.btn_toggle_style.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_style.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                text-align: left;
                font-weight: 700;
                font-size: 13px;
                color: #F8FAFC;
            }
            QPushButton:hover { color: #818CF8; }
        """)
        self.btn_toggle_style.clicked.connect(self._toggle_style_body)
        style_layout.addWidget(self.btn_toggle_style)

        # Style Body Container
        self.style_body = QWidget()
        body_layout = QVBoxLayout(self.style_body)
        body_layout.setContentsMargins(2, 2, 2, 2)
        body_layout.setSpacing(14)

        # Row A: Quick Style Preset & Resolution
        row_a = QHBoxLayout()
        row_a.setSpacing(16)

        preset_box = QVBoxLayout()
        preset_box.setSpacing(4)
        preset_lbl = QLabel("Quick Style Preset (Applied to all batch items):")
        preset_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #818CF8;")
        self.combo_presets = QComboBox()
        self.combo_presets.setView(_make_combo_view())
        self.combo_presets.addItems([
            "Viral Shorts (Yellow Text + Semi-Transparent Black Box)",
            "Classic YouTube (White Text + Black Outline)",
            "High-Contrast Box (Yellow Text + Solid Black Box)",
            "Cinematic Minimalist (Pure White + Soft Box)",
            "Custom Style"
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        preset_box.addWidget(preset_lbl)
        preset_box.addWidget(self.combo_presets)
        row_a.addLayout(preset_box, stretch=3)

        res_box = QVBoxLayout()
        res_box.setSpacing(4)
        res_lbl = QLabel("Output Resolution:")
        res_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #CBD5E1;")
        self.combo_res = QComboBox()
        self.combo_res.setView(_make_combo_view())
        self.combo_res.addItems([
            "1080p (1920x1080) - Default",
            "2K (2560x1440) - Recommended",
            "720p (1280x720) - Compact",
        ])
        self.combo_res.currentIndexChanged.connect(self._on_res_changed)
        res_box.addWidget(res_lbl)
        res_box.addWidget(self.combo_res)
        row_a.addLayout(res_box, stretch=2)

        # Clip Duration Selector in Batch Bar
        dur_box = QVBoxLayout()
        dur_box.setSpacing(4)
        dur_hdr = QHBoxLayout()
        dur_lbl = QLabel("Clip Hold Duration:")
        dur_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #CBD5E1;")
        self.lbl_clip_dur_val = QLabel("5.0 s")
        self.lbl_clip_dur_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #FB923C;")
        dur_hdr.addWidget(dur_lbl)
        dur_hdr.addStretch()
        dur_hdr.addWidget(self.lbl_clip_dur_val)
        dur_box.addLayout(dur_hdr)

        self.slider_clip_dur = QSlider(Qt.Horizontal)
        self.slider_clip_dur.setRange(1, 20)
        self.slider_clip_dur.setValue(5)
        self.slider_clip_dur.valueChanged.connect(lambda v: self.lbl_clip_dur_val.setText(f"{v}.0 s"))
        dur_box.addWidget(self.slider_clip_dur)
        row_a.addLayout(dur_box, stretch=2)

        body_layout.addLayout(row_a)

        # Row B: Background Box Style & Font Name
        row_b = QHBoxLayout()
        row_b.setSpacing(16)

        bg_box = QVBoxLayout()
        bg_box.setSpacing(4)
        bg_lbl = QLabel("Background Box Style:")
        bg_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #CBD5E1;")
        self.combo_bg = QComboBox()
        self.combo_bg.setView(_make_combo_view())
        self.combo_bg.addItems([
            "Semi-Transparent Black Box (YouTube / CapCut Style)",
            "Solid Black Box (Maximum Contrast)",
            "Clean Outline Only (No Box)"
        ])
        self.combo_bg.currentIndexChanged.connect(self._on_bg_changed)
        bg_box.addWidget(bg_lbl)
        bg_box.addWidget(self.combo_bg)
        row_b.addLayout(bg_box, stretch=3)

        font_box = QVBoxLayout()
        font_box.setSpacing(4)
        font_lbl = QLabel("Font Name:")
        font_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #CBD5E1;")
        self.combo_font = QComboBox()
        self.combo_font.setView(_make_combo_view())
        self.combo_font.addItems([
            "Montserrat",
            "Poppins",
            "Arial Black",
            "Roboto",
            "Impact",
            "Segoe UI",
            "Georgia",
            "Arial"
        ])
        self.combo_font.currentTextChanged.connect(self._update_preview)
        font_box.addWidget(font_lbl)
        font_box.addWidget(self.combo_font)
        row_b.addLayout(font_box, stretch=3)

        body_layout.addLayout(row_b)

        # Row C: Font Size Slider & Color Pickers
        row_c = QHBoxLayout()
        row_c.setSpacing(16)

        fs_box = QVBoxLayout()
        fs_box.setSpacing(4)
        fs_header = QHBoxLayout()
        fs_lbl = QLabel("Font Size:")
        fs_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #CBD5E1;")
        self.lbl_fs_val = QLabel(f"{self._font_size} pt")
        self.lbl_fs_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #818CF8;")
        fs_header.addWidget(fs_lbl)
        fs_header.addStretch()
        fs_header.addWidget(self.lbl_fs_val)

        self.slider_fs = QSlider(Qt.Horizontal)
        self.slider_fs.setRange(28, 96)
        self.slider_fs.setValue(self._font_size)
        self.slider_fs.valueChanged.connect(self._on_fs_changed)
        fs_box.addLayout(fs_header)
        fs_box.addWidget(self.slider_fs)
        row_c.addLayout(fs_box, stretch=3)

        # Text Color Swatch
        col_box = QVBoxLayout()
        col_box.setSpacing(4)
        col_lbl = QLabel("Subtitle Color")
        col_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #94A3B8;")
        self.btn_col_swatch = QPushButton("")
        self.btn_col_swatch.setFixedSize(38, 28)
        self.btn_col_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_col_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 6px;")
        self.btn_col_swatch.clicked.connect(self._pick_color)
        col_box.addWidget(col_lbl)
        col_box.addWidget(self.btn_col_swatch)
        row_c.addLayout(col_box)

        # Highlight Color Swatch
        hl_box = QVBoxLayout()
        hl_box.setSpacing(4)
        hl_lbl = QLabel("Highlight Color")
        hl_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #94A3B8;")
        self.btn_hl_swatch = QPushButton("")
        self.btn_hl_swatch.setFixedSize(38, 28)
        self.btn_hl_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 6px;")
        self.btn_hl_swatch.clicked.connect(self._pick_hl_color)
        hl_box.addWidget(hl_lbl)
        hl_box.addWidget(self.btn_hl_swatch)
        row_c.addLayout(hl_box)

        body_layout.addLayout(row_c)

        # Row D: Checkboxes
        check_row = QHBoxLayout()
        self.chk_chapters = QCheckBox("Auto-Generate YouTube Chapters (.txt) for each video")
        self.chk_chapters.setChecked(True)
        self.chk_ken_burns = QCheckBox("Include Images (Apply Dynamic Ken Burns Zoom & Pan)")
        self.chk_ken_burns.setChecked(True)
        check_row.addWidget(self.chk_chapters)
        check_row.addSpacing(16)
        check_row.addWidget(self.chk_ken_burns)
        check_row.addStretch()
        body_layout.addLayout(check_row)

        # Row E: Preview Banner
        self.preview_banner = QFrame()
        self.preview_banner.setFixedHeight(76)
        self.preview_banner.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #09090D, stop:1 #16161E);
                border: 1px solid #2C2C35;
                border-radius: 10px;
            }
        """)
        prev_layout = QHBoxLayout(self.preview_banner)
        prev_layout.setContentsMargins(16, 6, 16, 6)
        self.lbl_preview = QLabel("Batch Subtitle Style Preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        prev_layout.addWidget(self.lbl_preview)
        body_layout.addWidget(self.preview_banner)

        style_layout.addWidget(self.style_body)
        content_layout.addWidget(self.style_frame)
        self._is_style_open = True
        self._update_preview()

        # ── 4. Action Buttons Row ──────────────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.setSpacing(12)

        self.btn_start = QPushButton("Start Batch AutoCut")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F97316, stop:1 #FB923C);
                color: #FFFFFF;
                border: 1px solid #EA580C;
                border-radius: 8px;
                font-weight: 700;
                font-size: 14px;
                padding: 10px 24px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EA580C, stop:1 #F97316); }
            QPushButton:disabled { background: #27272A; border-color: #3F3F46; color: #71717A; }
        """)
        self.btn_start.clicked.connect(self._start_queue)
        act_row.addWidget(self.btn_start, stretch=2)

        self.btn_add_manual = QPushButton("+ Add Audio Files Manually")
        self.btn_add_manual.setObjectName("secondaryButton")
        self.btn_add_manual.setFixedHeight(46)
        self.btn_add_manual.setCursor(Qt.PointingHandCursor)
        self.btn_add_manual.clicked.connect(self._add_audio_files_manually)
        act_row.addWidget(self.btn_add_manual, stretch=1)

        self.btn_pause = QPushButton("Pause Queue")
        self.btn_pause.setObjectName("secondaryButton")
        self.btn_pause.setFixedHeight(46)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        act_row.addWidget(self.btn_pause)

        self.btn_clear = QPushButton("Clear Completed")
        self.btn_clear.setObjectName("secondaryButton")
        self.btn_clear.setFixedHeight(46)
        self.btn_clear.clicked.connect(self._clear_completed)
        act_row.addWidget(self.btn_clear)

        content_layout.addLayout(act_row)

        # ── 5. Batch Queue Table ───────────────────────────────────────────────
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "#", "Title", "Audio File", "Assets Folder", "Style Preset", "Progress", "Status", "Actions"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(7, QHeaderView.Fixed)

        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(1, 190)
        self.table.setColumnWidth(2, 160)
        self.table.setColumnWidth(3, 160)
        self.table.setColumnWidth(4, 150)
        self.table.setColumnWidth(5, 120)
        self.table.setColumnWidth(6, 110)
        self.table.setColumnWidth(7, 85)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(260)
        self.table.itemDoubleClicked.connect(self._on_table_double_clicked)
        content_layout.addWidget(self.table)

        # ── 6. Batch Progress Footer Card ──────────────────────────────────────
        footer_card = QFrame()
        footer_card.setObjectName("cardFrame")
        foot_layout = QVBoxLayout(footer_card)
        foot_layout.setContentsMargins(16, 12, 16, 12)
        foot_layout.setSpacing(6)

        f_header = QHBoxLayout()
        self.lbl_batch_status = QLabel("Batch Progress: 0 of 0 videos completed")
        self.lbl_batch_status.setStyleSheet("font-size: 12px; font-weight: 600; color: #D4D4D8;")
        self.lbl_batch_pct = QLabel("0%")
        self.lbl_batch_pct.setStyleSheet("font-size: 12px; font-weight: 700; color: #FB923C;")
        f_header.addWidget(self.lbl_batch_status)
        f_header.addStretch()
        f_header.addWidget(self.lbl_batch_pct)
        foot_layout.addLayout(f_header)

        self.bar_batch = QProgressBar()
        self.bar_batch.setValue(0)
        foot_layout.addWidget(self.bar_batch)

        content_layout.addWidget(footer_card)

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    # ── Style & Preset Interactions ────────────────────────────────────────────

    def _toggle_style_body(self) -> None:
        self._is_style_open = not self._is_style_open
        self.style_body.setVisible(self._is_style_open)
        arrow = "▲" if self._is_style_open else "▼"
        self.btn_toggle_style.setText(f"{arrow}  Global Typography & Output Settings For Batch")

    def _on_preset_changed(self, idx: int) -> None:
        if idx == 0:  # Viral Shorts (Yellow + Semi-Transparent Box)
            self._font_color = "#FFFF00"
            self._highlight_color = "#00FFFF"
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_opacity = 0.5
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(0)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Montserrat Bold")
            self.slider_fs.setValue(58)
            self._preset_name = "Viral Shorts (Yellow + Box)"
        elif idx == 1:  # Classic YouTube (White + Outline)
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FFFF00"
            self._bg_style = "outline"
            self._has_bg_box = False
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(2)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Arial (Classic YouTube Bold)")
            self.slider_fs.setValue(56)
            self._preset_name = "Classic YouTube (White + Outline)"
        elif idx == 2:  # High-Contrast Box (Yellow + Solid Black Box)
            self._font_color = "#FFFF00"
            self._highlight_color = "#FFFFFF"
            self._bg_style = "solid_black"
            self._has_bg_box = True
            self._bg_opacity = 1.0
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(1)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Impact")
            self.slider_fs.setValue(54)
            self._preset_name = "High-Contrast Box (Yellow + Solid Box)"
        elif idx == 3:  # Cinematic Minimalist
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FFFF00"
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_opacity = 0.4
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(0)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Segoe UI Bold")
            self.slider_fs.setValue(52)
            self._preset_name = "Cinematic Minimalist"

        self.btn_col_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 6px;")
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 6px;")
        self._update_preview()

    def _on_bg_changed(self, idx: int) -> None:
        if idx == 0:
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_opacity = 0.5
        elif idx == 1:
            self._bg_style = "solid_black"
            self._has_bg_box = True
            self._bg_opacity = 1.0
        elif idx == 2:
            self._bg_style = "outline"
            self._has_bg_box = False
        self._update_preview()

    def _on_res_changed(self, idx: int) -> None:
        txt = self.combo_res.currentText().lower()
        if "720p" in txt:
            self._resolution = "720p"
        elif "2k" in txt:
            self._resolution = "2k"
        else:
            self._resolution = "1080p"

    def _on_fs_changed(self, val: int) -> None:
        self._font_size = val
        self.lbl_fs_val.setText(f"{val} pt")
        self._update_preview()

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._font_color), self, "Select Subtitle Color")
        if c.isValid():
            self._font_color = c.name().upper()
            self.btn_col_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 6px;")
            self._update_preview()

    def _pick_hl_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._highlight_color), self, "Select Highlight Color")
        if c.isValid():
            self._highlight_color = c.name().upper()
            self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 6px;")
            self._update_preview()

    def _update_preview(self) -> None:
        font_name = self.combo_font.currentText().split(" ")[0]
        base_size = max(int(self._font_size * 0.34), 14)

        if self._bg_style == "solid_black":
            box_style = f"background-color: #000000; color: {self._font_color}; border: 1.5px solid #1E293B; border-radius: 6px; padding: 6px 16px;"
        elif self._bg_style == "semi_transparent_black":
            box_style = f"background-color: rgba(0, 0, 0, 0.70); color: {self._font_color}; border: 1px solid rgba(255, 255, 255, 0.20); border-radius: 6px; padding: 6px 16px;"
        else:  # Outline
            box_style = f"background-color: transparent; color: {self._font_color}; border: 1.5px dashed #334155; border-radius: 6px; padding: 6px 16px;"

        self.lbl_preview.setStyleSheet(f"""
            font-family: '{font_name}', 'Segoe UI', Arial, sans-serif;
            font-size: {base_size}px;
            font-weight: 800;
            letter-spacing: 0.5px;
            {box_style}
        """)

    # ── Folder Helpers ─────────────────────────────────────────────────────────

    def _browse_audio_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Audio Voiceovers Directory")
        if dir_path:
            self.edit_audio_dir.setText(dir_path)
            self._scan_current_directory()

    def _browse_footage_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Footage Library Directory")
        if dir_path:
            self.edit_footage_dir.setText(dir_path)

    def _browse_output_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Destination Output Directory")
        if dir_path:
            self.edit_output_dir.setText(dir_path)

    def _open_in_explorer(self, path_str: str) -> None:
        p = Path(path_str.strip())
        target = p if p.is_dir() else p.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QMessageBox.warning(self, "Path Not Found", f"Directory does not exist:\n{target}")

    def _add_audio_files_manually(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files to Queue", "", "Audio Files (*.mp3 *.wav *.m4a *.flac)"
        )
        for f in files:
            self._insert_row(Path(f))
        self._update_badge()

    def _scan_current_directory(self) -> None:
        dir_str = self.edit_audio_dir.text().strip()
        if not dir_str or not Path(dir_str).is_dir():
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid directory containing audio files.")
            return

        p = Path(dir_str)
        audio_exts = {".mp3", ".wav", ".m4a", ".flac"}
        count = 0
        for audio_file in sorted(p.iterdir()):
            if audio_file.is_file() and audio_file.suffix.lower() in audio_exts:
                self._insert_row(audio_file)
                count += 1

        self._update_badge()
        if count == 0:
            QMessageBox.information(self, "Scan Complete", f"No audio files found in:\n{p}")
        else:
            QMessageBox.information(self, "Scan Complete", f"Successfully detected and queued {count} audio file(s) for bulk processing!")

    def _insert_row(self, audio_path: Path) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        title = audio_path.stem.replace("_", " ").title()

        footage_base = Path(self.edit_footage_dir.text().strip()) if self.edit_footage_dir.text().strip() else get_config().project_root / "assets" / "sample_footage"
        specific_subfolder = footage_base / audio_path.stem
        assets_folder = specific_subfolder if specific_subfolder.is_dir() else footage_base

        # 0. Job #
        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, idx_item)

        # 1. Title
        title_item = QTableWidgetItem(title)
        self.table.setItem(row, 1, title_item)

        # 2. Audio File
        audio_item = QTableWidgetItem(audio_path.name)
        audio_item.setToolTip(str(audio_path))
        self.table.setItem(row, 2, audio_item)

        # 3. Assets Folder
        assets_item = QTableWidgetItem(assets_folder.name)
        assets_item.setToolTip(str(assets_folder))
        self.table.setItem(row, 3, assets_item)

        # 4. Style Preset
        style_item = QTableWidgetItem(self._preset_name)
        self.table.setItem(row, 4, style_item)

        # 5. Progress Bar
        bar = QProgressBar()
        bar.setValue(0)
        bar.setFixedHeight(12)
        bar.setStyleSheet("QProgressBar { border: none; background: #1E293B; border-radius: 3px; } QProgressBar::chunk { background: #06B6D4; border-radius: 3px; }")
        self.table.setCellWidget(row, 5, bar)

        # 6. Status Badge
        status_lbl = QLabel("Ready")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("background: #1E293B; color: #CBD5E1; border: 1px solid #334155; border-radius: 4px; font-weight: 600; font-size: 11px; padding: 2px 6px;")
        self.table.setCellWidget(row, 6, status_lbl)

        # 7. Actions Button
        btn_remove = QPushButton("Remove")
        btn_remove.setStyleSheet("background: #1E293B; color: #CBD5E1; border: 1px solid #334155; border-radius: 4px; padding: 3px 8px; font-size: 11px;")
        btn_remove.clicked.connect(lambda chk, r=row: self._remove_row(r))
        self.table.setCellWidget(row, 7, btn_remove)

    def _remove_row(self, row: int) -> None:
        self.table.removeRow(row)
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setText(str(r + 1))
        self._update_badge()

    def _clear_completed(self) -> None:
        for r in range(self.table.rowCount() - 1, -1, -1):
            w = self.table.cellWidget(r, 6)
            if isinstance(w, QLabel) and "Done" in w.text():
                self.table.removeRow(r)
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setText(str(r + 1))
        self._update_badge()

    def _update_badge(self) -> None:
        count = self.table.rowCount()
        self.queue_badge.setText(f"{count} Videos Queued")

    def _on_table_double_clicked(self, item: QTableWidgetItem) -> None:
        row = item.row()
        out_dir = Path(self.edit_output_dir.text().strip())
        audio_name = self.table.item(row, 2).text()
        safe_stem = Path(audio_name).stem.replace(" ", "_")
        out_file = out_dir / f"{safe_stem}_{self._resolution}.mp4"
        if out_file.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_file)))

    # ── Queue Execution ────────────────────────────────────────────────────────

    def _start_queue(self) -> None:
        if self.table.rowCount() == 0:
            QMessageBox.information(self, "Queue Empty", "Please scan a directory or add audio files to the queue.")
            return

        out_dir = Path(self.edit_output_dir.text().strip())
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_base = Path(self.edit_audio_dir.text().strip())
        footage_base = Path(self.edit_footage_dir.text().strip())

        jobs: list[BulkJob] = []
        for r in range(self.table.rowCount()):
            title = self.table.item(r, 1).text()
            audio_name = self.table.item(r, 2).text()
            audio_path = Path(self.table.item(r, 2).toolTip()) if self.table.item(r, 2).toolTip() else audio_base / audio_name

            assets_path = Path(self.table.item(r, 3).toolTip()) if self.table.item(r, 3).toolTip() else footage_base

            safe_stem = audio_path.stem.replace(" ", "_")
            out_file = out_dir / f"{safe_stem}_{self._resolution}.mp4"

            jobs.append(BulkJob(
                job_id=r,
                title=title,
                audio_path=audio_path,
                images_dir=assets_path,
                output_path=out_file,
                resolution=self._resolution,
                clip_duration=float(self.slider_clip_dur.value()) if hasattr(self, "slider_clip_dur") else 5.0,
                text_color=self._font_color,
                highlight_color=self._highlight_color,
                has_bg_box=self._has_bg_box,
                bg_style=self._bg_style,
                bg_color=self._bg_color,
                bg_opacity=self._bg_opacity,
                stroke_color=self._stroke_color,
                stroke_thickness=self._stroke_thickness,
                font_size=self._font_size,
                font_family=self.combo_font.currentText().split(" ")[0],
                bold=True,
                italic=False,
                box_padding=14.0,
                style_preset_name=self._preset_name,
                generate_chapters=self.chk_chapters.isChecked(),
                include_images=self.chk_ken_burns.isChecked(),
            ))

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause Queue")
        self._is_paused = False

        self._worker = BulkQueueWorker(jobs, parent=self)
        self._worker.job_started.connect(self._on_job_started)
        self._worker.job_progress.connect(self._on_job_progress)
        self._worker.job_completed.connect(self._on_job_completed)
        self._worker.job_failed.connect(self._on_job_failed)
        self._worker.batch_progress.connect(self._on_batch_progress)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.status_message.connect(lambda msg: self.lbl_batch_status.setText(msg))

        self._worker.start()

    def _toggle_pause(self) -> None:
        if self._worker is None:
            return
        if not self._is_paused:
            self._worker.pause()
            self._is_paused = True
            self.btn_pause.setText("Resume Queue")
        else:
            self._worker.resume()
            self._is_paused = False
            self.btn_pause.setText("Pause Queue")

    def _on_job_started(self, row: int, title: str) -> None:
        status_lbl = QLabel("Processing...")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("background: #0C4A6E; color: #38BDF8; border: 1px solid #0284C7; border-radius: 4px; font-weight: 600; font-size: 11px; padding: 2px 6px;")
        self.table.setCellWidget(row, 6, status_lbl)

    def _on_job_progress(self, row: int, pct: int, status_text: str) -> None:
        w = self.table.cellWidget(row, 5)
        if isinstance(w, QProgressBar):
            w.setValue(pct)
        status_lbl = self.table.cellWidget(row, 6)
        if isinstance(status_lbl, QLabel):
            status_lbl.setText(f"{pct}%")
            status_lbl.setToolTip(status_text)

    def _on_job_completed(self, row: int, out_path: str) -> None:
        w = self.table.cellWidget(row, 5)
        if isinstance(w, QProgressBar):
            w.setValue(100)
            w.setStyleSheet("QProgressBar { border: none; background: #1E293B; border-radius: 3px; } QProgressBar::chunk { background: #10B981; border-radius: 3px; }")
        status_lbl = QLabel("Done ✓")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("background: #064E3B; color: #34D399; border: 1px solid #059669; border-radius: 4px; font-weight: 600; font-size: 11px; padding: 2px 6px;")
        status_lbl.setToolTip(f"Completed: {out_path}\nDouble click row to play")
        self.table.setCellWidget(row, 6, status_lbl)

    def _on_job_failed(self, row: int, err: str) -> None:
        status_lbl = QLabel("Failed ✗")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("background: #450A0A; color: #F87171; border: 1px solid #DC2626; border-radius: 4px; font-weight: 600; font-size: 11px; padding: 2px 6px;")
        status_lbl.setToolTip(err)
        self.table.setCellWidget(row, 6, status_lbl)

    def _on_batch_progress(self, completed: int, total: int) -> None:
        pct = int((completed / total) * 100) if total > 0 else 0
        self.bar_batch.setValue(pct)
        self.lbl_batch_pct.setText(f"{pct}%")
        self.lbl_batch_status.setText(f"Batch Progress: {completed} of {total} videos completed ({pct}%)")

    def _on_batch_finished(self, success: int, fail: int) -> None:
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause Queue")
        self._is_paused = False
        self.lbl_batch_status.setText(f"Batch Complete • {success} succeeded, {fail} failed.")
        QMessageBox.information(
            self,
            "Batch Generation Finished",
            f"<b>Batch Processing Complete!</b><br><br>• Successful: {success}<br>• Failed: {fail}<br><br>Videos saved to destination folder."
        )

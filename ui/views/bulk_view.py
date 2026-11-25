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
    QComboBox, QSlider, QColorDialog, QListView, QSpinBox, QGridLayout
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QPalette

from utils.config import get_config
from core.ffmpeg_wrapper import probe_media
from services.character_matcher import CharacterMatcher, natural_sort_key
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

        # Global typography & pipeline settings
        self._font_color = "#FFFF00"          # Yellow default
        self._highlight_color = "#FFFF00"
        self._stroke_color = "#000000"
        self._stroke_thickness = 3.5
        self._bg_style = "semi_transparent_black"
        self._has_bg_box = True
        self._bg_color = "#000000"
        self._bg_opacity = 0.75
        self._font_size = 58
        self._font_family = "Montserrat"
        self._is_bold = True
        self._is_italic = False
        self._font_case = "title"
        self._box_padding = 14.0
        self._resolution = "1080p"
        self._preset_name = "Viral Shorts (Yellow + Box)"
        self._clip_duration = 5.0
        self._image_ratio = 0.5
        self._batch_limit = 0

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

        # ── Step 1: Bulk Template Pipeline Selector Card ───────────────────────
        card_template = QFrame()
        card_template.setObjectName("cardFrame")
        tpl_layout = QVBoxLayout(card_template)
        tpl_layout.setContentsMargins(18, 16, 18, 16)
        tpl_layout.setSpacing(8)

        tpl_header = QLabel("Step 1: Pipeline Template Mode")
        tpl_header.setStyleSheet("font-size: 13px; font-weight: 700; color: #F8FAFC;")

        self.combo_template = QComboBox()
        self.combo_template.setView(_make_combo_view())
        self.combo_template.addItems([
            "Template 1 (Hybrid Images & Video Mix)",
            "Template 2 (Episode Scene Matching)"
        ])
        self.combo_template.setStyleSheet("""
            QComboBox {
                background-color: #1A1A1F;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: 600;
            }
            QComboBox:hover { border-color: #818CF8; }
        """)

        self.combo_template.currentIndexChanged.connect(self._update_template_status_badge)

        tpl_desc = QLabel("Select rendering workflow: Template 1 mixes Ken Burns images and video b-roll clips, while Template 2 performs smart scene-matching across episode footage.")
        tpl_desc.setStyleSheet("font-size: 11px; color: #94A3B8;")

        self.lbl_gemini_status = QLabel()
        self.lbl_gemini_status.setWordWrap(True)

        self.btn_api_dashboard = QPushButton("View API Dashboard ↗")
        self.btn_api_dashboard.setObjectName("secondaryButton")
        self.btn_api_dashboard.setFixedHeight(24)
        self.btn_api_dashboard.setStyleSheet("font-size: 10px; color: #38BDF8; border: 1px solid #0369A1; background-color: #0B132B; border-radius: 4px; padding: 2px 8px; font-weight: 600;")
        self.btn_api_dashboard.setToolTip("Open Google AI Studio dashboard in browser")
        self.btn_api_dashboard.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://aistudio.google.com/")))

        self._update_template_status_badge()

        tpl_layout.addWidget(tpl_header)
        tpl_layout.addWidget(self.combo_template)
        tpl_layout.addWidget(self.lbl_gemini_status)
        tpl_layout.addWidget(self.btn_api_dashboard)
        tpl_layout.addWidget(tpl_desc)
        content_layout.addWidget(card_template)

        # ── Step 2: Source Folders & Smart Auto-Pairing Setup Card ─────────────
        cfg = get_config()
        card_folders = QFrame()
        card_folders.setObjectName("cardFrame")
        folders_layout = QVBoxLayout(card_folders)
        folders_layout.setContentsMargins(18, 16, 18, 16)
        folders_layout.setSpacing(14)

        folders_header = QLabel("Step 2: Source Folders & Smart Auto-Pairing Setup")
        folders_header.setStyleSheet("font-size: 13px; font-weight: 700; color: #F8FAFC;")
        folders_layout.addWidget(folders_header)

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

        audio_hint = QLabel("Select a folder containing voiceover audios (e.g. audio 1.mp3, audio 2.mp3). The scanner auto-pairs with matching numbered footage subfolders.")
        audio_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        audio_in_box.addWidget(audio_in_lbl)
        audio_in_box.addLayout(audio_in_row)
        audio_in_box.addWidget(audio_hint)
        folders_layout.addLayout(audio_in_box)

        # Batch Limit Control Row
        limit_box = QVBoxLayout()
        limit_box.setSpacing(4)
        limit_header = QHBoxLayout()
        limit_lbl = QLabel("Max Videos to Generate (Batch Limit):")
        limit_lbl.setObjectName("fieldLabel")
        limit_hint = QLabel("(0 = Process All Queued)")
        limit_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")
        limit_header.addWidget(limit_lbl)
        limit_header.addWidget(limit_hint)
        limit_header.addStretch()

        limit_row = QHBoxLayout()
        limit_row.setSpacing(8)

        self.spin_batch_limit = QSpinBox()
        self.spin_batch_limit.setRange(0, 500)
        self.spin_batch_limit.setValue(0)
        self.spin_batch_limit.setSpecialValueText("All (No Limit)")
        self.spin_batch_limit.setFixedWidth(140)
        self.spin_batch_limit.setStyleSheet("""
            QSpinBox {
                background-color: #1A1A1F;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QSpinBox:hover { border-color: #818CF8; }
        """)

        limit_row.addWidget(self.spin_batch_limit)
        for val, lbl in [(0, "All"), (5, "5"), (10, "10"), (20, "20"), (50, "50")]:
            btn_quick = QPushButton(lbl)
            btn_quick.setFixedWidth(42)
            btn_quick.setFixedHeight(28)
            btn_quick.setCursor(Qt.PointingHandCursor)
            btn_quick.setStyleSheet(BTN_INACTIVE_STYLE)
            btn_quick.clicked.connect(lambda chk=False, v=val: self.spin_batch_limit.setValue(v))
            limit_row.addWidget(btn_quick)
        limit_row.addStretch()

        limit_box.addLayout(limit_header)
        limit_box.addLayout(limit_row)
        folders_layout.addLayout(limit_box)

        # Folder 2: Footage Assets Root Directory
        footage_box = QVBoxLayout()
        footage_box.setSpacing(4)
        footage_lbl = QLabel("Main Footage Root Folder (Visual Assets Root)")
        footage_lbl.setObjectName("fieldLabel")
        footage_row = QHBoxLayout()
        footage_row.setSpacing(8)

        self.edit_footage_dir = QLineEdit()
        self.edit_footage_dir.setPlaceholderText("Select root folder containing numbered subfolders or visual assets...")
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

        footage_hint = QLabel("Root folder containing numbered video/character subfolders (e.g. 'video 1 content/', 'video 2 content/'). Auto-pairs or falls back to root.")
        footage_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        footage_box.addWidget(footage_lbl)
        footage_box.addLayout(footage_row)
        footage_box.addWidget(footage_hint)
        folders_layout.addLayout(footage_box)

        # Folder 3: Output Destination & Resolution
        out_box = QVBoxLayout()
        out_box.setSpacing(4)
        out_lbl = QLabel("Output Destination & Export Resolution")
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

        self.combo_res = QComboBox()
        self.combo_res.setView(_make_combo_view())
        self.combo_res.addItems([
            "1080p Full HD (1920x1080)",
            "720p HD (1280x720)",
            "2K QHD (2560x1440)",
        ])
        self.combo_res.setFixedWidth(200)
        self.combo_res.setStyleSheet("""
            QComboBox {
                background-color: #1A1A1F;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QComboBox:hover { border-color: #818CF8; }
        """)
        self.combo_res.currentIndexChanged.connect(self._on_res_changed)

        out_row.addWidget(self.edit_output_dir, stretch=2)
        out_row.addWidget(btn_browse_out)
        out_row.addWidget(btn_open_out)
        out_row.addWidget(self.combo_res)

        out_hint = QLabel("Select destination folder and export resolution preset (1080p Full HD default, 2K QHD, or 720p HD).")
        out_hint.setStyleSheet("font-size: 11px; color: #94A3B8;")

        out_box.addWidget(out_lbl)
        out_box.addLayout(out_row)
        out_box.addWidget(out_hint)
        folders_layout.addLayout(out_box)

        content_layout.addWidget(card_folders)

        # ── Step 3: Timing & Media Mix Settings Card ───────────────────────────
        card_timing = QFrame()
        card_timing.setObjectName("cardFrame")
        timing_layout = QVBoxLayout(card_timing)
        timing_layout.setContentsMargins(18, 16, 18, 16)
        timing_layout.setSpacing(14)

        t_hdr = QLabel("Step 3: Timing & Media Mix Settings")
        t_hdr.setStyleSheet("font-size: 13px; font-weight: 700; color: #F8FAFC;")
        timing_layout.addWidget(t_hdr)

        # Control 1: Clip / Image Hold Duration
        dur_header = QHBoxLayout()
        dur_lbl = QLabel("Clip / Image Duration (sec):")
        dur_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #E2E8F0;")
        self.lbl_clip_dur_val = QLabel("5.0 s")
        self.lbl_clip_dur_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #FB923C;")
        dur_header.addWidget(dur_lbl)
        dur_header.addStretch()
        dur_header.addWidget(self.lbl_clip_dur_val)
        timing_layout.addLayout(dur_header)

        self.slider_clip_dur = QSlider(Qt.Horizontal)
        self.slider_clip_dur.setRange(1, 20)
        self.slider_clip_dur.setValue(5)
        self.slider_clip_dur.valueChanged.connect(self._on_clip_dur_changed)
        timing_layout.addWidget(self.slider_clip_dur)

        dur_hint = QLabel("Controls continuous display length before transitioning to the next visual still or video b-roll.")
        dur_hint.setStyleSheet("font-size: 11px; color: #71717A;")
        timing_layout.addWidget(dur_hint)

        # Control 2: Media Mix Ratio
        mix_header = QHBoxLayout()
        mix_lbl = QLabel("Media Mix Ratio (Images vs. Video Footage):")
        mix_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #E2E8F0;")
        self.lbl_mix_ratio_val = QLabel("Images: 50% | Video Clips: 50%")
        self.lbl_mix_ratio_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8;")
        mix_header.addWidget(mix_lbl)
        mix_header.addStretch()
        mix_header.addWidget(self.lbl_mix_ratio_val)
        timing_layout.addLayout(mix_header)

        self.slider_mix_ratio = QSlider(Qt.Horizontal)
        self.slider_mix_ratio.setRange(0, 100)
        self.slider_mix_ratio.setValue(50)
        self.slider_mix_ratio.valueChanged.connect(self._on_mix_ratio_changed)
        timing_layout.addWidget(self.slider_mix_ratio)

        mix_hint = QLabel("Controls ratio of Ken Burns character stills vs. sliced raw episode video clips for Template 1.")
        mix_hint.setStyleSheet("font-size: 11px; color: #71717A;")
        timing_layout.addWidget(mix_hint)

        content_layout.addWidget(card_timing)

        # ── Step 4: Subtitles & Typography Studio (Global Batch Settings) ─────
        self.style_frame = QFrame()
        self.style_frame.setObjectName("cardFrame")
        style_layout = QVBoxLayout(self.style_frame)
        style_layout.setContentsMargins(18, 16, 18, 16)
        style_layout.setSpacing(14)

        # Header Toggle Button
        self.btn_toggle_style = QPushButton("▲  Step 4: Subtitles & Typography Studio (Global Batch Settings)")
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
        body_layout.setSpacing(12)

        # Row 1: Quick Style Presets
        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        p_lbl = QLabel("Style Preset:")
        p_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_presets = QComboBox()
        self.combo_presets.setView(_make_combo_view())
        self.combo_presets.addItems([
            "Classic YouTube (Bold Yellow Highlight)",
            "Viral Shorts (Yellow + Box)",
            "High-Contrast Box (White + Solid Black)",
            "Minimal Clean (White + Outline)",
            "Cinematic Teal"
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(p_lbl)
        preset_row.addWidget(self.combo_presets, stretch=1)
        body_layout.addLayout(preset_row)

        # Row 2: Aa Quick Swatches Row
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        swatches = [
            ("#FFFF00", "#000000", "Yellow"),
            ("#FFFFFF", "#000000", "White"),
            ("#38BDF8", "#000000", "Cyan"),
            ("#34D399", "#000000", "Green"),
            ("#FB923C", "#000000", "Coral"),
            ("#A78BFA", "#000000", "Violet"),
        ]
        for font_c, stroke_c, lbl in swatches:
            btn_s = QPushButton(f"Aa\n{lbl}")
            btn_s.setFixedSize(54, 38)
            btn_s.setCursor(Qt.PointingHandCursor)
            btn_s.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1A1A20;
                    color: {font_c};
                    border: 1px solid #32323D;
                    border-radius: 6px;
                    font-size: 9px;
                    font-weight: 700;
                    padding: 0;
                }}
                QPushButton:hover {{
                    border: 1.5px solid {font_c};
                    background-color: #22222A;
                }}
            """)
            btn_s.clicked.connect(lambda chk=False, fc=font_c, sc=stroke_c: self._apply_preset_color(fc, sc))
            swatch_row.addWidget(btn_s)
        swatch_row.addStretch()
        body_layout.addLayout(swatch_row)

        # Row 3: Font Family, Background Box Mode, Box Opacity Grid
        font_grid = QGridLayout()
        font_grid.setSpacing(10)

        ff_lbl = QLabel("Font Family")
        ff_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
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
            "Arial",
        ])
        self.combo_font.currentTextChanged.connect(lambda: self._update_preview())
        font_grid.addWidget(ff_lbl, 0, 0)
        font_grid.addWidget(self.combo_font, 1, 0)

        bg_lbl = QLabel("Background Box Mode")
        bg_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_bg = QComboBox()
        self.combo_bg.setView(_make_combo_view())
        self.combo_bg.addItems([
            "Outline Mode",
            "Semi-Transparent Box",
            "Solid Box",
            "Custom Box",
        ])
        self.combo_bg.currentIndexChanged.connect(self._on_bg_changed)
        font_grid.addWidget(bg_lbl, 0, 1)
        font_grid.addWidget(self.combo_bg, 1, 1)

        opac_lbl = QLabel("Box Opacity")
        opac_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_box_opacity = QComboBox()
        self.combo_box_opacity.setView(_make_combo_view())
        self.combo_box_opacity.addItems([
            "Semi-Transparent (75%)",
            "Solid (100%)",
            "Subtle (50%)",
        ])
        self.combo_box_opacity.currentIndexChanged.connect(self._on_opacity_changed)
        font_grid.addWidget(opac_lbl, 0, 2)
        font_grid.addWidget(self.combo_box_opacity, 1, 2)

        body_layout.addLayout(font_grid)

        # Row 4: Font Size Slider Row
        fs_row = QHBoxLayout()
        fs_lbl = QLabel("Font Size:")
        fs_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_fs_val = QLabel(f"{self._font_size} pt")
        self.lbl_fs_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #FB923C;")
        self.slider_fs = QSlider(Qt.Horizontal)
        self.slider_fs.setRange(32, 90)
        self.slider_fs.setValue(self._font_size)
        self.slider_fs.valueChanged.connect(self._on_fs_changed)
        fs_row.addWidget(fs_lbl)
        fs_row.addWidget(self.slider_fs, stretch=1)
        fs_row.addWidget(self.lbl_fs_val)
        body_layout.addLayout(fs_row)

        # Row 5: Patterns, Case & Color Pickers
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        # Pattern buttons (B, I, U)
        pat_box = QHBoxLayout()
        pat_box.setSpacing(4)
        self.btn_bold = QPushButton("B")
        self.btn_bold.setFixedSize(30, 28)
        self.btn_bold.setCursor(Qt.PointingHandCursor)
        self.btn_bold.setStyleSheet(BTN_ACTIVE_STYLE)
        self.btn_bold.clicked.connect(lambda: self._toggle_pattern_btn(self.btn_bold))

        self.btn_italic = QPushButton("I")
        self.btn_italic.setFixedSize(30, 28)
        self.btn_italic.setCursor(Qt.PointingHandCursor)
        self.btn_italic.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_italic.clicked.connect(lambda: self._toggle_pattern_btn(self.btn_italic))

        self.btn_underline = QPushButton("U")
        self.btn_underline.setFixedSize(30, 28)
        self.btn_underline.setCursor(Qt.PointingHandCursor)
        self.btn_underline.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_underline.clicked.connect(lambda: self._toggle_pattern_btn(self.btn_underline))

        pat_box.addWidget(self.btn_bold)
        pat_box.addWidget(self.btn_italic)
        pat_box.addWidget(self.btn_underline)
        controls_row.addLayout(pat_box)

        # Case buttons (TT, tt, Tt)
        case_box = QHBoxLayout()
        case_box.setSpacing(4)
        self.btn_case_upper = QPushButton("TT")
        self.btn_case_upper.setFixedSize(30, 28)
        self.btn_case_upper.setCursor(Qt.PointingHandCursor)
        self.btn_case_upper.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_case_upper.clicked.connect(lambda: self._set_case("upper"))

        self.btn_case_lower = QPushButton("tt")
        self.btn_case_lower.setFixedSize(30, 28)
        self.btn_case_lower.setCursor(Qt.PointingHandCursor)
        self.btn_case_lower.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_case_lower.clicked.connect(lambda: self._set_case("lower"))

        self.btn_case_title = QPushButton("Tt")
        self.btn_case_title.setFixedSize(30, 28)
        self.btn_case_title.setCursor(Qt.PointingHandCursor)
        self.btn_case_title.setStyleSheet(BTN_ACTIVE_STYLE)
        self.btn_case_title.clicked.connect(lambda: self._set_case("title"))

        case_box.addWidget(self.btn_case_upper)
        case_box.addWidget(self.btn_case_lower)
        case_box.addWidget(self.btn_case_title)
        controls_row.addLayout(case_box)

        # Color Swatches Pickers
        col_box = QHBoxLayout()
        col_box.setSpacing(6)
        col_lbl = QLabel("Text:")
        col_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        self.btn_col_swatch = QPushButton()
        self.btn_col_swatch.setFixedSize(26, 26)
        self.btn_col_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_col_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_col_swatch.setToolTip("Font Text Color")
        self.btn_col_swatch.clicked.connect(self._pick_color)

        hl_lbl = QLabel("Word:")
        hl_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        self.btn_hl_swatch = QPushButton()
        self.btn_hl_swatch.setFixedSize(26, 26)
        self.btn_hl_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_hl_swatch.setToolTip("Active Word Highlight Color")
        self.btn_hl_swatch.clicked.connect(self._pick_hl_color)

        bg_col_lbl = QLabel("Box:")
        bg_col_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        self.btn_bg_swatch = QPushButton()
        self.btn_bg_swatch.setFixedSize(26, 26)
        self.btn_bg_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_bg_swatch.setStyleSheet(f"background-color: {self._bg_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_bg_swatch.setToolTip("Background Box Fill Color")
        self.btn_bg_swatch.clicked.connect(self._pick_bg_color)

        col_box.addWidget(col_lbl)
        col_box.addWidget(self.btn_col_swatch)
        col_box.addWidget(hl_lbl)
        col_box.addWidget(self.btn_hl_swatch)
        col_box.addWidget(bg_col_lbl)
        col_box.addWidget(self.btn_bg_swatch)
        controls_row.addLayout(col_box)
        controls_row.addStretch()
        body_layout.addLayout(controls_row)

        # Row 6: Stroke Outline & Box Padding Controls
        stroke_pad_row = QHBoxLayout()
        stroke_pad_row.setSpacing(16)

        # Stroke Outline
        s_box = QHBoxLayout()
        s_box.setSpacing(6)
        s_lbl = QLabel("Stroke:")
        s_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.btn_stroke_swatch = QPushButton()
        self.btn_stroke_swatch.setFixedSize(26, 26)
        self.btn_stroke_swatch.setCursor(Qt.PointingHandCursor)
        self.btn_stroke_swatch.setStyleSheet(f"background-color: {self._stroke_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_stroke_swatch.clicked.connect(self._pick_stroke_color)
        self.slider_stroke = QSlider(Qt.Horizontal)
        self.slider_stroke.setRange(1, 10)
        self.slider_stroke.setValue(4)
        self.slider_stroke.valueChanged.connect(self._on_stroke_changed)
        self.lbl_stroke_val = QLabel(f"{self._stroke_thickness:.1f} px")
        self.lbl_stroke_val.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        s_box.addWidget(s_lbl)
        s_box.addWidget(self.btn_stroke_swatch)
        s_box.addWidget(self.slider_stroke, stretch=1)
        s_box.addWidget(self.lbl_stroke_val)
        stroke_pad_row.addLayout(s_box, stretch=1)

        # Box Padding
        p_box = QHBoxLayout()
        p_box.setSpacing(6)
        p_lbl = QLabel("Box Padding:")
        p_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.slider_box_padding = QSlider(Qt.Horizontal)
        self.slider_box_padding.setRange(6, 28)
        self.slider_box_padding.setValue(int(self._box_padding))
        self.slider_box_padding.valueChanged.connect(self._on_box_padding_changed)
        self.lbl_box_pad_val = QLabel(f"{int(self._box_padding)} px")
        self.lbl_box_pad_val.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        p_box.addWidget(p_lbl)
        p_box.addWidget(self.slider_box_padding, stretch=1)
        p_box.addWidget(self.lbl_box_pad_val)
        stroke_pad_row.addLayout(p_box, stretch=1)

        body_layout.addLayout(stroke_pad_row)

        # Row 7: Checkboxes
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

        # Row 8: Preview Banner
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
            "#", "Audio File", "Matched Footage Folder", "Template", "Output Path", "Progress", "Status", "Actions"
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
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 220)
        self.table.setColumnWidth(3, 110)
        self.table.setColumnWidth(4, 180)
        self.table.setColumnWidth(5, 110)
        self.table.setColumnWidth(6, 100)
        self.table.setColumnWidth(7, 80)

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
        self.btn_toggle_style.setText(f"{arrow}  Step 4: Subtitles & Typography Studio (Global Batch Settings)")

    def _on_preset_changed(self, idx: int) -> None:
        if idx == 0:  # Classic YouTube (Bold Yellow Highlight)
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FFFF00"
            self._stroke_color = "#000000"
            self._stroke_thickness = 3.5
            self._bg_style = "outline"
            self._has_bg_box = False
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(0)  # Outline
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Arial Black")
            self.slider_fs.setValue(56)
            self._is_bold = True
            self._is_italic = False
            self._preset_name = "Classic YouTube (Bold Yellow Highlight)"
        elif idx == 1:  # Viral Shorts (Yellow + Box)
            self._font_color = "#FFFF00"
            self._highlight_color = "#00FFFF"
            self._stroke_color = "#000000"
            self._stroke_thickness = 3.5
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_color = "#000000"
            self._bg_opacity = 0.75
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(1)  # Semi-Transparent Box
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Montserrat")
            self.slider_fs.setValue(58)
            self._is_bold = True
            self._is_italic = False
            self._preset_name = "Viral Shorts (Yellow + Box)"
        elif idx == 2:  # High-Contrast Box (White + Solid Black)
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FFFF00"
            self._stroke_color = "#000000"
            self._stroke_thickness = 3.5
            self._bg_style = "solid_black"
            self._has_bg_box = True
            self._bg_color = "#000000"
            self._bg_opacity = 1.0
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(2)  # Solid Box
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Impact")
            self.slider_fs.setValue(54)
            self._is_bold = True
            self._is_italic = False
            self._preset_name = "High-Contrast Box (White + Solid Black)"
        elif idx == 3:  # Minimal Clean (White + Outline)
            self._font_color = "#F8FAFC"
            self._highlight_color = "#38BDF8"
            self._stroke_color = "#000000"
            self._stroke_thickness = 2.5
            self._bg_style = "outline"
            self._has_bg_box = False
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(0)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Poppins")
            self.slider_fs.setValue(50)
            self._is_bold = False
            self._is_italic = False
            self._preset_name = "Minimal Clean (White + Outline)"
        elif idx == 4:  # Cinematic Teal
            self._font_color = "#2DD4BF"
            self._highlight_color = "#FDE047"
            self._stroke_color = "#042F2E"
            self._stroke_thickness = 3.0
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_color = "#022C22"
            self._bg_opacity = 0.70
            self.combo_bg.blockSignals(True)
            self.combo_bg.setCurrentIndex(1)
            self.combo_bg.blockSignals(False)
            self.combo_font.setCurrentText("Segoe UI")
            self.slider_fs.setValue(52)
            self._is_bold = True
            self._is_italic = False
            self._preset_name = "Cinematic Teal"

        self.btn_bold.setStyleSheet(BTN_ACTIVE_STYLE if self._is_bold else BTN_INACTIVE_STYLE)
        self.btn_italic.setStyleSheet(BTN_ACTIVE_STYLE if self._is_italic else BTN_INACTIVE_STYLE)
        self.btn_col_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_bg_swatch.setStyleSheet(f"background-color: {self._bg_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_stroke_swatch.setStyleSheet(f"background-color: {self._stroke_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self._update_preview()

    def _apply_preset_color(self, font_c: str, stroke_c: str) -> None:
        self._font_color = font_c
        self._stroke_color = stroke_c
        self.btn_col_swatch.setStyleSheet(f"background-color: {font_c}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_stroke_swatch.setStyleSheet(f"background-color: {stroke_c}; border: 1.5px solid #334155; border-radius: 5px;")
        self._update_preview()

    def _on_bg_changed(self, idx: int) -> None:
        if idx == 0:  # Outline Mode
            self._bg_style = "outline"
            self._has_bg_box = False
        elif idx == 1:  # Semi-Transparent Box
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_opacity = 0.75
            if hasattr(self, "combo_box_opacity"):
                self.combo_box_opacity.setCurrentIndex(0)
        elif idx == 2:  # Solid Box
            self._bg_style = "solid_black"
            self._has_bg_box = True
            self._bg_opacity = 1.0
            if hasattr(self, "combo_box_opacity"):
                self.combo_box_opacity.setCurrentIndex(1)
        elif idx == 3:  # Custom Box
            self._bg_style = "custom_box"
            self._has_bg_box = True
        self._update_preview()

    def _on_opacity_changed(self, idx: int) -> None:
        opacities = [0.75, 1.0, 0.50]
        if 0 <= idx < len(opacities):
            self._bg_opacity = opacities[idx]
        self._update_preview()

    def _pick_bg_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._bg_color), self, "Select Background Box Color")
        if c.isValid():
            self._bg_color = c.name().upper()
            self.btn_bg_swatch.setStyleSheet(f"background-color: {self._bg_color}; border: 1.5px solid #334155; border-radius: 6px;")
            self._update_preview()

    def _pick_stroke_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._stroke_color), self, "Select Outline Stroke Color")
        if c.isValid():
            self._stroke_color = c.name().upper()
            self.btn_stroke_swatch.setStyleSheet(f"background-color: {self._stroke_color}; border: 1.5px solid #334155; border-radius: 6px;")
            self._update_preview()

    def _on_box_padding_changed(self, val: int) -> None:
        self._box_padding = float(val)
        self.lbl_box_pad_val.setText(f"{val} px")
        self._update_preview()

    def _on_stroke_changed(self, val: int) -> None:
        self._stroke_thickness = val * 0.8
        self.lbl_stroke_val.setText(f"{self._stroke_thickness:.1f} px")
        self._update_preview()

    def _on_clip_dur_changed(self, val: int) -> None:
        self._clip_duration = float(val)
        self.lbl_clip_dur_val.setText(f"{val}.0 s")

    def _on_mix_ratio_changed(self, val: int) -> None:
        self._image_ratio = float(val) / 100.0
        self.lbl_mix_ratio_val.setText(f"Images: {val}% | Video Clips: {100 - val}%")

    def _toggle_pattern_btn(self, btn: QPushButton) -> None:
        if btn is self.btn_bold:
            self._is_bold = not self._is_bold
            btn.setStyleSheet(BTN_ACTIVE_STYLE if self._is_bold else BTN_INACTIVE_STYLE)
        elif btn is self.btn_italic:
            self._is_italic = not self._is_italic
            btn.setStyleSheet(BTN_ACTIVE_STYLE if self._is_italic else BTN_INACTIVE_STYLE)
        elif btn is self.btn_underline:
            is_active = (btn.styleSheet() == BTN_ACTIVE_STYLE)
            btn.setStyleSheet(BTN_INACTIVE_STYLE if is_active else BTN_ACTIVE_STYLE)
        self._update_preview()

    def _set_case(self, case_type: str) -> None:
        self._font_case = case_type
        self.btn_case_upper.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "upper" else BTN_INACTIVE_STYLE)
        self.btn_case_lower.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "lower" else BTN_INACTIVE_STYLE)
        self.btn_case_title.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "title" else BTN_INACTIVE_STYLE)
        self._update_preview()

    def _on_res_changed(self, idx: int) -> None:
        txt = self.combo_res.currentText().lower()
        if "720p" in txt:
            self._resolution = "720p"
        elif "2k" in txt:
            self._resolution = "2k"
        else:
            self._resolution = "1080p"

        out_dir = Path(self.edit_output_dir.text().strip()) if self.edit_output_dir.text().strip() else get_config().output_dir
        for r in range(self.table.rowCount()):
            audio_item = self.table.item(r, 1)
            if audio_item:
                audio_path = Path(audio_item.toolTip()) if audio_item.toolTip() else Path(audio_item.text())
                safe_stem = audio_path.stem.replace(" ", "_")
                new_out = out_dir / f"{safe_stem}_{self._resolution}.mp4"
                out_item = self.table.item(r, 4)
                if out_item:
                    out_item.setText(new_out.name)
                    out_item.setToolTip(str(new_out.resolve()))

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
        raw_font = self.combo_font.currentText().split("(")[0].strip()
        font_name = raw_font.replace(" Bold", "").strip() or "Montserrat"
        base_size = max(int(self._font_size * 0.36), 14)

        font_weight = "800" if self._is_bold else "normal"
        font_style = "italic" if self._is_italic else "normal"

        sample_text = "Batch Subtitle Typography Preview"
        if self._font_case == "upper":
            sample_text = sample_text.upper()
        elif self._font_case == "lower":
            sample_text = sample_text.lower()
        elif self._font_case == "title":
            sample_text = sample_text.title()

        self.lbl_preview.setText(sample_text)

        if self._has_bg_box:
            hex_c = self._bg_color.lstrip("#")
            if len(hex_c) == 6:
                try:
                    r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                except ValueError:
                    r, g, b = 0, 0, 0
            else:
                r, g, b = 0, 0, 0

            pad_v = max(4, int(self._box_padding * 0.45))
            pad_h = max(8, int(self._box_padding * 1.1))
            box_style = (
                f"background-color: rgba({r}, {g}, {b}, {self._bg_opacity:.2f}); "
                f"color: {self._font_color}; "
                f"border: 1px solid rgba(255, 255, 255, 0.20); "
                f"border-radius: 6px; "
                f"padding: {pad_v}px {pad_h}px;"
            )
        else:  # Outline
            box_style = (
                f"background-color: transparent; "
                f"color: {self._font_color}; "
                f"border: 1.5px dashed #334155; "
                f"border-radius: 6px; "
                f"padding: 6px 16px;"
            )

        self.lbl_preview.setStyleSheet(f"""
            font-family: '{font_name}', 'Segoe UI', Arial, sans-serif;
            font-size: {base_size}px;
            font-weight: {font_weight};
            font-style: {font_style};
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
        audio_files = [
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in audio_exts
        ]
        audio_files.sort(key=natural_sort_key)

        limit = self.spin_batch_limit.value() if hasattr(self, "spin_batch_limit") else 0
        if limit > 0:
            audio_files = audio_files[:limit]

        self.table.setRowCount(0)
        count = 0
        for audio_file in audio_files:
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

        footage_base = Path(self.edit_footage_dir.text().strip()) if self.edit_footage_dir.text().strip() else get_config().project_root / "assets" / "sample_footage"
        matched_folder, scan_summary = CharacterMatcher.find_matching_footage_folder(audio_path, footage_base)

        # Build clean summary label for column 2
        pools = scan_summary.get("character_pools", {})
        video_files = scan_summary.get("video_files", [])
        general_images = scan_summary.get("general_images", 0)

        summary_parts = []
        if pools:
            summary_parts.append(f"{len(pools)} pool{'s' if len(pools) > 1 else ''}")
        if general_images:
            summary_parts.append(f"{general_images} img{'s' if general_images > 1 else ''}")
        if video_files:
            summary_parts.append(f"{len(video_files)} video{'s' if len(video_files) > 1 else ''}")

        summary_suffix = f" ({', '.join(summary_parts)})" if summary_parts else ""
        folder_display_text = f"{matched_folder.name}{summary_suffix}"

        # Template selection
        selected_template_label = self.combo_template.currentText()
        template_short = "Template 1" if "Template 1" in selected_template_label else "Template 2"

        # Output path
        out_dir = Path(self.edit_output_dir.text().strip()) if self.edit_output_dir.text().strip() else get_config().output_dir
        safe_stem = audio_path.stem.replace(" ", "_")
        out_file = out_dir / f"{safe_stem}_{self._resolution}.mp4"

        # 0. Job #
        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, idx_item)

        # 1. Audio File
        audio_item = QTableWidgetItem(audio_path.name)
        audio_item.setToolTip(str(audio_path.resolve()))
        self.table.setItem(row, 1, audio_item)

        # 2. Matched Footage Folder
        matched_item = QTableWidgetItem(folder_display_text)
        matched_item.setToolTip(str(matched_folder.resolve()))
        self.table.setItem(row, 2, matched_item)

        # 3. Template
        tpl_item = QTableWidgetItem(template_short)
        tpl_item.setToolTip(selected_template_label)
        self.table.setItem(row, 3, tpl_item)

        # 4. Output Path
        out_item = QTableWidgetItem(out_file.name)
        out_item.setToolTip(str(out_file.resolve()))
        self.table.setItem(row, 4, out_item)

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
        out_item = self.table.item(row, 4)
        if out_item and out_item.toolTip():
            out_file = Path(out_item.toolTip())
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

        limit = self.spin_batch_limit.value() if hasattr(self, "spin_batch_limit") else 0
        total_rows = self.table.rowCount()
        rows_to_process = min(total_rows, limit) if limit > 0 else total_rows

        jobs: list[BulkJob] = []
        for r in range(rows_to_process):
            audio_item = self.table.item(r, 1)
            audio_path = Path(audio_item.toolTip()) if (audio_item and audio_item.toolTip()) else Path(audio_item.text())
            title = audio_path.stem.replace("_", " ").title()

            matched_item = self.table.item(r, 2)
            assets_path = Path(matched_item.toolTip()) if (matched_item and matched_item.toolTip()) else footage_base

            tpl_item = self.table.item(r, 3)
            tpl_text = tpl_item.text() if tpl_item else "Template 1"
            tpl_type = "template_2" if "Template 2" in tpl_text else "template_1"
            tpl_label = tpl_item.toolTip() if (tpl_item and tpl_item.toolTip()) else (
                "Template 2 (Episode Scene Matching)" if tpl_type == "template_2" else "Template 1 (Hybrid Images & Video Mix)"
            )

            out_item = self.table.item(r, 4)
            safe_stem = audio_path.stem.replace(" ", "_")
            out_file = Path(out_item.toolTip()) if (out_item and out_item.toolTip()) else (out_dir / f"{safe_stem}_{self._resolution}.mp4")

            jobs.append(BulkJob(
                job_id=r,
                title=title,
                audio_path=audio_path,
                images_dir=assets_path,
                output_path=out_file,
                template_type=tpl_type,
                template_label=tpl_label,
                resolution=self._resolution,
                clip_duration=float(self.slider_clip_dur.value()) if hasattr(self, "slider_clip_dur") else 5.0,
                image_ratio=float(self.slider_mix_ratio.value()) / 100.0 if hasattr(self, "slider_mix_ratio") else 0.5,
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
                bold=self._is_bold,
                italic=self._is_italic,
                box_padding=self._box_padding,
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
        self._update_template_status_badge()
        QMessageBox.information(
            self,
            "Batch Generation Finished",
            f"<b>Batch Processing Complete!</b><br><br>• Successful: {success}<br>• Failed: {fail}<br><br>Videos saved to destination folder."
        )

    def _update_template_status_badge(self) -> None:
        """Update subtle badge displaying Gemini AI vs Offline Heuristic matcher mode & lifetime tokens."""
        from database.db import get_lifetime_token_stats
        from utils.config import reload_config
        cfg = reload_config()
        is_t2 = (self.combo_template.currentIndex() == 1)
        api_key = os.getenv("GEMINI_API_KEY") or cfg.gemini_api_key or ""
        from services.gemini_matcher import SmartSceneMatcher
        has_valid_key = SmartSceneMatcher.is_api_key_valid(api_key)

        if hasattr(self, "btn_api_dashboard"):
            self.btn_api_dashboard.setVisible(is_t2)

        if is_t2:
            stats = get_lifetime_token_stats()
            total_toks = stats.get("total_tokens", 0)
            cost_usd = stats.get("total_cost_usd", 0.0)
            if has_valid_key:
                tok_str = f"{total_toks:,}" if total_toks > 0 else "0"
                self.lbl_gemini_status.setText(
                    f"🤖 Gemini AI Matcher Active | Lifetime: {tok_str} tokens (~${cost_usd:.4f})"
                )
                self.lbl_gemini_status.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; border: 1px solid #059669; "
                    "border-radius: 5px; font-size: 11px; font-weight: 700; padding: 5px 8px;"
                )
            else:
                self.lbl_gemini_status.setText("⚡ Offline Smart Scene Matcher Active (Heuristic)")
                self.lbl_gemini_status.setStyleSheet(
                    "background-color: #1E293B; color: #38BDF8; border: 1px solid #0284C7; "
                    "border-radius: 5px; font-size: 11px; font-weight: 700; padding: 5px 8px;"
                )
        else:
            if has_valid_key:
                self.lbl_gemini_status.setText("🤖 Hybrid Media Engine + Gemini Vision Scene Matching")
                self.lbl_gemini_status.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; border: 1px solid #059669; "
                    "border-radius: 5px; font-size: 11px; font-weight: 700; padding: 5px 8px;"
                )
            else:
                self.lbl_gemini_status.setText("⚡ Hybrid Media Engine (Smart Keyword Matching)")
                self.lbl_gemini_status.setStyleSheet(
                    "background-color: #1E293B; color: #38BDF8; border: 1px solid #0284C7; "
                    "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
                )


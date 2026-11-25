"""
ui/views/studio_view.py
=======================
Studio workspace for CineForge AI Video Editor Studio.
Split-Screen 2-Column Workbench:
  - Left Dock (45% width, scrollable): Step 1 Project Setup, Step 2 Visual & Audio Dropzones, Step 3 Caption Typography Studio.
  - Right Stage (55% width, sticky preview & render center): 16:9 Cinema Preview Stage, Output Bar, Dual Pipeline Telemetry, and Huge Pinned Primary Action Button.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QComboBox, QCheckBox, QProgressBar, QFrame, QScrollArea,
    QSlider, QMessageBox, QColorDialog, QListView, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QColor, QPalette, QPixmap

from utils.config import get_config
from core.ffmpeg_wrapper import _nvenc_available
from ui.workers.video_worker import VideoRenderWorker
from utils.file_helpers import is_video_file, is_image_file


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


# Button styles for pattern and case controls
BTN_INACTIVE_STYLE = """
    QPushButton {
        background-color: #222228;
        color: #A1A1AA;
        border: 1px solid #32323D;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
    }
    QPushButton:hover {
        background-color: #2A2A34;
        color: #F4F4F5;
        border-color: #444454;
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
    }
"""


class StudioView(QWidget):
    """
    Split-Screen 2-Column Creative Workbench for CineForge AI.
    Left Dock handles configuration cards; Right Stage pins live preview & render controls.
    """

    request_play_video = Signal(str)
    request_scan_youtube = Signal(str)
    interview_segments_scanned = Signal(str, list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[VideoRenderWorker] = None
        self._scan_worker: Optional[Any] = None

        # Caption Styling State
        self._font_color: str = "#FFFF00"
        self._highlight_color: str = "#FFFF00"
        self._stroke_color: str = "#000000"
        self._stroke_thickness: float = 3.5
        self._font_size: int = 56
        self._font_case: str = "none"  # "none", "upper", "lower", "title"
        self._has_bg_box: bool = False
        self._bg_style: str = "outline"  # "outline", "semi_transparent_black", "solid_black", "custom_box"
        self._bg_color: str = "#000000"
        self._bg_opacity: float = 0.75
        self._box_padding: float = 14.0
        self._audio_paths: list[Path] = []

        # Template 3: Interview & Framing State
        self._interview_border_color: str = "#EF4444"
        self._interview_corner_shape: str = "rounded"
        self._interview_clip_paths: list[Path] = []
        self._scanned_interview_segments: list[Any] = []
        self._confirmed_interview_clips: list[Any] = []
        self._interview_source_video: str = ""

        self._setup_ui()

    def _setup_ui(self) -> None:
        cfg = get_config()
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ══════════════════════════════════════════════════════════════════════
        # ZONE 1: LEFT CONFIGURATION DOCK (Scrollable Cards, 46% width)
        # ══════════════════════════════════════════════════════════════════════
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        dock_container = QWidget()
        dock_layout = QVBoxLayout(dock_container)
        dock_layout.setContentsMargins(0, 0, 10, 0)
        dock_layout.setSpacing(14)

        # ── Card 1: Project & Video Format Setup ─────────────────────────────
        card_setup = QFrame()
        card_setup.setObjectName("cardFrame")
        setup_layout = QVBoxLayout(card_setup)
        setup_layout.setContentsMargins(16, 14, 16, 16)
        setup_layout.setSpacing(10)

        # Header
        h_box = QHBoxLayout()
        step1_badge = QLabel("STEP 1")
        step1_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;")
        step1_title = QLabel("Video Rendering Engine & Template")
        step1_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")
        h_box.addWidget(step1_badge)
        h_box.addWidget(step1_title)
        h_box.addStretch()
        setup_layout.addLayout(h_box)

        # Internal title holder (for backwards compatibility and test assertions)
        self.edit_title = QLineEdit()
        self.edit_title.setText("Untitled Video Project")

        # Single Clean Format Dropdown
        fmt_lbl = QLabel("Creative Pipeline Template")
        fmt_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_format = QComboBox()
        self.combo_format.setView(_make_combo_view())
        self.combo_format.addItems([
            "Template 1 (Hybrid Images & Video Mix)",
            "Template 2 (Episode Scene Matching)",
            "Template 3 + Interview",
        ])
        self.combo_format.currentIndexChanged.connect(self._update_template_status_badge)

        # Narration Language Dropdown for Template 2 (Urdu / English)
        self.box_t2_language = QFrame()
        self.box_t2_language.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 6px; padding: 4px 10px;")
        t2_lang_layout = QHBoxLayout(self.box_t2_language)
        t2_lang_layout.setContentsMargins(2, 2, 2, 2)
        t2_lang_layout.setSpacing(8)

        lbl_lang = QLabel("🗣️ Narration Language:")
        lbl_lang.setStyleSheet("font-size: 11px; font-weight: 700; color: #F4F4F5;")
        self.combo_language = QComboBox()
        self.combo_language.setView(_make_combo_view())
        self.combo_language.addItems([
            "Urdu (اردو)",
            "English",
        ])
        self.combo_language.setFixedHeight(26)
        self.combo_language.setStyleSheet("font-size: 11px; font-weight: 600;")
        t2_lang_layout.addWidget(lbl_lang)
        t2_lang_layout.addWidget(self.combo_language, stretch=1)

        self.lbl_gemini_status = QLabel()
        self.lbl_gemini_status.setWordWrap(True)

        self.btn_api_dashboard = QPushButton("View API Dashboard ↗")
        self.btn_api_dashboard.setObjectName("secondaryButton")
        self.btn_api_dashboard.setFixedHeight(24)
        self.btn_api_dashboard.setStyleSheet("font-size: 10px; color: #38BDF8; border: 1px solid #0369A1; background-color: #0B132B; border-radius: 4px; padding: 2px 8px; font-weight: 600;")
        self.btn_api_dashboard.setToolTip("Open Google AI Studio dashboard in browser")
        self.btn_api_dashboard.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://aistudio.google.com/")))

        self._update_template_status_badge()

        setup_layout.addWidget(fmt_lbl)
        setup_layout.addWidget(self.combo_format)
        setup_layout.addWidget(self.box_t2_language)
        setup_layout.addWidget(self.lbl_gemini_status)
        setup_layout.addWidget(self.btn_api_dashboard)
        dock_layout.addWidget(card_setup)

        # ── Card 2: Interactive Media Dropzones ───────────────────────────────
        card_media = QFrame()
        card_media.setObjectName("cardFrame")
        media_layout = QVBoxLayout(card_media)
        media_layout.setContentsMargins(16, 14, 16, 16)
        media_layout.setSpacing(12)

        h_box2 = QHBoxLayout()
        step2_badge = QLabel("STEP 2")
        step2_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;")
        step2_title = QLabel("Media & Visual Assets")
        step2_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")
        h_box2.addWidget(step2_badge)
        h_box2.addWidget(step2_title)
        h_box2.addStretch()
        media_layout.addLayout(h_box2)

        # 1. Voiceover Audio Dropzone Box
        vo_box = QFrame()
        vo_box.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        vo_layout = QVBoxLayout(vo_box)
        vo_layout.setSpacing(6)

        vo_header = QHBoxLayout()
        vo_tag = QLabel("🎙️ Voiceover Audio Track (.mp3, .wav)")
        vo_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        self.lbl_vo_badge = QLabel("Select Audio")
        self.lbl_vo_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;")
        vo_header.addWidget(vo_tag)
        vo_header.addStretch()
        vo_header.addWidget(self.lbl_vo_badge)
        vo_layout.addLayout(vo_header)

        # Audio file input line & buttons (Supports Single Audio & Multi-Part Audio)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(6)
        self.edit_audio = QLineEdit()
        self.edit_audio.setPlaceholderText("Select voiceover audio (or multi-part: part 1.mp3, part 2.mp3)...")
        self.edit_audio.textChanged.connect(self._on_audio_changed)

        btn_browse_audio = QPushButton("Browse Audio")
        btn_browse_audio.setObjectName("secondaryButton")
        btn_browse_audio.setFixedWidth(105)
        btn_browse_audio.setToolTip("Select single audio or multiple audio parts")
        btn_browse_audio.clicked.connect(self._browse_audio)

        btn_add_parts = QPushButton("+ Add Parts")
        btn_add_parts.setObjectName("secondaryButton")
        btn_add_parts.setFixedWidth(85)
        btn_add_parts.setToolTip("Add more audio parts to the queue")
        btn_add_parts.clicked.connect(self._add_more_parts)

        btn_clear_audio = QPushButton("✕")
        btn_clear_audio.setObjectName("secondaryButton")
        btn_clear_audio.setFixedWidth(30)
        btn_clear_audio.setToolTip("Clear selected audio")
        btn_clear_audio.clicked.connect(lambda: self.edit_audio.clear())

        audio_row.addWidget(self.edit_audio, stretch=1)
        audio_row.addWidget(btn_browse_audio)
        audio_row.addWidget(btn_add_parts)
        audio_row.addWidget(btn_clear_audio)
        vo_layout.addLayout(audio_row)
        media_layout.addWidget(vo_box)

        # 2. Unified Footage & Assets Folder Box
        assets_box = QFrame()
        assets_box.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        assets_layout = QVBoxLayout(assets_box)
        assets_layout.setSpacing(8)

        assets_header = QHBoxLayout()
        self.assets_tag = QLabel("📁 Footage & Visual Assets Folder (Images & Videos)")
        self.assets_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        assets_header.addWidget(self.assets_tag)
        assets_header.addStretch()
        assets_layout.addLayout(assets_header)

        assets_row = QHBoxLayout()
        assets_row.setSpacing(8)
        self.edit_assets = QLineEdit()
        self.edit_assets.setPlaceholderText("Select visual assets folder (with character subfolders and/or raw videos)")
        sample_footage_dir = cfg.project_root / "assets" / "sample_footage"
        if sample_footage_dir.exists():
            self.edit_assets.setText(str(sample_footage_dir))
        self.edit_assets.textChanged.connect(lambda: self._update_assets_scan_summary())

        # Backwards compatibility aliases
        self.edit_images = self.edit_assets
        self.edit_videos = self.edit_assets

        btn_browse_assets = QPushButton("Browse Folder")
        btn_browse_assets.setObjectName("secondaryButton")
        btn_browse_assets.setFixedWidth(110)
        btn_browse_assets.clicked.connect(self._browse_assets_folder)

        btn_open_assets = QPushButton("Open")
        btn_open_assets.setObjectName("secondaryButton")
        btn_open_assets.setFixedWidth(60)
        btn_open_assets.clicked.connect(lambda: self._open_in_explorer(self.edit_assets.text()))

        assets_row.addWidget(self.edit_assets, stretch=1)
        assets_row.addWidget(btn_browse_assets)
        assets_row.addWidget(btn_open_assets)
        assets_layout.addLayout(assets_row)

        # Live Folder Scan Summary Badge right below path input
        self.lbl_assets_badge = QLabel("Scanning folder...")
        self.lbl_assets_badge.setStyleSheet(
            "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
            "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
        )
        assets_layout.addWidget(self.lbl_assets_badge)

        # Motion pill toggle
        self.chk_include_images = QCheckBox("Apply Ken Burns Cinematic Zoom & Pan on static images")
        self.chk_include_images.setChecked(True)
        assets_layout.addWidget(self.chk_include_images)

        media_layout.addWidget(assets_box)
        dock_layout.addWidget(card_media)

        # ── Card: Template 3 Interview & Framing Settings (Conditional) ───────
        self.card_interview = QFrame()
        self.card_interview.setObjectName("cardFrame")
        interview_layout = QVBoxLayout(self.card_interview)
        interview_layout.setContentsMargins(16, 14, 16, 16)
        interview_layout.setSpacing(12)

        # Header with toggle button
        h_box_interview = QHBoxLayout()
        step_interview_badge = QLabel("TEMPLATE 3")
        step_interview_badge.setStyleSheet(
            "background-color: #1E1B4B; color: #818CF8; border: 1px solid #4338CA; "
            "border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;"
        )
        step_interview_title = QLabel("Interview & Framing Settings")
        step_interview_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")

        self.btn_toggle_interview_card = QPushButton("▼")
        self.btn_toggle_interview_card.setFixedSize(24, 24)
        self.btn_toggle_interview_card.setStyleSheet(
            "background-color: #222228; color: #A1A1AA; border: 1px solid #32323D; "
            "border-radius: 4px; font-size: 10px; font-weight: bold;"
        )
        self.btn_toggle_interview_card.setToolTip("Toggle Interview & Framing Settings")
        self.btn_toggle_interview_card.clicked.connect(self._toggle_interview_panel)

        h_box_interview.addWidget(step_interview_badge)
        h_box_interview.addWidget(step_interview_title)
        h_box_interview.addStretch()
        h_box_interview.addWidget(self.btn_toggle_interview_card)
        interview_layout.addLayout(h_box_interview)

        # Collapsible Body Container
        self.interview_body = QWidget()
        body_layout = QVBoxLayout(self.interview_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        # a. Interview-Only Background Music
        box_music = QFrame()
        box_music.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        box_music_layout = QVBoxLayout(box_music)
        box_music_layout.setSpacing(8)

        music_header = QHBoxLayout()
        music_tag = QLabel("🎵 Interview Background Music")
        music_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        self.lbl_interview_music_badge = QLabel("Optional BGM")
        self.lbl_interview_music_badge.setStyleSheet(
            "background-color: #1E1B4B; color: #A5B4FC; border: 1px solid #4338CA; "
            "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
        )
        music_header.addWidget(music_tag)
        music_header.addStretch()
        music_header.addWidget(self.lbl_interview_music_badge)
        box_music_layout.addLayout(music_header)

        # File selector input with Browse and Clear buttons
        music_file_row = QHBoxLayout()
        music_file_row.setSpacing(6)
        self.edit_interview_music = QLineEdit()
        self.edit_interview_music.setPlaceholderText("Select background music audio (.mp3, .wav, .m4a)...")
        self.edit_interview_music.setReadOnly(True)

        self.btn_browse_interview_music = QPushButton("Browse")
        self.btn_browse_interview_music.setObjectName("secondaryButton")
        self.btn_browse_interview_music.setFixedHeight(30)
        self.btn_browse_interview_music.clicked.connect(self._browse_interview_music)

        self.btn_clear_interview_music = QPushButton("Clear")
        self.btn_clear_interview_music.setObjectName("secondaryButton")
        self.btn_clear_interview_music.setFixedHeight(30)
        self.btn_clear_interview_music.clicked.connect(self._clear_interview_music)

        music_file_row.addWidget(self.edit_interview_music, stretch=1)
        music_file_row.addWidget(self.btn_browse_interview_music)
        music_file_row.addWidget(self.btn_clear_interview_music)
        box_music_layout.addLayout(music_file_row)

        # Volume slider (0% to 100%, default 15%) with live percentage label
        vol_header = QHBoxLayout()
        vol_lbl = QLabel("Volume:")
        vol_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_interview_music_vol = QLabel("15%")
        self.lbl_interview_music_vol.setStyleSheet("font-size: 11px; font-weight: 700; color: #818CF8;")
        vol_header.addWidget(vol_lbl)
        vol_header.addStretch()
        vol_header.addWidget(self.lbl_interview_music_vol)
        box_music_layout.addLayout(vol_header)

        self.slider_interview_music_vol = QSlider(Qt.Horizontal)
        self.slider_interview_music_vol.setRange(0, 100)
        self.slider_interview_music_vol.setValue(15)
        self.slider_interview_music_vol.valueChanged.connect(
            lambda val: self.lbl_interview_music_vol.setText(f"{val}%")
        )
        box_music_layout.addWidget(self.slider_interview_music_vol)
        body_layout.addWidget(box_music)

        # b. Background Image & Blur Section (Template 3 Exclusive)
        box_bg = QFrame()
        box_bg.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        box_bg_layout = QVBoxLayout(box_bg)
        box_bg_layout.setSpacing(10)

        bg_header = QHBoxLayout()
        bg_tag = QLabel("🖼️ Background Image & Blur")
        bg_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        self.lbl_interview_bg_badge = QLabel("Template 3")
        self.lbl_interview_bg_badge.setStyleSheet(
            "background-color: #1E1B4B; color: #A5B4FC; border: 1px solid #4338CA; "
            "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
        )
        bg_header.addWidget(bg_tag)
        bg_header.addStretch()
        bg_header.addWidget(self.lbl_interview_bg_badge)
        box_bg_layout.addLayout(bg_header)

        # File selector row with Thumbnail preview, Path/filename line edit, Browse and Clear buttons
        bg_file_row = QHBoxLayout()
        bg_file_row.setSpacing(8)

        # Thumbnail preview label
        self.lbl_interview_bg_preview = QLabel("No Image")
        self.lbl_interview_bg_preview.setFixedSize(68, 44)
        self.lbl_interview_bg_preview.setAlignment(Qt.AlignCenter)
        self.lbl_interview_bg_preview.setStyleSheet(
            "background-color: #0E0E12; border: 1px dashed #3F3F46; border-radius: 4px; color: #71717A; font-size: 10px; font-weight: 600;"
        )
        bg_file_row.addWidget(self.lbl_interview_bg_preview)

        bg_input_col = QVBoxLayout()
        bg_input_col.setSpacing(4)

        self.edit_interview_bg_image = QLineEdit()
        self.edit_interview_bg_image.setPlaceholderText("Select custom background image (.png, .jpg, .jpeg, .webp)...")
        self.edit_interview_bg_image.setReadOnly(True)

        bg_btn_row = QHBoxLayout()
        bg_btn_row.setSpacing(6)
        self.btn_browse_interview_bg = QPushButton("Browse")
        self.btn_browse_interview_bg.setObjectName("secondaryButton")
        self.btn_browse_interview_bg.setFixedHeight(28)
        self.btn_browse_interview_bg.clicked.connect(self._browse_interview_bg)

        self.btn_clear_interview_bg = QPushButton("Clear")
        self.btn_clear_interview_bg.setObjectName("secondaryButton")
        self.btn_clear_interview_bg.setFixedHeight(28)
        self.btn_clear_interview_bg.clicked.connect(self._clear_interview_bg)

        bg_btn_row.addWidget(self.edit_interview_bg_image, stretch=1)
        bg_btn_row.addWidget(self.btn_browse_interview_bg)
        bg_btn_row.addWidget(self.btn_clear_interview_bg)
        bg_input_col.addLayout(bg_btn_row)
        bg_file_row.addLayout(bg_input_col, stretch=1)

        box_bg_layout.addLayout(bg_file_row)

        # Blur slider (0% to 100%, default 0%) with live numeric readout
        blur_header = QHBoxLayout()
        blur_lbl = QLabel("Background Blur:")
        blur_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_interview_blur = QLabel("0%")
        self.lbl_interview_blur.setStyleSheet("font-size: 11px; font-weight: 700; color: #38BDF8;")
        blur_header.addWidget(blur_lbl)
        blur_header.addStretch()
        blur_header.addWidget(self.lbl_interview_blur)
        box_bg_layout.addLayout(blur_header)

        self.slider_interview_blur = QSlider(Qt.Horizontal)
        self.slider_interview_blur.setRange(0, 100)
        self.slider_interview_blur.setValue(0)
        self.slider_interview_blur.valueChanged.connect(
            lambda val: self.lbl_interview_blur.setText(f"{val}%")
        )
        box_bg_layout.addWidget(self.slider_interview_blur)
        body_layout.addWidget(box_bg)

        # c. Mobile / Vertical Clip Framing Controls
        box_framing = QFrame()
        box_framing.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        box_framing_layout = QVBoxLayout(box_framing)
        box_framing_layout.setSpacing(10)

        framing_tag = QLabel("📱 Mobile / Vertical Clip Framing")
        framing_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        box_framing_layout.addWidget(framing_tag)

        # 1. Interview Clip Scale (10% to 100%, default 70%)
        scale_header = QHBoxLayout()
        scale_lbl = QLabel("Interview Clip Scale:")
        scale_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_interview_scale = QLabel("70%")
        self.lbl_interview_scale.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")
        scale_header.addWidget(scale_lbl)
        scale_header.addStretch()
        scale_header.addWidget(self.lbl_interview_scale)
        box_framing_layout.addLayout(scale_header)

        self.slider_interview_scale = QSlider(Qt.Horizontal)
        self.slider_interview_scale.setRange(10, 100)
        self.slider_interview_scale.setValue(70)
        self.slider_interview_scale.valueChanged.connect(
            lambda val: self.lbl_interview_scale.setText(f"{val}%")
        )
        box_framing_layout.addWidget(self.slider_interview_scale)

        # 2. Frame Border: Checkbox (Border), clickable color swatch, Line Thickness slider (0px to 20px, default 4px)
        border_header = QHBoxLayout()
        self.chk_interview_border = QCheckBox("Border")
        self.chk_interview_border.setChecked(True)
        self.chk_interview_border.setStyleSheet("font-size: 11px; font-weight: 600; color: #F4F4F5;")

        self.btn_interview_border_color = QPushButton()
        self.btn_interview_border_color.setFixedSize(26, 26)
        self.btn_interview_border_color.setStyleSheet(
            f"background-color: {self._interview_border_color}; border: 1.5px solid #334155; border-radius: 5px;"
        )
        self.btn_interview_border_color.setToolTip("Frame Border Color")
        self.btn_interview_border_color.clicked.connect(self._pick_interview_border_color)

        self.lbl_interview_border_thickness = QLabel("4 px")
        self.lbl_interview_border_thickness.setStyleSheet("font-size: 11px; font-weight: 700; color: #EF4444;")

        border_header.addWidget(self.chk_interview_border)
        border_header.addSpacing(6)
        border_header.addWidget(self.btn_interview_border_color)
        border_header.addStretch()
        border_header.addWidget(self.lbl_interview_border_thickness)
        box_framing_layout.addLayout(border_header)

        self.slider_interview_border_thickness = QSlider(Qt.Horizontal)
        self.slider_interview_border_thickness.setRange(0, 20)
        self.slider_interview_border_thickness.setValue(4)
        self.slider_interview_border_thickness.valueChanged.connect(
            lambda val: self.lbl_interview_border_thickness.setText(f"{val} px")
        )
        box_framing_layout.addWidget(self.slider_interview_border_thickness)
        self.chk_interview_border.toggled.connect(self._on_interview_border_toggled)

        # 4. Corner Shape: Toggle button pair (Sharp Corners / Rounded Corners) and Corner Curve slider
        corner_row = QHBoxLayout()
        corner_lbl = QLabel("Corner Shape:")
        corner_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")

        self.btn_corner_sharp = QPushButton("Sharp Corners")
        self.btn_corner_sharp.setFixedHeight(30)
        self.btn_corner_sharp.setMinimumWidth(115)
        self.btn_corner_sharp.setStyleSheet(
            "background-color: #1E1E24; color: #A1A1AA; border: 1px solid #32323D; "
            "border-radius: 6px; font-weight: 600; font-size: 11px; padding: 4px 12px;"
        )
        self.btn_corner_sharp.clicked.connect(lambda: self._set_corner_shape("sharp"))

        self.btn_corner_rounded = QPushButton("Rounded Corners")
        self.btn_corner_rounded.setFixedHeight(30)
        self.btn_corner_rounded.setMinimumWidth(125)
        self.btn_corner_rounded.setStyleSheet(
            "background-color: #2A201A; color: #FB923C; border: 1.5px solid #EA580C; "
            "border-radius: 6px; font-weight: 800; font-size: 11px; padding: 4px 12px;"
        )
        self.btn_corner_rounded.clicked.connect(lambda: self._set_corner_shape("rounded"))

        self.lbl_interview_curve = QLabel("24 px")
        self.lbl_interview_curve.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")

        corner_row.addWidget(corner_lbl)
        corner_row.addSpacing(6)
        corner_row.addWidget(self.btn_corner_sharp)
        corner_row.addWidget(self.btn_corner_rounded)
        corner_row.addStretch()
        corner_row.addWidget(self.lbl_interview_curve)
        box_framing_layout.addLayout(corner_row)

        self.slider_interview_curve = QSlider(Qt.Horizontal)
        self.slider_interview_curve.setRange(0, 60)
        self.slider_interview_curve.setValue(24)
        self.slider_interview_curve.valueChanged.connect(
            lambda val: self.lbl_interview_curve.setText(f"{val} px")
        )
        box_framing_layout.addWidget(self.slider_interview_curve)
        body_layout.addWidget(box_framing)

        # c. Interview Clips & YouTube Link Bar
        box_clips = QFrame()
        box_clips.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        box_clips_layout = QVBoxLayout(box_clips)
        box_clips_layout.setSpacing(8)

        clips_tag = QLabel("🎬 Interview Clips & YouTube Link Bar")
        clips_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        box_clips_layout.addWidget(clips_tag)

        # Summary box with Browse Clips button
        clips_row = QHBoxLayout()
        clips_row.setSpacing(6)
        self.edit_interview_clips = QLineEdit()
        self.edit_interview_clips.setPlaceholderText("No interview clips added yet...")
        self.edit_interview_clips.setReadOnly(True)

        self.btn_browse_interview_clips = QPushButton("Browse Clips")
        self.btn_browse_interview_clips.setObjectName("secondaryButton")
        self.btn_browse_interview_clips.setFixedHeight(30)
        self.btn_browse_interview_clips.clicked.connect(self._browse_interview_clips)

        self.btn_review_interview_clips = QPushButton("Review / Edit Clips")
        self.btn_review_interview_clips.setObjectName("secondaryButton")
        self.btn_review_interview_clips.setFixedHeight(30)
        self.btn_review_interview_clips.setEnabled(False)
        self.btn_review_interview_clips.setToolTip("Select, trim start/end times, and set timeline placement")
        self.btn_review_interview_clips.clicked.connect(self._open_interview_selector_dialog)

        clips_row.addWidget(self.edit_interview_clips, stretch=1)
        clips_row.addWidget(self.btn_browse_interview_clips)
        clips_row.addWidget(self.btn_review_interview_clips)
        box_clips_layout.addLayout(clips_row)

        # Paste Comp YouTube Video Link text input paired with a primary Scan Clips button
        yt_row = QHBoxLayout()
        yt_row.setSpacing(6)
        self.edit_youtube_link = QLineEdit()
        self.edit_youtube_link.setPlaceholderText("Paste YouTube link (e.g. https://www.youtube.co...")

        self.chk_interview_relax_filter = QCheckBox("Include all scenes")
        self.chk_interview_relax_filter.setToolTip("Disable strict face/interview filtering and include all detected video scenes/cuts.")
        self.chk_interview_relax_filter.setStyleSheet("font-size: 11px; color: #94A3B8;")

        self.btn_scan_youtube = QPushButton("Scan Clips")
        self.btn_scan_youtube.setFixedHeight(30)
        self.btn_scan_youtube.setStyleSheet(
            "background-color: #EA580C; color: #FFFFFF; font-weight: 700; border-radius: 5px; padding: 4px 14px; font-size: 11px;"
        )
        self.btn_scan_youtube.clicked.connect(self._on_scan_youtube_clicked)

        yt_row.addWidget(self.edit_youtube_link, stretch=1)
        yt_row.addWidget(self.chk_interview_relax_filter)
        yt_row.addWidget(self.btn_scan_youtube)
        box_clips_layout.addLayout(yt_row)

        self.bar_interview_scan = QProgressBar()
        self.bar_interview_scan.setFixedHeight(4)
        self.bar_interview_scan.setRange(0, 100)
        self.bar_interview_scan.setValue(0)
        self.bar_interview_scan.setTextVisible(False)
        self.bar_interview_scan.setVisible(False)
        box_clips_layout.addWidget(self.bar_interview_scan)

        self.lbl_interview_scan_status = QLabel("")
        self.lbl_interview_scan_status.setStyleSheet("font-size: 11px; color: #38BDF8; font-weight: 600;")
        self.lbl_interview_scan_status.setVisible(False)
        box_clips_layout.addWidget(self.lbl_interview_scan_status)

        body_layout.addWidget(box_clips)

        interview_layout.addWidget(self.interview_body)
        dock_layout.addWidget(self.card_interview)
        self.card_interview.setVisible(False)

        # ── Card 3: Timing & Media Mix ────────────────────────────────────────
        card_timing = QFrame()
        card_timing.setObjectName("cardFrame")
        timing_layout = QVBoxLayout(card_timing)
        timing_layout.setContentsMargins(16, 14, 16, 16)
        timing_layout.setSpacing(12)

        h_box_timing = QHBoxLayout()
        step_timing_badge = QLabel("STEP 3")
        step_timing_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;")
        step_timing_title = QLabel("Timing & Media Mix Engine")
        step_timing_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")
        h_box_timing.addWidget(step_timing_badge)
        h_box_timing.addWidget(step_timing_title)
        h_box_timing.addStretch()
        timing_layout.addLayout(h_box_timing)

        # Control 1: Clip / Image Hold Duration (sec)
        dur_header = QHBoxLayout()
        dur_lbl = QLabel("Clip / Image Hold Duration:")
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

        dur_hint = QLabel("Decouples visual changes from short subtitle words for smooth cinematic pace (1s to 20s).")
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

        mix_hint = QLabel("Controls ratio of Ken Burns character stills vs. sliced raw episode video clips.")
        mix_hint.setStyleSheet("font-size: 11px; color: #71717A;")
        timing_layout.addWidget(mix_hint)

        # Control 3: Scene Transition Sound Effects (SFX)
        box_sfx = QFrame()
        box_sfx.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        box_sfx_layout = QVBoxLayout(box_sfx)
        box_sfx_layout.setSpacing(8)

        sfx_header = QHBoxLayout()
        sfx_title = QLabel("🔊 Scene Transition Sound Effects (SFX)")
        sfx_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        self.lbl_sfx_badge = QLabel("6 Active")
        self.lbl_sfx_badge.setStyleSheet(
            "background-color: #1E1B4B; color: #818CF8; border: 1px solid #4338CA; "
            "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
        )
        sfx_header.addWidget(sfx_title)
        sfx_header.addStretch()
        sfx_header.addWidget(self.lbl_sfx_badge)
        box_sfx_layout.addLayout(sfx_header)

        # Actions row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        lbl_sfx_info = QLabel("Random non-repeating placement across scene cuts:")
        lbl_sfx_info.setStyleSheet("font-size: 11px; color: #71717A;")

        btn_sfx_select_all = QPushButton("Select All")
        btn_sfx_select_all.setObjectName("secondaryButton")
        btn_sfx_select_all.setFixedHeight(24)
        btn_sfx_select_all.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_sfx_select_all.clicked.connect(self._select_all_sfx)

        btn_sfx_clear_all = QPushButton("Clear All")
        btn_sfx_clear_all.setObjectName("secondaryButton")
        btn_sfx_clear_all.setFixedHeight(24)
        btn_sfx_clear_all.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_sfx_clear_all.clicked.connect(self._clear_all_sfx)

        actions_row.addWidget(lbl_sfx_info)
        actions_row.addStretch()
        actions_row.addWidget(btn_sfx_select_all)
        actions_row.addWidget(btn_sfx_clear_all)
        box_sfx_layout.addLayout(actions_row)

        # 6 Curated Sound Effects checkboxes
        from services.sfx_manager import CURATED_SFX_CATALOG
        self._sfx_checkboxes: dict[str, QCheckBox] = {}
        sfx_grid = QGridLayout()
        sfx_grid.setSpacing(6)

        for idx, item in enumerate(CURATED_SFX_CATALOG):
            cb = QCheckBox(item["name"])
            cb.setChecked(True)
            cb.setStyleSheet("font-size: 11px; font-weight: 600; color: #E2E8F0;")
            cb.setToolTip(item["description"])
            cb.toggled.connect(self._update_sfx_badge)
            self._sfx_checkboxes[item["id"]] = cb
            r = idx // 3
            c = idx % 3
            sfx_grid.addWidget(cb, r, c)

        box_sfx_layout.addLayout(sfx_grid)

        # Volume / Intensity control slider
        vol_header = QHBoxLayout()
        vol_lbl = QLabel("SFX Volume / Intensity:")
        vol_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_sfx_vol_val = QLabel("35%")
        self.lbl_sfx_vol_val.setStyleSheet("font-size: 11px; font-weight: 700; color: #38BDF8;")
        vol_header.addWidget(vol_lbl)
        vol_header.addStretch()
        vol_header.addWidget(self.lbl_sfx_vol_val)
        box_sfx_layout.addLayout(vol_header)

        self.slider_sfx_vol = QSlider(Qt.Horizontal)
        self.slider_sfx_vol.setRange(0, 100)
        self.slider_sfx_vol.setValue(35)
        self.slider_sfx_vol.valueChanged.connect(lambda v: self.lbl_sfx_vol_val.setText(f"{v}%"))
        box_sfx_layout.addWidget(self.slider_sfx_vol)

        timing_layout.addWidget(box_sfx)

        dock_layout.addWidget(card_timing)

        # ── Card 4: Caption Typography Studio ────────────────────────────────
        card_typo = QFrame()
        card_typo.setObjectName("cardFrame")
        typo_layout = QVBoxLayout(card_typo)
        typo_layout.setContentsMargins(16, 14, 16, 16)
        typo_layout.setSpacing(12)

        h_box3 = QHBoxLayout()
        step3_badge = QLabel("STEP 4")
        step3_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;")
        step3_title = QLabel("Subtitles & Typography Studio")
        step3_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")
        h_box3.addWidget(step3_badge)
        h_box3.addWidget(step3_title)
        h_box3.addStretch()
        typo_layout.addLayout(h_box3)

        # Quick Presets Row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(10)
        p_lbl = QLabel("Style Preset:")
        p_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_quick_presets = QComboBox()
        self.combo_quick_presets.setView(_make_combo_view())
        self.combo_quick_presets.addItems([
            "Classic YouTube (Bold Yellow Highlight)",
            "Viral Shorts (Yellow + Box)",
            "High-Contrast Box (White + Solid Black)",
            "Minimal Clean (White + Outline)",
            "Cinematic Teal"
        ])
        self.combo_quick_presets.currentIndexChanged.connect(self._on_quick_preset_changed)
        preset_row.addWidget(p_lbl)
        preset_row.addWidget(self.combo_quick_presets, stretch=1)
        typo_layout.addLayout(preset_row)

        # Aa Quick Swatches Row
        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        swatches = [
            ("#FFFF00", "#000000", "Yellow"),
            ("#FFFFFF", "#000000", "White"),
            ("#38BDF8", "#000000", "Cyan"),
            ("#34D399", "#000000", "Green"),
            ("#FB923C", "#000000", "Coral"),
            ("#A78BFA", "#000000", "Violet")
        ]
        for font_c, stroke_c, lbl in swatches:
            btn_s = QPushButton(f"Aa\n{lbl}")
            btn_s.setFixedSize(54, 40)
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
        typo_layout.addLayout(swatch_row)

        # Font & Style Grid
        font_grid = QGridLayout()
        font_grid.setSpacing(10)

        # Font family
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
            "Arial"
        ])
        self.combo_font.currentTextChanged.connect(lambda: self._update_caption_preview())
        font_grid.addWidget(ff_lbl, 0, 0)
        font_grid.addWidget(self.combo_font, 1, 0)

        # Background Box Mode
        bg_lbl = QLabel("Background Box Mode")
        bg_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_bg_style = QComboBox()
        self.combo_bg_style.setView(_make_combo_view())
        self.combo_bg_style.addItems(["Outline Mode", "Semi-Transparent Box", "Solid Box", "Custom Box"])
        self.combo_bg_style.currentIndexChanged.connect(self._on_bg_style_changed)
        font_grid.addWidget(bg_lbl, 0, 1)
        font_grid.addWidget(self.combo_bg_style, 1, 1)

        # Box Opacity
        opac_lbl = QLabel("Box Opacity")
        opac_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.combo_box_opacity = QComboBox()
        self.combo_box_opacity.setView(_make_combo_view())
        self.combo_box_opacity.addItems(["Semi-Transparent (75%)", "Solid (100%)", "Subtle (50%)"])
        self.combo_box_opacity.currentIndexChanged.connect(self._on_opacity_changed)
        font_grid.addWidget(opac_lbl, 0, 2)
        font_grid.addWidget(self.combo_box_opacity, 1, 2)

        typo_layout.addLayout(font_grid)

        # Font Size Slider Row
        fs_row = QHBoxLayout()
        fs_lbl = QLabel("Font Size:")
        fs_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_font_size_val = QLabel(f"{self._font_size} pt")
        self.lbl_font_size_val.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")
        self.slider_font_size = QSlider(Qt.Horizontal)
        self.slider_font_size.setRange(32, 90)
        self.slider_font_size.setValue(self._font_size)
        self.slider_font_size.valueChanged.connect(self._on_font_size_changed)

        fs_row.addWidget(fs_lbl)
        fs_row.addWidget(self.slider_font_size, stretch=1)
        fs_row.addWidget(self.lbl_font_size_val)
        typo_layout.addLayout(fs_row)

        # Caption Length Slider Row
        len_row = QHBoxLayout()
        len_lbl = QLabel("Caption Length:")
        len_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_caption_length = QLabel("4 words / line")
        self.lbl_caption_length.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")
        self.slider_caption_length = QSlider(Qt.Horizontal)
        self.slider_caption_length.setRange(2, 10)
        self.slider_caption_length.setValue(4)
        self.slider_caption_length.valueChanged.connect(
            lambda v: self.lbl_caption_length.setText(f"{v} words / line")
        )
        len_row.addWidget(len_lbl)
        len_row.addWidget(self.slider_caption_length, stretch=1)
        len_row.addWidget(self.lbl_caption_length)
        typo_layout.addLayout(len_row)

        # Pattern, Case, and Color Pickers
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        # Pattern buttons (B, I, U)
        pat_box = QHBoxLayout()
        pat_box.setSpacing(4)
        self.btn_bold = QPushButton("B")
        self.btn_bold.setFixedSize(30, 28)
        self.btn_bold.setStyleSheet(BTN_ACTIVE_STYLE)
        self.btn_bold.clicked.connect(lambda: self._toggle_pattern_btn(self.btn_bold))

        self.btn_italic = QPushButton("I")
        self.btn_italic.setFixedSize(30, 28)
        self.btn_italic.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_italic.clicked.connect(lambda: self._toggle_pattern_btn(self.btn_italic))

        self.btn_underline = QPushButton("U")
        self.btn_underline.setFixedSize(30, 28)
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
        self.btn_case_upper.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_case_upper.clicked.connect(lambda: self._set_case("upper"))

        self.btn_case_lower = QPushButton("tt")
        self.btn_case_lower.setFixedSize(30, 28)
        self.btn_case_lower.setStyleSheet(BTN_INACTIVE_STYLE)
        self.btn_case_lower.clicked.connect(lambda: self._set_case("lower"))

        self.btn_case_title = QPushButton("Tt")
        self.btn_case_title.setFixedSize(30, 28)
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
        self.btn_color_swatch = QPushButton()
        self.btn_color_swatch.setFixedSize(26, 26)
        self.btn_color_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_color_swatch.setToolTip("Font Text Color")
        self.btn_color_swatch.clicked.connect(self._pick_font_color)

        hl_lbl = QLabel("Word:")
        hl_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        self.btn_hl_swatch = QPushButton()
        self.btn_hl_swatch.setFixedSize(26, 26)
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_hl_swatch.setToolTip("Active Word Highlight Color")
        self.btn_hl_swatch.clicked.connect(self._pick_highlight_color)

        bg_col_lbl = QLabel("Box:")
        bg_col_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        self.btn_bg_swatch = QPushButton()
        self.btn_bg_swatch.setFixedSize(26, 26)
        self.btn_bg_swatch.setStyleSheet(f"background-color: {self._bg_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_bg_swatch.setToolTip("Background Box Fill Color")
        self.btn_bg_swatch.clicked.connect(self._pick_bg_color)

        col_box.addWidget(col_lbl)
        col_box.addWidget(self.btn_color_swatch)
        col_box.addWidget(hl_lbl)
        col_box.addWidget(self.btn_hl_swatch)
        col_box.addWidget(bg_col_lbl)
        col_box.addWidget(self.btn_bg_swatch)
        controls_row.addLayout(col_box)

        controls_row.addStretch()
        typo_layout.addLayout(controls_row)

        # Stroke Outline Controls
        stroke_row = QHBoxLayout()
        stroke_row.setSpacing(10)
        s_lbl = QLabel("Stroke:")
        s_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.btn_stroke_swatch = QPushButton()
        self.btn_stroke_swatch.setFixedSize(26, 26)
        self.btn_stroke_swatch.setStyleSheet(f"background-color: {self._stroke_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_stroke_swatch.clicked.connect(self._pick_stroke_color)

        self.slider_stroke = QSlider(Qt.Horizontal)
        self.slider_stroke.setRange(1, 10)
        self.slider_stroke.setValue(4)
        self.slider_stroke.valueChanged.connect(self._on_stroke_changed)

        self.lbl_stroke_val = QLabel(f"{self._stroke_thickness:.1f} px")
        self.lbl_stroke_val.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")

        stroke_row.addWidget(s_lbl)
        stroke_row.addWidget(self.btn_stroke_swatch)
        stroke_row.addWidget(self.slider_stroke, stretch=1)
        stroke_row.addWidget(self.lbl_stroke_val)
        typo_layout.addLayout(stroke_row)

        # Box Padding Slider Row
        pad_row = QHBoxLayout()
        pad_row.setSpacing(10)
        p_lbl = QLabel("Box Padding:")
        p_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.slider_box_padding = QSlider(Qt.Horizontal)
        self.slider_box_padding.setRange(6, 28)
        self.slider_box_padding.setValue(int(self._box_padding))
        self.slider_box_padding.valueChanged.connect(self._on_box_padding_changed)
        self.lbl_box_pad_val = QLabel(f"{int(self._box_padding)} px")
        self.lbl_box_pad_val.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")

        pad_row.addWidget(p_lbl)
        pad_row.addWidget(self.slider_box_padding, stretch=1)
        pad_row.addWidget(self.lbl_box_pad_val)
        typo_layout.addLayout(pad_row)
        typo_layout.addLayout(stroke_row)

        # YouTube SEO Chapters checkbox
        chap_row = QHBoxLayout()
        self.chk_chapters = QCheckBox("Auto-Generate YouTube Chapters (Timestamps .txt for Description SEO)")
        self.chk_chapters.setChecked(True)
        self.btn_clear_completed = QPushButton("Clear Completed")
        self.btn_clear_completed.setObjectName("ghostButton")
        self.btn_clear_completed.clicked.connect(self._reset_progress)
        chap_row.addWidget(self.chk_chapters)
        chap_row.addStretch()
        chap_row.addWidget(self.btn_clear_completed)
        typo_layout.addLayout(chap_row)

        dock_layout.addWidget(card_typo)
        dock_layout.addStretch()

        scroll_area.setWidget(dock_container)
        root_layout.addWidget(scroll_area, stretch=46)

        # ══════════════════════════════════════════════════════════════════════
        # ZONE 2: RIGHT STAGE (Sticky Cinema Preview & Render Center, 54% width)
        # ══════════════════════════════════════════════════════════════════════
        stage_card = QFrame()
        stage_card.setObjectName("cardFrame")
        stage_layout = QVBoxLayout(stage_card)
        stage_layout.setContentsMargins(18, 16, 18, 18)
        stage_layout.setSpacing(14)

        # ── 1. Live 16:9 Cinema Preview Stage ─────────────────────────────────
        preview_header = QHBoxLayout()
        prev_title = QLabel("Cinema Subtitle Live Stage")
        prev_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F4F4F5;")
        res_badge = QLabel("2K QHD (2560x1440)")
        res_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 8px;")
        preview_header.addWidget(prev_title)
        preview_header.addStretch()
        preview_header.addWidget(res_badge)
        stage_layout.addLayout(preview_header)

        # Cinema Stage Screen (16:9 vignette canvas)
        self.stage_screen = QFrame()
        self.stage_screen.setMinimumHeight(240)
        self.stage_screen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stage_screen.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #09090D, stop:0.5 #121218, stop:1 #09090D);
                border: 1.5px solid #2C2C35;
                border-radius: 10px;
            }
        """)
        stage_screen_layout = QVBoxLayout(self.stage_screen)
        stage_screen_layout.setContentsMargins(20, 20, 20, 30)
        stage_screen_layout.addStretch()

        # Dynamic Subtitle Overlay Label
        self.lbl_preview_text = QLabel("Princess Anne's Untold Decision")
        self.lbl_preview_text.setAlignment(Qt.AlignCenter)
        self.lbl_preview_text.setWordWrap(True)
        stage_screen_layout.addWidget(self.lbl_preview_text)
        stage_layout.addWidget(self.stage_screen, stretch=1)

        # ── 2. Output Destination & Resolution Bar ──────────────────────────────
        out_box = QFrame()
        out_box.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 6px 12px;")
        out_layout = QHBoxLayout(out_box)
        out_layout.setSpacing(8)

        out_lbl = QLabel("Output:")
        out_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")
        self.edit_output = QLineEdit()
        self.edit_output.setPlaceholderText("Select output .mp4 destination")
        default_out = cfg.output_dir / "autocut_studio_render.mp4"
        self.edit_output.setText(str(default_out))

        btn_browse_out = QPushButton("Change")
        btn_browse_out.setObjectName("secondaryButton")
        btn_browse_out.setFixedHeight(28)
        btn_browse_out.clicked.connect(self._browse_output)

        btn_open_out = QPushButton("📁")
        btn_open_out.setObjectName("secondaryButton")
        btn_open_out.setFixedSize(28, 28)
        btn_open_out.clicked.connect(lambda: self._open_in_explorer(self.edit_output.text()))

        res_lbl = QLabel("Resolution:")
        res_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C; margin-left: 6px;")
        self.combo_resolution = QComboBox()
        self.combo_resolution.setView(_make_combo_view())
        self.combo_resolution.addItems([
            "1080p Full HD (1920x1080)",
            "2K QHD (2560x1440)",
            "720p HD (1280x720)",
        ])
        self.combo_resolution.setCurrentIndex(0)
        self.combo_resolution.setFixedHeight(28)

        out_layout.addWidget(out_lbl)
        out_layout.addWidget(self.edit_output, stretch=1)
        out_layout.addWidget(btn_browse_out)
        out_layout.addWidget(btn_open_out)
        out_layout.addWidget(res_lbl)
        out_layout.addWidget(self.combo_resolution)
        stage_layout.addWidget(out_box)

        # ── 3. Dual Pipeline Telemetry Cards ──────────────────────────────────
        prog_card = QFrame()
        prog_card.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 10px; padding: 12px 14px;")
        prog_layout = QVBoxLayout(prog_card)
        prog_layout.setSpacing(10)

        # Stage 1: AI Transcription
        vo_p_header = QHBoxLayout()
        vo_p_title = QLabel("Stage 1 • Speech-to-Text Voiceover Transcription")
        vo_p_title.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_vo_pct = QLabel("0%")
        self.lbl_vo_pct.setStyleSheet("font-size: 11px; font-weight: 700; color: #06B6D4;")
        vo_p_header.addWidget(vo_p_title)
        vo_p_header.addStretch()
        vo_p_header.addWidget(self.lbl_vo_pct)
        prog_layout.addLayout(vo_p_header)

        self.bar_transcription = QProgressBar()
        self.bar_transcription.setObjectName("audioProgress")
        self.bar_transcription.setFixedHeight(7)
        self.bar_transcription.setRange(0, 100)
        self.bar_transcription.setValue(0)
        prog_layout.addWidget(self.bar_transcription)

        # Stage 2: 2K Video Render
        exp_p_header = QHBoxLayout()
        exp_p_title = QLabel("Stage 2 • 2K Master Video & Subtitle Rendering")
        exp_p_title.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")
        self.lbl_exp_pct = QLabel("0%")
        self.lbl_exp_pct.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")
        exp_p_header.addWidget(exp_p_title)
        exp_p_header.addStretch()
        exp_p_header.addWidget(self.lbl_exp_pct)
        prog_layout.addLayout(exp_p_header)

        self.bar_render = QProgressBar()
        self.bar_render.setObjectName("renderProgress")
        self.bar_render.setFixedHeight(7)
        self.bar_render.setRange(0, 100)
        self.bar_render.setValue(0)
        prog_layout.addWidget(self.bar_render)

        # Status text
        self.lbl_status = QLabel("Ready • Select a voiceover or enter a video title to begin.")
        self.lbl_status.setStyleSheet("font-size: 12px; font-weight: 500; color: #D4D4D8;")
        prog_layout.addWidget(self.lbl_status)

        # Token usage banner (shown when Gemini AI tokens are consumed)
        self.lbl_token_usage_banner = QLabel()
        self.lbl_token_usage_banner.setStyleSheet(
            "background-color: #064E3B; color: #6EE7B7; border: 1px solid #059669; "
            "border-radius: 6px; font-size: 11px; font-weight: 600; padding: 6px 10px;"
        )
        self.lbl_token_usage_banner.setWordWrap(True)
        self.lbl_token_usage_banner.setVisible(False)
        prog_layout.addWidget(self.lbl_token_usage_banner)
        stage_layout.addWidget(prog_card)

        # ── 4. Primary Action Dock (Huge Pinned CTA Button) ─────────────────────
        self.btn_start = QPushButton("Start AutoCut")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F97316, stop:1 #FB923C);
                color: #FFFFFF;
                border: 1px solid #EA580C;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EA580C, stop:1 #F97316);
                border-color: #F97316;
            }
            QPushButton#primaryButton:disabled {
                background: #27272A;
                color: #71717A;
                border: 1px solid #3F3F46;
            }
        """)
        self.btn_start.clicked.connect(self._start_render)
        stage_layout.addWidget(self.btn_start)

        root_layout.addWidget(stage_card, stretch=54)

        # Initialize visual preview and asset scan summary
        self._update_caption_preview()
        self._update_assets_scan_summary()

    # ── Quick Preset and Color Handlers ────────────────────────────────────────

    def _on_quick_preset_changed(self, idx: int) -> None:
        if idx == 0:  # Classic YouTube (White + Outline)
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FFFF00"
            self._stroke_color = "#000000"
            self._stroke_thickness = 3.5
            self._has_bg_box = False
            self._bg_style = "outline"
            self.combo_bg_style.setCurrentIndex(0)
            self._font_size = 56
        elif idx == 1:  # Viral Shorts (Yellow + Semi-Transparent Box)
            self._font_color = "#FFFF00"
            self._highlight_color = "#00FFFF"
            self._stroke_color = "#000000"
            self._stroke_thickness = 2.0
            self._has_bg_box = True
            self._bg_style = "semi_transparent_black"
            self.combo_bg_style.setCurrentIndex(1)
            self._font_size = 58
        elif idx == 2:  # High-Contrast Box (Yellow + Solid Black Box)
            self._font_color = "#FFFF00"
            self._highlight_color = "#FFFF00"
            self._stroke_color = "#000000"
            self._stroke_thickness = 0.0
            self._has_bg_box = True
            self._bg_style = "solid_black"
            self.combo_bg_style.setCurrentIndex(2)
            self._font_size = 56
        elif idx == 3:  # Minimal Clean
            self._font_color = "#FFFFFF"
            self._highlight_color = "#FB923C"
            self._stroke_color = "#000000"
            self._stroke_thickness = 2.5
            self.combo_bg_style.setCurrentIndex(0)
            self._font_size = 52
        elif idx == 4:  # Cinematic Teal
            self._font_color = "#38BDF8"
            self._highlight_color = "#38BDF8"
            self._stroke_color = "#000000"
            self._stroke_thickness = 3.5
            self.combo_bg_style.setCurrentIndex(0)
            self._font_size = 56

        self.slider_font_size.setValue(self._font_size)
        self.btn_color_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self._update_caption_preview()

    def _on_bg_style_changed(self, idx: int) -> None:
        if idx == 0:
            self._bg_style = "outline"
            self._has_bg_box = False
        elif idx == 1:
            self._bg_style = "semi_transparent_black"
            self._has_bg_box = True
            self._bg_opacity = 0.75
            if hasattr(self, "combo_box_opacity"):
                self.combo_box_opacity.setCurrentIndex(0)
        elif idx == 2:
            self._bg_style = "solid_black"
            self._has_bg_box = True
            self._bg_opacity = 1.0
            if hasattr(self, "combo_box_opacity"):
                self.combo_box_opacity.setCurrentIndex(1)
        elif idx == 3:
            self._bg_style = "custom_box"
            self._has_bg_box = True
        self._update_caption_preview()

    def _on_opacity_changed(self, idx: int) -> None:
        opacities = [0.75, 1.0, 0.50]
        if 0 <= idx < len(opacities):
            self._bg_opacity = opacities[idx]
        self._update_caption_preview()

    def _pick_bg_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._bg_color), self, "Select Background Box Color")
        if color.isValid():
            self._bg_color = color.name().upper()
            self.btn_bg_swatch.setStyleSheet(f"background-color: {self._bg_color}; border: 1.5px solid #334155; border-radius: 5px;")
            self._update_caption_preview()

    def _on_box_padding_changed(self, val: int) -> None:
        self._box_padding = float(val)
        self.lbl_box_pad_val.setText(f"{val} px")
        self._update_caption_preview()

    def _on_font_size_changed(self, val: int) -> None:
        self._font_size = val
        self.lbl_font_size_val.setText(f"{val} pt")
        self._update_caption_preview()

    def _on_stroke_changed(self, val: int) -> None:
        self._stroke_thickness = val * 0.8
        self.lbl_stroke_val.setText(f"{self._stroke_thickness:.1f} px")
        self._update_caption_preview()

    def _apply_preset_color(self, font_color: str, stroke_color: str) -> None:
        self._font_color = font_color
        self._stroke_color = stroke_color
        self.btn_color_swatch.setStyleSheet(f"background-color: {font_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self.btn_stroke_swatch.setStyleSheet(f"background-color: {stroke_color}; border: 1.5px solid #334155; border-radius: 5px;")
        self._update_caption_preview()

    def _pick_font_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._font_color), self, "Select Subtitle Color")
        if color.isValid():
            self._font_color = color.name().upper()
            self.btn_color_swatch.setStyleSheet(f"background-color: {self._font_color}; border: 1.5px solid #334155; border-radius: 5px;")
            self._update_caption_preview()

    def _pick_highlight_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._highlight_color), self, "Select Word Highlight Color")
        if color.isValid():
            self._highlight_color = color.name().upper()
            self.btn_hl_swatch.setStyleSheet(f"background-color: {self._highlight_color}; border: 1.5px solid #334155; border-radius: 5px;")
            self._update_caption_preview()

    def _pick_stroke_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._stroke_color), self, "Select Outline Color")
        if color.isValid():
            self._stroke_color = color.name().upper()
            self.btn_stroke_swatch.setStyleSheet(f"background-color: {self._stroke_color}; border: 1.5px solid #334155; border-radius: 5px;")
            self._update_caption_preview()

    def _toggle_pattern_btn(self, btn: QPushButton) -> None:
        is_active = (btn.styleSheet() == BTN_ACTIVE_STYLE)
        btn.setStyleSheet(BTN_INACTIVE_STYLE if is_active else BTN_ACTIVE_STYLE)
        self._update_caption_preview()

    def _set_case(self, case_type: str) -> None:
        self._font_case = case_type
        self.btn_case_upper.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "upper" else BTN_INACTIVE_STYLE)
        self.btn_case_lower.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "lower" else BTN_INACTIVE_STYLE)
        self.btn_case_title.setStyleSheet(BTN_ACTIVE_STYLE if case_type == "title" else BTN_INACTIVE_STYLE)
        self._update_caption_preview()

    def _update_caption_preview(self) -> None:
        text = self.edit_title.text().strip() or "Princess Anne's Untold Decision"
        if len(text) > 34:
            text = text[:34] + "..."
        if self._font_case == "upper":
            text = text.upper()
        elif self._font_case == "lower":
            text = text.lower()
        elif self._font_case == "title":
            text = text.title()

        self.lbl_preview_text.setText(text)

        raw_font = self.combo_font.currentText().split("(")[0].strip()
        font_name = raw_font.replace(" Bold", "").strip() or "Montserrat"
        base_size = max(int(self._font_size * 0.36), 14)

        is_bold = (self.btn_bold.styleSheet() == BTN_ACTIVE_STYLE)
        is_italic = (self.btn_italic.styleSheet() == BTN_ACTIVE_STYLE)
        font_weight = "800" if is_bold else "normal"
        font_style = "italic" if is_italic else "normal"

        if self._has_bg_box:
            # Parse hex bg color to RGBA
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
                f"border: 1px solid rgba(255, 255, 255, 0.15); "
                f"border-radius: 6px; "
                f"padding: {pad_v}px {pad_h}px;"
            )
        else:  # Outline
            box_style = (
                f"background-color: transparent; "
                f"color: {self._font_color}; "
                f"border: 1.5px dashed #3F3F4E; "
                f"border-radius: 6px; "
                f"padding: 6px 18px;"
            )

        self.lbl_preview_text.setStyleSheet(f"""
            font-family: '{font_name}', 'Segoe UI', Arial, sans-serif;
            font-size: {base_size}px;
            font-weight: {font_weight};
            font-style: {font_style};
            letter-spacing: 0.5px;
            {box_style}
        """)

    # ── Browse & Audio Handlers ────────────────────────────────────────────────

    def _browse_audio(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Voiceover Audio Part(s)",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.aac);;All Files (*.*)",
        )
        if files:
            from services.character_matcher import natural_sort_key
            self._audio_paths = sorted([Path(f) for f in files], key=lambda p: natural_sort_key(p.name))
            if len(self._audio_paths) == 1:
                self.edit_audio.setText(str(self._audio_paths[0]))
            else:
                self.edit_audio.setText("; ".join(str(p) for p in self._audio_paths))

    def _add_more_parts(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Voiceover Audio Part(s)",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.aac);;All Files (*.*)",
        )
        if files:
            from services.character_matcher import natural_sort_key
            existing = list(self._audio_paths)
            for f in files:
                p = Path(f)
                if p not in existing:
                    existing.append(p)
            self._audio_paths = sorted(existing, key=lambda p: natural_sort_key(p.name))
            if len(self._audio_paths) == 1:
                self.edit_audio.setText(str(self._audio_paths[0]))
            else:
                self.edit_audio.setText("; ".join(str(p) for p in self._audio_paths))

    def _on_audio_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self._audio_paths = []
            self.lbl_vo_badge.setText("Select Audio")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
            return

        from services.character_matcher import natural_sort_key
        raw_parts = [p.strip() for p in text.split(";") if p.strip()]
        paths = sorted([Path(p) for p in raw_parts], key=lambda p: natural_sort_key(p.name))
        self._audio_paths = paths

        all_exist = all(p.exists() and p.is_file() for p in paths)
        if not all_exist or not paths:
            self.lbl_vo_badge.setText("File Not Found")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #2A1D16; color: #EF4444; border: 1px solid #7F1D1D; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
            return

        cfg = get_config()
        from core.ffmpeg_wrapper import probe_media
        import re

        if len(paths) == 1:
            p = paths[0]
            try:
                dur = probe_media(p).duration
                dur_str = f"{int(dur // 60):02d}:{int(dur % 60):02d}"
            except Exception:
                dur_str = "00:00"
            self.lbl_vo_badge.setText(f"✓ Single Audio: {p.name} ({dur_str})")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
            title = p.stem.replace("_", " ").title()
            self.edit_title.setText(title)
            self._refresh_output_path()
            self._update_caption_preview()
        else:
            # Multi-part mode
            total_dur = 0.0
            for p in paths:
                try:
                    total_dur += probe_media(p).duration
                except Exception:
                    pass
            mins = int(total_dur // 60)
            secs = int(total_dur % 60)
            dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

            first_name = paths[0].stem
            last_name = paths[-1].stem
            self.lbl_vo_badge.setText(
                f"✓ Multi-Part Mode: {len(paths)} Parts ({first_name} → {last_name} • Total: {dur_str}) — Auto-merges"
            )
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #1A1A3A; color: #818CF8; border: 1px solid #4338CA; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )

            # Derive clean default output filename based on the first part's stem (strip Part 1 / pt 1)
            base_stem = re.sub(r'[\s_\-]*(?:part|pt)[\s_\-]*\d+', '', paths[0].stem, flags=re.IGNORECASE).strip() or paths[0].stem
            title = base_stem.replace("_", " ").title()
            self.edit_title.setText(title)
            self._refresh_output_path()
            self._update_caption_preview()

    def _get_default_export_dir(self) -> Path:
        """Return the export subfolder within the selected assets dir, or global cfg.output_dir."""
        assets_text = self.edit_assets.text().strip() if hasattr(self, "edit_assets") else ""
        if assets_text:
            assets_path = Path(assets_text)
            if assets_path.exists() and assets_path.is_dir():
                export_dir = assets_path / "export"
                export_dir.mkdir(parents=True, exist_ok=True)
                return export_dir
        cfg = get_config()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        return cfg.output_dir

    def _refresh_output_path(self) -> None:
        """Route default output save path into the input folder's export/ subfolder."""
        if not hasattr(self, "edit_output"):
            return
        export_dir = self._get_default_export_dir()
        import re

        if getattr(self, "_audio_paths", None):
            paths = self._audio_paths
            if len(paths) == 1:
                filename = f"{paths[0].stem}.mp4"
            else:
                base_stem = re.sub(r'[\s_\-]*(?:part|pt)[\s_\-]*\d+', '', paths[0].stem, flags=re.IGNORECASE).strip() or paths[0].stem
                filename = f"{base_stem}_Full.mp4"
        elif hasattr(self, "edit_assets") and self.edit_assets.text().strip():
            folder = Path(self.edit_assets.text().strip())
            filename = f"{folder.name}_Final.mp4"
        else:
            filename = "autocut_studio_render.mp4"

        target_out = export_dir / filename
        self.edit_output.setText(str(target_out))

    def _browse_assets_folder(self) -> None:
        is_t2 = (getattr(self, "combo_format", None) is not None and self.combo_format.currentIndex() == 1)
        dialog_title = "Select Episode Footage Folder" if is_t2 else "Select Footage & Visual Assets Folder"
        dir_path = QFileDialog.getExistingDirectory(self, dialog_title)
        if dir_path:
            self.edit_assets.setText(dir_path)
            self._update_assets_scan_summary()

    def _update_assets_scan_summary(self) -> None:
        if not hasattr(self, "lbl_assets_badge"):
            return
        path_str = self.edit_assets.text().strip() if hasattr(self, "edit_assets") else ""
        if not path_str:
            self.lbl_assets_badge.setText("⚠️ Please select an assets directory")
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #2A1D16; color: #EF4444; border: 1px solid #7F1D1D; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )
            return

        folder = Path(path_str)
        if not folder.exists() or not folder.is_dir():
            self.lbl_assets_badge.setText(f"⚠️ Directory does not exist: {folder.name}")
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #2A1D16; color: #EF4444; border: 1px solid #7F1D1D; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )
            return

        # 1. Automatic export/ subfolder generation inside the selected folder
        export_dir = folder / "export"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not auto-create export directory: %s", exc)

        # 2. Template-aware asset scanning
        is_t2 = (getattr(self, "combo_format", None) is not None and self.combo_format.currentIndex() == 1)
        tmpl_key = "template_2" if is_t2 else "template_1"

        from services.character_matcher import CharacterMatcher
        summary = CharacterMatcher.scan_asset_folder(folder, template_type=tmpl_key)
        badge_text = summary.get("badge_text", "")
        self.lbl_assets_badge.setText(badge_text)

        has_assets = (summary.get("total_images", 0) > 0 or len(summary.get("video_files", [])) > 0)
        if has_assets:
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )
        else:
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #2A1D16; color: #F59E0B; border: 1px solid #78350F; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )

        # 3. Route default output save path into export/ folder
        self._refresh_output_path()

    def _update_template_status_badge(self) -> None:
        """Update subtle badge displaying Gemini AI vs Offline Heuristic matcher mode & lifetime tokens."""
        from database.db import get_lifetime_token_stats
        from utils.config import reload_config
        cfg = reload_config()
        format_idx = self.combo_format.currentIndex()
        is_t1 = (format_idx == 0)
        is_t2 = (format_idx == 1)
        is_t3 = (format_idx == 2 or "Template 3" in self.combo_format.currentText())

        api_key = os.getenv("GEMINI_API_KEY") or cfg.gemini_api_key or ""
        from services.gemini_matcher import SmartSceneMatcher
        has_valid_key = SmartSceneMatcher.is_api_key_valid(api_key)

        if hasattr(self, "btn_api_dashboard"):
            self.btn_api_dashboard.setVisible(is_t2)
        if hasattr(self, "box_t2_language"):
            self.box_t2_language.setVisible(is_t2)
        if hasattr(self, "card_interview"):
            self.card_interview.setVisible(is_t3)
        if hasattr(self, "assets_tag"):
            if is_t2:
                self.assets_tag.setText("🎬 Episode Footage Folder (Raw Episode Videos)")
            elif is_t3:
                self.assets_tag.setText("📁 Main Footage & Visual Assets Folder (Background Loop / Images)")
            else:
                self.assets_tag.setText("📁 Footage & Visual Assets Folder (Images & Videos)")

        # Re-scan asset summary for the selected template
        self._update_assets_scan_summary()

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
        elif is_t3:
            self.lbl_gemini_status.setText("🎙️ Template 3: Hybrid Media + Interview Overlay Engine Active")
            self.lbl_gemini_status.setStyleSheet(
                "background-color: #1E1B4B; color: #818CF8; border: 1px solid #4338CA; "
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

    def _on_clip_dur_changed(self, val: int) -> None:
        self.lbl_clip_dur_val.setText(f"{val}.0 s")

    def _on_mix_ratio_changed(self, val: int) -> None:
        self.lbl_mix_ratio_val.setText(f"Images: {val}% | Video Clips: {100 - val}%")

    def _browse_output(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output MP4 Destination", "", "MP4 Video (*.mp4)"
        )
        if file_path:
            if not file_path.endswith(".mp4"):
                file_path += ".mp4"
            self.edit_output.setText(file_path)

    def _open_in_explorer(self, path_str: str) -> None:
        p = Path(path_str)
        target = p if p.is_dir() else p.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        else:
            QMessageBox.warning(self, "Path Not Found", f"Directory does not exist:\n{target}")

    def _reset_progress(self) -> None:
        self.bar_transcription.setValue(0)
        self.bar_render.setValue(0)
        self.lbl_vo_pct.setText("0%")
        self.lbl_exp_pct.setText("0%")
        self.lbl_status.setText("Ready • Select a voiceover to begin.")

    # ── Render Worker Pipeline ─────────────────────────────────────────────────

    def get_selected_resolution(self) -> str:
        """Return '720p', '1080p', or '2k' from resolution selector."""
        txt = self.combo_resolution.currentText().lower()
        if "720p" in txt:
            return "720p"
        elif "2k" in txt:
            return "2k"
        return "1080p"

    def get_selected_language(self) -> str:
        """Return 'ur' or 'en' from the narration language selector."""
        if hasattr(self, "combo_language"):
            txt = self.combo_language.currentText().lower()
            if "ur" in txt or "اردو" in txt:
                return "ur"
        return "en"

    def _select_all_sfx(self) -> None:
        """Select all available transition sound effects."""
        for cb in getattr(self, "_sfx_checkboxes", {}).values():
            cb.setChecked(True)
        self._update_sfx_badge()

    def _clear_all_sfx(self) -> None:
        """Deselect all transition sound effects."""
        for cb in getattr(self, "_sfx_checkboxes", {}).values():
            cb.setChecked(False)
        self._update_sfx_badge()

    def _update_sfx_badge(self) -> None:
        """Update badge text and color to reflect active transition SFX count."""
        selected = self.get_selected_sfx_ids()
        count = len(selected)
        if not hasattr(self, "lbl_sfx_badge"):
            return
        if count == 0:
            self.lbl_sfx_badge.setText("Off (0)")
            self.lbl_sfx_badge.setStyleSheet(
                "background-color: #26262B; color: #71717A; border: 1px solid #3F3F46; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
        else:
            self.lbl_sfx_badge.setText(f"{count} Active")
            self.lbl_sfx_badge.setStyleSheet(
                "background-color: #1E1B4B; color: #818CF8; border: 1px solid #4338CA; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )

    def get_selected_sfx_ids(self) -> list[str]:
        """Return IDs of user-selected transition sound effects."""
        return [
            sfx_id for sfx_id, cb in getattr(self, "_sfx_checkboxes", {}).items()
            if cb.isChecked()
        ]

    def set_selected_sfx_ids(self, sfx_ids: Sequence[str]) -> None:
        """Programmatically select specific transition sound effects."""
        id_set = set(sfx_ids)
        for sfx_id, cb in getattr(self, "_sfx_checkboxes", {}).items():
            cb.setChecked(sfx_id in id_set)
        self._update_sfx_badge()

    def get_transition_sfx_config(self) -> dict:
        """Return serializable transition SFX configuration dict."""
        selected_ids = self.get_selected_sfx_ids()
        volume = float(self.slider_sfx_vol.value()) / 100.0 if hasattr(self, "slider_sfx_vol") else 0.35
        return {
            "enabled": len(selected_ids) > 0 and volume > 0.0,
            "selected_sfx": selected_ids,
            "volume": max(0.0, min(volume, 1.0)),
        }

    def _browse_interview_music(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Interview Background Music",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.aac *.flac);;All Files (*.*)",
        )
        if path:
            self.edit_interview_music.setText(path)

    def _clear_interview_music(self) -> None:
        self.edit_interview_music.clear()

    def _browse_interview_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*.*)",
        )
        if path:
            self.set_interview_bg_image(path)

    def _clear_interview_bg(self) -> None:
        self.set_interview_bg_image("")

    def set_interview_bg_image(self, path: str) -> None:
        clean_path = str(path or "").strip()
        self.edit_interview_bg_image.setText(clean_path)
        self._update_interview_bg_thumbnail(clean_path)

    def _update_interview_bg_thumbnail(self, path_str: str) -> None:
        if path_str and Path(path_str).is_file():
            pix = QPixmap(path_str)
            if not pix.isNull():
                scaled = pix.scaled(
                    64,
                    40,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.lbl_interview_bg_preview.setPixmap(scaled)
                self.lbl_interview_bg_preview.setText("")
                return
        self.lbl_interview_bg_preview.setPixmap(QPixmap())
        self.lbl_interview_bg_preview.setText("No Image")

    def set_interview_blur(self, val: int) -> None:
        self.slider_interview_blur.setValue(max(0, min(100, int(val))))

    def _pick_interview_border_color(self) -> None:
        from PySide6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(QColor(self._interview_border_color), self, "Select Border Color")
        if color.isValid():
            self._interview_border_color = color.name().upper()
            self.btn_interview_border_color.setStyleSheet(
                f"background-color: {self._interview_border_color}; border: 1px solid #52525B; border-radius: 4px;"
            )

    def _on_interview_border_toggled(self, checked: bool) -> None:
        self.btn_interview_border_color.setEnabled(checked)
        self.slider_interview_border_thickness.setEnabled(checked)
        self.lbl_interview_border_thickness.setEnabled(checked)

    def _set_corner_shape(self, shape: str) -> None:
        self._interview_corner_shape = shape
        if shape == "sharp":
            self.btn_corner_sharp.setStyleSheet(
                "background-color: #2A201A; color: #FB923C; border: 1.5px solid #EA580C; "
                "border-radius: 6px; font-weight: 800; font-size: 11px; padding: 4px 12px;"
            )
            self.btn_corner_rounded.setStyleSheet(
                "background-color: #1E1E24; color: #A1A1AA; border: 1px solid #32323D; "
                "border-radius: 6px; font-weight: 600; font-size: 11px; padding: 4px 12px;"
            )
            self.slider_interview_curve.setEnabled(False)
            self.lbl_interview_curve.setEnabled(False)
        else:
            self._interview_corner_shape = "rounded"
            self.btn_corner_rounded.setStyleSheet(
                "background-color: #2A201A; color: #FB923C; border: 1.5px solid #EA580C; "
                "border-radius: 6px; font-weight: 800; font-size: 11px; padding: 4px 12px;"
            )
            self.btn_corner_sharp.setStyleSheet(
                "background-color: #1E1E24; color: #A1A1AA; border: 1px solid #32323D; "
                "border-radius: 6px; font-weight: 600; font-size: 11px; padding: 4px 12px;"
            )
            self.slider_interview_curve.setEnabled(True)
            self.lbl_interview_curve.setEnabled(True)

    def _browse_interview_clips(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Interview Footage Clips",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm);;All Files (*.*)",
        )
        if files:
            self._interview_clip_paths = files
            self.edit_interview_clips.setText(f"{len(files)} interview clip(s) selected")

    def _on_scan_youtube_clicked(self) -> None:
        url = self.edit_youtube_link.text().strip()
        if not url:
            QMessageBox.warning(self, "Video Link Required", "Please enter a valid YouTube or video URL to scan.")
            return

        from services.interview_scanner import is_valid_video_url
        if not is_valid_video_url(url):
            QMessageBox.warning(
                self,
                "Invalid URL",
                f"'{url}' does not appear to be a valid video URL.\nPlease enter an http:// or https:// link."
            )
            return

        self.btn_scan_youtube.setEnabled(False)
        self.btn_scan_youtube.setText("⏳ Scanning...")
        self.bar_interview_scan.setValue(5)
        self.bar_interview_scan.setVisible(True)
        self.lbl_interview_scan_status.setText("Connecting to video source...")
        self.lbl_interview_scan_status.setVisible(True)
        self.lbl_status.setText(f"Scanning interview clips: {url}")

        from ui.workers.interview_scan_worker import InterviewScanWorker
        api_key = os.getenv("GEMINI_API_KEY")
        filter_mode = "all" if getattr(self, "chk_interview_relax_filter", None) and self.chk_interview_relax_filter.isChecked() else "strict"
        self._scan_worker = InterviewScanWorker(url=url, api_key=api_key, filter_mode=filter_mode)
        self._scan_worker.progress.connect(self._on_interview_scan_progress)
        self._scan_worker.status_message.connect(self._on_interview_scan_status_message)
        self._scan_worker.scan_finished.connect(self._on_interview_scan_finished)
        self._scan_worker.scan_failed.connect(self._on_interview_scan_failed)
        self._scan_worker.start()

        self.request_scan_youtube.emit(url)

    def _on_interview_scan_progress(self, pct: int) -> None:
        self.bar_interview_scan.setValue(pct)

    def _on_interview_scan_status_message(self, msg: str) -> None:
        self.lbl_interview_scan_status.setText(msg)
        self.lbl_status.setText(msg)

    def _on_interview_scan_finished(self, video_path: str, segments: list) -> None:
        self.btn_scan_youtube.setEnabled(True)
        self.btn_scan_youtube.setText("Scan Clips")
        self.bar_interview_scan.setValue(100)
        self.bar_interview_scan.setVisible(False)

        self._scanned_interview_segments = segments
        self._interview_source_video = video_path
        self._interview_clip_paths = [Path(video_path)]
        self.btn_review_interview_clips.setEnabled(True)

        v_name = Path(video_path).name
        summary_text = f"✓ {len(segments)} segment(s) detected • {v_name}"
        self.edit_interview_clips.setText(summary_text)
        self.lbl_interview_scan_status.setText(f"✓ Found {len(segments)} interview segment(s)")
        self.lbl_status.setText(f"Interview scan complete: {len(segments)} segment(s) ready in {v_name}")

        self.interview_segments_scanned.emit(video_path, segments)

        # Automatically open Selection & Trimming Dialog
        self._open_interview_selector_dialog()

    def _open_interview_selector_dialog(self) -> None:
        if not getattr(self, "_scanned_interview_segments", []):
            QMessageBox.information(
                self,
                "No Interview Clips",
                "No scanned interview clips available yet.\nPlease enter a video URL and click 'Scan Clips' first."
            )
            return

        from ui.dialogs.interview_selector_dialog import InterviewSelectorDialog
        source_vid = getattr(self, "_interview_source_video", "") or (str(self._interview_clip_paths[0]) if self._interview_clip_paths else "")
        dlg = InterviewSelectorDialog(
            parent=self,
            source_video=source_vid,
            segments=self._scanned_interview_segments,
            confirmed_clips=getattr(self, "_confirmed_interview_clips", None),
        )
        if dlg.exec():
            confirmed = dlg.get_confirmed_clips()
            self._apply_confirmed_interview_clips(confirmed)

    def _apply_confirmed_interview_clips(self, confirmed: list) -> None:
        self._confirmed_interview_clips = confirmed
        total_dur = sum(c.duration_sec for c in confirmed)
        total_detected = len(self._scanned_interview_segments)
        summary_text = f"✓ {len(confirmed)} of {total_detected} clips selected ({total_dur:.1f}s total)"
        self.edit_interview_clips.setText(summary_text)
        self.lbl_interview_scan_status.setText(f"✓ {len(confirmed)} clips confirmed ({total_dur:.1f}s)")
        self.lbl_status.setText(f"Confirmed {len(confirmed)} interview clips for video ({total_dur:.1f}s).")

    def _on_interview_scan_failed(self, err_msg: str) -> None:
        self.btn_scan_youtube.setEnabled(True)
        self.btn_scan_youtube.setText("Scan Clips")
        self.bar_interview_scan.setVisible(False)
        self.lbl_interview_scan_status.setText(f"✗ Scan failed: {err_msg[:40]}...")
        self.lbl_status.setText(f"Interview scan failed: {err_msg[:60]}")

        QMessageBox.warning(
            self,
            "Interview Scan Failed",
            f"Could not download or scan the video link:\n\n{err_msg}\n\n"
            "Please check the URL or your network connection."
        )

    def _toggle_interview_panel(self) -> None:
        is_collapsed = getattr(self, "_interview_panel_collapsed", False)
        new_state = not is_collapsed
        self._interview_panel_collapsed = new_state
        if hasattr(self, "interview_body"):
            self.interview_body.setVisible(not new_state)
        self.btn_toggle_interview_card.setText("▲" if new_state else "▼")

    def get_interview_config(self) -> dict[str, Any]:
        """Return Template 3 specific interview framing & audio configuration."""
        return {
            "music_path": self.edit_interview_music.text().strip(),
            "music_volume": self.slider_interview_music_vol.value() / 100.0,
            "bg_image_path": self.edit_interview_bg_image.text().strip() if hasattr(self, "edit_interview_bg_image") else "",
            "bg_blur_amount": self.slider_interview_blur.value() if hasattr(self, "slider_interview_blur") else 0,
            "clip_scale": self.slider_interview_scale.value() / 100.0,
            "bg_blur": self.slider_interview_blur.value() / 100.0,
            "border_enabled": self.chk_interview_border.isChecked(),
            "border_color": self._interview_border_color,
            "border_thickness": self.slider_interview_border_thickness.value(),
            "corner_shape": self._interview_corner_shape,
            "corner_curve": self.slider_interview_curve.value() if self._interview_corner_shape == "rounded" else 0,
            "interview_clips": [str(p) for p in self._interview_clip_paths],
            "youtube_url": self.edit_youtube_link.text().strip(),
            "scanned_segments": [
                {
                    "segment_id": s.segment_id,
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "duration_sec": s.duration_sec,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "label": s.label,
                    "speaker": getattr(s, "speaker", "Speaker"),
                    "transcript": getattr(s, "transcript", ""),
                }
                for s in getattr(self, "_scanned_interview_segments", [])
            ],
            "confirmed_clips": [
                {
                    "segment_id": c.segment_id,
                    "source_video": c.source_video,
                    "trim_start_sec": c.trim_start_sec,
                    "trim_end_sec": c.trim_end_sec,
                    "duration_sec": c.duration_sec,
                    "target_start_sec": c.target_start_sec,
                    "target_end_sec": c.target_end_sec,
                    "label": c.label,
                    "speaker": getattr(c, "speaker", "Speaker"),
                    "transcript": getattr(c, "transcript", ""),
                    "crop_top": getattr(c, "crop_top", 0),
                    "crop_bottom": getattr(c, "crop_bottom", 0),
                    "crop_left": getattr(c, "crop_left", 0),
                    "crop_right": getattr(c, "crop_right", 0),
                }
                for c in getattr(self, "_confirmed_interview_clips", [])
            ],
        }

    def get_render_config(self) -> dict[str, Any]:
        """Return the comprehensive render configuration dictionary."""
        format_idx = self.combo_format.currentIndex()
        template_name = "template_1"
        if format_idx == 1:
            template_name = "template_2"
        elif format_idx == 2 or "Template 3" in self.combo_format.currentText():
            template_name = "template_3"

        cfg = {
            "template": template_name,
            "audio_path": self.edit_audio.text().strip(),
            "assets_dir": self.edit_assets.text().strip(),
            "output_path": self.edit_output.text().strip(),
            "title": self.edit_title.text().strip(),
            "resolution": self.get_selected_resolution(),
            "language": self.get_selected_language(),
            "clip_duration": float(self.slider_clip_dur.value()) if hasattr(self, "slider_clip_dur") else 5.0,
            "font_size": self._font_size,
            "font_color": self._font_color,
            "transition_sfx": self.get_transition_sfx_config(),
        }
        if template_name == "template_3":
            cfg["interview"] = self.get_interview_config()
        return cfg

    # Convenient alias
    get_config = get_render_config

    def _start_render(self) -> None:
        audio_text = self.edit_audio.text().strip()
        from services.character_matcher import natural_sort_key
        raw_parts = [p.strip() for p in audio_text.split(";") if p.strip()]
        audio_paths = sorted([Path(p) for p in raw_parts], key=lambda p: natural_sort_key(p.name)) if raw_parts else []
        audio_path = audio_paths[0] if audio_paths else Path(audio_text)

        assets_dir = Path(self.edit_assets.text().strip()) if hasattr(self, "edit_assets") and self.edit_assets.text().strip() else None
        output_path = Path(self.edit_output.text().strip())
        title = self.edit_title.text().strip() or (audio_path.stem.replace("_", " ").title() if audio_path.name else "Untitled Video Project")
        target_res = self.get_selected_resolution()
        selected_lang = self.get_selected_language()
        clip_duration = float(self.slider_clip_dur.value()) if hasattr(self, "slider_clip_dur") else 5.0
        image_ratio = (float(self.slider_mix_ratio.value()) / 100.0) if hasattr(self, "slider_mix_ratio") else 0.5

        if not audio_paths or not all(p.exists() for p in audio_paths):
            missing = [str(p) for p in audio_paths if not p.exists()] or [str(audio_path)]
            QMessageBox.critical(self, "Missing Audio", f"Voiceover audio file(s) not found:\n" + "\n".join(missing))
            return
        if not assets_dir or not assets_dir.exists():
            QMessageBox.critical(self, "Missing Assets", "Please select a valid Footage & Assets Folder.")
            return

        self.btn_start.setEnabled(False)
        self._reset_progress()
        self.lbl_token_usage_banner.setVisible(False)
        self.lbl_status.setText(f"Initializing studio render ({target_res.upper()}) for '{title}'...")

        raw_font = self.combo_font.currentText().split("(")[0].strip()
        font_family = raw_font.replace(" Bold", "").strip() or "Montserrat"
        is_bold = (self.btn_bold.styleSheet() == BTN_ACTIVE_STYLE)
        is_italic = (self.btn_italic.styleSheet() == BTN_ACTIVE_STYLE)

        format_idx = self.combo_format.currentIndex()
        if format_idx == 1:
            # Check Gemini API Key validity for Template 2 (Episode Scene Matching)
            from utils.config import reload_config
            cfg = reload_config()
            from services.gemini_matcher import SmartSceneMatcher
            api_key = os.getenv("GEMINI_API_KEY") or cfg.gemini_api_key or ""
            is_valid_key = SmartSceneMatcher.is_api_key_valid(api_key)

            if not is_valid_key:
                lang_display = "Urdu" if selected_lang == "ur" else "English"
                QMessageBox.critical(
                    self,
                    "Gemini API Key Required",
                    f"⚠️ <b>Google Gemini API Key Required for Template 2 ({lang_display} Narration)</b><br><br>"
                    f"Template 2 (Episode Scene Matching) requires an active Google Gemini 1.5 Flash API key "
                    f"to semantically understand {lang_display} narrative context and match relevant episode scenes.<br><br>"
                    "Silent fallback to blind keyword matching has been disabled to prevent incorrect or misaligned scene cuts.<br><br>"
                    "Please configure a valid <code>GEMINI_API_KEY</code> in your <code>.env</code> file or Settings."
                )
                self.btn_start.setEnabled(True)
                return

            from ui.workers.episode_worker import EpisodeCompilerWorker
            self._worker = EpisodeCompilerWorker(
                audio_path=audio_path,
                footage_path=assets_dir,
                output_path=output_path,
                title=title,
                resolution=target_res,
                language=selected_lang,
                text_color=self._font_color,
                highlight_color=self._highlight_color,
                has_bg_box=self._has_bg_box,
                bg_style=self._bg_style,
                bg_color=self._bg_color,
                bg_opacity=self._bg_opacity,
                box_padding=self._box_padding,
                stroke_color=self._stroke_color,
                stroke_thickness=self._stroke_thickness,
                font_size=self._font_size,
                font_family=font_family,
                bold=is_bold,
                italic=is_italic,
                clip_duration=clip_duration,
                generate_chapters=self.chk_chapters.isChecked(),
                transition_sfx=self.get_transition_sfx_config(),
            )
        else:
            is_template_3 = (format_idx == 2 or "Template 3" in self.combo_format.currentText())
            self._worker = VideoRenderWorker(
                audio_path=audio_path,
                audio_paths=audio_paths if len(audio_paths) > 1 else None,
                images_dir=assets_dir,
                videos_path=assets_dir,
                clip_duration=clip_duration,
                image_ratio=image_ratio,
                language=selected_lang,
                output_path=output_path,
                title=title,
                resolution=target_res,
                text_color=self._font_color,
                highlight_color=self._highlight_color,
                has_bg_box=self._has_bg_box,
                bg_style=self._bg_style,
                bg_color=self._bg_color,
                bg_opacity=self._bg_opacity,
                box_padding=self._box_padding,
                stroke_color=self._stroke_color,
                stroke_thickness=self._stroke_thickness,
                font_size=self._font_size,
                font_family=font_family,
                bold=is_bold,
                italic=is_italic,
                generate_chapters=self.chk_chapters.isChecked(),
                include_images=self.chk_include_images.isChecked(),
                template_type="template_3" if is_template_3 else "template_1",
                transition_sfx=self.get_transition_sfx_config(),
            )
            if is_template_3:
                interview_cfg = self.get_interview_config()
                self._worker.interview_config = interview_cfg
                self._worker.bg_image_path = interview_cfg.get("bg_image_path", "")
                self._worker.bg_blur_amount = interview_cfg.get("bg_blur_amount", 0)

        self._worker.status_message.connect(lambda msg: self.lbl_status.setText(msg))
        self._worker.audio_progress.connect(self._on_audio_progress)
        self._worker.render_progress.connect(self._on_render_progress)
        self._worker.render_finished.connect(self._on_render_finished)
        self._worker.render_failed.connect(self._on_render_error)
        if hasattr(self._worker, "token_usage_updated"):
            self._worker.token_usage_updated.connect(self._on_token_usage_updated)

        self._worker.start()

    def _on_audio_progress(self, pct: int) -> None:
        self.bar_transcription.setValue(pct)
        self.lbl_vo_pct.setText(f"{pct}%")

    def _on_render_progress(self, pct: int) -> None:
        self.bar_render.setValue(pct)
        self.lbl_exp_pct.setText(f"{pct}%")

    def _on_token_usage_updated(self, stats: dict) -> None:
        if stats and stats.get("total_tokens", 0) > 0:
            tot = stats.get("total_tokens", 0)
            inp = stats.get("prompt_tokens", 0)
            outp = stats.get("output_tokens", 0)
            cost = stats.get("estimated_cost_usd", 0.0)
            self.lbl_token_usage_banner.setText(
                f"📊 Video Token Usage: {tot:,} tokens (In: {inp:,} | Out: {outp:,}) • Est. Cost: ${cost:.4f}"
            )
            self.lbl_token_usage_banner.setVisible(True)

    def _on_render_finished(self, out_path: str) -> None:
        self.btn_start.setEnabled(True)
        self.bar_transcription.setValue(100)
        self.bar_render.setValue(100)
        self.lbl_vo_pct.setText("100%")
        self.lbl_exp_pct.setText("100%")
        self.lbl_status.setText(f"✓ Video rendered successfully: {Path(out_path).name}")

        format_idx = self.combo_format.currentIndex()
        token_stats = getattr(self._worker, "token_stats", {})
        tot = token_stats.get("total_tokens", 0)
        inp = token_stats.get("prompt_tokens", 0)
        outp = token_stats.get("output_tokens", 0)
        cost = token_stats.get("estimated_cost_usd", 0.0)
        used_gemini = token_stats.get("used_gemini", False) or tot > 0

        # Refresh lifetime counter
        self._update_template_status_badge()

        msg = QMessageBox(self)

        if format_idx == 1:
            # Dedicated Template 2 Completion & Telemetry Dialog
            msg.setWindowTitle("🎬 Template 2 Render Complete!")
            banner_text = f"🤖 Gemini AI: {tot:,} tokens (In: {inp:,} | Out: {outp:,}) • Est. Cost: ${cost:.4f}"
            self.lbl_token_usage_banner.setText(banner_text)
            self.lbl_token_usage_banner.setStyleSheet(
                "background-color: #064E3B; color: #34D399; border: 1px solid #059669; "
                "border-radius: 5px; font-size: 11px; font-weight: 700; padding: 6px 12px;"
            )
            self.lbl_token_usage_banner.setVisible(True)

            engine_label = "Gemini 1.5 Flash Vision (Online AI Matcher)" if used_gemini else "Offline Smart Scene Matcher (Heuristics)"
            status_badge = (
                "<span style='color:#34D399; font-weight:bold;'>✓ 100% Online Semantic Processing Verified</span>"
                if used_gemini else
                "<span style='color:#38BDF8; font-weight:bold;'>⚡ Heuristic Offline Matching</span>"
            )

            html_body = f"""
            <h3 style='color:#38BDF8; margin:0;'>🎬 Template 2 Render Complete!</h3>
            <p style='color:#E2E8F0; font-size:12px; margin:4px 0 12px 0;'>
                Your episode-matched master video has been generated and synchronized.
            </p>
            <div style='background-color:#16161A; border:1px solid #2C2C35; border-radius:6px; padding:10px; margin-bottom:12px;'>
                <b style='color:#94A3B8; font-size:11px;'>📁 Master Output Video:</b><br>
                <code style='color:#34D399; font-size:11px;'>{out_path}</code>
            </div>
            <div style='background-color:#0D2818; border:1px solid #059669; border-radius:6px; padding:12px;'>
                <b style='color:#34D399; font-size:13px;'>🤖 Engine: {engine_label}</b><br><br>
                <span style='color:#F1F5F9; font-size:12px; line-height:1.6;'>
                • <b>📊 Tokens Used:</b> {tot:,} (Input: {inp:,} | Output: {outp:,})<br>
                • <b>💰 Estimated Cost:</b> ${cost:.4f} USD<br>
                • <b>⚡ Status:</b> {status_badge}
                </span>
            </div>
            """
            msg.setText(html_body)
        elif format_idx == 2 or "Template 3" in self.combo_format.currentText():
            msg.setWindowTitle("🎬 Template 3 + Interview Render Complete")
            msg.setText(
                f"<b>Your Template 3 + Interview Master Video has been generated!</b><br><br>"
                f"Saved to:<br><code>{out_path}</code>"
            )
        else:
            # Template 1
            if tot > 0:
                banner_text = f"📊 Video Token Usage: {tot:,} tokens (In: {inp:,} | Out: {outp:,}) • Est. Cost: ${cost:.4f}"
                self.lbl_token_usage_banner.setText(banner_text)
                self.lbl_token_usage_banner.setVisible(True)
                tok_html = f"<br><br><span style='color:#34D399; font-weight:bold;'>{banner_text}</span>"
            else:
                self.lbl_token_usage_banner.setVisible(False)
                tok_html = ""

            msg.setWindowTitle("AutoCut Render Complete")
            msg.setText(f"<b>Your Master Video has been generated!</b><br><br>Saved to:<br><code>{out_path}</code>{tok_html}")

        btn_preview = msg.addButton("🎬 Play in Studio", QMessageBox.ActionRole)
        btn_open = msg.addButton("Open in OS Player", QMessageBox.ActionRole)
        btn_folder = msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton("Close", QMessageBox.RejectRole)
        msg.exec()

        if msg.clickedButton() == btn_preview:
            self.request_play_video.emit(out_path)
        elif msg.clickedButton() == btn_open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(out_path))
        elif msg.clickedButton() == btn_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(out_path).parent)))

    def _on_render_error(self, err_msg: str) -> None:
        self.btn_start.setEnabled(True)
        self.lbl_status.setText(f"✗ Render Failed: {err_msg[:60]}...")

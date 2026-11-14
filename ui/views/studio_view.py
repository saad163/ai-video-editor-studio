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
from PySide6.QtGui import QDesktopServices, QColor, QPalette

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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[VideoRenderWorker] = None

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
            "Template 2 (Episode Scene Matching)"
        ])
        setup_layout.addWidget(fmt_lbl)
        setup_layout.addWidget(self.combo_format)
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

        # Audio file input line & button (Empty by default)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)
        self.edit_audio = QLineEdit()
        self.edit_audio.setPlaceholderText("Select your voiceover audio (.mp3, .wav)...")
        self.edit_audio.textChanged.connect(self._on_audio_changed)

        btn_browse_audio = QPushButton("Browse Audio")
        btn_browse_audio.setObjectName("secondaryButton")
        btn_browse_audio.setFixedWidth(110)
        btn_browse_audio.clicked.connect(self._browse_audio)
        audio_row.addWidget(self.edit_audio, stretch=1)
        audio_row.addWidget(btn_browse_audio)
        vo_layout.addLayout(audio_row)
        media_layout.addWidget(vo_box)

        # 2. Unified Footage & Assets Folder Box
        assets_box = QFrame()
        assets_box.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px;")
        assets_layout = QVBoxLayout(assets_box)
        assets_layout.setSpacing(8)

        assets_header = QHBoxLayout()
        assets_tag = QLabel("📁 Footage & Visual Assets Folder (Images & Videos)")
        assets_tag.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        assets_header.addWidget(assets_tag)
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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voiceover Audio", "", "Audio Files (*.mp3 *.wav *.m4a *.flac)"
        )
        if file_path:
            self.edit_audio.setText(file_path)

    def _on_audio_changed(self, text: str) -> None:
        p = Path(text.strip())
        if p.exists() and p.is_file():
            self.lbl_vo_badge.setText("Audio Ready")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
            # Auto-generate title and output filename from audio stem
            title = p.stem.replace("_", " ").title()
            self.edit_title.setText(title)
            cfg = get_config()
            target_out = cfg.output_dir / f"{p.stem}.mp4"
            self.edit_output.setText(str(target_out))
            self._update_caption_preview()
        elif text.strip():
            self.lbl_vo_badge.setText("File Not Found")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #2A1D16; color: #EF4444; border: 1px solid #7F1D1D; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )
        else:
            self.lbl_vo_badge.setText("Select Audio")
            self.lbl_vo_badge.setStyleSheet(
                "background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; "
                "border-radius: 4px; font-size: 9px; font-weight: 700; padding: 1px 6px;"
            )

    def _browse_assets_folder(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Footage & Assets Folder")
        if dir_path:
            self.edit_assets.setText(dir_path)
            self._update_assets_scan_summary()

    def _update_assets_scan_summary(self) -> None:
        path_str = self.edit_assets.text().strip() if hasattr(self, "edit_assets") else ""
        if not path_str:
            self.lbl_assets_badge.setText("⚠️ Please select an assets directory")
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #2A1D16; color: #EF4444; border: 1px solid #7F1D1D; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )
            return

        from services.character_matcher import CharacterMatcher
        summary = CharacterMatcher.scan_asset_folder(Path(path_str))
        badge_text = summary.get("badge_text", "")
        self.lbl_assets_badge.setText(badge_text)
        if summary.get("total_images", 0) > 0 or len(summary.get("video_files", [])) > 0:
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
                "border-radius: 5px; font-size: 11px; font-weight: 600; padding: 4px 8px;"
            )
        else:
            self.lbl_assets_badge.setStyleSheet(
                "background-color: #2A1D16; color: #F59E0B; border: 1px solid #78350F; "
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

    def _start_render(self) -> None:
        audio_path = Path(self.edit_audio.text().strip())
        assets_dir = Path(self.edit_assets.text().strip()) if hasattr(self, "edit_assets") and self.edit_assets.text().strip() else None
        output_path = Path(self.edit_output.text().strip())
        title = self.edit_title.text().strip() or (audio_path.stem.replace("_", " ").title() if audio_path.name else "Untitled Video Project")
        target_res = self.get_selected_resolution()
        clip_duration = float(self.slider_clip_dur.value()) if hasattr(self, "slider_clip_dur") else 5.0
        image_ratio = (float(self.slider_mix_ratio.value()) / 100.0) if hasattr(self, "slider_mix_ratio") else 0.5

        if not audio_path.exists():
            QMessageBox.critical(self, "Missing Audio", f"Voiceover audio file not found:\n{audio_path}")
            return
        if not assets_dir or not assets_dir.exists():
            QMessageBox.critical(self, "Missing Assets", "Please select a valid Footage & Assets Folder.")
            return

        self.btn_start.setEnabled(False)
        self._reset_progress()
        self.lbl_status.setText(f"Initializing studio render ({target_res.upper()}) for '{title}'...")

        raw_font = self.combo_font.currentText().split("(")[0].strip()
        font_family = raw_font.replace(" Bold", "").strip() or "Montserrat"
        is_bold = (self.btn_bold.styleSheet() == BTN_ACTIVE_STYLE)
        is_italic = (self.btn_italic.styleSheet() == BTN_ACTIVE_STYLE)

        format_idx = self.combo_format.currentIndex()
        if format_idx == 1:
            from ui.workers.episode_worker import EpisodeCompilerWorker
            self._worker = EpisodeCompilerWorker(
                audio_path=audio_path,
                footage_path=assets_dir,
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
                generate_chapters=self.chk_chapters.isChecked()
            )
        else:
            self._worker = VideoRenderWorker(
                audio_path=audio_path,
                images_dir=assets_dir,
                videos_path=assets_dir,
                clip_duration=clip_duration,
                image_ratio=image_ratio,
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
                include_images=self.chk_include_images.isChecked()
            )

        self._worker.status_message.connect(lambda msg: self.lbl_status.setText(msg))
        self._worker.audio_progress.connect(self._on_audio_progress)
        self._worker.render_progress.connect(self._on_render_progress)
        self._worker.render_finished.connect(self._on_render_finished)
        self._worker.render_failed.connect(self._on_render_error)

        self._worker.start()

    def _on_audio_progress(self, pct: int) -> None:
        self.bar_transcription.setValue(pct)
        self.lbl_vo_pct.setText(f"{pct}%")

    def _on_render_progress(self, pct: int) -> None:
        self.bar_render.setValue(pct)
        self.lbl_exp_pct.setText(f"{pct}%")

    def _on_render_finished(self, out_path: str) -> None:
        self.btn_start.setEnabled(True)
        self.bar_transcription.setValue(100)
        self.bar_render.setValue(100)
        self.lbl_vo_pct.setText("100%")
        self.lbl_exp_pct.setText("100%")
        self.lbl_status.setText(f"✓ Video rendered successfully in 2K: {Path(out_path).name}")

        msg = QMessageBox(self)
        msg.setWindowTitle("AutoCut Render Complete")
        msg.setText(f"<b>Your Master Video has been generated!</b><br><br>Saved to:<br><code>{out_path}</code>")
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

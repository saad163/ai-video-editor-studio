"""
ui/views/templates_view.py
==========================
Templates & Formats view for AI Video Editor Studio.
Displays Module A (Template 1) and Module B (Template 2) specifications.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal


class TemplatesView(QWidget):
    """Templates selection and presets view."""

    use_template = Signal(str)

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

        title_lbl = QLabel("Production Templates & Formats")
        title_lbl.setObjectName("viewTitle")
        subtitle_lbl = QLabel("Choose a creation pipeline tailored for your channel's storytelling style")
        subtitle_lbl.setObjectName("viewSubtitle")

        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header_layout.addLayout(title_box)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # ── Template 1 Card ────────────────────────────────────────────────────
        t1_card = QFrame()
        t1_card.setObjectName("cardFrame")
        t1_layout = QVBoxLayout(t1_card)
        t1_layout.setContentsMargins(20, 20, 20, 20)
        t1_layout.setSpacing(10)

        t1_header = QHBoxLayout()
        t1_title = QLabel("Template 1: Script & Character-Based Video Assembler")
        t1_title.setObjectName("sectionTitle")
        t1_badge = QLabel(" Fully Operational ")
        t1_badge.setObjectName("badgeActive")
        t1_header.addWidget(t1_title)
        t1_header.addStretch()
        t1_header.addWidget(t1_badge)
        t1_layout.addLayout(t1_header)

        t1_desc = QLabel(
            "Designed for character-driven narratives, educational documentaries, and story videos. "
            "Automatically transcribes voiceovers, identifies speaking characters from the script, assigns visual "
            "assets with anti-repetition rules, applies smooth Ken Burns pan/zoom camera effects, and burns styled subtitles."
        )
        t1_desc.setWordWrap(True)
        t1_desc.setStyleSheet("color: #CBD5E1; font-size: 13px; line-height: 1.4;")
        t1_layout.addWidget(t1_desc)

        t1_features = QLabel(
            "• Inputs: Voiceover audio (.mp3/.wav), Character Images folders, Subtitle styling.\n"
            "• Engine: faster-whisper (CPU int8 / RTX float16) + FFmpeg Ken Burns Lanczos matrix.\n"
            "• Output: 1080p / 2K 60FPS MP4 video."
        )
        t1_features.setStyleSheet("color: #94A3B8; font-size: 12px;")
        t1_layout.addWidget(t1_features)

        btn_launch_t1 = QPushButton("⚡ Launch Template 1 in Studio")
        btn_launch_t1.setObjectName("primaryButton")
        btn_launch_t1.setFixedHeight(38)
        btn_launch_t1.clicked.connect(lambda: self.use_template.emit("template_1"))
        t1_layout.addWidget(btn_launch_t1, alignment=Qt.AlignLeft)

        layout.addWidget(t1_card)

        # ── Template 2 Card ────────────────────────────────────────────────────
        t2_card = QFrame()
        t2_card.setObjectName("cardFrame")
        t2_layout = QVBoxLayout(t2_card)
        t2_layout.setContentsMargins(20, 20, 20, 20)
        t2_layout.setSpacing(10)

        t2_header = QHBoxLayout()
        t2_title = QLabel("Template 2: Episode Scene Compiler & Context Matcher")
        t2_title.setObjectName("sectionTitle")
        t2_badge = QLabel(" Module B (Phase 3) ")
        t2_badge.setObjectName("badgeNeutral")
        t2_header.addWidget(t2_title)
        t2_header.addStretch()
        t2_header.addWidget(t2_badge)
        t2_layout.addLayout(t2_header)

        t2_desc = QLabel(
            "Splits multi-episode source videos into individual scene shots using PySceneDetect, extracts "
            "representative middle keyframes, and matches video scenes to 15–30s narration chunks via "
            "Gemini 1.5 Flash batch-context prompts."
        )
        t2_desc.setWordWrap(True)
        t2_desc.setStyleSheet("color: #CBD5E1; font-size: 13px; line-height: 1.4;")
        t2_layout.addWidget(t2_desc)

        t2_features = QLabel(
            "• Inputs: 20-35 min Voiceover audio, 6-8 Source Episode MP4s, Reference script.\n"
            "• Engine: PySceneDetect ContentAware + Gemini 1.5 Flash API + Lossless stream cuts.\n"
            "• Running cost: ~$0.05 - $0.15 per 30m video."
        )
        t2_features.setStyleSheet("color: #94A3B8; font-size: 12px;")
        t2_layout.addWidget(t2_features)

        btn_launch_t2 = QPushButton("Configure Template 2 Sources")
        btn_launch_t2.setObjectName("secondaryButton")
        btn_launch_t2.setFixedHeight(38)
        btn_launch_t2.clicked.connect(lambda: self.use_template.emit("template_2"))
        t2_layout.addWidget(btn_launch_t2, alignment=Qt.AlignLeft)

        layout.addWidget(t2_card)

        layout.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)


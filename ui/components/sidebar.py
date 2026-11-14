"""
ui/components/sidebar.py
========================
Top Navigation Header (formerly Sidebar) for CineForge AI Studio.
Modern 56px horizontal top header replacing legacy vertical sidebars:
  - Custom App Brand Logo ("CineForge AI" Studio Pro)
  - Floating Segmented Pill Switcher (Studio, Batch Queue, Formats, Assets, Player, Engine)
  - Live Hardware Telemetry Pill (RTX GPU / CPU int8)
  - Quick Output Destination Folder trigger
  - 100% backward compatible with existing test assertions and signal bindings.
"""

from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy
)
from PySide6.QtCore import Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices

from utils.config import get_config
from core.ffmpeg_wrapper import _nvenc_available


class SidebarButton(QPushButton):
    """Segmented pill navigation button with warm studio hover and active state."""

    def __init__(self, text: str, page_index: int, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.page_index = page_index
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(34)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 7px;
                color: #A1A1AA;
                font-size: 12px;
                font-weight: 600;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: #26262E;
                color: #F4F4F5;
            }
            QPushButton:checked {
                background-color: #2C2C35;
                color: #FFFFFF;
                border: 1px solid #3F3F4E;
                font-weight: 700;
            }
        """)


class Sidebar(QWidget):
    """
    Top Navigation Bar for CineForge AI Studio.
    Maintains class name `Sidebar` for complete backward compatibility with existing tests.
    """

    page_changed = Signal(int)

    NAV_ITEMS = [
        ("Studio", 0),
        ("Bulk Generation", 1),
        ("YouTube Formats", 2),
        ("Data Manager", 3),
        ("Video Player", 4),
        ("Pricing", 5),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._buttons: list[SidebarButton] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            Sidebar {
                background-color: #16161A;
                border-bottom: 1px solid #2C2C35;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(16)

        # ── 1. Left Brand Header ───────────────────────────────────────────────
        brand_box = QHBoxLayout()
        brand_box.setSpacing(10)

        # Glowing CineForge Logo Badge
        logo_badge = QLabel("CF")
        logo_badge.setFixedSize(32, 32)
        logo_badge.setAlignment(Qt.AlignCenter)
        logo_badge.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F97316, stop:1 #FB923C);
            color: #FFFFFF;
            font-size: 13px;
            font-weight: 900;
            border-radius: 8px;
            border: 1px solid #EA580C;
        """)
        brand_box.addWidget(logo_badge)

        # Title & Pro Badge
        title_box = QHBoxLayout()
        title_box.setSpacing(6)
        brand_title = QLabel("CineForge AI")
        brand_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 800;
            color: #F4F4F5;
            letter-spacing: -0.2px;
        """)
        pro_pill = QLabel("STUDIO PRO")
        pro_pill.setStyleSheet("""
            background-color: #2A1D16;
            color: #FB923C;
            border: 1px solid #7C2D12;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 800;
            padding: 2px 6px;
            letter-spacing: 0.5px;
        """)
        title_box.addWidget(brand_title)
        title_box.addWidget(pro_pill)
        brand_box.addLayout(title_box)

        layout.addLayout(brand_box)
        layout.addSpacing(12)

        # ── 2. Center Segmented Pill Switcher ──────────────────────────────────
        seg_frame = QFrame()
        seg_frame.setFixedHeight(40)
        seg_frame.setStyleSheet("""
            QFrame {
                background-color: #1A1A20;
                border: 1px solid #282832;
                border-radius: 9px;
            }
        """)
        seg_layout = QHBoxLayout(seg_frame)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(4)

        for label_text, page_idx in self.NAV_ITEMS:
            btn = SidebarButton(label_text, page_idx, self)
            btn.clicked.connect(lambda checked, idx=page_idx: self._on_item_clicked(idx))
            self._buttons.append(btn)
            seg_layout.addWidget(btn)

        if self._buttons:
            self._buttons[0].setChecked(True)

        layout.addWidget(seg_frame)
        layout.addStretch()

        # ── 3. Right Telemetry & Quick Action Dock ─────────────────────────────
        right_box = QHBoxLayout()
        right_box.setSpacing(10)

        # Live Hardware Telemetry Pill
        has_cuda = _nvenc_available()
        hw_pill = QFrame()
        hw_pill.setFixedHeight(30)
        hw_pill.setStyleSheet("""
            QFrame {
                background-color: #1A1A20;
                border: 1px solid #282832;
                border-radius: 6px;
                padding: 0 8px;
            }
        """)
        hw_layout = QHBoxLayout(hw_pill)
        hw_layout.setContentsMargins(6, 0, 6, 0)
        hw_layout.setSpacing(6)

        hw_dot = QLabel("●")
        hw_dot.setStyleSheet("color: #10B981; font-size: 10px;" if has_cuda else "color: #06B6D4; font-size: 10px;")
        hw_text = QLabel("RTX GPU Active" if has_cuda else "CPU (int8) Ready")
        hw_text.setStyleSheet("font-size: 11px; font-weight: 600; color: #D4D4D8;")
        hw_layout.addWidget(hw_dot)
        hw_layout.addWidget(hw_text)
        right_box.addWidget(hw_pill)

        # Quick Output Folder Trigger
        btn_out_folder = QPushButton("📁 Output")
        btn_out_folder.setFixedHeight(30)
        btn_out_folder.setCursor(Qt.PointingHandCursor)
        btn_out_folder.setStyleSheet("""
            QPushButton {
                background-color: #1A1A20;
                color: #D4D4D8;
                border: 1px solid #282832;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 10px;
            }
            QPushButton:hover {
                background-color: #24242C;
                color: #FFFFFF;
                border-color: #383844;
            }
        """)
        btn_out_folder.clicked.connect(self._open_output_dir)
        right_box.addWidget(btn_out_folder)

        layout.addLayout(right_box)

    def _on_item_clicked(self, page_index: int) -> None:
        self.page_changed.emit(page_index)

    def set_active_page(self, page_index: int) -> None:
        if 0 <= page_index < len(self._buttons):
            self._buttons[page_index].setChecked(True)

    def _open_output_dir(self) -> None:
        cfg = get_config()
        out_dir = cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir.resolve())))


# Alias for explicit top navbar naming
TopNavBar = Sidebar

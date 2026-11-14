"""
ui/views/assets_view.py
=======================
Asset & Data Manager tab for AI Video Editor Studio.
Displays:
  - SQLite database statistics
  - Project history records
  - Media asset hash deduplication table
  - One-click cache cleanup utility (Storage Guard Engine)
"""

from __future__ import annotations

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QMessageBox
)
from PySide6.QtCore import Qt

from utils.config import get_config
from utils.file_helpers import clean_directory, get_directory_size_gb
from database.db import get_connection, list_projects


class AssetsView(QWidget):
    """Asset deduplication and storage management view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        # ── Header ─────────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(3)

        title_lbl = QLabel("Asset & Data Manager")
        title_lbl.setObjectName("viewTitle")
        subtitle_lbl = QLabel("Manage SQLite media hash registry, track project history & reclaim cache storage")
        subtitle_lbl.setObjectName("viewSubtitle")

        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.btn_refresh = QPushButton("🔄 Refresh")
        self.btn_refresh.setObjectName("secondaryButton")
        self.btn_refresh.clicked.connect(self.load_data)
        header_layout.addWidget(self.btn_refresh)

        layout.addLayout(header_layout)

        # ── Storage Guard Card ─────────────────────────────────────────────────
        storage_card = QFrame()
        storage_card.setObjectName("cardFrame")
        st_layout = QHBoxLayout(storage_card)
        st_layout.setContentsMargins(18, 14, 18, 14)
        st_layout.setSpacing(14)

        info_box = QVBoxLayout()
        info_box.setSpacing(2)
        st_title = QLabel("Storage Guard Engine")
        st_title.setObjectName("sectionTitle")
        self.lbl_cache_size = QLabel("Current Temp Cache: Calculating...")
        self.lbl_cache_size.setStyleSheet("font-size: 12px; color: #94A3B8;")
        info_box.addWidget(st_title)
        info_box.addWidget(self.lbl_cache_size)
        st_layout.addLayout(info_box)
        st_layout.addStretch()

        btn_clean_temp = QPushButton("🧹  Purge Temp Chunks & Frames")
        btn_clean_temp.setObjectName("secondaryButton")
        btn_clean_temp.clicked.connect(self._clean_temp)
        st_layout.addWidget(btn_clean_temp)

        layout.addWidget(storage_card)

        # ── Projects Table ─────────────────────────────────────────────────────
        sec_lbl = QLabel("Project Registry (SQLite)")
        sec_lbl.setObjectName("sectionTitle")
        layout.addWidget(sec_lbl)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "ID", "Title", "Template", "Status", "Created At"
        ])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, stretch=1)

        self.load_data()

    def load_data(self) -> None:
        """Load SQLite projects and compute storage usage."""
        cfg = get_config()
        # Compute temp size
        try:
            sz_gb = get_directory_size_gb(cfg.temp_dir)
            self.lbl_cache_size.setText(f"Current Temp Cache: {sz_gb * 1024:.1f} MB in {cfg.temp_dir}")
        except Exception:
            self.lbl_cache_size.setText("Current Temp Cache: 0 MB")

        # Load projects
        try:
            projects = list_projects()
            self.table.setRowCount(0)
            for p in projects:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(p["id"])))
                self.table.setItem(row, 1, QTableWidgetItem(p["title"]))
                self.table.setItem(row, 2, QTableWidgetItem(p["template_type"]))

                status_item = QTableWidgetItem(p["status"].capitalize())
                status_item.setTextAlignment(Qt.AlignCenter)
                if p["status"] == "done":
                    status_item.setForeground(Qt.darkGreen)
                elif p["status"] == "error":
                    status_item.setForeground(Qt.red)
                self.table.setItem(row, 3, status_item)

                self.table.setItem(row, 4, QTableWidgetItem(p["created_at"]))
        except Exception as e:
            pass

    def _clean_temp(self) -> None:
        cfg = get_config()
        deleted = clean_directory(cfg.temp_chunks_dir) + clean_directory(cfg.temp_frames_dir)
        QMessageBox.information(
            self, "Storage Cleaned", f"Successfully removed {deleted} temporary render files."
        )
        self.load_data()


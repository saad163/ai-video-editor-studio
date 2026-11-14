"""
ui/main_window.py
=================
Main application window for AI Video Editor Studio (AutoCut Edition).
Assembles the Sidebar navigation and QStackedWidget pages:
  - Page 0: StudioView
  - Page 1: BulkView
  - Page 2: TemplatesView (YouTube Formats)
  - Page 3: AssetsView (Data Manager)
  - Page 4: PlayerView (Video Player)
  - Page 5: SettingsView (Pricing / AI Settings)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QStackedWidget, QStatusBar, QLabel, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QScreen

from ui.components.sidebar import Sidebar
from ui.views.studio_view import StudioView
from ui.views.bulk_view import BulkView
from ui.views.templates_view import TemplatesView
from ui.views.assets_view import AssetsView
from ui.views.player_view import PlayerView
from ui.views.settings_view import SettingsView


class MainWindow(QMainWindow):
    """Primary application shell."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoCut - AI Video Editor")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 640)
        self._center_window()
        self._setup_ui()

    def _center_window(self) -> None:
        """Center the main window on screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - 1280) // 2
            y = (geo.height() - 800) // 2
            self.move(max(0, x), max(0, y))

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top Navigation Bar ─────────────────────────────────────────────────
        self.sidebar = Sidebar(self)
        self.navbar = self.sidebar
        main_layout.addWidget(self.sidebar)

        # ── Stacked Views ──────────────────────────────────────────────────────
        self.stack = QStackedWidget(self)

        self.studio_view = StudioView(self)
        self.bulk_view = BulkView(self)
        self.templates_view = TemplatesView(self)
        self.assets_view = AssetsView(self)
        self.player_view = PlayerView(self)
        self.settings_view = SettingsView(self)

        self.stack.addWidget(self.studio_view)       # Index 0
        self.stack.addWidget(self.bulk_view)         # Index 1
        self.stack.addWidget(self.templates_view)    # Index 2
        self.stack.addWidget(self.assets_view)       # Index 3
        self.stack.addWidget(self.player_view)       # Index 4
        self.stack.addWidget(self.settings_view)     # Index 5

        main_layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central_widget)

        # ── Signals ────────────────────────────────────────────────────────────
        self.sidebar.page_changed.connect(self._on_page_changed)
        self.templates_view.use_template.connect(self._on_use_template)
        self.studio_view.request_play_video.connect(self._play_in_player)

    def _play_in_player(self, video_path: str) -> None:
        self.sidebar.set_active_page(4)
        self.stack.setCurrentIndex(4)
        self.player_view.play_video(Path(video_path))

    def _on_page_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        # Auto-refresh assets view or player view on entry
        if index == 3:
            self.assets_view.load_data()
        elif index == 4:
            self.player_view.scan_output_folder()

    def _on_use_template(self, template_name: str) -> None:
        self.sidebar.set_active_page(0)
        self.stack.setCurrentIndex(0)
        if template_name == "template_1":
            self.studio_view.chk_include_images.setChecked(True)
        elif template_name == "template_2":
            self.studio_view.combo_format.setCurrentIndex(0)

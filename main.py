"""
main.py
=======
AI Video Editor Studio — Application Entry Point
AutoCut Studio Desktop Edition (PySide6 / Qt6)

Responsibilities:
  1. Bootstrap logging and config
  2. Initialise the SQLite database
  3. Load and apply modern Emerald Studio design theme
  4. Launch the PySide6 main window with fluid multi-tab navigation

Run:
    python main.py
"""

from __future__ import annotations

import sys
import logging

# ── Bootstrap config & logging before anything else ──────────────────────────
from utils.config import setup_logging, get_config

setup_logging()
logger = logging.getLogger(__name__)

# ── Database init ─────────────────────────────────────────────────────────────
from database.db import init_db

# ── PySide6 ───────────────────────────────────────────────────────────────────
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor

# ── UI Components & Stylesheet ───────────────────────────────────────────────
from ui.styles.theme import GLOBAL_STYLESHEET
from ui.main_window import MainWindow


def main() -> None:
    cfg = get_config()
    logger.info("=" * 60)
    logger.info("AI Video Editor Studio starting …")
    logger.info("Project root : %s", cfg.project_root)
    logger.info("Temp dir     : %s", cfg.temp_dir)
    logger.info("Output dir   : %s", cfg.output_dir)
    logger.info("DB path      : %s", cfg.db_path)
    logger.info("Whisper model: %s | device: %s", cfg.whisper_model_size, cfg.compute_device)
    logger.info("=" * 60)

    # Initialise database & sample assets
    init_db()
    from utils.file_helpers import ensure_sample_assets
    ensure_sample_assets()
    logger.info("Database & sample assets ready.")

    # ── PySide6 Application ───────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("AI Video Editor Studio")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("AutoCut Studio")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Set explicit valid base font so Windows High-DPI never produces setPointSize(-1) warning
    base_font = QFont("Segoe UI")
    base_font.setPointSize(10)
    app.setFont(base_font)

    # Set dark application palette for 100% consistent text and popup visibility on Windows
    dark_pal = app.palette()
    dark_pal.setColor(QPalette.Window, QColor("#121214"))
    dark_pal.setColor(QPalette.WindowText, QColor("#F4F4F5"))
    dark_pal.setColor(QPalette.Base, QColor("#1A1A1F"))
    dark_pal.setColor(QPalette.AlternateBase, QColor("#222228"))
    dark_pal.setColor(QPalette.ToolTipBase, QColor("#1A1A1F"))
    dark_pal.setColor(QPalette.ToolTipText, QColor("#F4F4F5"))
    dark_pal.setColor(QPalette.Text, QColor("#F4F4F5"))
    dark_pal.setColor(QPalette.Button, QColor("#222228"))
    dark_pal.setColor(QPalette.ButtonText, QColor("#F4F4F5"))
    dark_pal.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    dark_pal.setColor(QPalette.Highlight, QColor("#F97316"))
    dark_pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(dark_pal)

    # Apply global modern studio theme
    app.setStyleSheet(GLOBAL_STYLESHEET)

    # Launch Primary MainWindow
    window = MainWindow()
    window.show()

    logger.info("Main studio window presented.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

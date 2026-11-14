"""
ui/styles/theme.py
==================
CineForge AI — Modern Warm-Charcoal & Studio Coral Design System.
Flagship creative editing suite styling:
  - Rich Carbon background (#121214)
  - Elevated Warm-Charcoal surfaces & cards (#1A1A1F) with crisp borders (#2C2C35)
  - Studio Coral-Orange accents (#F97316 to #FB923C gradient)
  - Inset dark input fields (#222228) with glowing studio focus rings (#F97316)
  - Crisp high-contrast zinc typography (#F4F4F5, #A1A1AA, #71717A)
  - Explicit Windows QComboBox popup styling with zero dark-mode artifacts
"""

from __future__ import annotations

# ── Color Palette Constants ──────────────────────────────────────────────────
PRIMARY_COLOR = "#F97316"            # Studio Coral-Orange
PRIMARY_HOVER = "#EA580C"            # Deeper Coral hover
PRIMARY_ACTIVE = "#C2410C"           # Pressed Coral
PRIMARY_GRADIENT = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #F97316, stop:1 #FB923C)"
PRIMARY_GRADIENT_HOVER = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EA580C, stop:1 #F97316)"
PRIMARY_LIGHT = "#2A201A"            # Subtle warm selection background
PRIMARY_BORDER = "#F97316"

BG_WINDOW = "#121214"                # Rich Carbon
BG_CARD = "#1A1A1F"                  # Elevated Warm-Charcoal Surface
BG_CARD_BORDER = "#2C2C35"           # Crisp border for cards
BG_TINT_CARD = "#16161A"             # Header filter card
BG_INPUT = "#222228"                 # Inset Dark Surface
BG_NAVBAR = "#16161A"                # Sleek Top Navigation Header
BG_NAVBAR_PILL = "#1E1E24"           # Segmented pill background
BG_NAVBAR_ACTIVE = "#2C2C35"         # Active pill background

TEXT_PRIMARY = "#F4F4F5"             # Crisp bright text (Zinc 50)
TEXT_SECONDARY = "#A1A1AA"           # Medium zinc text (Zinc 400)
TEXT_MUTED = "#71717A"               # Helper / Subtitle text (Zinc 500)
TEXT_ACTIVE_NAV = "#FFFFFF"          # Bright white for active nav items

BORDER_DEFAULT = "#2C2C35"           # Zinc 800
BORDER_LIGHT = "#383844"             # Input borders
BORDER_FOCUS = "#F97316"             # Coral focus ring

ACCENT_CYAN = "#06B6D4"              # Electric Cyan for Audio
ACCENT_EMERALD = "#10B981"           # Status / Success Emerald
ACCENT_AMBER = "#F59E0B"             # Highlights / Badges
ACCENT_CORAL = "#F97316"             # Studio Coral


GLOBAL_STYLESHEET = f"""
/* ── Reset & Global Typography ───────────────────────────── */
QWidget {{
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QMainWindow, QDialog {{
    background-color: {BG_WINDOW};
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}

/* ── Typography Helpers ──────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {TEXT_PRIMARY};
}}

QLabel#viewTitle {{
    font-size: 22px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.4px;
}}

QLabel#viewSubtitle {{
    font-size: 13px;
    color: {TEXT_MUTED};
    font-weight: 400;
}}

QLabel#fieldLabel {{
    font-size: 12px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin-bottom: 2px;
}}

QLabel#statusText {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}

/* ── Badges & Status Pills ────────────────────────────────── */
QLabel#creditBadge {{
    background-color: #241A16;
    color: #FB923C;
    border: 1px solid #7C2D12;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}}

QLabel#proBadge {{
    background: {PRIMARY_GRADIENT};
    color: #FFFFFF;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.5px;
}}

QLabel#tagPill {{
    background-color: #222228;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
}}

QLabel#tagPillActive {{
    background-color: #2A201A;
    color: #FB923C;
    border: 1px solid #C2410C;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 700;
}}

/* ── Card Containers ──────────────────────────────────────── */
QFrame#cardFrame {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 12px;
}}

QFrame#statusCard {{
    background-color: #16161A;
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 10px;
    padding: 10px 14px;
}}

/* ── Form Inputs ─────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    selection-background-color: #7C2D12;
    selection-color: #FFFFFF;
}}

QLineEdit:hover, QTextEdit:hover, QSpinBox:hover {{
    border-color: #444452;
}}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{
    border: 1.5px solid {BORDER_FOCUS};
    background-color: #18181E;
}}

QLineEdit:read-only {{
    background-color: #141418;
    color: {TEXT_MUTED};
}}

/* ── ComboBox ────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    min-height: 20px;
}}

QComboBox:hover {{
    border-color: #444452;
}}

QComboBox:focus {{
    border: 1.5px solid {BORDER_FOCUS};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: none;
}}

QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    margin-right: 8px;
}}

QComboBox QAbstractItemView,
QComboBox QListView,
QListView {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 8px;
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: #FB923C;
    padding: 4px;
    outline: none;
}}

QComboBoxPrivateContainer,
QComboBoxPrivateContainer QFrame {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 8px;
}}

QComboBox QAbstractItemView::item,
QListView::item {{
    color: {TEXT_PRIMARY};
    background-color: {BG_CARD};
    min-height: 28px;
    padding: 6px 12px;
    border-radius: 6px;
}}

QComboBox QAbstractItemView::item:hover,
QComboBox QAbstractItemView::item:selected,
QListView::item:hover,
QListView::item:selected {{
    background-color: {PRIMARY_LIGHT};
    color: #FB923C;
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {{
    background-color: #222228;
    border: 1px solid #32323D;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}

QPushButton:hover {{
    background-color: #2A2A32;
    border-color: #444452;
    color: #FFFFFF;
}}

QPushButton:pressed {{
    background-color: #1A1A20;
}}

QPushButton:disabled {{
    background-color: #16161A;
    border-color: #24242C;
    color: #52525B;
}}

QPushButton#primaryButton {{
    background: {PRIMARY_GRADIENT};
    color: #FFFFFF;
    border: 1px solid #EA580C;
    font-weight: 700;
    font-size: 14px;
    padding: 10px 22px;
    border-radius: 8px;
}}

QPushButton#primaryButton:hover {{
    background: {PRIMARY_GRADIENT_HOVER};
    border-color: #F97316;
}}

QPushButton#primaryButton:pressed {{
    background: #C2410C;
}}

QPushButton#primaryButton:disabled {{
    background: #27272A;
    color: #71717A;
    border-color: #3F3F46;
}}

QPushButton#secondaryButton {{
    background-color: #222228;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}}

QPushButton#secondaryButton:hover {{
    background-color: #2E2E38;
    border-color: #52525B;
    color: #FFFFFF;
}}

QPushButton#ghostButton {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
}}

QPushButton#ghostButton:hover {{
    background-color: #222228;
    color: {TEXT_PRIMARY};
}}

QPushButton#dangerButton {{
    background-color: #2D1414;
    color: #F87171;
    border: 1px solid #7F1D1D;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}}

QPushButton#dangerButton:hover {{
    background-color: #3D1A1A;
    border-color: #991B1B;
    color: #FCA5A5;
}}

/* ── CheckBox ────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_PRIMARY};
    font-size: 12px;
    font-weight: 500;
    spacing: 8px;
    background: transparent;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {BORDER_LIGHT};
    border-radius: 5px;
    background-color: {BG_INPUT};
}}

QCheckBox::indicator:hover {{
    border-color: {PRIMARY_COLOR};
}}

QCheckBox::indicator:checked {{
    background-color: {PRIMARY_COLOR};
    border-color: {PRIMARY_COLOR};
    image: none;
}}

/* ── ProgressBar ─────────────────────────────────────────── */
QProgressBar {{
    background-color: #121216;
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background: {PRIMARY_GRADIENT};
    border-radius: 5px;
}}

QProgressBar#audioProgress::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #06B6D4, stop:1 #38BDF8);
    border-radius: 5px;
}}

QProgressBar#renderProgress::chunk {{
    background: {PRIMARY_GRADIENT};
    border-radius: 5px;
}}

/* ── Slider ──────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: #24242C;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {PRIMARY_GRADIENT};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {PRIMARY_COLOR};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #FB923C;
    border-color: #EA580C;
}}

/* ── ScrollBar ───────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: #2C2C35;
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: #3F3F46;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
}}

QScrollBar::handle:horizontal {{
    background: #2C2C35;
    border-radius: 4px;
    min-width: 24px;
}}

/* ── Table & Tree Views ──────────────────────────────────── */
QTableWidget, QTableView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_DEFAULT};
    border-radius: 10px;
    gridline-color: #24242C;
    color: {TEXT_PRIMARY};
    selection-background-color: {PRIMARY_LIGHT};
    selection-color: #FB923C;
    outline: none;
}}

QHeaderView::section {{
    background-color: #16161A;
    color: {TEXT_SECONDARY};
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 700;
    border: none;
    border-bottom: 1px solid {BORDER_DEFAULT};
    border-right: 1px solid #24242C;
    letter-spacing: 0.3px;
}}

QHeaderView::section:last {{
    border-right: none;
}}

QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid #222228;
    color: {TEXT_PRIMARY};
}}

QTableWidget::item:selected {{
    background-color: {PRIMARY_LIGHT};
    color: #FB923C;
}}

/* ── Tooltip ─────────────────────────────────────────────── */
QToolTip {{
    background-color: #1E1E24;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_LIGHT};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}
"""

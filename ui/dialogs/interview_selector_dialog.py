"""
ui/dialogs/interview_selector_dialog.py
======================================
Interactive Clip Selection, Custom Duration Trimming, Placement Timing,
Live "View Clip" Preview Player, and 4-Side Screen Area / Crop Selection Dialog
for Template 3 (+ Interview) in AI Video Editor Studio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QUrl, QRect, QPoint, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QCursor,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QSplitter,
    QWidget,
    QMessageBox,
    QFrame,
    QSizePolicy,
)

from services.interview_scanner import InterviewSegment, format_timestamp_mmss
from core.ffmpeg_wrapper import probe_media, build_crop_filter
from utils.logger import get_logger

logger = get_logger(__name__)


def format_seconds_to_mmss(seconds: float, show_decimals: bool = False) -> str:
    """
    Format floating point seconds into clean 'MM:SS' or 'MM:SS.s' format.
    Handles rollover and boundary rounding safely.
    Examples:
      7.6 (show_decimals=True)  -> '00:07.6'
      19.4 (show_decimals=True) -> '00:19.4'
      70.0 (show_decimals=False) -> '01:10'
      125.6 (show_decimals=True) -> '02:05.6'
    """
    sec = max(0.0, float(seconds))
    if show_decimals:
        rounded_sec = round(sec, 1)
        mins = int(rounded_sec) // 60
        rem_sec = rounded_sec - (mins * 60)
        return f"{mins:02d}:{rem_sec:04.1f}"
    total_sec = int(round(sec))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


@dataclass
class TrimmedInterviewClip:
    """Represents a user-confirmed interview clip trimmed and placed on the timeline."""
    segment_id: int
    source_video: str
    trim_start_sec: float
    trim_end_sec: float
    duration_sec: float
    target_start_sec: float
    target_end_sec: float
    label: str
    speaker: str = "Speaker"
    transcript: str = ""
    is_selected: bool = True
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0
    norm_x: float = 0.0
    norm_y: float = 0.0
    norm_w: float = 1.0
    norm_h: float = 1.0


class CropOverlayWidget(QWidget):
    """
    Interactive 4-side screen area / crop rectangle overlay drawn on top of the preview video.
    Allows visual dragging of top, bottom, left, right borders and corner handles, as well as
    click-and-drag rubberband screen area selection.
    Synchronizes two-way with Top, Bottom, Left, Right spinboxes.
    """
    cropChanged = Signal(int, int, int, int)  # (crop_top, crop_bottom, crop_left, crop_right)

    HANDLE_SIZE = 8
    HIT_MARGIN = 10

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        self.source_width: int = 1920
        self.source_height: int = 1080
        self.crop_top: int = 0
        self.crop_bottom: int = 0
        self.crop_left: int = 0
        self.crop_right: int = 0

        self._drag_mode: Optional[str] = None
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_crops: Optional[tuple[int, int, int, int]] = None

    def set_source_dimensions(self, width: int, height: int) -> None:
        self.source_width = max(64, int(width))
        self.source_height = max(64, int(height))
        self._clamp_crops()
        self.update()

    def set_crop(self, top: int, bottom: int, left: int, right: int) -> None:
        self.crop_top = max(0, int(top))
        self.crop_bottom = max(0, int(bottom))
        self.crop_left = max(0, int(left))
        self.crop_right = max(0, int(right))
        self._clamp_crops()
        self.update()

    def get_crop(self) -> tuple[int, int, int, int]:
        return (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)

    def get_normalized_crop(self) -> tuple[float, float, float, float]:
        """Return resolution-independent normalized rectangle (norm_x, norm_y, norm_w, norm_h) in [0.0, 1.0]."""
        sw = max(1, self.source_width)
        sh = max(1, self.source_height)
        cw = max(0, sw - self.crop_left - self.crop_right)
        ch = max(0, sh - self.crop_top - self.crop_bottom)
        return (
            round(self.crop_left / sw, 4),
            round(self.crop_top / sh, 4),
            round(cw / sw, 4),
            round(ch / sh, 4),
        )

    def set_crop_from_normalized(self, norm_x: float, norm_y: float, norm_w: float, norm_h: float) -> None:
        """Set pixel crops from normalized ratios in [0.0, 1.0]."""
        sw = max(64, self.source_width)
        sh = max(64, self.source_height)
        nx = max(0.0, min(1.0, float(norm_x)))
        ny = max(0.0, min(1.0, float(norm_y)))
        nw = max(0.0, min(1.0 - nx, float(norm_w)))
        nh = max(0.0, min(1.0 - ny, float(norm_h)))

        crop_l = int(round(nx * sw))
        crop_t = int(round(ny * sh))
        crop_r = max(0, sw - crop_l - int(round(nw * sw)))
        crop_b = max(0, sh - crop_t - int(round(nh * sh)))
        self.set_crop(crop_t, crop_b, crop_l, crop_r)

    def reset_crop(self) -> None:
        self.set_crop(0, 0, 0, 0)
        self.cropChanged.emit(0, 0, 0, 0)

    def _clamp_crops(self) -> None:
        sw, sh = self.source_width, self.source_height
        max_crop_x = max(0, sw - 64)
        if self.crop_left + self.crop_right > max_crop_x:
            scale_x = max_crop_x / max(1, self.crop_left + self.crop_right)
            self.crop_left = int(self.crop_left * scale_x)
            self.crop_right = int(self.crop_right * scale_x)

        max_crop_y = max(0, sh - 64)
        if self.crop_top + self.crop_bottom > max_crop_y:
            scale_y = max_crop_y / max(1, self.crop_top + self.crop_bottom)
            self.crop_top = int(self.crop_top * scale_y)
            self.crop_bottom = int(self.crop_bottom * scale_y)

        self.crop_left = max(0, self.crop_left)
        self.crop_right = max(0, self.crop_right)
        self.crop_top = max(0, self.crop_top)
        self.crop_bottom = max(0, self.crop_bottom)

    def get_video_rect(self) -> QRect:
        """Calculate the letterbox/pillarbox rectangle of the video within this widget."""
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0 or self.source_width <= 0 or self.source_height <= 0:
            return self.rect()

        vid_ar = self.source_width / self.source_height
        w_ar = w / h
        if w_ar > vid_ar:
            # Pillarbox (black bars on left & right)
            vh = h
            vw = int(round(vh * vid_ar))
            vx = (w - vw) // 2
            vy = 0
        else:
            # Letterbox (black bars on top & bottom)
            vw = w
            vh = int(round(vw / vid_ar))
            vx = 0
            vy = (h - vh) // 2
        return QRect(vx, vy, max(1, vw), max(1, vh))

    def screen_to_source_coords(self, pos: QPoint) -> tuple[int, int]:
        """Convert widget screen coordinates to source video pixel coordinates, respecting letterbox/pillarbox."""
        vr = self.get_video_rect()
        if vr.width() <= 0 or vr.height() <= 0:
            return 0, 0
        clamped_x = max(vr.left(), min(vr.right(), pos.x()))
        clamped_y = max(vr.top(), min(vr.bottom(), pos.y()))
        norm_x = (clamped_x - vr.left()) / vr.width()
        norm_y = (clamped_y - vr.top()) / vr.height()
        src_x = int(round(norm_x * self.source_width))
        src_y = int(round(norm_y * self.source_height))
        return max(0, min(self.source_width, src_x)), max(0, min(self.source_height, src_y))

    def get_crop_screen_rect(self) -> QRect:
        """Map the source crop (px) into screen coordinates within the video rect."""
        vr = self.get_video_rect()
        sw = max(64, self.source_width)
        sh = max(64, self.source_height)

        crop_w_src = max(64, sw - self.crop_left - self.crop_right)
        crop_h_src = max(64, sh - self.crop_top - self.crop_bottom)

        sx = vr.x() + int(round(self.crop_left * vr.width() / sw))
        sy = vr.y() + int(round(self.crop_top * vr.height() / sh))
        rw = max(10, int(round(crop_w_src * vr.width() / sw)))
        rh = max(10, int(round(crop_h_src * vr.height() / sh)))

        return QRect(sx, sy, rw, rh)

    def _hit_test(self, pos: QPoint) -> Optional[str]:
        cr = self.get_crop_screen_rect()
        m = self.HIT_MARGIN
        x, y = pos.x(), pos.y()

        if not QRect(cr.x() - m, cr.y() - m, cr.width() + 2 * m, cr.height() + 2 * m).contains(pos):
            return None

        on_left = abs(x - cr.left()) <= m
        on_right = abs(x - cr.right()) <= m
        on_top = abs(y - cr.top()) <= m
        on_bottom = abs(y - cr.bottom()) <= m

        if on_top and on_left:
            return "NW"
        if on_top and on_right:
            return "NE"
        if on_bottom and on_left:
            return "SW"
        if on_bottom and on_right:
            return "SE"
        if on_top:
            return "N"
        if on_bottom:
            return "S"
        if on_left:
            return "W"
        if on_right:
            return "E"

        if cr.contains(pos):
            return "MOVE"
        return None

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if self._drag_mode is None:
            mode = self._hit_test(pos)
            if mode in ("NW", "SE"):
                self.setCursor(Qt.SizeFDiagCursor)
            elif mode in ("NE", "SW"):
                self.setCursor(Qt.SizeBDiagCursor)
            elif mode in ("N", "S"):
                self.setCursor(Qt.SizeVerCursor)
            elif mode in ("W", "E"):
                self.setCursor(Qt.SizeHorCursor)
            elif mode == "MOVE":
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.unsetCursor()
        else:
            self._handle_drag(pos)
            event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            vr = self.get_video_rect()
            if not vr.contains(pos):
                super().mousePressEvent(event)
                return

            mode = self._hit_test(pos)
            is_cropped = (self.crop_top > 0 or self.crop_bottom > 0 or self.crop_left > 0 or self.crop_right > 0)

            # 1. Border or corner resize handles
            if mode in ("NW", "NE", "SW", "SE", "N", "S", "W", "E"):
                self._drag_mode = mode
                self._drag_start_pos = pos
                self._drag_start_crops = (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
                event.accept()
                return

            # 2. Inside existing cropped box -> Move entire box
            if mode == "MOVE" and is_cropped and not (event.modifiers() & Qt.ShiftModifier):
                self._drag_mode = "MOVE"
                self._drag_start_pos = pos
                self._drag_start_crops = (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
                event.accept()
                return

            # 3. Rubberband screen area selection (drag new box)
            self._drag_mode = "SELECT"
            self._drag_start_pos = pos
            self._drag_start_crops = (self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_mode is not None:
            self._drag_mode = None
            self._drag_start_pos = None
            self._drag_start_crops = None
            self.unsetCursor()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _handle_drag(self, pos: QPoint) -> None:
        if not self._drag_start_pos or not self._drag_start_crops:
            return

        sw, sh = self.source_width, self.source_height
        min_dim = 64

        if self._drag_mode == "SELECT":
            x1, y1 = self.screen_to_source_coords(self._drag_start_pos)
            x2, y2 = self.screen_to_source_coords(pos)
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)

            if (max_x - min_x >= 8) or (max_y - min_y >= 8):
                self.crop_left = min_x
                self.crop_right = max(0, sw - max_x)
                self.crop_top = min_y
                self.crop_bottom = max(0, sh - max_y)
                self._clamp_crops()
                self.update()
                self.cropChanged.emit(self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
            return

        orig_top, orig_bottom, orig_left, orig_right = self._drag_start_crops
        if self._drag_mode == "MOVE":
            dx = pos.x() - self._drag_start_pos.x()
            dy = pos.y() - self._drag_start_pos.y()
            vr = self.get_video_rect()
            if vr.width() <= 0 or vr.height() <= 0:
                return

            dx_src = int(round(dx * sw / vr.width()))
            dy_src = int(round(dy * sh / vr.height()))

            box_w = sw - orig_left - orig_right
            box_h = sh - orig_top - orig_bottom

            new_left = max(0, min(sw - box_w, orig_left + dx_src))
            new_right = sw - box_w - new_left
            new_top = max(0, min(sh - box_h, orig_top + dy_src))
            new_bottom = sh - box_h - new_top

            self.crop_left = new_left
            self.crop_right = new_right
            self.crop_top = new_top
            self.crop_bottom = new_bottom
            self._clamp_crops()
            self.update()
            self.cropChanged.emit(self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
            return

        # Resize handles (N, S, W, E, NW, NE, SW, SE)
        cur_x, cur_y = self.screen_to_source_coords(pos)
        if "N" in self._drag_mode:
            max_top = sh - orig_bottom - min_dim
            self.crop_top = max(0, min(max_top, cur_y))
        if "S" in self._drag_mode:
            max_bottom = sh - orig_top - min_dim
            self.crop_bottom = max(0, min(max_bottom, sh - cur_y))
        if "W" in self._drag_mode:
            max_left = sw - orig_right - min_dim
            self.crop_left = max(0, min(max_left, cur_x))
        if "E" in self._drag_mode:
            max_right = sw - orig_left - min_dim
            self.crop_right = max(0, min(max_right, sw - cur_x))

        self._clamp_crops()
        self.update()
        self.cropChanged.emit(self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        vr = self.get_video_rect()
        cr = self.get_crop_screen_rect()

        is_cropped = (self.crop_top > 0 or self.crop_bottom > 0 or self.crop_left > 0 or self.crop_right > 0)

        # Opaque cut matte on areas outside crop rectangle:
        # Ensures only the selected screen area is visible and outside is immediately cut away
        cut_color = QColor(6, 6, 10, 248) if is_cropped else QColor(0, 0, 0, 0)
        # Top mask
        if cr.top() > vr.top():
            p.fillRect(QRect(vr.left(), vr.top(), vr.width(), cr.top() - vr.top()), cut_color)
        # Bottom mask
        if cr.bottom() < vr.bottom():
            p.fillRect(QRect(vr.left(), cr.bottom(), vr.width(), vr.bottom() - cr.bottom()), cut_color)
        # Left mask
        if cr.left() > vr.left():
            p.fillRect(QRect(vr.left(), cr.top(), cr.left() - vr.left(), cr.height()), cut_color)
        # Right mask
        if cr.right() < vr.right():
            p.fillRect(QRect(cr.right(), cr.top(), vr.right() - cr.right(), cr.height()), cut_color)

        # Draw Crop Rectangle Border
        border_pen = QPen(QColor("#38BDF8") if is_cropped else QColor("#FB923C"), 2, Qt.SolidLine)
        p.setPen(border_pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(cr)

        # Draw Rule-of-Thirds Guide Lines (subtle dashed guide lines inside crop box)
        guide_pen = QPen(QColor(255, 255, 255, 45), 1, Qt.DashLine)
        p.setPen(guide_pen)
        w3 = cr.width() / 3.0
        h3 = cr.height() / 3.0
        p.drawLine(int(cr.left() + w3), cr.top(), int(cr.left() + w3), cr.bottom())
        p.drawLine(int(cr.left() + 2 * w3), cr.top(), int(cr.left() + 2 * w3), cr.bottom())
        p.drawLine(cr.left(), int(cr.top() + h3), cr.right(), int(cr.top() + h3))
        p.drawLine(cr.left(), int(cr.top() + 2 * h3), cr.right(), int(cr.top() + 2 * h3))

        # Draw 8 Grab Handles
        handle_size = self.HANDLE_SIZE
        hs2 = handle_size // 2
        handle_brush = QBrush(QColor("#38BDF8") if is_cropped else QColor("#FB923C"))
        handle_pen = QPen(QColor("#FFFFFF"), 1.5)
        p.setPen(handle_pen)
        p.setBrush(handle_brush)

        points = [
            (cr.left(), cr.top()),                       # NW
            (cr.center().x(), cr.top()),                 # N
            (cr.right(), cr.top()),                      # NE
            (cr.right(), cr.center().y()),               # E
            (cr.right(), cr.bottom()),                   # SE
            (cr.center().x(), cr.bottom()),              # S
            (cr.left(), cr.bottom()),                    # SW
            (cr.left(), cr.center().y()),                # W
        ]
        for px, py in points:
            p.drawRect(px - hs2, py - hs2, handle_size, handle_size)

        # Draw Dimension & Mode Pill
        cw = max(64, self.source_width - self.crop_left - self.crop_right)
        ch = max(64, self.source_height - self.crop_top - self.crop_bottom)
        pill_text = f"✂ KEPT SCREEN AREA: {cw} × {ch} (OUTSIDE CUT)" if is_cropped else f"Full Frame: {cw} × {ch}"
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        text_rect = p.fontMetrics().boundingRect(pill_text)
        pill_w = text_rect.width() + 20
        pill_h = text_rect.height() + 8
        pill_x = cr.center().x() - pill_w // 2
        pill_y = max(vr.top() + 4, cr.top() - pill_h - 4)
        if pill_y < vr.top() + 4:
            pill_y = cr.top() + 6

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(18, 18, 22, 235))
        p.drawRoundedRect(pill_x, pill_y, pill_w, pill_h, 5, 5)

        p.setPen(QColor("#38BDF8") if is_cropped else QColor("#34D399"))
        p.drawText(QRect(pill_x, pill_y, pill_w, pill_h), Qt.AlignCenter, pill_text)
        p.end()


class VideoPreviewCropContainer(QFrame):
    """Container holding QVideoWidget and CropOverlayWidget stacked together."""
    def __init__(self, video_widget: QVideoWidget, overlay: CropOverlayWidget, parent=None):
        super().__init__(parent)
        self.video_widget = video_widget
        self.overlay = overlay
        self.video_widget.setParent(self)
        self.overlay.setParent(self)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #09090D; border: 1.5px solid #2C2C35; border-radius: 8px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        self.video_widget.setGeometry(0, 0, w, h)
        self.overlay.setGeometry(0, 0, w, h)
        self.overlay.raise_()


class _RowControls:
    """Internal container for widget references in a table row."""
    def __init__(
        self,
        segment: InterviewSegment,
        chk_include: QCheckBox,
        spin_start: QDoubleSpinBox,
        spin_end: QDoubleSpinBox,
        lbl_dur: QLabel,
        spin_target: QDoubleSpinBox,
        btn_view_clip: Optional[QPushButton] = None,
        lbl_start_time: Optional[QLabel] = None,
        lbl_end_time: Optional[QLabel] = None,
        lbl_target_time: Optional[QLabel] = None,
        crop_top: int = 0,
        crop_bottom: int = 0,
        crop_left: int = 0,
        crop_right: int = 0,
    ) -> None:
        self.segment = segment
        self.chk_include = chk_include
        self.spin_start = spin_start
        self.spin_end = spin_end
        self.lbl_dur = lbl_dur
        self.spin_target = spin_target
        self.btn_view_clip = btn_view_clip
        self.lbl_start_time = lbl_start_time
        self.lbl_end_time = lbl_end_time
        self.lbl_target_time = lbl_target_time
        self.crop_top = max(0, int(crop_top))
        self.crop_bottom = max(0, int(crop_bottom))
        self.crop_left = max(0, int(crop_left))
        self.crop_right = max(0, int(crop_right))


class InterviewSelectorDialog(QDialog):
    """
    Modal dialog allowing users to review AI-detected interview segments,
    watch/preview individual clips with playback & range seeking, and define
    interactive 4-side screen area / crop margins (Top, Bottom, Left, Right).
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        source_video: str = "",
        segments: Optional[list[InterviewSegment]] = None,
        confirmed_clips: Optional[list[TrimmedInterviewClip]] = None,
    ) -> None:
        super().__init__(parent)
        self.source_video = str(source_video)
        self._all_segments: list[InterviewSegment] = list(segments or [])
        self.segments = list(segments or [])
        self.confirmed_clips: list[TrimmedInterviewClip] = []
        self._rows: list[_RowControls] = []
        self._active_preview_row_idx: Optional[int] = None
        self._source_dims: Optional[tuple[int, int]] = None

        # Map previously confirmed clips by segment_id for pre-populating state
        self._prev_confirmed_map: dict[int, TrimmedInterviewClip] = {}
        if confirmed_clips:
            self._prev_confirmed_map = {c.segment_id: c for c in confirmed_clips}

        # Initialize QMediaPlayer & audio output engine
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.85)

        self.setWindowTitle("🎬 Interview Clip Selection, Trimming & Crop Studio")
        self.resize(1440, 840)
        self.setMinimumSize(1200, 680)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F0F13;
                color: #F4F4F5;
            }
            QTableWidget {
                background-color: #141419;
                border: 1px solid #27272A;
                border-radius: 8px;
                gridline-color: #202028;
                color: #F4F4F5;
                font-size: 12px;
                selection-background-color: #1E293B;
            }
            QTableWidget::item {
                padding: 4px 6px;
            }
            QTableWidget::item:selected {
                background-color: #1E293B;
            }
            QHeaderView::section {
                background-color: #181820;
                color: #94A3B8;
                font-weight: 700;
                font-size: 11px;
                border: none;
                border-bottom: 2px solid #27272A;
                padding: 8px 8px;
            }
            QScrollBar:vertical {
                background: #121216;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3F3F46;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #52525B;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #18181D;
                border: 1px solid #3F3F46;
                border-radius: 6px;
                color: #F4F4F5;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 4px;
                min-height: 26px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #38BDF8;
                background-color: #202028;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 17px;
                border-left: 1px solid #3F3F46;
                border-top-right-radius: 5px;
                background-color: #26262E;
            }
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 17px;
                border-left: 1px solid #3F3F46;
                border-bottom-right-radius: 5px;
                background-color: #26262E;
            }
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
                background-color: #38BDF8;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #27272A;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #FB923C;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #EA580C;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #FFEDD5;
                border-color: #F97316;
            }
            QCheckBox {
                font-size: 11px;
                color: #D4D4D8;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1.5px solid #52525B;
                background-color: #18181D;
            }
            QCheckBox::indicator:hover {
                border-color: #FB923C;
            }
            QCheckBox::indicator:checked {
                background-color: #EA580C;
                border-color: #F97316;
            }
            QPushButton {
                background-color: #22222A;
                color: #E4E4E7;
                border: 1px solid #3F3F46;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                padding: 3px 10px;
            }
            QPushButton:hover {
                background-color: #2C2C36;
                border-color: #71717A;
            }
            QPushButton#secondaryButton {
                background-color: #22222A;
                color: #E4E4E7;
                border: 1px solid #3F3F46;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                padding: 3px 10px;
            }
            QPushButton#secondaryButton:hover {
                background-color: #2C2C36;
                border-color: #71717A;
            }
            QFrame#rightPane {
                background-color: #15151A;
                border: 1px solid #27272A;
                border-radius: 8px;
            }
            QFrame#transportDock {
                background-color: #181820;
                border: 1px solid #27272A;
                border-radius: 6px;
            }
            QFrame#cropBox {
                background-color: #181820;
                border: 1px solid #27272A;
                border-radius: 6px;
            }
            QLabel#cropLabel {
                color: #94A3B8;
                font-size: 11px;
                font-weight: 600;
                background: transparent;
                border: none;
                padding: 0px;
            }
            QSpinBox#cropSpinBox {
                background-color: #18181D;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                color: #F4F4F5;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 4px;
                min-height: 24px;
            }
            QSpinBox#cropSpinBox:focus {
                border: 1px solid #38BDF8;
                background-color: #202028;
            }
            QPushButton#presetButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 5px;
                font-weight: 700;
                font-size: 11px;
                padding: 3px 8px;
                min-height: 24px;
            }
            QPushButton#presetButton:hover {
                background-color: #0284C7;
                color: #FFFFFF;
                border-color: #38BDF8;
            }
            QPushButton#presetButton:pressed {
                background-color: #0369A1;
                color: #FFFFFF;
            }
            QPushButton#btnResetCrop {
                background-color: #22222A;
                color: #E4E4E7;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                font-weight: 600;
                font-size: 11px;
                padding: 3px 12px;
                min-height: 26px;
            }
            QPushButton#btnResetCrop:hover {
                background-color: #2C2C36;
                border-color: #38BDF8;
                color: #38BDF8;
            }
            QPushButton#primaryButton {
                background-color: #EA580C;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 700;
                font-size: 12px;
                padding: 6px 18px;
            }
            QPushButton#primaryButton:hover {
                background-color: #F97316;
            }
        """)

        self._setup_ui()
        self._populate_table()
        self._connect_player_signals()
        self._update_summary_and_validation()

        # Pre-select and load the first clip if available
        if self._rows:
            self._load_clip_for_preview(0)

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(10)

        # ── Header ──────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_left = QVBoxLayout()
        header_left.setSpacing(3)

        title_lbl = QLabel("🎬 Select, Trim & Crop Interview Clips")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 800; color: #F4F4F5;")
        v_name = Path(self.source_video).name if self.source_video else "Scanned Footage"
        sub_lbl = QLabel(f"Source Footage: {v_name} • {len(self.segments)} detected dialogue segment(s)")
        sub_lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")

        header_left.addWidget(title_lbl)
        header_left.addWidget(sub_lbl)
        header_row.addLayout(header_left)
        header_row.addStretch()

        # Selection Toolbar Actions
        self.chk_show_all_scenes = QCheckBox("Show all scenes")
        self.chk_show_all_scenes.setToolTip("Toggle to view all detected clips, including those with lower interview framing confidence.")
        self.chk_show_all_scenes.setStyleSheet("font-size: 11px; color: #94A3B8; margin-right: 6px;")
        self.chk_show_all_scenes.toggled.connect(self._on_toggle_show_all_scenes)

        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.setObjectName("secondaryButton")
        self.btn_select_all.clicked.connect(self._select_all_clips)

        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.setObjectName("secondaryButton")
        self.btn_clear_all.clicked.connect(self._clear_all_clips)

        self.lbl_summary_badge = QLabel("0 clips selected")
        self.lbl_summary_badge.setStyleSheet(
            "background-color: #122B22; color: #34D399; border: 1px solid #065F46; "
            "border-radius: 5px; font-size: 11px; font-weight: 700; padding: 5px 10px;"
        )

        header_row.addWidget(self.chk_show_all_scenes)
        header_row.addWidget(self.btn_select_all)
        header_row.addWidget(self.btn_clear_all)
        header_row.addSpacing(6)
        header_row.addWidget(self.lbl_summary_badge)
        root_layout.addLayout(header_row)

        # ── Main Horizontal Splitter (Left: Table | Right: Preview & Crop) ───
        self.splitter = QSplitter(Qt.Horizontal)

        # ── Left Pane: Clips Table & Overlap Warning ─────────────────────────
        left_pane = QFrame()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Include",
            "Clip / Topic",
            "Source Start",
            "Source End",
            "Duration",
            "Target Video Placement",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(78)

        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)

        self.table.setColumnWidth(0, 65)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 140)
        self.table.setColumnWidth(5, 168)

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(lambda row, col: self._load_clip_for_preview(row))
        left_layout.addWidget(self.table)

        # Overlap Warning Card
        self.card_warning = QFrame()
        self.card_warning.setStyleSheet(
            "background-color: #2D1515; border: 1px solid #7F1D1D; border-radius: 6px; padding: 6px 12px;"
        )
        warning_layout = QHBoxLayout(self.card_warning)
        warning_layout.setContentsMargins(6, 4, 6, 4)
        warning_layout.setSpacing(8)

        self.lbl_warning_icon = QLabel("⚠️")
        self.lbl_warning_icon.setStyleSheet("font-size: 13px;")
        self.lbl_warning_text = QLabel("Timeline overlap detected.")
        self.lbl_warning_text.setStyleSheet("font-size: 11px; font-weight: 600; color: #FCA5A5;")

        warning_layout.addWidget(self.lbl_warning_icon)
        warning_layout.addWidget(self.lbl_warning_text, stretch=1)
        self.card_warning.setVisible(False)
        left_layout.addWidget(self.card_warning)

        self.splitter.addWidget(left_pane)

        # ── Right Pane: View Clip Player & 4-Side Crop Studio ─────────────────
        right_pane = QFrame()
        right_pane.setObjectName("rightPane")
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        # Preview Header (Title & Active Range)
        preview_header = QVBoxLayout()
        preview_header.setSpacing(3)
        self.lbl_preview_clip_title = QLabel("🎬 Preview Clip")
        self.lbl_preview_clip_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        self.lbl_preview_clip_title.setWordWrap(True)

        self.lbl_preview_clip_range = QLabel("⏱ Select a clip on the left to watch preview")
        self.lbl_preview_clip_range.setStyleSheet("font-size: 11px; font-weight: 600; color: #FB923C;")

        preview_header.addWidget(self.lbl_preview_clip_title)
        preview_header.addWidget(self.lbl_preview_clip_range)
        right_layout.addLayout(preview_header)

        # Video Player Canvas + Stacked Interactive Crop Overlay
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 6px;")
        self._player.setVideoOutput(self.video_widget)

        self.crop_overlay = CropOverlayWidget()
        self.crop_overlay.cropChanged.connect(self._on_crop_overlay_changed)

        self.video_container = VideoPreviewCropContainer(self.video_widget, self.crop_overlay)
        self.video_container.setMinimumHeight(170)
        right_layout.addWidget(self.video_container, stretch=1)

        # Media Player Transport Controls (2-Row Layout: Scrubber on Top, Buttons Below)
        transport_dock = QFrame()
        transport_dock.setObjectName("transportDock")
        transport_dock.setFixedHeight(64)
        transport_layout = QVBoxLayout(transport_dock)
        transport_layout.setContentsMargins(6, 4, 6, 4)
        transport_layout.setSpacing(4)

        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.setValue(0)
        self.slider_seek.sliderMoved.connect(self._on_seek_moved)
        transport_layout.addWidget(self.slider_seek)

        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setObjectName("primaryButton")
        self.btn_play_pause.setFixedHeight(26)
        self.btn_play_pause.setFixedWidth(78)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)

        self.lbl_player_time = QLabel("00:00.0 / 00:00.0")
        self.lbl_player_time.setStyleSheet(
            "background-color: #121216; color: #38BDF8; border: 1px solid #27272A; "
            "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
            "font-size: 11px; font-weight: 700; padding: 2px 8px; min-width: 110px;"
        )
        self.lbl_player_time.setAlignment(Qt.AlignCenter)

        self.chk_loop = QCheckBox("Loop Playback")
        self.chk_loop.setChecked(True)
        self.chk_loop.setStyleSheet("font-size: 11px; font-weight: 600; color: #D4D4D8;")

        controls_row.addWidget(self.btn_play_pause)
        controls_row.addWidget(self.lbl_player_time)
        controls_row.addStretch()
        controls_row.addWidget(self.chk_loop)
        transport_layout.addLayout(controls_row)

        right_layout.addWidget(transport_dock, stretch=0)

        # ── 4-Side Screen Area / Crop Margins Controls ────────────────────────
        crop_box = QFrame()
        crop_box.setObjectName("cropBox")
        crop_box.setFixedHeight(210)
        crop_layout = QVBoxLayout(crop_box)
        crop_layout.setContentsMargins(10, 8, 10, 8)
        crop_layout.setSpacing(6)

        crop_title_row = QHBoxLayout()
        crop_tag = QLabel("✂ Interview Screen / Crop Area")
        crop_tag.setStyleSheet("font-size: 11px; font-weight: 700; color: #38BDF8;")
        self.lbl_crop_summary = QLabel("1920×1080 (Full Frame)")
        self.lbl_crop_summary.setStyleSheet(
            "background-color: #121216; color: #34D399; border: 1px solid #27272A; "
            "border-radius: 4px; font-family: 'Consolas', monospace; font-size: 10px; font-weight: 700; padding: 2px 6px;"
        )
        crop_title_row.addWidget(crop_tag)
        crop_title_row.addStretch()
        crop_title_row.addWidget(self.lbl_crop_summary)
        crop_layout.addLayout(crop_title_row)

        # Quick Framing Preset Buttons Row
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(6)

        self.btn_preset_full = QPushButton("Full Frame")
        self.btn_preset_full.setObjectName("presetButton")
        self.btn_preset_full.setFixedHeight(24)
        self.btn_preset_full.clicked.connect(lambda: self._apply_framing_preset("full"))

        self.btn_preset_left = QPushButton("Left Speaker")
        self.btn_preset_left.setObjectName("presetButton")
        self.btn_preset_left.setFixedHeight(24)
        self.btn_preset_left.clicked.connect(lambda: self._apply_framing_preset("left"))

        self.btn_preset_center = QPushButton("Center Focus")
        self.btn_preset_center.setObjectName("presetButton")
        self.btn_preset_center.setFixedHeight(24)
        self.btn_preset_center.clicked.connect(lambda: self._apply_framing_preset("center"))

        self.btn_preset_right = QPushButton("Right Speaker")
        self.btn_preset_right.setObjectName("presetButton")
        self.btn_preset_right.setFixedHeight(24)
        self.btn_preset_right.clicked.connect(lambda: self._apply_framing_preset("right"))

        preset_row.addWidget(self.btn_preset_full)
        preset_row.addWidget(self.btn_preset_left)
        preset_row.addWidget(self.btn_preset_center)
        preset_row.addWidget(self.btn_preset_right)
        crop_layout.addLayout(preset_row)

        # Spinbox grid (Generous, unclipped layout directly placed in grid)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setContentsMargins(2, 2, 2, 2)

        # Top
        top_box = QHBoxLayout()
        top_box.setContentsMargins(0, 0, 0, 0)
        top_box.setSpacing(6)
        lbl_top = QLabel("Top:")
        lbl_top.setObjectName("cropLabel")
        lbl_top.setMinimumWidth(44)
        lbl_top.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_crop_top = QSpinBox()
        self.spin_crop_top.setObjectName("cropSpinBox")
        self.spin_crop_top.setRange(0, 9999)
        self.spin_crop_top.setSingleStep(2)
        self.spin_crop_top.setSuffix(" px")
        self.spin_crop_top.setFixedWidth(98)
        self.spin_crop_top.setFixedHeight(26)
        self.spin_crop_top.valueChanged.connect(self._on_crop_spinbox_changed)
        top_box.addWidget(lbl_top)
        top_box.addWidget(self.spin_crop_top)
        grid.addLayout(top_box, 0, 0, 1, 4, Qt.AlignCenter)

        # Left & Right
        lbl_left = QLabel("Left:")
        lbl_left.setObjectName("cropLabel")
        lbl_left.setMinimumWidth(44)
        lbl_left.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_crop_left = QSpinBox()
        self.spin_crop_left.setObjectName("cropSpinBox")
        self.spin_crop_left.setRange(0, 9999)
        self.spin_crop_left.setSingleStep(2)
        self.spin_crop_left.setSuffix(" px")
        self.spin_crop_left.setFixedWidth(98)
        self.spin_crop_left.setFixedHeight(26)
        self.spin_crop_left.valueChanged.connect(self._on_crop_spinbox_changed)
        grid.addWidget(lbl_left, 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self.spin_crop_left, 1, 1, Qt.AlignLeft | Qt.AlignVCenter)

        lbl_right = QLabel("Right:")
        lbl_right.setObjectName("cropLabel")
        lbl_right.setMinimumWidth(44)
        lbl_right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_crop_right = QSpinBox()
        self.spin_crop_right.setObjectName("cropSpinBox")
        self.spin_crop_right.setRange(0, 9999)
        self.spin_crop_right.setSingleStep(2)
        self.spin_crop_right.setSuffix(" px")
        self.spin_crop_right.setFixedWidth(98)
        self.spin_crop_right.setFixedHeight(26)
        self.spin_crop_right.valueChanged.connect(self._on_crop_spinbox_changed)
        grid.addWidget(lbl_right, 1, 2, Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(self.spin_crop_right, 1, 3, Qt.AlignLeft | Qt.AlignVCenter)

        # Bottom
        bot_box = QHBoxLayout()
        bot_box.setContentsMargins(0, 0, 0, 0)
        bot_box.setSpacing(6)
        lbl_bot = QLabel("Bottom:")
        lbl_bot.setObjectName("cropLabel")
        lbl_bot.setMinimumWidth(44)
        lbl_bot.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin_crop_bottom = QSpinBox()
        self.spin_crop_bottom.setObjectName("cropSpinBox")
        self.spin_crop_bottom.setRange(0, 9999)
        self.spin_crop_bottom.setSingleStep(2)
        self.spin_crop_bottom.setSuffix(" px")
        self.spin_crop_bottom.setFixedWidth(98)
        self.spin_crop_bottom.setFixedHeight(26)
        self.spin_crop_bottom.valueChanged.connect(self._on_crop_spinbox_changed)
        bot_box.addWidget(lbl_bot)
        bot_box.addWidget(self.spin_crop_bottom)
        grid.addLayout(bot_box, 2, 0, 1, 4, Qt.AlignCenter)

        crop_layout.addLayout(grid)

        # Reset button & hint row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        lbl_crop_hint = QLabel("💡 Drag box on video or click presets to crop speaker")
        lbl_crop_hint.setStyleSheet("font-size: 10px; color: #64748B;")
        self.btn_reset_crop = QPushButton("↺ Reset Full Frame")
        self.btn_reset_crop.setObjectName("btnResetCrop")
        self.btn_reset_crop.setFixedHeight(26)
        self.btn_reset_crop.setMinimumWidth(130)
        self.btn_reset_crop.clicked.connect(self._reset_crop)
        btn_row.addWidget(lbl_crop_hint)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_reset_crop)
        crop_layout.addLayout(btn_row)

        right_layout.addWidget(crop_box, stretch=0)

        self.splitter.addWidget(right_pane)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([900, 520])
        root_layout.addWidget(self.splitter, stretch=1)

        # ── Bottom Actions ───────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        tip_lbl = QLabel("💡 Tip: Click any row or '👁 View Clip' to preview footage and crop out logos, watermarks, or black bars.")
        tip_lbl.setStyleSheet("font-size: 11px; color: #94A3B8; font-style: normal;")

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setFixedHeight(30)
        self.btn_cancel.setFixedWidth(84)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_confirm.setObjectName("primaryButton")
        self.btn_confirm.setFixedHeight(30)
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)

        bottom_row.addWidget(tip_lbl)
        bottom_row.addStretch()
        bottom_row.addWidget(self.btn_cancel)
        bottom_row.addWidget(self.btn_confirm)
        root_layout.addLayout(bottom_row)

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.segments))
        self._rows.clear()

        for row_idx, seg in enumerate(self.segments):
            prev_clip = self._prev_confirmed_map.get(seg.segment_id)
            is_checked = prev_clip.is_selected if prev_clip else True
            init_start = prev_clip.trim_start_sec if prev_clip else seg.start_sec
            init_end = prev_clip.trim_end_sec if prev_clip else seg.end_sec
            init_target = prev_clip.target_start_sec if prev_clip else seg.start_sec
            init_top = getattr(prev_clip, "crop_top", getattr(seg, "crop_top", 0))
            init_bot = getattr(prev_clip, "crop_bottom", getattr(seg, "crop_bottom", 0))
            init_left = getattr(prev_clip, "crop_left", getattr(seg, "crop_left", 0))
            init_right = getattr(prev_clip, "crop_right", getattr(seg, "crop_right", 0))

            # Col 0: Checkbox
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(6, 0, 6, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setChecked(is_checked)
            chk.toggled.connect(self._on_row_value_changed)
            chk_layout.addWidget(chk)
            self.table.setCellWidget(row_idx, 0, chk_widget)

            # Col 1: Topic / Summary & Speaker + "View Clip" action button
            topic_widget = QWidget()
            topic_layout = QVBoxLayout(topic_widget)
            topic_layout.setContentsMargins(10, 6, 10, 6)
            topic_layout.setSpacing(4)
            topic_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)
            top_row.setSpacing(8)
            lbl_title = QLabel(f"<b>#{seg.segment_id}</b> {seg.label}")
            lbl_title.setWordWrap(True)
            lbl_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5; line-height: 1.25;")
            top_row.addWidget(lbl_title, stretch=1)

            btn_view = QPushButton("👁 View Clip")
            btn_view.setObjectName("secondaryButton")
            btn_view.setFixedHeight(24)
            btn_view.setFixedWidth(86)
            btn_view.setStyleSheet(
                "QPushButton { background-color: #0369A1; color: #FFFFFF; border: 1px solid #38BDF8; "
                "border-radius: 4px; font-size: 10px; font-weight: 700; padding: 2px 6px; } "
                "QPushButton:hover { background-color: #0284C7; border-color: #7DD3FC; }"
            )
            btn_view.clicked.connect(lambda _, r=row_idx: self._load_clip_for_preview(r))
            top_row.addWidget(btn_view, 0, Qt.AlignRight | Qt.AlignVCenter)
            topic_layout.addLayout(top_row)

            spk_text = getattr(seg, "speaker", "Speaker")
            clean_tr = seg.transcript.strip()
            short_tr = clean_tr[:115] + "..." if len(clean_tr) > 115 else clean_tr
            lbl_sub = QLabel(f"<span style='color:#38BDF8; font-weight:700;'>[{spk_text}]</span> <span style='color:#A1A1AA;'>{short_tr}</span>")
            lbl_sub.setWordWrap(True)
            lbl_sub.setStyleSheet("font-size: 11px; line-height: 1.3;")
            topic_layout.addWidget(lbl_sub)

            full_tooltip = (
                f"Clip #{seg.segment_id}: {seg.label}\n"
                f"Speaker: {spk_text}\n"
                f"Transcript: {seg.transcript}\n"
                f"Detected Range: {format_seconds_to_mmss(seg.start_sec, True)} - {format_seconds_to_mmss(seg.end_sec, True)} ({seg.duration_sec:.1f}s)"
            )
            topic_widget.setToolTip(full_tooltip)
            lbl_title.setToolTip(full_tooltip)
            lbl_sub.setToolTip(full_tooltip)
            self.table.setCellWidget(row_idx, 1, topic_widget)

            # Col 2: Source Start
            start_widget = QWidget()
            start_layout = QVBoxLayout(start_widget)
            start_layout.setContentsMargins(4, 4, 4, 4)
            start_layout.setSpacing(4)
            start_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            spin_start = QDoubleSpinBox()
            spin_start.setRange(0.0, 99999.0)
            spin_start.setDecimals(1)
            spin_start.setSingleStep(0.5)
            spin_start.setValue(init_start)
            spin_start.setSuffix(" s")
            spin_start.setFixedWidth(108)

            lbl_start_time = QLabel(f"⏱ {format_seconds_to_mmss(init_start, show_decimals=True)}")
            lbl_start_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_start_time.setStyleSheet(
                "background-color: #181820; color: #38BDF8; border: 1px solid #27272A; "
                "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                "font-size: 11px; font-weight: 600; padding: 2px 6px; min-height: 18px;"
            )

            spin_start.valueChanged.connect(self._on_row_value_changed)
            start_layout.addWidget(spin_start)
            start_layout.addWidget(lbl_start_time)
            self.table.setCellWidget(row_idx, 2, start_widget)

            # Col 3: Source End
            end_widget = QWidget()
            end_layout = QVBoxLayout(end_widget)
            end_layout.setContentsMargins(4, 4, 4, 4)
            end_layout.setSpacing(4)
            end_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            spin_end = QDoubleSpinBox()
            spin_end.setRange(0.0, 99999.0)
            spin_end.setDecimals(1)
            spin_end.setSingleStep(0.5)
            spin_end.setValue(init_end)
            spin_end.setSuffix(" s")
            spin_end.setFixedWidth(108)

            lbl_end_time = QLabel(f"⏱ {format_seconds_to_mmss(init_end, show_decimals=True)}")
            lbl_end_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_end_time.setStyleSheet(
                "background-color: #181820; color: #38BDF8; border: 1px solid #27272A; "
                "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                "font-size: 11px; font-weight: 600; padding: 2px 6px; min-height: 18px;"
            )

            spin_end.valueChanged.connect(self._on_row_value_changed)
            end_layout.addWidget(spin_end)
            end_layout.addWidget(lbl_end_time)
            self.table.setCellWidget(row_idx, 3, end_widget)

            # Col 4: Live Duration Label
            dur_widget = QWidget()
            dur_layout = QVBoxLayout(dur_widget)
            dur_layout.setContentsMargins(4, 4, 4, 4)
            dur_layout.setSpacing(4)
            dur_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            dur_lbl = QLabel()
            dur_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dur_lbl.setStyleSheet(
                "background-color: #064E3B; color: #34D399; border: 1px solid #059669; "
                "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                "font-size: 11px; font-weight: 700; padding: 3px 6px;"
            )
            dur_layout.addWidget(dur_lbl)
            self.table.setCellWidget(row_idx, 4, dur_widget)

            # Col 5: Target Video Placement
            target_widget = QWidget()
            target_layout = QVBoxLayout(target_widget)
            target_layout.setContentsMargins(4, 4, 4, 4)
            target_layout.setSpacing(4)
            target_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            spin_target = QDoubleSpinBox()
            spin_target.setRange(0.0, 99999.0)
            spin_target.setDecimals(1)
            spin_target.setSingleStep(1.0)
            spin_target.setValue(init_target)
            spin_target.setSuffix(" s")
            spin_target.setFixedWidth(118)

            lbl_target_time = QLabel()
            lbl_target_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_target_time.setStyleSheet(
                "background-color: #271C0C; color: #FBBF24; border: 1px solid #78350F; "
                "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                "font-size: 10px; font-weight: 700; padding: 2px 6px; min-height: 18px;"
            )

            spin_target.valueChanged.connect(self._on_row_value_changed)
            target_layout.addWidget(spin_target)
            target_layout.addWidget(lbl_target_time)
            self.table.setCellWidget(row_idx, 5, target_widget)

            self.table.setRowHeight(row_idx, 80)

            row_controls = _RowControls(
                segment=seg,
                chk_include=chk,
                spin_start=spin_start,
                spin_end=spin_end,
                lbl_dur=dur_lbl,
                spin_target=spin_target,
                btn_view_clip=btn_view,
                lbl_start_time=lbl_start_time,
                lbl_end_time=lbl_end_time,
                lbl_target_time=lbl_target_time,
                crop_top=init_top,
                crop_bottom=init_bot,
                crop_left=init_left,
                crop_right=init_right,
            )
            self._rows.append(row_controls)

    def _connect_player_signals(self) -> None:
        self._player.positionChanged.connect(self._on_player_position_changed)

    def _load_clip_for_preview(self, row_idx: int) -> None:
        if row_idx < 0 or row_idx >= len(self._rows):
            return

        # Save previous active clip's crop values first
        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            prev_row = self._rows[self._active_preview_row_idx]
            prev_row.crop_top = self.spin_crop_top.value()
            prev_row.crop_bottom = self.spin_crop_bottom.value()
            prev_row.crop_left = self.spin_crop_left.value()
            prev_row.crop_right = self.spin_crop_right.value()

        self._active_preview_row_idx = row_idx
        row = self._rows[row_idx]

        # Highlight table row
        self.table.blockSignals(True)
        self.table.selectRow(row_idx)
        self.table.blockSignals(False)

        spk = getattr(row.segment, "speaker", "Speaker")
        self.lbl_preview_clip_title.setText(f"🎬 Clip #{row.segment.segment_id}: {row.segment.label}")
        start_s = row.spin_start.value()
        end_s = row.spin_end.value()
        dur_s = max(0.0, end_s - start_s)
        self.lbl_preview_clip_range.setText(
            f"[{spk}] ⏱ {format_seconds_to_mmss(start_s, True)} → {format_seconds_to_mmss(end_s, True)} ({dur_s:.1f}s)"
        )

        # Source video probing and media player setup
        vid_path = Path(self.source_video) if self.source_video else None
        if vid_path and vid_path.is_file():
            if self._source_dims is None:
                try:
                    info = probe_media(vid_path)
                    w = info.width if info.width > 0 else 1920
                    h = info.height if info.height > 0 else 1080
                    self._source_dims = (w, h)
                except Exception:
                    self._source_dims = (1920, 1080)

            sw, sh = self._source_dims
            self.crop_overlay.set_source_dimensions(sw, sh)

            self.spin_crop_left.setMaximum(max(0, sw - 64))
            self.spin_crop_right.setMaximum(max(0, sw - 64))
            self.spin_crop_top.setMaximum(max(0, sh - 64))
            self.spin_crop_bottom.setMaximum(max(0, sh - 64))

            # Set media source and seek
            current_source = self._player.source().toLocalFile()
            if not current_source or Path(current_source).resolve() != vid_path.resolve():
                self._player.setSource(QUrl.fromLocalFile(str(vid_path)))

            start_ms = int(start_s * 1000)
            self._player.setPosition(start_ms)
        else:
            self._source_dims = (1920, 1080)
            self.crop_overlay.set_source_dimensions(1920, 1080)

        # Sync 4-side crop controls with row values
        self._sync_crop_controls(row.crop_top, row.crop_bottom, row.crop_left, row.crop_right)

        # Reset seek slider position
        self.slider_seek.blockSignals(True)
        self.slider_seek.setValue(0)
        self.slider_seek.blockSignals(False)
        self.lbl_player_time.setText(f"00:00.0 / {format_seconds_to_mmss(dur_s, True)}")
        self.btn_play_pause.setText("▶ Play")

    def _sync_crop_controls(self, top: int, bottom: int, left: int, right: int) -> None:
        self.spin_crop_top.blockSignals(True)
        self.spin_crop_bottom.blockSignals(True)
        self.spin_crop_left.blockSignals(True)
        self.spin_crop_right.blockSignals(True)

        self.spin_crop_top.setValue(top)
        self.spin_crop_bottom.setValue(bottom)
        self.spin_crop_left.setValue(left)
        self.spin_crop_right.setValue(right)

        self.spin_crop_top.blockSignals(False)
        self.spin_crop_bottom.blockSignals(False)
        self.spin_crop_left.blockSignals(False)
        self.spin_crop_right.blockSignals(False)

        self.crop_overlay.set_crop(top, bottom, left, right)
        self._update_crop_summary_label()

    def _on_crop_spinbox_changed(self) -> None:
        top = self.spin_crop_top.value()
        bot = self.spin_crop_bottom.value()
        left = self.spin_crop_left.value()
        right = self.spin_crop_right.value()

        self.crop_overlay.set_crop(top, bottom=bot, left=left, right=right)
        actual_top, actual_bot, actual_left, actual_right = self.crop_overlay.get_crop()

        if (actual_top, actual_bot, actual_left, actual_right) != (top, bot, left, right):
            self.spin_crop_top.blockSignals(True)
            self.spin_crop_bottom.blockSignals(True)
            self.spin_crop_left.blockSignals(True)
            self.spin_crop_right.blockSignals(True)
            self.spin_crop_top.setValue(actual_top)
            self.spin_crop_bottom.setValue(actual_bot)
            self.spin_crop_left.setValue(actual_left)
            self.spin_crop_right.setValue(actual_right)
            self.spin_crop_top.blockSignals(False)
            self.spin_crop_bottom.blockSignals(False)
            self.spin_crop_left.blockSignals(False)
            self.spin_crop_right.blockSignals(False)

        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            row = self._rows[self._active_preview_row_idx]
            row.crop_top = actual_top
            row.crop_bottom = actual_bot
            row.crop_left = actual_left
            row.crop_right = actual_right

        self._update_crop_summary_label()

    def _on_crop_overlay_changed(self, top: int, bottom: int, left: int, right: int) -> None:
        self.spin_crop_top.blockSignals(True)
        self.spin_crop_bottom.blockSignals(True)
        self.spin_crop_left.blockSignals(True)
        self.spin_crop_right.blockSignals(True)

        self.spin_crop_top.setValue(top)
        self.spin_crop_bottom.setValue(bottom)
        self.spin_crop_left.setValue(left)
        self.spin_crop_right.setValue(right)

        self.spin_crop_top.blockSignals(False)
        self.spin_crop_bottom.blockSignals(False)
        self.spin_crop_left.blockSignals(False)
        self.spin_crop_right.blockSignals(False)

        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            row = self._rows[self._active_preview_row_idx]
            row.crop_top = top
            row.crop_bottom = bottom
            row.crop_left = left
            row.crop_right = right

        self._update_crop_summary_label()

    def _apply_framing_preset(self, preset: str) -> None:
        """Apply a framing preset (full, left, center, right) to the active clip."""
        sw = max(64, self.crop_overlay.source_width)
        sh = max(64, self.crop_overlay.source_height)

        if preset == "full":
            self.crop_overlay.reset_crop()
            self._sync_crop_controls(0, 0, 0, 0)
        elif preset == "left":
            # Left speaker: focus on left 55% of width, full height
            self.crop_overlay.set_crop_from_normalized(norm_x=0.0, norm_y=0.0, norm_w=0.55, norm_h=1.0)
            t, b, l, r = self.crop_overlay.get_crop()
            self._sync_crop_controls(t, b, l, r)
        elif preset == "right":
            # Right speaker: focus on right 55% of width, full height
            self.crop_overlay.set_crop_from_normalized(norm_x=0.45, norm_y=0.0, norm_w=0.55, norm_h=1.0)
            t, b, l, r = self.crop_overlay.get_crop()
            self._sync_crop_controls(t, b, l, r)
        elif preset == "center":
            # Center focus: center 60% width, central 85% height
            self.crop_overlay.set_crop_from_normalized(norm_x=0.20, norm_y=0.08, norm_w=0.60, norm_h=0.84)
            t, b, l, r = self.crop_overlay.get_crop()
            self._sync_crop_controls(t, b, l, r)

        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            row = self._rows[self._active_preview_row_idx]
            row.crop_top, row.crop_bottom, row.crop_left, row.crop_right = self.crop_overlay.get_crop()

    def _reset_crop(self) -> None:
        self._sync_crop_controls(0, 0, 0, 0)
        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            row = self._rows[self._active_preview_row_idx]
            row.crop_top = 0
            row.crop_bottom = 0
            row.crop_left = 0
            row.crop_right = 0

    def _update_crop_summary_label(self) -> None:
        sw = self.crop_overlay.source_width
        sh = self.crop_overlay.source_height
        top = self.spin_crop_top.value()
        bot = self.spin_crop_bottom.value()
        left = self.spin_crop_left.value()
        right = self.spin_crop_right.value()

        cw = max(64, sw - left - right)
        ch = max(64, sh - top - bot)

        if top == 0 and bot == 0 and left == 0 and right == 0:
            self.lbl_crop_summary.setText(f"{sw}×{sh} (Full Frame)")
            self.lbl_crop_summary.setStyleSheet(
                "background-color: #121216; color: #34D399; border: 1px solid #27272A; "
                "border-radius: 4px; font-family: 'Consolas', monospace; font-size: 10px; font-weight: 700; padding: 2px 6px;"
            )
        else:
            self.lbl_crop_summary.setText(f"{cw}×{ch} (Cropped)")
            self.lbl_crop_summary.setStyleSheet(
                "background-color: #121216; color: #FB923C; border: 1px solid #27272A; "
                "border-radius: 4px; font-family: 'Consolas', monospace; font-size: 10px; font-weight: 700; padding: 2px 6px;"
            )

    def _toggle_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
                row = self._rows[self._active_preview_row_idx]
                start_ms = int(row.spin_start.value() * 1000)
                end_ms = int(max(row.spin_start.value() + 0.5, row.spin_end.value()) * 1000)
                cur_pos = self._player.position()
                if cur_pos >= end_ms or cur_pos < start_ms - 200:
                    self._player.setPosition(start_ms)
            self._player.play()
            self.btn_play_pause.setText("⏸ Pause")

    def _on_seek_moved(self, slider_val: int) -> None:
        if self._active_preview_row_idx is None or self._active_preview_row_idx >= len(self._rows):
            return
        row = self._rows[self._active_preview_row_idx]
        start_ms = int(row.spin_start.value() * 1000)
        end_ms = int(max(row.spin_start.value() + 0.5, row.spin_end.value()) * 1000)
        dur_ms = max(1, end_ms - start_ms)
        target_ms = start_ms + int(slider_val * dur_ms / 1000)
        self._player.setPosition(target_ms)

    def _on_player_position_changed(self, pos_ms: int) -> None:
        if self._active_preview_row_idx is None or self._active_preview_row_idx >= len(self._rows):
            return
        row = self._rows[self._active_preview_row_idx]
        start_ms = int(row.spin_start.value() * 1000)
        end_ms = int(max(row.spin_start.value() + 0.5, row.spin_end.value()) * 1000)
        dur_ms = max(1, end_ms - start_ms)

        if pos_ms >= end_ms or pos_ms < start_ms - 200:
            if self.chk_loop.isChecked():
                self._player.setPosition(start_ms)
                return
            else:
                self._player.pause()
                self._player.setPosition(end_ms)
                self.btn_play_pause.setText("▶ Play")
                return

        rel_pos = max(0, min(dur_ms, pos_ms - start_ms))
        val = int(rel_pos * 1000 / dur_ms)
        self.slider_seek.blockSignals(True)
        self.slider_seek.setValue(val)
        self.slider_seek.blockSignals(False)

        cur_sec = rel_pos / 1000.0
        dur_sec = dur_ms / 1000.0
        self.lbl_player_time.setText(f"{format_seconds_to_mmss(cur_sec, True)} / {format_seconds_to_mmss(dur_sec, True)}")

    def _on_row_value_changed(self) -> None:
        """Triggered whenever any checkbox, start, end, or target placement spinbox changes."""
        self._update_summary_and_validation()

        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            active_row = self._rows[self._active_preview_row_idx]
            dur_s = max(0.0, active_row.spin_end.value() - active_row.spin_start.value())
            spk = getattr(active_row.segment, "speaker", "Speaker")
            self.lbl_preview_clip_range.setText(
                f"[{spk}] ⏱ {format_seconds_to_mmss(active_row.spin_start.value(), True)} → "
                f"{format_seconds_to_mmss(active_row.spin_end.value(), True)} ({dur_s:.1f}s)"
            )

    def _update_summary_and_validation(self) -> None:
        selected_count = 0
        total_duration = 0.0
        has_invalid_range = False

        for row in self._rows:
            is_active = row.chk_include.isChecked()
            start_s = row.spin_start.value()
            end_s = row.spin_end.value()
            target_s = row.spin_target.value()

            start_mmss = format_seconds_to_mmss(start_s, show_decimals=True)
            end_mmss = format_seconds_to_mmss(end_s, show_decimals=True)

            if row.lbl_start_time is not None:
                row.lbl_start_time.setText(f"⏱ {start_mmss}")
            if row.lbl_end_time is not None:
                row.lbl_end_time.setText(f"⏱ {end_mmss}")

            row.spin_start.setToolTip(f"Start: {start_mmss} ({start_s:.1f} s)")
            row.spin_end.setToolTip(f"End: {end_mmss} ({end_s:.1f} s)")

            dur = max(0.0, end_s - start_s)
            if end_s <= start_s or dur <= 0:
                row.lbl_dur.setText("Invalid (≤ 0s)")
                row.lbl_dur.setStyleSheet(
                    "background-color: #2D1515; color: #EF4444; border: 1px solid #7F1D1D; "
                    "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                    "font-size: 11px; font-weight: 700; padding: 3px 6px;"
                )
                if row.lbl_target_time is not None:
                    row.lbl_target_time.setText(f"📍 {format_seconds_to_mmss(target_s, show_decimals=True)} (Invalid)")
                    row.lbl_target_time.setStyleSheet(
                        "background-color: #2D1515; color: #EF4444; border: 1px solid #7F1D1D; "
                        "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                        "font-size: 10px; font-weight: 700; padding: 2px 6px; min-height: 18px;"
                    )
                if is_active:
                    has_invalid_range = True
            else:
                dur_mmss = format_seconds_to_mmss(dur, show_decimals=True)
                row.lbl_dur.setText(f"{dur:.1f}s ({dur_mmss})")
                row.lbl_dur.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; border: 1px solid #059669; "
                    "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                    "font-size: 11px; font-weight: 700; padding: 3px 6px;"
                )
                if row.lbl_target_time is not None:
                    t_end = target_s + dur
                    target_start_mmss = format_seconds_to_mmss(target_s, show_decimals=True)
                    target_end_mmss = format_seconds_to_mmss(t_end, show_decimals=True)
                    row.lbl_target_time.setText(f"📍 {target_start_mmss} → {target_end_mmss}")
                    row.lbl_target_time.setStyleSheet(
                        "background-color: #271C0C; color: #FBBF24; border: 1px solid #78350F; "
                        "border-radius: 4px; font-family: 'Consolas', 'Segoe UI', monospace; "
                        "font-size: 10px; font-weight: 700; padding: 2px 6px; min-height: 18px;"
                    )
                if is_active:
                    selected_count += 1
                    total_duration += dur

            row.spin_target.setToolTip(
                f"Timeline Placement: {format_seconds_to_mmss(target_s, show_decimals=True)} → "
                f"{format_seconds_to_mmss(target_s + dur, show_decimals=True)} ({dur:.1f}s duration)"
            )

        self.lbl_summary_badge.setText(
            f"✓ {selected_count} of {len(self._rows)} clips selected ({total_duration:.1f}s total)"
        )

        overlaps = self.check_target_overlaps()
        if overlaps:
            pair_strs = [f"Clip #{c1} & Clip #{c2}" for c1, c2 in overlaps[:3]]
            overlap_desc = ", ".join(pair_strs)
            self.lbl_warning_text.setText(
                f"Timeline Overlap Detected: {overlap_desc}. Please adjust target placement timestamps."
            )
            self.card_warning.setVisible(True)
        elif has_invalid_range:
            self.lbl_warning_text.setText("Warning: One or more selected clips have End Time ≤ Start Time.")
            self.card_warning.setVisible(True)
        else:
            self.card_warning.setVisible(False)

    def check_target_overlaps(self) -> list[tuple[int, int]]:
        """Detect if any two selected clips overlap in target video placement timeline."""
        active_intervals = []
        for row in self._rows:
            if not row.chk_include.isChecked():
                continue
            dur = max(0.0, row.spin_end.value() - row.spin_start.value())
            t_start = row.spin_target.value()
            t_end = t_start + dur
            active_intervals.append((row.segment.segment_id, t_start, t_end))

        overlaps: list[tuple[int, int]] = []
        for i in range(len(active_intervals)):
            id_a, start_a, end_a = active_intervals[i]
            for j in range(i + 1, len(active_intervals)):
                id_b, start_b, end_b = active_intervals[j]
                if max(start_a, start_b) < min(end_a, end_b):
                    overlaps.append((id_a, id_b))

        return overlaps

    def _on_toggle_show_all_scenes(self, checked: bool) -> None:
        """Filter table rows by interview confidence or show all detected scenes."""
        self._apply_scene_filter(checked)

    def _apply_scene_filter(self, show_all: bool) -> None:
        """Apply active scene filter based on interview framing confidence."""
        if show_all:
            self.segments = list(self._all_segments)
        else:
            # Filter to clips meeting interview confidence and verified human presence
            filtered = [
                s for s in self._all_segments
                if getattr(s, "is_human_verified", True) and getattr(s, "interview_score", 1.0) >= 0.38
            ]
            self.segments = filtered if filtered else list(self._all_segments)

        self._populate_table()
        self._update_summary_and_validation()
        if self._rows:
            self._load_clip_for_preview(0)

    def _select_all_clips(self) -> None:
        for row in self._rows:
            row.chk_include.setChecked(True)
        self._update_summary_and_validation()

    def _clear_all_clips(self) -> None:
        for row in self._rows:
            row.chk_include.setChecked(False)
        self._update_summary_and_validation()

    def _release_player(self) -> None:
        """Properly stop playback on Windows."""
        try:
            self._player.stop()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self._release_player()
        super().closeEvent(event)

    def reject(self) -> None:
        self._release_player()
        super().reject()

    def _on_confirm_clicked(self) -> None:
        """Validate selections, save crop values, and build confirmed clip models."""
        selected_rows = [r for r in self._rows if r.chk_include.isChecked()]

        if not selected_rows:
            QMessageBox.warning(
                self,
                "No Clips Selected",
                "Please select at least one interview clip to include in the video.",
            )
            return

        for r in selected_rows:
            if r.spin_end.value() <= r.spin_start.value():
                QMessageBox.warning(
                    self,
                    "Invalid Trim Range",
                    f"Clip #{r.segment.segment_id} ('{r.segment.label}') has an End Time earlier than or equal to Start Time.\n"
                    "Please adjust the start and end values.",
                )
                return

        overlaps = self.check_target_overlaps()
        if overlaps:
            pair_strs = [f"Clip #{c1} & Clip #{c2}" for c1, c2 in overlaps]
            msg = (
                "The following selected clips have overlapping target video placement ranges:\n\n"
                + "\n".join(pair_strs)
                + "\n\nPlease adjust target placement start times so clips do not clash."
            )
            QMessageBox.warning(self, "Timeline Placement Overlap", msg)
            return

        # Save active preview row's crops before building output
        if self._active_preview_row_idx is not None and self._active_preview_row_idx < len(self._rows):
            active_r = self._rows[self._active_preview_row_idx]
            active_r.crop_top = self.spin_crop_top.value()
            active_r.crop_bottom = self.spin_crop_bottom.value()
            active_r.crop_left = self.spin_crop_left.value()
            active_r.crop_right = self.spin_crop_right.value()

        sw, sh = self._source_dims if self._source_dims else (1920, 1080)
        sw = max(64, sw)
        sh = max(64, sh)

        self.confirmed_clips.clear()
        for r in selected_rows:
            s_val = round(r.spin_start.value(), 2)
            e_val = round(r.spin_end.value(), 2)
            dur_val = round(e_val - s_val, 2)
            tgt_val = round(r.spin_target.value(), 2)
            tgt_end = round(tgt_val + dur_val, 2)

            cl = max(0, r.crop_left)
            ct = max(0, r.crop_top)
            cr = max(0, r.crop_right)
            cb = max(0, r.crop_bottom)
            cw = max(0, sw - cl - cr)
            ch = max(0, sh - ct - cb)

            norm_x = round(cl / sw, 4)
            norm_y = round(ct / sh, 4)
            norm_w = round(cw / sw, 4)
            norm_h = round(ch / sh, 4)

            self.confirmed_clips.append(
                TrimmedInterviewClip(
                    segment_id=r.segment.segment_id,
                    source_video=self.source_video,
                    trim_start_sec=s_val,
                    trim_end_sec=e_val,
                    duration_sec=dur_val,
                    target_start_sec=tgt_val,
                    target_end_sec=tgt_end,
                    label=r.segment.label,
                    speaker=getattr(r.segment, "speaker", "Speaker"),
                    transcript=getattr(r.segment, "transcript", ""),
                    is_selected=True,
                    crop_top=ct,
                    crop_bottom=cb,
                    crop_left=cl,
                    crop_right=cr,
                    norm_x=norm_x,
                    norm_y=norm_y,
                    norm_w=norm_w,
                    norm_h=norm_h,
                )
            )

        logger.info(
            "User confirmed %d interview clips (Total: %.1fs)",
            len(self.confirmed_clips),
            sum(c.duration_sec for c in self.confirmed_clips),
        )
        self._release_player()
        self.accept()

    def get_confirmed_clips(self) -> list[TrimmedInterviewClip]:
        """Return confirmed TrimmedInterviewClip objects."""
        return list(self.confirmed_clips)

    def get_confirmed_clips_data(self) -> list[dict[str, Any]]:
        """Return serialized list of confirmed trimmed interview clips with crop metadata."""
        return [
            {
                "segment_id": clip.segment_id,
                "source_video": clip.source_video,
                "trim_start_sec": clip.trim_start_sec,
                "trim_end_sec": clip.trim_end_sec,
                "duration_sec": clip.duration_sec,
                "target_start_sec": clip.target_start_sec,
                "target_end_sec": clip.target_end_sec,
                "label": clip.label,
                "speaker": clip.speaker,
                "transcript": clip.transcript,
                "crop_top": getattr(clip, "crop_top", 0),
                "crop_bottom": getattr(clip, "crop_bottom", 0),
                "crop_left": getattr(clip, "crop_left", 0),
                "crop_right": getattr(clip, "crop_right", 0),
                "norm_x": getattr(clip, "norm_x", 0.0),
                "norm_y": getattr(clip, "norm_y", 0.0),
                "norm_w": getattr(clip, "norm_w", 1.0),
                "norm_h": getattr(clip, "norm_h", 1.0),
            }
            for clip in self.confirmed_clips
        ]

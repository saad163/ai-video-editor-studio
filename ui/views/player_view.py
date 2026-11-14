"""
ui/views/player_view.py
=======================
Video Player & Preview panel for CineForge AI Studio.
Embedded media player with native QVideoWidget & QMediaPlayer playback,
scrubbing timeline, volume control, and file inspection metadata.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QFileDialog, QListWidget, QListWidgetItem, QSplitter, QSlider, QSizePolicy
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from utils.config import get_config
from core.ffmpeg_wrapper import probe_media


def _format_time_ms(ms: int) -> str:
    """Format milliseconds into MM:SS format."""
    total_sec = max(0, ms // 1000)
    minutes = total_sec // 60
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:02d}"


class PlayerView(QWidget):
    """Integrated cinema player and media inspector panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_video: Path | None = None
        self._duration_ms: int = 0

        # Qt Multimedia engine
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.85)

        self._setup_ui()
        self._connect_player_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # ── Header ─────────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title_lbl = QLabel("Video Player & Quality Inspector")
        title_lbl.setObjectName("viewTitle")
        subtitle_lbl = QLabel("Inspect 2K master renders, verify speech sync, and review subtitle burn-in")
        subtitle_lbl.setObjectName("viewSubtitle")

        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        btn_browse = QPushButton("📁  Open Video File")
        btn_browse.setObjectName("secondaryButton")
        btn_browse.clicked.connect(self._browse_video)
        header_layout.addWidget(btn_browse)

        layout.addLayout(header_layout)

        # ── Main Splitter ──────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left Column: Video Library List
        left_card = QFrame()
        left_card.setObjectName("cardFrame")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        left_lbl = QLabel("Rendered Library (output/)")
        left_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #F4F4F5;")
        left_layout.addWidget(left_lbl)

        self.list_videos = QListWidget()
        self.list_videos.setStyleSheet("""
            QListWidget {
                background-color: #16161A;
                border: 1px solid #2C2C35;
                border-radius: 8px;
                color: #F4F4F5;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 6px;
                color: #D4D4D8;
            }
            QListWidget::item:hover {
                background-color: #24242C;
                color: #FFFFFF;
            }
            QListWidget::item:selected {
                background-color: #2A201A;
                color: #FB923C;
                border-left: 3px solid #F97316;
            }
        """)
        self.list_videos.itemClicked.connect(self._on_item_clicked)
        left_layout.addWidget(self.list_videos)

        btn_refresh_list = QPushButton("🔄 Refresh Library")
        btn_refresh_list.setObjectName("secondaryButton")
        btn_refresh_list.clicked.connect(self.scan_output_folder)
        left_layout.addWidget(btn_refresh_list)

        splitter.addWidget(left_card)

        # Right Column: Embedded Cinema Stage & Specs
        right_card = QFrame()
        right_card.setObjectName("cardFrame")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # Title bar
        title_bar = QHBoxLayout()
        self.lbl_selected_title = QLabel("No Video Selected")
        self.lbl_selected_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #F4F4F5;")
        self.lbl_selected_badge = QLabel("READY")
        self.lbl_selected_badge.setStyleSheet("background-color: #2A1D16; color: #FB923C; border: 1px solid #7C2D12; border-radius: 4px; font-size: 10px; font-weight: 800; padding: 2px 6px;")
        title_bar.addWidget(self.lbl_selected_title)
        title_bar.addStretch()
        title_bar.addWidget(self.lbl_selected_badge)
        right_layout.addLayout(title_bar)

        # Embedded Cinema Screen
        self.video_container = QFrame()
        self.video_container.setMinimumHeight(280)
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_container.setStyleSheet("background-color: #09090D; border: 1.5px solid #2C2C35; border-radius: 10px;")
        vid_layout = QVBoxLayout(self.video_container)
        vid_layout.setContentsMargins(2, 2, 2, 2)

        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 8px;")
        vid_layout.addWidget(self.video_widget)
        self._player.setVideoOutput(self.video_widget)

        right_layout.addWidget(self.video_container, stretch=1)

        # ── Playback Controls Dock ─────────────────────────────────────────────
        ctrl_dock = QFrame()
        ctrl_dock.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 8px 12px;")
        ctrl_layout = QVBoxLayout(ctrl_dock)
        ctrl_layout.setSpacing(6)

        # Row 1: Timeline Slider & Timestamps
        time_row = QHBoxLayout()
        time_row.setSpacing(10)
        self.lbl_time_cur = QLabel("00:00")
        self.lbl_time_cur.setStyleSheet("font-size: 11px; font-weight: 700; color: #FB923C;")

        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.setValue(0)
        self.slider_seek.sliderMoved.connect(self._on_seek_moved)

        self.lbl_time_dur = QLabel("00:00")
        self.lbl_time_dur.setStyleSheet("font-size: 11px; font-weight: 600; color: #A1A1AA;")

        time_row.addWidget(self.lbl_time_cur)
        time_row.addWidget(self.slider_seek, stretch=1)
        time_row.addWidget(self.lbl_time_dur)
        ctrl_layout.addLayout(time_row)

        # Row 2: Transport Buttons (Play/Pause, Stop, Volume, Open External)
        trans_row = QHBoxLayout()
        trans_row.setSpacing(8)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setObjectName("primaryButton")
        self.btn_play.setFixedWidth(85)
        self.btn_play.setFixedHeight(32)
        self.btn_play.clicked.connect(self._toggle_playback)
        trans_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("secondaryButton")
        self.btn_stop.setFixedWidth(75)
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.clicked.connect(self._stop_playback)
        trans_row.addWidget(self.btn_stop)

        # Volume
        vol_lbl = QLabel("🔊")
        vol_lbl.setStyleSheet("font-size: 13px;")
        self.slider_vol = QSlider(Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(85)
        self.slider_vol.setFixedWidth(80)
        self.slider_vol.valueChanged.connect(lambda v: self._audio.setVolume(v / 100.0))
        trans_row.addWidget(vol_lbl)
        trans_row.addWidget(self.slider_vol)

        trans_row.addStretch()

        self.btn_play_system = QPushButton("External Player")
        self.btn_play_system.setObjectName("secondaryButton")
        self.btn_play_system.setFixedHeight(32)
        self.btn_play_system.clicked.connect(self._open_in_system_player)
        trans_row.addWidget(self.btn_play_system)

        self.btn_open_folder = QPushButton("📂 Open Location")
        self.btn_open_folder.setObjectName("secondaryButton")
        self.btn_open_folder.setFixedHeight(32)
        self.btn_open_folder.clicked.connect(self._open_containing_folder)
        trans_row.addWidget(self.btn_open_folder)

        ctrl_layout.addLayout(trans_row)
        right_layout.addWidget(ctrl_dock)

        # Technical Specs Info Box
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #16161A; border: 1px solid #2C2C35; border-radius: 8px; padding: 10px 14px;")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setSpacing(16)

        self.lbl_info_res = QLabel("Resolution: —")
        self.lbl_info_dur = QLabel("Duration: —")
        self.lbl_info_fps = QLabel("FPS: —")
        self.lbl_info_size = QLabel("Size: —")
        self.lbl_info_codecs = QLabel("Codecs: —")

        for lbl in (self.lbl_info_res, self.lbl_info_dur, self.lbl_info_fps, self.lbl_info_size, self.lbl_info_codecs):
            lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #D4D4D8;")
            info_layout.addWidget(lbl)
        info_layout.addStretch()

        right_layout.addWidget(info_frame)

        splitter.addWidget(right_card)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter, stretch=1)

        self.scan_output_folder()

    def _connect_player_signals(self) -> None:
        self._player.positionChanged.connect(self._on_player_position_changed)
        self._player.durationChanged.connect(self._on_player_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

    def scan_output_folder(self) -> None:
        """Scan output/ directory for mp4 files."""
        self.list_videos.clear()
        cfg = get_config()
        if not cfg.output_dir.exists():
            return

        videos = sorted(cfg.output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Also include bulk_test videos if available
        bulk_test_dir = cfg.output_dir / "bulk_test"
        if bulk_test_dir.exists():
            videos.extend(sorted(bulk_test_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True))

        for v in videos:
            item = QListWidgetItem(f"🎬  {v.name}")
            item.setData(Qt.UserRole, str(v))
            self.list_videos.addItem(item)

        if videos and not self._current_video:
            self.list_videos.setCurrentRow(0)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        p_str = item.data(Qt.UserRole)
        if p_str:
            self._load_video_details(Path(p_str))

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video (*.mp4 *.mkv *.mov)")
        if path:
            self._load_video_details(Path(path))

    def _load_video_details(self, path: Path) -> None:
        self._current_video = path
        self.lbl_selected_title.setText(path.name)
        self.btn_play_system.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        # Load into QMediaPlayer
        self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))

        try:
            info = probe_media(path)
            sz_mb = path.stat().st_size / (1024 * 1024)
            self.lbl_info_res.setText(f"Res: {info.resolution_label}")
            self.lbl_info_dur.setText(f"Dur: {info.duration:.1f}s")
            self.lbl_info_fps.setText(f"FPS: {info.fps:.0f}")
            self.lbl_info_size.setText(f"Size: {sz_mb:.1f}MB")
            self.lbl_info_codecs.setText(f"{info.video_codec}/{info.audio_codec}")
        except Exception as e:
            self.lbl_info_res.setText(f"Probed: {path.name}")

    def _toggle_playback(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _stop_playback(self) -> None:
        self._player.stop()
        self.slider_seek.setValue(0)
        self.lbl_time_cur.setText("00:00")

    def _on_player_position_changed(self, pos_ms: int) -> None:
        if self._duration_ms > 0:
            fraction = pos_ms / self._duration_ms
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(int(fraction * 1000))
            self.slider_seek.blockSignals(False)
        self.lbl_time_cur.setText(_format_time_ms(pos_ms))

    def _on_player_duration_changed(self, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        self.lbl_time_dur.setText(_format_time_ms(dur_ms))

    def _on_seek_moved(self, slider_val: int) -> None:
        if self._duration_ms > 0:
            target_ms = int((slider_val / 1000.0) * self._duration_ms)
            self._player.setPosition(target_ms)

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlayingState:
            self.btn_play.setText("⏸ Pause")
        else:
            self.btn_play.setText("▶ Play")

    def refresh_library(self) -> None:
        """Public alias to reload and update output directory video list."""
        self.scan_output_folder()

    def play_video(self, path: Path | str) -> None:
        """Load and start playback of a specified video file."""
        target = Path(path)
        self.scan_output_folder()
        self._load_video_details(target)
        for i in range(self.list_videos.count()):
            item = self.list_videos.item(i)
            if item.data(Qt.UserRole) == str(target):
                self.list_videos.setCurrentItem(item)
                break
        self._player.play()

    def _open_in_system_player(self) -> None:
        if self._current_video and self._current_video.exists():
            if sys.platform == "win32":
                os.startfile(str(self._current_video.resolve()))
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_video.resolve())))

    def _open_containing_folder(self) -> None:
        if self._current_video and self._current_video.exists():
            folder = self._current_video.parent
            if sys.platform == "win32":
                subprocess.Popen(["explorer", f"/select,{self._current_video.resolve()}"])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

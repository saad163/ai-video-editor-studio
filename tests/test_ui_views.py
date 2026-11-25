"""
tests/test_ui_views.py
======================
Unit tests for PySide6 Desktop GUI components:
  - StudioView layout, controls, and responsive scroll wrapper
  - Subtitle accordion style matrix and color picker
  - BulkView batch queue table and summary counters
  - Sidebar navigation and hardware indicators
"""

import pytest
from PySide6.QtWidgets import QApplication, QScrollArea
from PySide6.QtCore import Qt

from ui.main_window import MainWindow
from ui.views.studio_view import StudioView
from ui.views.bulk_view import BulkView
from ui.components.sidebar import Sidebar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_structure(qapp):
    win = MainWindow()
    assert win.windowTitle() == "AutoCut - AI Video Editor"
    assert win.width() >= 1024
    assert win.height() >= 640
    assert win.stack.count() == 6
    win.close()


def test_sidebar_items(qapp):
    sidebar = Sidebar()
    assert len(sidebar._buttons) == 6
    assert sidebar._buttons[0].text().strip() == "Studio"
    assert sidebar._buttons[1].text().strip() == "Bulk Generation"
    assert sidebar._buttons[2].text().strip() == "YouTube Formats"
    sidebar.close()


def test_studio_view_scroll_and_inputs(qapp):
    view = StudioView()
    # Ensure view has a QScrollArea as its top-level child for responsiveness
    scroll_areas = view.findChildren(QScrollArea)
    assert len(scroll_areas) >= 1
    assert scroll_areas[0].widgetResizable() is True

    # Test inputs and defaults
    assert view.edit_title.text() != ""
    assert view.chk_include_images.isChecked() is True
    assert view.btn_start.text() == "Start AutoCut"

    # Test subtitle preset changes
    view._apply_preset_color("#facc15", "#000000")
    assert view._font_color == "#facc15"
    assert view._stroke_color == "#000000"
    view.close()


def test_bulk_view_queue(qapp, tmp_path):
    view = BulkView()
    assert view.table.columnCount() == 8

    # Insert a dummy audio file
    fake_audio = tmp_path / "chapter_01.mp3"
    fake_audio.write_bytes(b"dummy audio bytes")
    view._insert_row(fake_audio)

    assert view.table.rowCount() == 1
    assert "chapter_01" in view.table.item(0, 1).text().lower()
    view.close()


def test_worker_signals_compatibility(qapp, tmp_path):
    from ui.workers.video_worker import VideoRenderWorker
    from ui.workers.episode_worker import EpisodeCompilerWorker
    from pathlib import Path

    fake_audio = tmp_path / "test.mp3"
    fake_audio.write_bytes(b"audio")
    fake_footage = tmp_path / "footage"
    fake_footage.mkdir()
    fake_out = tmp_path / "out.mp4"

    v_worker = VideoRenderWorker(
        audio_path=fake_audio,
        images_dir=fake_footage,
        output_path=fake_out,
        title="Test"
    )

    # Verify all expected signals exist on VideoRenderWorker
    assert hasattr(v_worker, "audio_progress")
    assert hasattr(v_worker, "render_progress")
    assert hasattr(v_worker, "status_message")
    assert hasattr(v_worker, "step_changed")
    assert hasattr(v_worker, "render_finished")
    assert hasattr(v_worker, "render_failed")
    assert hasattr(v_worker, "progress_transcription")
    assert hasattr(v_worker, "progress_render")

    e_worker = EpisodeCompilerWorker(
        audio_path=fake_audio,
        footage_path=fake_footage,
        output_path=fake_out,
        title="Episode Test"
    )

    # Verify all expected signals exist on EpisodeCompilerWorker
    assert hasattr(e_worker, "audio_progress")
    assert hasattr(e_worker, "render_progress")
    assert hasattr(e_worker, "status_message")
    assert hasattr(e_worker, "step_changed")
    assert hasattr(e_worker, "render_finished")
    assert hasattr(e_worker, "render_failed")


def test_player_view_and_play_signal(qapp, tmp_path):
    from unittest.mock import patch
    from core.ffmpeg_wrapper import MediaInfo
    from ui.main_window import MainWindow

    win = MainWindow()
    fake_video = tmp_path / "test_preview.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    # Initial page is Studio View (0)
    assert win.stack.currentIndex() == 0

    # Emitting request_play_video should transition to Player View (4)
    with patch.object(win.player_view._player, "setSource"), \
         patch("ui.views.player_view.probe_media", return_value=MediaInfo(path=fake_video, width=1920, height=1080, duration=10.0, video_codec="h264", audio_codec="aac")):
        win.studio_view.request_play_video.emit(str(fake_video))
        assert win.stack.currentIndex() == 4
        assert win.player_view._current_video == fake_video


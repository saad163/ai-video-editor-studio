"""
tests/test_resolution_and_timeline.py
======================================
Tests for:
  1. Resolution mapping (720p, 1080p, 2K) in FFmpeg wrapper and SubtitleEngine.
  2. Full audio duration timeline continuous gap-filling in CharacterMatcher.
  3. UI resolution selectors in StudioView and BulkView.
"""

from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from core.ffmpeg_wrapper import RESOLUTIONS, DEFAULT_RESOLUTION
from services.character_matcher import CharacterMatcher, SegmentAssignment
from services.subtitle_engine import SubtitleEngine, SUBTITLE_RESOLUTIONS
from services.transcriber import TranscriptionResult, TranscriptionSegment


def test_resolutions_mapping():
    assert "720p" in RESOLUTIONS
    assert "1080p" in RESOLUTIONS
    assert "2k" in RESOLUTIONS
    assert RESOLUTIONS["720p"] == (1280, 720)
    assert RESOLUTIONS["1080p"] == (1920, 1080)
    assert RESOLUTIONS["2k"] == (2560, 1440)


def test_subtitle_engine_resolution_scaling(tmp_path: Path):
    sub_720 = SubtitleEngine(resolution="720p")
    assert sub_720.play_res_x == 1280
    assert sub_720.play_res_y == 720
    header_720 = sub_720._ass_header()
    assert "PlayResX: 1280" in header_720
    assert "PlayResY: 720" in header_720

    sub_1080 = SubtitleEngine(resolution="1080p")
    assert sub_1080.play_res_x == 1920
    assert sub_1080.play_res_y == 1080
    header_1080 = sub_1080._ass_header()
    assert "PlayResX: 1920" in header_1080
    assert "PlayResY: 1080" in header_1080

    sub_2k = SubtitleEngine(resolution="2k")
    assert sub_2k.play_res_x == 2560
    assert sub_2k.play_res_y == 1440


def test_character_matcher_full_timeline_matching(tmp_path: Path):
    # Setup dummy images
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img1.write_text("dummy")
    img2.write_text("dummy")

    matcher = CharacterMatcher(images_dir=tmp_path)

    # 3 segments with speech pauses and silence gaps:
    # Seg 0: 1.5s - 4.0s (pause at beginning: 0.0s - 1.5s)
    # Seg 1: 7.0s - 10.0s (silence gap between seg 0 and 1: 4.0s - 7.0s)
    # Seg 2: 13.0s - 16.0s (silence gap between seg 1 and 2: 10.0s - 13.0s)
    # Total audio duration is 20.0s (silence outro: 16.0s - 20.0s)
    segs = [
        TranscriptionSegment(start=1.5, end=4.0, text="First line"),
        TranscriptionSegment(start=7.0, end=10.0, text="Second line"),
        TranscriptionSegment(start=13.0, end=16.0, text="Third line"),
    ]
    mock_res = TranscriptionResult(
        audio_path=tmp_path / "voiceover.mp3",
        audio_hash="ABC",
        language="en",
        full_text="First line Second line Third line",
        segments=segs,
        model_used="small",
        elapsed_seconds=1.0,
    )

    total_audio_dur = 20.0
    assignments = matcher.assign(mock_res, total_duration=total_audio_dur)

    assert len(assignments) >= 3

    # Total duration of all visual clips must equal full voiceover duration (20.0s)
    total_video_dur = sum(a.duration for a in assignments)
    assert abs(total_video_dur - total_audio_dur) < 0.01

    # Clip 0 covers lead-in (0.0s -> seg 1 start 7.0s = 7.0s)
    assert abs(assignments[0].duration - 7.0) < 0.01

    # Clip 1 covers gap to seg 2 start (7.0s -> 13.0s = 6.0s)
    assert abs(assignments[1].duration - 6.0) < 0.01

    # Clip 2 covers remaining time to total_audio_dur (13.0s -> 20.0s = 7.0s)
    assert abs(assignments[2].duration - 7.0) < 0.01


def test_ui_studio_view_resolution_selector(qtbot):
    from ui.views.studio_view import StudioView
    view = StudioView()
    qtbot.addWidget(view)

    assert hasattr(view, "combo_resolution")
    assert view.combo_resolution.count() >= 3

    # Default is 1080p Full HD
    assert "1080p" in view.combo_resolution.currentText().lower()
    assert view.get_selected_resolution() == "1080p"

    # Select 2K
    view.combo_resolution.setCurrentIndex(1)
    assert view.get_selected_resolution() == "2k"

    # Select 720p
    view.combo_resolution.setCurrentIndex(2)
    assert view.get_selected_resolution() == "720p"

    # Primary Start Button must have visible text and high-contrast styling
    assert view.btn_start.text() == "Start AutoCut"
    assert view.btn_start.isEnabled() is True


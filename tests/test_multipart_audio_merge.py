"""
tests/test_multipart_audio_merge.py
===================================
Tests for Multi-Part Voiceover Audio Workflow and Auto-Merge:
  1. Natural sorting of audio parts (part 1.mp3, part 2.mp3, part 10.mp3).
  2. StudioView single vs multi-part audio selection, dynamic status badges, and output naming.
  3. VideoRenderWorker multi-part parameter initialization and backwards compatibility.
  4. HybridMediaMixer cross-part visual continuity (used_images accumulator).
  5. VideoRenderWorker._run_multipart end-to-end execution and lossless concatenation merge.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication

from core.ffmpeg_wrapper import KenBurnsPreset
from database.db import init_db
from services.character_matcher import natural_sort_key
from services.hybrid_mixer import HybridMediaMixer, VisualSlot
from services.transcriber import TranscriptionResult, TranscriptionSegment, WordTimestamp
from ui.views.studio_view import StudioView
from ui.workers.video_worker import VideoRenderWorker
from utils.config import get_config


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def mock_qmessagebox():
    """Prevent blocking modal dialogs during automated tests."""
    with patch("PySide6.QtWidgets.QMessageBox.information"), \
         patch("PySide6.QtWidgets.QMessageBox.warning"), \
         patch("PySide6.QtWidgets.QMessageBox.critical"):
        yield


def test_natural_sorting_audio_parts():
    """Verify natural sorting orders part 1, part 2, part 10 correctly."""
    unordered = [
        "chapter_part 10.mp3",
        "chapter_part 2.mp3",
        "chapter_part 1.mp3",
        "chapter_part 20.mp3",
        "chapter_part 3.mp3",
    ]
    ordered = sorted(unordered, key=natural_sort_key)
    expected = [
        "chapter_part 1.mp3",
        "chapter_part 2.mp3",
        "chapter_part 3.mp3",
        "chapter_part 10.mp3",
        "chapter_part 20.mp3",
    ]
    assert ordered == expected


def test_studio_view_single_vs_multipart_badge(qapp, tmp_path):
    """Verify StudioView updates status badge and default output name for single vs multi-part."""
    view = StudioView()

    # Create dummy audio files
    part1 = tmp_path / "Episode_7_Part_1.mp3"
    part2 = tmp_path / "Episode_7_Part_2.mp3"
    part1.write_bytes(b"dummy audio data 1")
    part2.write_bytes(b"dummy audio data 2")

    with patch("core.ffmpeg_wrapper.probe_media") as mock_probe:
        mock_info = MagicMock()
        mock_info.duration = 125.0  # 2m 5s each
        mock_probe.return_value = mock_info

        # 1. Single audio mode
        view.edit_audio.setText(str(part1))
        assert "Single Audio" in view.lbl_vo_badge.text()
        assert "02:05" in view.lbl_vo_badge.text()
        assert "Episode 7 Part 1" in view.edit_title.text()
        assert view.edit_output.text().endswith("Episode_7_Part_1.mp4")

        # 2. Multi-part audio mode
        view.edit_audio.setText(f"{part1}; {part2}")
        assert "Multi-Part Mode" in view.lbl_vo_badge.text()
        assert "2 Parts" in view.lbl_vo_badge.text()
        assert "Auto-merges" in view.lbl_vo_badge.text()
        # Cleaned title & output name without "Part 1"
        assert "Episode 7" in view.edit_title.text()
        assert view.edit_output.text().endswith("Episode_7_Full.mp4")


def test_video_render_worker_initialization(tmp_path):
    """Verify VideoRenderWorker initializes with both single audio_path and multi audio_paths."""
    p1 = tmp_path / "part1.mp3"
    p2 = tmp_path / "part2.mp3"
    p1.touch()
    p2.touch()
    out = tmp_path / "final.mp4"

    # Single mode
    w1 = VideoRenderWorker(
        audio_path=p1,
        images_dir=tmp_path,
        output_path=out,
    )
    assert len(w1.audio_paths) == 1
    assert w1.audio_path == p1.resolve()

    # Multi-part mode
    w2 = VideoRenderWorker(
        audio_path=p1,
        audio_paths=[p1, p2],
        images_dir=tmp_path,
        output_path=out,
    )
    assert len(w2.audio_paths) == 2
    assert w2.audio_paths[0] == p1.resolve()
    assert w2.audio_paths[1] == p2.resolve()
    assert w2.audio_path == p1.resolve()  # Backward compatibility


def test_hybrid_mixer_cross_part_used_images(tmp_path):
    """Verify HybridMediaMixer accumulates and respects used_images across multiple parts."""
    # Create 3 images
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    img3 = tmp_path / "img3.jpg"
    img1.touch()
    img2.touch()
    img3.touch()

    mixer = HybridMediaMixer(
        images_dir=tmp_path,
        clip_duration=3.0,
        image_ratio=1.0,  # 100% images
    )

    shared_used_images: set[Path] = set()

    # Part 1: duration 6s -> 2 slots
    slots_part1 = mixer.create_mix(total_duration=6.0, used_images=shared_used_images)
    assert len(slots_part1) == 2
    assert len(shared_used_images) == 2

    # Part 2: duration 3s -> 1 slot, should pick the unused image (img3)
    slots_part2 = mixer.create_mix(total_duration=3.0, used_images=shared_used_images)
    assert len(slots_part2) == 1
    assert slots_part2[0].source_path not in [s.source_path for s in slots_part1]
    assert len(shared_used_images) == 3


def test_video_render_worker_multipart_run(tmp_path):
    """Verify end-to-end execution of _run_multipart with lossless concatenation."""
    init_db()
    p1 = tmp_path / "part_01.mp3"
    p2 = tmp_path / "part_02.mp3"
    p1.touch()
    p2.touch()
    out = tmp_path / "output_master.mp4"
    dummy_img = tmp_path / "image.jpg"
    dummy_img.touch()

    worker = VideoRenderWorker(
        audio_path=p1,
        audio_paths=[p1, p2],
        images_dir=tmp_path,
        output_path=out,
    )

    # Track emitted progress and steps
    emitted_steps = []
    worker.step_changed.connect(emitted_steps.append)

    # Mock dependencies
    dummy_seg = TranscriptionSegment(start=0.0, end=5.0, text="Hello world from part", words=[])
    dummy_res = TranscriptionResult(
        audio_path=p1,
        audio_hash="dummy_hash",
        language="en",
        full_text="Hello world from part",
        segments=[dummy_seg],
        model_used="whisper",
        elapsed_seconds=1.0,
    )

    with patch("services.transcriber.TranscriptionService.transcribe", return_value=dummy_res), \
         patch("ui.workers.video_worker.probe_media") as mock_probe, \
         patch("services.subtitle_engine.SubtitleEngine.generate_ass"), \
         patch("services.subtitle_engine.SubtitleEngine.generate_srt"), \
         patch("services.hybrid_mixer.HybridMediaMixer.create_mix") as mock_mix, \
         patch("ui.workers.video_worker.build_ken_burns_clip") as mock_kb, \
         patch("ui.workers.video_worker.concatenate_clips") as mock_concat, \
         patch("ui.workers.video_worker.lossless_concat_videos") as mock_lossless:

        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_probe.return_value = mock_info

        # Hybrid mix returns 1 slot per part
        mock_slot = VisualSlot(
            slot_index=0,
            media_type="image",
            start_time=0.0,
            duration=5.0,
            source_path=dummy_img,
            preset=KenBurnsPreset.ZOOM_IN_CENTER,
        )
        mock_mix.return_value = [mock_slot]

        # Run worker directly
        worker.run()

        # Assertions
        assert mock_concat.call_count == 2, "Should render 2 intermediate part videos"
        assert mock_lossless.call_count == 1, "Should merge parts using lossless_concat_videos"
        assert any("Merging Audio Parts" in s for s in emitted_steps)
        assert any("Export Completed" in s for s in emitted_steps)

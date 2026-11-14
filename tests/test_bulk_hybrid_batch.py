"""
tests/test_bulk_hybrid_batch.py
===============================
Integration and unit tests for Step 3 & Step 4 Bulk Generation features:
- Batch Video Count Limit (capping scan and queue execution)
- Timing & Media Mix Controls (Clip Duration slider & Media Mix Ratio)
- Export Resolution Dropdown (1080p, 720p, 2K updating paths)
- Unified Subtitle Typography Studio (Presets, Aa colors, B/I formatting, Box padding, live preview)
- Parameter propagation to BulkJob and BulkQueueWorker
"""

from pathlib import Path
import pytest
from PySide6.QtCore import Qt

from ui.views.bulk_view import BulkView
from ui.workers.bulk_worker import BulkJob, BulkQueueWorker


def test_bulk_limit_and_scan(qapp, tmp_path):
    """Verify that spin_batch_limit caps the number of scanned items added to the table."""
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Create 6 audio files
    for i in range(1, 7):
        (audio_dir / f"audio {i}.mp3").write_bytes(b"fake audio")
        sub = footage_dir / f"video {i} content"
        sub.mkdir()
        (sub / "pic.jpg").write_bytes(b"fake pic")

    view = BulkView()
    view.edit_audio_dir.setText(str(audio_dir))
    view.edit_footage_dir.setText(str(footage_dir))
    view.edit_output_dir.setText(str(output_dir))

    # Test 1: Set limit to 3 -> only 3 files scanned
    view.spin_batch_limit.setValue(3)
    view._scan_current_directory()
    assert view.table.rowCount() == 3

    # Verify rows 1, 2, 3 were matched
    assert "audio 1.mp3" in view.table.item(0, 1).text()
    assert "audio 2.mp3" in view.table.item(1, 1).text()
    assert "audio 3.mp3" in view.table.item(2, 1).text()

    # Test 2: Set limit to 0 (All) -> all 6 files scanned
    view.spin_batch_limit.setValue(0)
    view._scan_current_directory()
    assert view.table.rowCount() == 6


def test_bulk_resolution_updates_queued_paths(qapp, tmp_path):
    """Verify changing resolution dropdown updates the table output path column."""
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    (audio_dir / "audio 1.mp3").write_bytes(b"fake audio")

    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    view = BulkView()
    view.edit_audio_dir.setText(str(audio_dir))
    view.edit_footage_dir.setText(str(footage_dir))
    view.edit_output_dir.setText(str(output_dir))

    view.spin_batch_limit.setValue(0)
    view._scan_current_directory()
    assert view.table.rowCount() == 1

    # Default is 1080p
    out_item = view.table.item(0, 4)
    assert "1080p.mp4" in out_item.text()

    # Change to 720p (index 1)
    view.combo_res.setCurrentIndex(1)
    out_item = view.table.item(0, 4)
    assert "720p.mp4" in out_item.text()

    # Change to 2k (index 2)
    view.combo_res.setCurrentIndex(2)
    out_item = view.table.item(0, 4)
    assert "2k.mp4" in out_item.text()


def test_bulk_timing_and_typography_job_propagation(qapp, tmp_path):
    """Verify Timing, Mix Ratio, and Subtitle Studio settings pass accurately into BulkJob."""
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    (audio_dir / "audio 1.mp3").write_bytes(b"fake audio")
    (audio_dir / "audio 2.mp3").write_bytes(b"fake audio")

    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    view = BulkView()
    view.edit_audio_dir.setText(str(audio_dir))
    view.edit_footage_dir.setText(str(footage_dir))
    view.edit_output_dir.setText(str(output_dir))

    # Scan 2 files
    view.spin_batch_limit.setValue(0)
    view._scan_current_directory()
    assert view.table.rowCount() == 2

    # Configure Timing & Media Mix
    view.slider_clip_dur.setValue(8) # 8s duration
    view.slider_mix_ratio.setValue(20) # 20% images / 80% video

    # Configure Typography Studio
    view.combo_res.setCurrentIndex(1) # 720p
    view._is_bold = True
    view._is_italic = True
    view.slider_box_padding.setValue(22)
    view._box_padding = 22.0
    view._font_color = "#38BDF8"
    view._bg_style = "solid"
    view._has_bg_box = True
    view._bg_color = "#1E293B"
    view._bg_opacity = 0.85
    view.slider_stroke.setValue(5)
    view._stroke_thickness = 5.0

    # Spy on BulkQueueWorker instantiation inside _start_queue
    captured_jobs = []
    original_init = BulkQueueWorker.__init__

    def mock_init(self, jobs, parent=None):
        nonlocal captured_jobs
        captured_jobs = jobs
        original_init(self, jobs, parent)

    BulkQueueWorker.__init__ = mock_init
    try:
        # Also patch QThread.start to prevent actual background rendering during this unit test
        original_start = BulkQueueWorker.start
        BulkQueueWorker.start = lambda self: None

        view._start_queue()

        assert len(captured_jobs) == 2
        job = captured_jobs[0]
        assert job.clip_duration == 8.0
        assert job.image_ratio == 0.20
        assert job.resolution == "720p"
        assert job.bold is True
        assert job.italic is True
        assert job.box_padding == 22.0
        assert job.text_color == "#38BDF8"
        assert job.bg_style == "solid"
        assert job.has_bg_box is True
        assert job.bg_color == "#1E293B"
        assert job.bg_opacity == 0.85
        assert job.stroke_thickness == 5.0
        assert "720p.mp4" in str(job.output_path)
    finally:
        BulkQueueWorker.__init__ = original_init
        BulkQueueWorker.start = original_start


def test_bulk_queue_respects_batch_limit_on_start(qapp, tmp_path):
    """Verify that if table has 5 items but batch limit is 2, only 2 jobs are submitted."""
    audio_dir = tmp_path / "audios"
    audio_dir.mkdir()
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    for i in range(1, 6):
        (audio_dir / f"audio {i}.mp3").write_bytes(b"fake audio")

    view = BulkView()
    view.edit_audio_dir.setText(str(audio_dir))
    view.edit_footage_dir.setText(str(footage_dir))
    view.edit_output_dir.setText(str(output_dir))

    # Scan with limit=0 to fill table with all 5 items
    view.spin_batch_limit.setValue(0)
    view._scan_current_directory()
    assert view.table.rowCount() == 5

    # Now set limit to 2 before starting queue
    view.spin_batch_limit.setValue(2)

    captured_jobs = []
    original_init = BulkQueueWorker.__init__
    original_start = BulkQueueWorker.start

    def mock_init(self, jobs, parent=None):
        nonlocal captured_jobs
        captured_jobs = jobs
        original_init(self, jobs, parent)

    BulkQueueWorker.__init__ = mock_init
    BulkQueueWorker.start = lambda self: None
    try:
        view._start_queue()
        assert len(captured_jobs) == 2
        assert captured_jobs[0].job_id == 0
        assert captured_jobs[1].job_id == 1
    finally:
        BulkQueueWorker.__init__ = original_init
        BulkQueueWorker.start = original_start

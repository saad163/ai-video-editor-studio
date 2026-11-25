"""
tests/test_interview_scanner.py
================================
Unit and integration tests for Part 2: Video Link Downloader & AI Interview Timestamp Scanner.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QMessageBox

from services.interview_scanner import (
    InterviewSegment,
    VideoDownloader,
    InterviewScanner,
    format_timestamp_mmss,
    parse_timestamp_to_seconds,
    is_valid_video_url,
    get_url_cache_hash,
)
from services.transcriber import TranscriptionSegment
from ui.workers.interview_scan_worker import InterviewScanWorker
from ui.views.studio_view import StudioView


def test_url_validation_and_hash():
    """Verify URL validation and deterministic caching hash."""
    assert is_valid_video_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert is_valid_video_url("http://vimeo.com/12345678") is True
    assert is_valid_video_url("https://example.com/media/interview.mp4") is True
    assert is_valid_video_url("ftp://invalid.com") is False
    assert is_valid_video_url("not_a_valid_url") is False
    assert is_valid_video_url("") is False
    assert is_valid_video_url(None) is False

    h1 = get_url_cache_hash("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    h2 = get_url_cache_hash("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    h3 = get_url_cache_hash("https://www.youtube.com/watch?v=other_video")

    assert len(h1) == 12
    assert h1 == h2
    assert h1 != h3


def test_timestamp_formatting_and_parsing():
    """Verify MM:SS formatting and bidirectional timestamp parsing."""
    assert format_timestamp_mmss(0.0) == "00:00"
    assert format_timestamp_mmss(59.4) == "00:59"
    assert format_timestamp_mmss(60.0) == "01:00"
    assert format_timestamp_mmss(125.6) == "02:06"
    assert format_timestamp_mmss(3605.0) == "60:05"

    assert parse_timestamp_to_seconds("00:00") == 0.0
    assert parse_timestamp_to_seconds("01:05") == 65.0
    assert parse_timestamp_to_seconds("02:30") == 150.0
    assert parse_timestamp_to_seconds("01:00:00") == 3600.0
    assert parse_timestamp_to_seconds("42.5") == 42.5
    assert parse_timestamp_to_seconds("invalid") == 0.0


def test_video_downloader_cache_reuse(tmp_path):
    """Verify VideoDownloader returns existing cached file without invoking yt-dlp."""
    downloader = VideoDownloader(download_dir=tmp_path)
    url = "https://www.youtube.com/watch?v=cached123"
    url_hash = get_url_cache_hash(url)

    cached_file = tmp_path / f"interview_{url_hash}.mp4"
    # Create fake video file > 50KB to pass validity check
    cached_file.write_bytes(b"\x00" * 60_000)

    progress_calls = []
    result_path = downloader.download(
        url,
        progress_cb=lambda pct, msg: progress_calls.append((pct, msg)),
        force_redownload=False,
    )

    assert result_path == cached_file
    assert any("Found cached video" in msg for _, msg in progress_calls)


def test_video_downloader_invalid_url_raises(tmp_path):
    """Verify VideoDownloader rejects invalid URLs immediately."""
    downloader = VideoDownloader(download_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid video URL"):
        downloader.download("ftp://broken_link")


def test_transcription_service_supports_mp4_video_format(tmp_path):
    """Verify TranscriptionService validates .mp4 video files without raising format error."""
    from services.transcriber import TranscriptionService
    svc = TranscriptionService()
    fake_video = tmp_path / "interview_sample.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    # Should not raise ValueError("Unsupported audio format")
    with patch.object(svc, "_load_model"), patch.object(svc, "_model") as mock_model:
        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_model.transcribe.return_value = ([], mock_info)
        res = svc.transcribe(fake_video, use_cache=False)
        assert res.audio_path == fake_video


def test_interview_scanner_heuristic_fallback():
    """Verify speech segments are grouped into coherent interview blocks by heuristic timing."""
    scanner = InterviewScanner(api_key="")

    # Create dummy Whisper segments spanning ~60 seconds with sentences
    mock_segments = [
        TranscriptionSegment(start=0.0, end=5.0, text="Welcome everyone to today's interview session."),
        TranscriptionSegment(start=5.5, end=11.0, text="Today we are discussing groundbreaking AI advances."),
        TranscriptionSegment(start=11.2, end=18.0, text="Our special guest has spent a decade building neural systems."),
        TranscriptionSegment(start=18.2, end=24.5, text="Could you tell us where this journey began?"),
        TranscriptionSegment(start=26.0, end=35.0, text="It all began when we realized standard models lacked context."),
        TranscriptionSegment(start=35.5, end=45.0, text="So we designed continuous timeline matching algorithms."),
        TranscriptionSegment(start=45.2, end=55.0, text="That completely transformed our video editing speed."),
    ]

    detected = scanner._detect_with_heuristics(mock_segments)
    assert len(detected) >= 2

    seg1 = detected[0]
    assert seg1.segment_id == 1
    assert seg1.start_sec == 0.0
    assert seg1.end_sec >= 20.0
    assert seg1.duration_sec == round(seg1.end_sec - seg1.start_sec, 2)
    assert seg1.start_time == "00:00"
    assert ":" in seg1.end_time
    assert len(seg1.label) > 0


def test_interview_scanner_gemini_parsing():
    """Verify Gemini JSON output is parsed into structured InterviewSegment objects."""
    scanner = InterviewScanner(api_key="AIzaSyDummyKeyForTestingPurposes12345")

    mock_segments = [
        TranscriptionSegment(start=10.0, end=20.0, text="We built the core rendering pipeline."),
        TranscriptionSegment(start=20.0, end=35.0, text="It handles dual aspect ratio conversion seamlessly."),
    ]

    mock_gemini_response = MagicMock()
    mock_gemini_response.text = """
    ```json
    [
      {
        "start_sec": 10.0,
        "end_sec": 35.0,
        "label": "Pipeline Architecture and Dual Aspect Ratio",
        "speaker": "Lead Architect"
      }
    ]
    ```
    """

    with patch("services.gemini_resolver.generate_content_with_fallback", return_value=(mock_gemini_response, "gemini-3.8-flash")):
        results = scanner._detect_with_gemini(mock_segments)

    assert len(results) == 1
    seg = results[0]
    assert seg.segment_id == 1
    assert seg.start_sec == 10.0
    assert seg.end_sec == 35.0
    assert seg.duration_sec == 25.0
    assert seg.start_time == "00:10"
    assert seg.end_time == "00:35"
    assert seg.label == "Pipeline Architecture and Dual Aspect Ratio"
    assert seg.speaker == "Lead Architect"
    assert "core rendering pipeline" in seg.transcript


def test_interview_scan_worker_execution(qtbot, tmp_path):
    """Verify InterviewScanWorker signals during download and scan execution."""
    test_video = tmp_path / "mock_interview.mp4"
    test_video.write_bytes(b"\x00" * 1000)

    dummy_segments = [
        InterviewSegment(
            segment_id=1,
            start_sec=5.0,
            end_sec=25.0,
            duration_sec=20.0,
            start_time="00:05",
            end_time="00:25",
            label="Discussing video innovation",
            speaker="Guest",
        )
    ]

    worker = InterviewScanWorker(url="https://www.youtube.com/watch?v=sample123")

    with patch.object(worker.scanner, "scan_url", return_value=(test_video, dummy_segments)):
        with qtbot.waitSignal(worker.scan_finished, timeout=2000) as blocker:
            worker.start()

        v_path, segments = blocker.args
        assert v_path == str(test_video)
        assert len(segments) == 1
        assert segments[0].start_sec == 5.0


def test_studio_view_scan_clips_wiring(qtbot, tmp_path, monkeypatch):
    """Verify clicking Scan Clips in StudioView launches scan and updates UI state."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.show()
    view.combo_format.setCurrentIndex(2)

    dummy_segments = [
        InterviewSegment(
            segment_id=1,
            start_sec=12.0,
            end_sec=42.0,
            duration_sec=30.0,
            start_time="00:12",
            end_time="00:42",
            label="Core breakthrough discussion",
            speaker="Guest",
        )
    ]
    test_video = tmp_path / "downloaded_interview.mp4"
    test_video.write_bytes(b"\x00" * 1000)

    # 1. Invalid URL shows warning and does not launch worker
    with patch.object(QMessageBox, "warning"):
        view.edit_youtube_link.setText("not_a_url")
        view.btn_scan_youtube.click()
    assert getattr(view, "_scan_worker", None) is None

    # 2. Valid URL triggers scan and finishes
    view.edit_youtube_link.setText("https://www.youtube.com/watch?v=valid_interview")

    # Simulate finishing callback directly to verify UI response (bypass modal dialog in headless test)
    with patch.object(view, "_open_interview_selector_dialog"):
        view._on_interview_scan_finished(str(test_video), dummy_segments)

    assert len(view._scanned_interview_segments) == 1
    assert "1 segment(s) detected" in view.edit_interview_clips.text()
    assert view.btn_scan_youtube.isEnabled() is True

    # 3. Verify get_interview_config serializes scanned segments
    cfg = view.get_interview_config()
    assert len(cfg["scanned_segments"]) == 1
    assert cfg["scanned_segments"][0]["start_sec"] == 12.0
    assert cfg["scanned_segments"][0]["label"] == "Core breakthrough discussion"

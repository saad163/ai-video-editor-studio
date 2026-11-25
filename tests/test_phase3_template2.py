"""
tests/test_phase3_template2.py
==============================
End-to-end integration tests for Template 2 (Episode Scene Compiler & Gemini Matcher).
Verifies:
  1. EpisodeCompiler service direct execution with 1080p export.
  2. SmartSceneMatcher narrative matching (both Gemini API mock & offline heuristic mode).
  3. VideoRenderWorker seamless routing when format_mode == "Template 2 (Episode Scene Matching)".
  4. EpisodeCompilerWorker non-blocking QThread execution & signal emission.
  5. StudioView UI template selection and Gemini AI status badge.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication

from core.ffmpeg_wrapper import probe_media, resolve_ffmpeg
from database.db import init_db
from services.episode_compiler import (
    EpisodeCompiler,
    EpisodeCompileConfig,
    EpisodeCompileResult,
)
from services.gemini_matcher import (
    SmartSceneMatcher,
    NarrativeChunk,
    NarrativeChunker,
)
from services.scene_detector import Scene
from services.transcriber import TranscriptionResult, TranscriptionSegment
from ui.views.studio_view import StudioView
from ui.workers.episode_worker import EpisodeCompilerWorker
from ui.workers.video_worker import VideoRenderWorker
from utils.config import get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def slice_audio(input_audio: Path, output_audio: Path, start_sec: float, duration_sec: float) -> Path:
    """Slice a short audio excerpt using FFmpeg."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe, _ = resolve_ffmpeg()
    import subprocess
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(input_audio),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output_audio),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to slice audio: {res.stderr}")
    return output_audio


def test_gemini_matcher_mock_api(tmp_path):
    """Test SmartSceneMatcher calling Gemini API with structured JSON response."""
    init_db()

    chunks = [
        NarrativeChunk(chunk_index=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0, text="Intro to battle"),
        NarrativeChunk(chunk_index=1, start_sec=5.0, end_sec=10.0, duration_sec=5.0, text="Climax confrontation"),
    ]

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    scenes = [
        Scene(scene_id=1, video_hash="h1", video_path=dummy_video, scene_index=1, start_sec=0.0, end_sec=8.0, duration_sec=8.0, description="Peaceful town"),
        Scene(scene_id=2, video_hash="h1", video_path=dummy_video, scene_index=2, start_sec=8.0, end_sec=16.0, duration_sec=8.0, description="Explosive sword fight"),
    ]

    # Test with valid dummy key and mocked genai
    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyMockKeyForTestingPurposes12345")

    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {"chunk_index": 0, "candidate_index": 0},
        {"chunk_index": 1, "candidate_index": 1},
    ])

    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model_instance

        matched = matcher.match_scenes(chunks, scenes, project_id=None)
        assert len(matched) == 2
        assert matched[0].matched_scene == scenes[0]
        assert matched[1].matched_scene == scenes[1]


def test_gemini_matcher_offline_heuristic(tmp_path):
    """Test SmartSceneMatcher offline fallback with anti-duplication."""
    init_db()

    chunks = [
        NarrativeChunk(chunk_index=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0, text="First scene"),
        NarrativeChunk(chunk_index=1, start_sec=5.0, end_sec=10.0, duration_sec=5.0, text="Second scene"),
        NarrativeChunk(chunk_index=2, start_sec=10.0, end_sec=15.0, duration_sec=5.0, text="Third scene"),
    ]

    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    scenes = [
        Scene(scene_id=10, video_hash="h1", video_path=dummy_video, scene_index=1, start_sec=0.0, end_sec=6.0, duration_sec=6.0),
        Scene(scene_id=20, video_hash="h1", video_path=dummy_video, scene_index=2, start_sec=6.0, end_sec=12.0, duration_sec=6.0),
        Scene(scene_id=30, video_hash="h1", video_path=dummy_video, scene_index=3, start_sec=12.0, end_sec=18.0, duration_sec=6.0),
    ]

    # No API key -> offline heuristics
    matcher = SmartSceneMatcher(gemini_api_key="")
    matched = matcher.match_scenes(chunks, scenes, project_id=None)

    assert len(matched) == 3
    # Check that consecutive scenes are not identical
    for i in range(1, len(matched)):
        assert matched[i].matched_scene.scene_id != matched[i - 1].matched_scene.scene_id


def test_studio_view_template2_badge(qapp):
    """Test StudioView Template 2 selection and Gemini status badge."""
    view = StudioView()
    assert hasattr(view, "lbl_gemini_status")

    # Template 1
    view.combo_format.setCurrentIndex(0)
    assert "Hybrid" in view.lbl_gemini_status.text()

    # Template 2
    view.combo_format.setCurrentIndex(1)
    status_text = view.lbl_gemini_status.text()
    assert "Matcher Active" in status_text


def test_template2_episode_compiler_end_to_end(qapp, tmp_path):
    """Test full EpisodeCompilerWorker pipeline with PySceneDetect, Whisper, and FFmpeg."""
    init_db()

    source_audio = PROJECT_ROOT / "sample.mp3"
    source_footage = PROJECT_ROOT / "Autocut.mp4"

    assert source_audio.exists(), f"sample.mp3 missing at {source_audio}"
    assert source_footage.exists(), f"Autocut.mp4 missing at {source_footage}"

    test_audio = tmp_path / "test_episode_voice.mp3"
    slice_audio(source_audio, test_audio, start_sec=0.0, duration_sec=6.0)

    test_out = tmp_path / "output_episode_master_1080p.mp4"

    audio_progs: list[int] = []
    render_progs: list[int] = []
    steps: list[str] = []
    finished_paths: list[str] = []
    failed_msgs: list[str] = []

    worker = EpisodeCompilerWorker(
        audio_path=test_audio,
        footage_path=source_footage,
        output_path=test_out,
        title="Test Episode Master",
        resolution="1080p",
        text_color="#FFFF00",
        highlight_color="#FFFF00",
        has_bg_box=True,
        bg_style="semi_transparent_black",
        bg_color="#000000",
        bg_opacity=0.5,
        font_size=56,
        font_family="Montserrat",
        generate_chapters=True,
        scene_threshold=27.0,
        clip_duration=4.0,
        gemini_api_key="",
    )

    worker.audio_progress.connect(lambda pct: audio_progs.append(pct))
    worker.render_progress.connect(lambda pct: render_progs.append(pct))
    worker.step_changed.connect(lambda s: steps.append(s))
    worker.render_finished.connect(lambda p: finished_paths.append(p))
    worker.render_failed.connect(lambda e: failed_msgs.append(e))

    worker.run()

    assert len(failed_msgs) == 0, f"Worker failed: {failed_msgs}"
    assert len(finished_paths) == 1
    assert Path(finished_paths[0]).exists()

    # Output file verification
    probe = probe_media(test_out)
    assert probe.has_video is True
    assert probe.has_audio is True
    assert probe.width == 1920
    assert probe.height == 1080
    assert probe.duration >= 2.0

    # Subtitles and chapters verification
    srt_file = tmp_path / "output_episode_master_1080p.srt"
    chapters_file = tmp_path / "output_episode_master_1080p_chapters.txt"
    assert srt_file.exists()
    assert chapters_file.exists()

    # Signals verification
    assert len(audio_progs) > 0
    assert 100 in audio_progs
    assert len(render_progs) > 0
    assert 100 in render_progs
    assert "Detecting Episode Scenes" in steps
    assert "Extracting Scene Subclips" in steps
    assert "Final Master Render" in steps


def test_video_render_worker_template2_routing(qapp, tmp_path):
    """Test that VideoRenderWorker cleanly executes Template 2 when format_mode is set."""
    init_db()

    source_audio = PROJECT_ROOT / "sample.mp3"
    source_footage = PROJECT_ROOT / "Autocut.mp4"

    test_audio = tmp_path / "test_vrw_voice.mp3"
    slice_audio(source_audio, test_audio, start_sec=0.0, duration_sec=5.0)

    test_out = tmp_path / "output_vrw_template2.mp4"

    finished_paths = []
    errors = []

    worker = VideoRenderWorker(
        audio_path=test_audio,
        images_dir=tmp_path,
        videos_path=source_footage,
        output_path=test_out,
        format_mode="Template 2 (Episode Scene Matching)",
        resolution="720p",
        clip_duration=4.0,
        gemini_api_key="",
    )
    worker.render_finished.connect(lambda p: finished_paths.append(p))
    worker.render_failed.connect(lambda e: errors.append(e))

    worker.run()

    assert len(errors) == 0, f"VideoRenderWorker failed: {errors}"
    assert len(finished_paths) == 1
    assert Path(finished_paths[0]).exists()

    probe = probe_media(test_out)
    assert probe.has_video is True
    assert probe.width == 1280
    assert probe.height == 720

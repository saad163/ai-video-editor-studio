"""
tests/test_template2_audio_stripping_and_language.py
===================================================
Tests for:
  1. Automatic source audio stripping (-an) during subclip extraction in core/ffmpeg_wrapper.py.
  2. Language selection in StudioView (Urdu vs English).
  3. Narration language configuration propagation in EpisodeCompiler and EpisodeCompilerWorker.
  4. Language context injection in SmartSceneMatcher (Gemini and heuristic matching).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication

from core.ffmpeg_wrapper import extract_subclip
from services.episode_compiler import (
    EpisodeCompiler,
    EpisodeCompileConfig,
    EpisodeCompileResult,
)
from services.gemini_matcher import SmartSceneMatcher, NarrativeChunk
from services.scene_detector import Scene
from services.transcriber import TranscriptionResult, TranscriptionSegment
from ui.views.studio_view import StudioView
from ui.workers.episode_worker import EpisodeCompilerWorker


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


def test_extract_subclip_forces_an_audio_stripping(tmp_path):
    """Verify that extract_subclip automatically passes -an to strip source audio by default."""
    dummy_src = tmp_path / "episode_source.mp4"
    dummy_src.touch()
    dummy_out = tmp_path / "subclip_01.mp4"

    with patch("core.ffmpeg_wrapper._run_with_progress") as mock_run, \
         patch("core.ffmpeg_wrapper.probe_media") as mock_probe:

        mock_info = MagicMock()
        mock_info.duration = 10.0
        mock_info.has_audio = True
        mock_probe.return_value = mock_info

        # 1. Default invocation: should force -an
        extract_subclip(
            video_path=dummy_src,
            start_sec=1.0,
            end_sec=5.0,
            output_path=dummy_out,
        )
        assert mock_run.called
        cmd1 = mock_run.call_args[0][0]
        assert "-an" in cmd1, "extract_subclip must include '-an' by default to strip source audio"
        assert "-c:a" not in cmd1

        mock_run.reset_mock()

        # 2. Explicit keep_audio=True: should retain audio
        extract_subclip(
            video_path=dummy_src,
            start_sec=1.0,
            end_sec=5.0,
            output_path=dummy_out,
            keep_audio=True,
            strip_source_audio=False,
        )
        assert mock_run.called
        cmd2 = mock_run.call_args[0][0]
        assert "-an" not in cmd2
        assert "-c:a" in cmd2


def test_studio_view_language_selector(qapp):
    """Verify StudioView provides Narration Language dropdown with Urdu and English options."""
    view = StudioView()
    assert hasattr(view, "combo_language")
    assert view.combo_language.count() >= 2

    # Test default Urdu selection
    view.combo_language.setCurrentIndex(0)
    assert view.get_selected_language() == "ur"

    # Test English selection
    view.combo_language.setCurrentIndex(1)
    assert view.get_selected_language() == "en"


def test_episode_compiler_propagates_urdu_language(tmp_path):
    """Verify EpisodeCompiler passes language='ur' to transcriber and scene matcher."""
    audio_file = tmp_path / "urdu_voiceover.mp3"
    audio_file.touch()
    footage_dir = tmp_path / "episodes"
    footage_dir.mkdir()
    (footage_dir / "ep01.mp4").touch()
    out_file = tmp_path / "output_master.mp4"

    config = EpisodeCompileConfig(
        audio_path=audio_file,
        footage_path=footage_dir,
        output_path=out_file,
        language="ur",
    )
    compiler = EpisodeCompiler(config)

    dummy_seg = TranscriptionSegment(start=0.0, end=5.0, text="یہ ایک کہانی ہے", words=[])
    dummy_res = TranscriptionResult(
        audio_path=audio_file,
        audio_hash="dummy_hash",
        language="ur",
        full_text="یہ ایک کہانی ہے",
        segments=[dummy_seg],
        model_used="whisper",
        elapsed_seconds=1.0,
    )

    with patch("services.transcriber.TranscriptionService.transcribe", return_value=dummy_res) as mock_transcribe, \
         patch("services.episode_compiler.probe_media") as mock_probe, \
         patch("services.scene_detector.SceneDetectorService.detect_scenes") as mock_detect, \
         patch("services.gemini_matcher.SmartSceneMatcher.match_scenes") as mock_match, \
         patch("services.episode_compiler.extract_subclip") as mock_extract, \
         patch("services.episode_compiler.concatenate_clips") as mock_concat, \
         patch("services.subtitle_engine.SubtitleEngine.generate_ass"), \
         patch("services.subtitle_engine.SubtitleEngine.generate_srt"):

        mock_info = MagicMock()
        mock_info.duration = 5.0
        mock_info.width = 1920
        mock_info.height = 1080
        mock_probe.return_value = mock_info

        dummy_scene = Scene(
            scene_id=1,
            video_hash="v1",
            video_path=footage_dir / "ep01.mp4",
            scene_index=1,
            start_sec=0.0,
            end_sec=10.0,
            duration_sec=10.0,
        )
        mock_detect.return_value = [dummy_scene]

        dummy_chunk = NarrativeChunk(
            chunk_index=0,
            start_sec=0.0,
            end_sec=5.0,
            duration_sec=5.0,
            text="یہ ایک کہانی ہے",
            segments=[dummy_seg],
            matched_scene=dummy_scene,
        )
        mock_match.return_value = [dummy_chunk]

        result = compiler.compile()
        assert result is not None

        # Verify transcribe was called with language="ur"
        mock_transcribe.assert_called_once()
        assert mock_transcribe.call_args[1]["language"] == "ur"

        # Verify scene matcher was called with language="ur"
        mock_match.assert_called_once()
        assert mock_match.call_args[1]["language"] == "ur"

        # Verify subclip extraction applied strip_source_audio=True
        mock_extract.assert_called_once()
        assert mock_extract.call_args[1]["strip_source_audio"] is True
        assert mock_extract.call_args[1]["keep_audio"] is False


def test_smart_scene_matcher_gemini_prompt_language():
    """Verify SmartSceneMatcher injects the narration language into Gemini prompt."""
    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyDummyKeyForTestingPurpose123")

    dummy_scene = Scene(
        scene_id=1,
        video_hash="h1",
        video_path=Path("scene.mp4"),
        scene_index=1,
        start_sec=0.0,
        end_sec=5.0,
        duration_sec=5.0,
        description="King entering castle",
    )
    dummy_chunk = NarrativeChunk(
        chunk_index=0,
        start_sec=0.0,
        end_sec=5.0,
        duration_sec=5.0,
        text="بادشاہ محل میں داخل ہوتا ہے",
    )

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:

        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '[{"chunk_index": 0, "candidate_index": 0}]'
        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        matcher._match_with_gemini([dummy_chunk], [dummy_scene], language="ur")

        prompt_sent = mock_instance.generate_content.call_args[0][0]
        assert "Urdu" in prompt_sent, "Gemini prompt must include Urdu language indication"


def test_episode_worker_propagates_language(tmp_path):
    """Verify EpisodeCompilerWorker passes language parameter to EpisodeCompileConfig."""
    audio = tmp_path / "audio.mp3"
    footage = tmp_path / "footage"
    out = tmp_path / "out.mp4"
    audio.touch()
    footage.mkdir()

    worker = EpisodeCompilerWorker(
        audio_path=audio,
        footage_path=footage,
        output_path=out,
        language="ur",
    )
    assert worker.language == "ur"

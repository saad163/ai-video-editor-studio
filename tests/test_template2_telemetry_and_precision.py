"""
tests/test_template2_telemetry_and_precision.py
===============================================
Unit tests for Template 2 (Episode Scene Matching):
1. Strict Gemini API key validation and error reporting.
2. Token usage metadata tracking & cost estimation.
3. Urdu and bilingual narrative chunking punctuation precision.
4. Precision timestamp alignment for subclip extraction.
5. StudioView UI validation before render start.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from services.scene_detector import Scene
from services.transcriber import TranscriptionResult, TranscriptionSegment
from services.gemini_matcher import NarrativeChunker, SmartSceneMatcher, NarrativeChunk
from services.episode_compiler import EpisodeCompiler, EpisodeCompileConfig
from ui.views.studio_view import StudioView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gemini_matcher_rejects_invalid_key_in_strict_mode():
    """Verify SmartSceneMatcher raises ValueError when API key is missing or invalid in strict mode."""
    # Empty key with strict_mode=True
    matcher = SmartSceneMatcher(gemini_api_key="", strict_mode=True)
    dummy_chunk = NarrativeChunk(0, 0.0, 5.0, 5.0, "Test narration")
    dummy_scene = Scene(1, "h1", Path("s.mp4"), 1, 0.0, 5.0, 5.0)

    with pytest.raises(ValueError, match="Missing or invalid Gemini API key"):
        matcher.match_scenes([dummy_chunk], [dummy_scene], language="en")

    # Invalid placeholder key
    matcher_bad = SmartSceneMatcher(gemini_api_key="your_api_key_here_placeholder_12345", strict_mode=True)
    with pytest.raises(ValueError, match="Missing or invalid Gemini API key"):
        matcher_bad.match_scenes([dummy_chunk], [dummy_scene], language="en")

    # Short key
    matcher_short = SmartSceneMatcher(gemini_api_key="short_key", strict_mode=True)
    with pytest.raises(ValueError, match="Missing or invalid Gemini API key"):
        matcher_short.match_scenes([dummy_chunk], [dummy_scene], language="en")

    # Valid key formats (AIzaSy... and AQ....)
    assert SmartSceneMatcher.is_api_key_valid("AIzaSyMockKeyForTelemetry12345") is True
    assert SmartSceneMatcher.is_api_key_valid("AQ.MockOAuthStyleTokenForTestingOnly_1234567890abcdef") is True

    # Urdu narration automatically enforces strict mode
    matcher_urdu = SmartSceneMatcher(gemini_api_key="", strict_mode=False)
    with pytest.raises(ValueError, match="Missing or invalid Gemini API key"):
        matcher_urdu.match_scenes([dummy_chunk], [dummy_scene], language="ur")


def test_gemini_matcher_offline_when_not_strict():
    """Verify SmartSceneMatcher falls back gracefully to heuristics when not in strict mode."""
    matcher = SmartSceneMatcher(gemini_api_key="", strict_mode=False)
    dummy_chunk = NarrativeChunk(0, 0.0, 5.0, 5.0, "Narration text without key")
    dummy_scene = Scene(1, "h1", Path("s.mp4"), 1, 0.0, 5.0, 5.0)

    result = matcher.match_scenes([dummy_chunk], [dummy_scene], language="en")
    assert len(result) == 1
    assert result[0].matched_scene == dummy_scene

    summary = matcher.get_usage_summary()
    assert summary["used_gemini"] is False
    assert summary["total_tokens"] == 0
    assert summary["estimated_cost_usd"] == 0.0


def test_gemini_matcher_logs_and_returns_usage_metadata():
    """Verify SmartSceneMatcher parses usage_metadata and calculates estimated cost."""
    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyMockKeyForTelemetry12345")
    dummy_chunk = NarrativeChunk(0, 0.0, 5.0, 5.0, "The king enters the grand hall")
    dummy_scene = Scene(1, "h1", Path("s.mp4"), 1, 0.0, 5.0, 5.0, description="King in grand hall")

    with patch("services.gemini_matcher.get_cached_scene_match", return_value=None), \
         patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:

        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '[{"chunk_index": 0, "candidate_index": 0}]'

        # Mock usage_metadata object
        mock_meta = MagicMock()
        mock_meta.prompt_token_count = 1450
        mock_meta.candidates_token_count = 250
        mock_meta.total_token_count = 1700
        mock_response.usage_metadata = mock_meta

        mock_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_instance

        matched = matcher.match_scenes([dummy_chunk], [dummy_scene], language="en")
        assert len(matched) == 1

        summary = matcher.get_usage_summary()
        assert summary["used_gemini"] is True
        assert summary["prompt_tokens"] == 1450
        assert summary["output_tokens"] == 250
        assert summary["total_tokens"] == 1700
        assert summary["estimated_cost_usd"] > 0.0


def test_narrative_chunker_urdu_punctuation_precision():
    """Verify NarrativeChunker splits on Urdu sentence full stops (۔) and question marks (؟)."""
    chunker = NarrativeChunker(target_chunk_sec=5.0, min_chunk_sec=2.0, max_chunk_sec=15.0)

    # 3 Urdu segments: seg 1 ends with Urdu full stop '۔' at 6.0s
    segments = [
        TranscriptionSegment(start=0.0, end=3.0, text="بادشاہ سلامت محل میں تشریف لائے"),
        TranscriptionSegment(start=3.0, end=6.0, text="اور سب نے جھک کر سلام کیا۔"),
        TranscriptionSegment(start=6.0, end=11.0, text="وزیر نے جنگ کی خبر سنائی۔"),
    ]
    transcription = TranscriptionResult(
        audio_path=Path("audio.mp3"),
        audio_hash="h1",
        language="ur",
        full_text="مکمل کہانی",
        segments=segments,
        model_used="base",
        elapsed_seconds=1.0,
    )

    chunks = chunker.chunk_transcription(transcription)
    assert len(chunks) == 2
    # Chunk 0 should end exactly at 6.0s on the Urdu sentence boundary
    assert chunks[0].end_sec == 6.0
    assert chunks[0].duration_sec == 6.0
    assert "سلام کیا۔" in chunks[0].text
    # Chunk 1 starts at 6.0s and ends at 11.0s
    assert chunks[1].start_sec == 6.0
    assert chunks[1].end_sec == 11.0


def test_episode_compiler_precision_subclip_slicing(tmp_path: Path):
    """
    Verify that EpisodeCompiler guarantees exact target_duration for each subclip,
    aligning extracted cuts precisely to narrative chunk timestamps.
    """
    audio_file = tmp_path / "voiceover.mp3"
    audio_file.write_bytes(b"dummy_audio")
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    (footage_dir / "ep1.mp4").write_bytes(b"dummy_video")
    out_file = tmp_path / "out.mp4"

    cfg = EpisodeCompileConfig(
        audio_path=audio_file,
        footage_path=footage_dir,
        output_path=out_file,
        language="ur",
        gemini_api_key="AIzaSyMockKeyForSlicing12345",
    )
    compiler = EpisodeCompiler(cfg)

    # Mock audio probe to 20.0 seconds
    mock_audio_media = MagicMock()
    mock_audio_media.duration = 20.0
    mock_audio_media.has_audio = True

    # Mock video probe to 60.0 seconds
    mock_video_media = MagicMock()
    mock_video_media.duration = 60.0
    mock_video_media.width = 1920
    mock_video_media.height = 1080
    mock_video_media.has_audio = False

    segments = [
        TranscriptionSegment(start=0.0, end=8.0, text="حصہ اول۔"),
        TranscriptionSegment(start=8.0, end=20.0, text="حصہ دوم۔"),
    ]
    trans_res = TranscriptionResult(
        audio_path=audio_file,
        audio_hash="h1",
        language="ur",
        full_text="پورا بیانیہ",
        segments=segments,
        model_used="base",
        elapsed_seconds=1.0,
    )

    scenes = [
        Scene(1, "h1", footage_dir / "ep1.mp4", 1, 10.0, 15.0, 5.0, description="Scene 1"),
        Scene(2, "h1", footage_dir / "ep1.mp4", 2, 25.0, 35.0, 10.0, description="Scene 2"),
    ]

    extracted_durations = []

    def mock_extract(video_path, start_sec, end_sec, output_path, target_duration, **kwargs):
        extracted_durations.append(target_duration)
        return output_path

    with patch("services.episode_compiler.probe_media", side_effect=[mock_audio_media, mock_video_media, mock_video_media, mock_video_media]), \
         patch("services.transcriber.TranscriptionService.transcribe", return_value=trans_res), \
         patch("services.subtitle_engine.SubtitleEngine.generate_ass"), \
         patch("services.subtitle_engine.SubtitleEngine.generate_srt"), \
         patch("services.scene_detector.SceneDetectorService.detect_scenes", return_value=scenes), \
         patch("services.gemini_matcher.SmartSceneMatcher._match_with_gemini") as mock_gemini, \
         patch("services.episode_compiler.extract_subclip", side_effect=mock_extract), \
         patch("services.episode_compiler.concatenate_clips"):

        def fake_gemini(chunks, scenes, language="ur"):
            for i, ch in enumerate(chunks):
                ch.matched_scene = scenes[i % len(scenes)]
            return chunks

        mock_gemini.side_effect = fake_gemini

        res = compiler.compile()
        assert res is not None
        # Sum of subclip target durations must match total audio duration
        assert sum(extracted_durations) == pytest.approx(20.0, abs=0.1)
        assert len(extracted_durations) == len(res.matched_chunks)


def test_studio_view_blocks_render_urdu_without_gemini_key(qapp, tmp_path: Path):
    """Verify StudioView immediately halts Template 2 render if Urdu is selected without a valid Gemini key."""
    audio_file = tmp_path / "urdu_voiceover.mp3"
    audio_file.write_bytes(b"dummy")
    assets_dir = tmp_path / "raw_footage"
    assets_dir.mkdir()
    (assets_dir / "ep.mp4").write_bytes(b"dummy_vid")

    view = StudioView()
    view.edit_audio.setText(str(audio_file))
    view.edit_assets.setText(str(assets_dir))

    # Set Template 2 (Index 1) and Urdu language (Index 0)
    view.combo_format.setCurrentIndex(1)
    view.combo_language.setCurrentIndex(0)

    dummy_cfg = MagicMock()
    dummy_cfg.gemini_api_key = ""
    with patch("utils.config.reload_config", return_value=dummy_cfg), \
         patch.dict("os.environ", {"GEMINI_API_KEY": ""}), \
         patch("ui.views.studio_view.QMessageBox.critical") as mock_crit:

        view._start_render()

        # Must display critical warning dialog and halt
        assert mock_crit.called
        assert "Gemini API Key Required" in mock_crit.call_args[0][1]
        assert "Urdu" in mock_crit.call_args[0][2]
        assert getattr(view, "_worker", None) is None or not view._worker.isRunning()


def test_studio_view_blocks_render_english_without_gemini_key(qapp, tmp_path: Path):
    """Verify StudioView also halts Template 2 render if English is selected without a valid Gemini key."""
    audio_file = tmp_path / "english_voiceover.mp3"
    audio_file.write_bytes(b"dummy")
    assets_dir = tmp_path / "raw_footage"
    assets_dir.mkdir()
    (assets_dir / "ep.mp4").write_bytes(b"dummy_vid")

    view = StudioView()
    view.edit_audio.setText(str(audio_file))
    view.edit_assets.setText(str(assets_dir))

    # Set Template 2 (Index 1) and English language (Index 1)
    view.combo_format.setCurrentIndex(1)
    view.combo_language.setCurrentIndex(1)

    dummy_cfg = MagicMock()
    dummy_cfg.gemini_api_key = ""
    with patch("utils.config.reload_config", return_value=dummy_cfg), \
         patch.dict("os.environ", {"GEMINI_API_KEY": ""}), \
         patch("ui.views.studio_view.QMessageBox.critical") as mock_crit:

        view._start_render()

        assert mock_crit.called
        assert "Gemini API Key Required" in mock_crit.call_args[0][1]
        assert "English" in mock_crit.call_args[0][2]
        assert getattr(view, "_worker", None) is None or not view._worker.isRunning()


def test_studio_view_template2_telemetry_dialog_on_finish(qapp, tmp_path: Path):
    """Verify StudioView displays the modal telemetry popup with exact tokens and estimated USD cost."""
    out_file = tmp_path / "render_done.mp4"
    out_file.write_bytes(b"dummy_video")

    view = StudioView()
    view.combo_format.setCurrentIndex(1)

    # Mock worker with token_stats
    mock_worker = MagicMock()
    mock_worker.token_stats = {
        "prompt_tokens": 4800,
        "output_tokens": 620,
        "total_tokens": 5420,
        "estimated_cost_usd": 0.000546,
        "used_gemini": True,
    }
    view._worker = mock_worker

    with patch.object(QMessageBox, "exec") as mock_exec:
        view.show()
        view._on_render_finished(str(out_file))

        assert mock_exec.called
        assert not view.lbl_token_usage_banner.isHidden()
        assert "5,420" in view.lbl_token_usage_banner.text()
        assert "$0.0005" in view.lbl_token_usage_banner.text()


def test_cached_audio_transcripts_bypass_whisper_and_cleanup_temp_wav(tmp_path: Path):
    """
    Verify that:
    1. Whisper transcription is persisted to SQLite.
    2. Repeat runs with identical audio hash bypass Whisper model inference completely.
    3. Ephemeral temporary WAV files are automatically deleted after transcription.
    """
    from services.transcriber import TranscriptionService, TranscriptionSegment, TranscriptionResult
    from database.db import get_cached_transcription

    import uuid
    svc = TranscriptionService()
    temp_wav = tmp_path / "temp_voiceover_extract.wav"
    audio_data = f"RIFF_FAKE_AUDIO_SAMPLE_{uuid.uuid4()}".encode("utf-8")
    temp_wav.write_bytes(audio_data)

    mock_whisper_model = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 4.5
    mock_seg.text = "This is a cached audio test."
    mock_seg.words = []
    mock_seg.avg_logprob = -0.2
    mock_seg.no_speech_prob = 0.01

    mock_info = MagicMock()
    mock_info.duration = 4.5
    mock_info.language = "en"

    mock_whisper_model.transcribe.return_value = ([mock_seg], mock_info)
    svc._model = mock_whisper_model

    # Run 1: First transcription (invokes Whisper and stores to SQLite)
    res1 = svc.transcribe(temp_wav, project_id=None, language="en", cleanup_temp_audio=False)
    assert res1.from_cache is False
    assert mock_whisper_model.transcribe.call_count == 1
    assert "This is a cached audio test." in res1.full_text

    # Verify SQLite cache entry exists
    cached_row = get_cached_transcription(res1.audio_hash)
    assert cached_row is not None
    assert cached_row["full_text"] == res1.full_text

    # Run 2: Repeat transcription on same audio content with cleanup_temp_audio=True
    # Recreate the file with the same content to simulate extracting or reusing
    temp_wav.write_bytes(audio_data)
    assert temp_wav.exists()

    res2 = svc.transcribe(temp_wav, project_id=None, language="en", cleanup_temp_audio=True)

    # Must be served from cache with elapsed_seconds == 0.0, completely bypassing Whisper
    assert res2.from_cache is True
    assert res2.elapsed_seconds == 0.0
    assert mock_whisper_model.transcribe.call_count == 1  # No additional Whisper inference call!
    assert res2.full_text == res1.full_text

    # Temporary WAV file must be automatically cleaned up from disk
    assert not temp_wav.exists(), "Temporary WAV file should have been cleaned up after cache hit"


def test_episode_compiler_cleans_up_intermediate_chunks(tmp_path: Path):
    """Verify EpisodeCompiler automatically deletes intermediate subclip chunks after master export."""
    audio_file = tmp_path / "voiceover_clean.mp3"
    audio_file.write_bytes(b"dummy_audio")
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    (footage_dir / "ep1.mp4").write_bytes(b"dummy_video")
    out_file = tmp_path / "final_out.mp4"

    cfg = EpisodeCompileConfig(
        audio_path=audio_file,
        footage_path=footage_dir,
        output_path=out_file,
        language="en",
        gemini_api_key="AIzaSyMockKeyForCleanupTest12345",
    )
    compiler = EpisodeCompiler(cfg)

    mock_audio_media = MagicMock(duration=10.0, has_audio=True)
    mock_video_media = MagicMock(duration=30.0, width=1920, height=1080, has_audio=False)

    segments = [TranscriptionSegment(start=0.0, end=10.0, text="Clean test segment.")]
    trans_res = TranscriptionResult(
        audio_path=audio_file,
        audio_hash="clean_hash",
        language="en",
        full_text="Clean test segment.",
        segments=segments,
        model_used="base",
        elapsed_seconds=0.0,
    )

    scenes = [Scene(1, "h1", footage_dir / "ep1.mp4", 1, 0.0, 10.0, 10.0)]

    created_clip_dir = []

    def mock_extract(video_path, start_sec, end_sec, output_path, **kwargs):
        created_clip_dir.append(output_path.parent)
        output_path.write_bytes(b"clip_data")
        return output_path

    def mock_concat(clip_paths, audio_path, output_path, **kwargs):
        output_path.write_bytes(b"final_mp4_data")
        return output_path

    with patch("services.episode_compiler.probe_media", side_effect=[mock_audio_media, mock_video_media, mock_video_media, mock_video_media]), \
         patch("services.transcriber.TranscriptionService.transcribe", return_value=trans_res), \
         patch("services.subtitle_engine.SubtitleEngine.generate_ass"), \
         patch("services.subtitle_engine.SubtitleEngine.generate_srt"), \
         patch("services.scene_detector.SceneDetectorService.detect_scenes", return_value=scenes), \
         patch("services.gemini_matcher.SmartSceneMatcher._match_with_gemini") as mock_gemini, \
         patch("services.episode_compiler.extract_subclip", side_effect=mock_extract), \
         patch("services.episode_compiler.concatenate_clips", side_effect=mock_concat):

        def fake_gemini(chunks, scenes, language="en"):
            for ch in chunks:
                ch.matched_scene = scenes[0]
            return chunks

        mock_gemini.side_effect = fake_gemini

        res = compiler.compile()
        assert res is not None

        # Verify intermediate clips directory was cleaned up
        assert len(created_clip_dir) > 0
        clips_folder = created_clip_dir[0]
        assert not clips_folder.exists(), "Intermediate clips directory should be cleaned up after export succeeds"


def test_persistent_scene_match_cache_instant_rerun():
    """
    Verify that completed Gemini narrative-to-scene matches are persisted to SQLite,
    and subsequent re-renders with the same voiceover + video hashes load instantly in 0.01s with 0 API calls.
    """
    import json
    from database.db import init_db

    init_db()

    chunks = [
        NarrativeChunk(0, 0.0, 5.0, 5.0, "Opening narrator statement"),
        NarrativeChunk(1, 5.0, 10.0, 5.0, "Second narrator statement"),
    ]
    scenes = [
        Scene(101, "vid_cache_test", Path("scene_101.mp4"), 1, 0.0, 5.0, 5.0, description="Scene 101"),
        Scene(102, "vid_cache_test", Path("scene_102.mp4"), 2, 5.0, 10.0, 5.0, description="Scene 102"),
    ]

    import uuid
    vo_hash = f"vo_test_hash_{uuid.uuid4().hex}"
    vid_hash = f"vid_test_hash_{uuid.uuid4().hex}"

    mock_resp = MagicMock()
    mock_resp.text = json.dumps([
        {"chunk_index": 0, "candidate_index": 0},
        {"chunk_index": 1, "candidate_index": 1},
    ])
    mock_resp.usage_metadata = MagicMock(prompt_token_count=150, candidates_token_count=30, total_token_count=180)

    # First Run: Cache miss -> calls Gemini once and caches result in SQLite
    matcher1 = SmartSceneMatcher(gemini_api_key="AIzaSyMockKeyForCachePersistence12345")
    with patch("services.gemini_resolver.generate_content_with_fallback", return_value=(mock_resp, "gemini-3-flash-preview")) as mock_gen:
        res1 = matcher1.match_scenes(
            chunks=chunks,
            available_scenes=scenes,
            voiceover_hash=vo_hash,
            video_hash=vid_hash,
            template_mode="template_2",
            language="en",
        )
        assert mock_gen.call_count == 1
        assert matcher1.from_cache is False
        assert matcher1.api_calls_count == 1
        assert len(res1) == 2
        assert res1[0].matched_scene.scene_id == 101
        assert res1[1].matched_scene.scene_id == 102

    # Second Run: Cache hit -> loads instantly in 0.01s with 0 API calls
    matcher2 = SmartSceneMatcher(gemini_api_key="AIzaSyMockKeyForCachePersistence12345")
    with patch("services.gemini_resolver.generate_content_with_fallback") as mock_gen_rerun:
        res2 = matcher2.match_scenes(
            chunks=chunks,
            available_scenes=scenes,
            voiceover_hash=vo_hash,
            video_hash=vid_hash,
            template_mode="template_2",
            language="en",
        )
        # Gemini was NEVER called!
        mock_gen_rerun.assert_not_called()
        assert matcher2.from_cache is True
        assert matcher2.api_calls_count == 0
        assert matcher2.total_tokens == 0
        assert len(res2) == 2
        assert res2[0].matched_scene.scene_id == 101
        assert res2[1].matched_scene.scene_id == 102



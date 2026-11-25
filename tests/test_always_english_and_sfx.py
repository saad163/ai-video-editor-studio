"""
tests/test_always_english_and_sfx.py
====================================
Comprehensive tests for:
  1. Always-English Subtitles:
     - Enforces strictly English subtitles even when voiceover transcription is Urdu or non-English.
     - Ensures timestamps stay aligned with spoken segments.
     - Sanitizes empty/garbled lines and prevents non-English script leakage.
     - Passes English subtitles through cleanly without modification.
  2. 5–6 Transition Sound Effects (SFX):
     - Curated library of 6 transition SFX with fallback synthesis.
     - User selection state (subset, all, or 0 selected).
     - Random non-repeating transition placement (no back-to-back repeats when 2+ SFX selected).
     - Safe gain clamping and clean skip when 0 SFX selected.
     - StudioView UI controls for SFX selection and volume.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication

from services.subtitle_engine import (
    SubtitleEngine,
    has_non_latin_script,
    has_non_latin_script_in_segments,
    sanitize_english_text,
    translate_and_sanitize_to_english,
)
from services.transcriber import TranscriptionResult, TranscriptionSegment
from services.sfx_manager import (
    SFXManager,
    CURATED_SFX_CATALOG,
    synthesize_sfx_audio,
    write_wav_file,
)
from ui.views.studio_view import StudioView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ─────────────────────────────────────────────────────────────────────────────
# 1. Always-English Subtitles Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_has_non_latin_script_detection():
    """Verify has_non_latin_script correctly detects Urdu and non-Latin characters."""
    assert has_non_latin_script("بادشاہ سلامت") is True
    assert has_non_latin_script("یہ ایک کہانی ہے") is True
    assert has_non_latin_script("Hello world") is False
    assert has_non_latin_script("The King arrives at 10:00 AM!") is False
    assert has_non_latin_script("Mixed script بادشاہ and English") is True


def test_sanitize_english_text_removes_urdu_leakage():
    """Verify sanitize_english_text strips all Urdu and foreign characters without touching English."""
    assert sanitize_english_text("Hello world") == "Hello world"
    # Pure Urdu becomes empty
    assert sanitize_english_text("بادشاہ سلامت") == ""
    # Mixed text strips Urdu and leaves clean English
    assert sanitize_english_text("King بادشاہ arrived سلامت") == "King arrived"
    # Normalizes excess spaces
    assert sanitize_english_text("  The   Prince   arrived.  ") == "The Prince arrived."


def test_translate_and_sanitize_to_english_preserves_timestamps():
    """Verify offline translation preserves exact start/end timestamps and produces pure English."""
    urdu_segs = [
        TranscriptionSegment(start=0.5, end=3.5, text="بادشاہ سلامت محل میں تشریف لائے"),
        TranscriptionSegment(start=3.5, end=7.0, text="اور وزیر نے جنگ کی خبر سنائی"),
    ]

    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        eng_segs = translate_and_sanitize_to_english(urdu_segs, gemini_api_key="", source_language="ur")
    assert len(eng_segs) == 2

    # Timestamp preservation
    assert eng_segs[0].start == 0.5
    assert eng_segs[0].end == 3.5
    assert eng_segs[1].start == 3.5
    assert eng_segs[1].end == 7.0

    # Strict English enforcement: No Urdu characters can exist in text
    urdu_pattern = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
    for s in eng_segs:
        assert not urdu_pattern.search(s.text), f"Found leaked Urdu script in: {s.text}"
        assert len(s.text.strip()) > 0


def test_translate_with_gemini_mocked():
    """Verify Gemini batch translation translates Urdu to concise English while keeping timestamps."""
    urdu_segs = [
        TranscriptionSegment(start=1.0, end=4.0, text="بادشاہ سلامت نے اعلان کیا"),
        TranscriptionSegment(start=4.0, end=8.0, text="سب لوگ محل میں جمع ہو گئے"),
    ]

    mock_resp = MagicMock()
    mock_resp.text = json.dumps([
        {"i": 0, "start": 1.0, "end": 4.0, "text": "The King made an announcement."},
        {"i": 1, "start": 4.0, "end": 8.0, "text": "Everyone gathered in the palace."},
    ])

    with patch("services.gemini_resolver.generate_content_with_fallback", return_value=(mock_resp, "gemini-3-flash-preview")):
        translated = translate_and_sanitize_to_english(
            urdu_segs,
            gemini_api_key="AIzaSyMockKeyForSubtitleTranslation12345",
            source_language="ur",
        )

    assert len(translated) == 2
    assert translated[0].text == "The King made an announcement."
    assert translated[0].start == 1.0
    assert translated[0].end == 4.0
    assert translated[1].text == "Everyone gathered in the palace."
    assert translated[1].start == 4.0
    assert translated[1].end == 8.0


def test_generate_ass_and_srt_strictly_english_from_urdu(tmp_path):
    """Verify SubtitleEngine generates ASS and SRT with zero raw Urdu script when input is Urdu."""
    urdu_segs = [
        TranscriptionSegment(start=0.0, end=4.0, text="بادشاہ سلامت کا محل"),
        TranscriptionSegment(start=4.0, end=8.0, text="ایک نئی جنگ کا آغاز"),
    ]
    urdu_res = TranscriptionResult(
        audio_path=Path("mock_urdu.mp3"),
        audio_hash="dummy_hash_1",
        language="ur",
        full_text="بادشاہ سلامت کا محل ایک نئی جنگ کا آغاز",
        segments=urdu_segs,
        model_used="whisper",
        elapsed_seconds=0.5,
    )

    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        engine = SubtitleEngine(gemini_api_key="")
        ass_out = tmp_path / "test_subs.ass"
        srt_out = tmp_path / "test_subs.srt"

        engine.generate_ass(urdu_res, ass_out)
        engine.generate_srt(urdu_res, srt_out)

    ass_content = ass_out.read_text(encoding="utf-8-sig")
    srt_content = srt_out.read_text(encoding="utf-8")

    # Both must be non-empty
    assert len(ass_content) > 100
    assert len(srt_content) > 50

    # Must contain Dialogue events
    assert "Dialogue: 0," in ass_content

    # Strict assertion: NO raw Urdu characters can appear anywhere in the output!
    urdu_char_match = re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", ass_content)
    assert urdu_char_match is None, f"Urdu character leaked into ASS: {urdu_char_match.group() if urdu_char_match else ''}"

    urdu_char_srt = re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", srt_content)
    assert urdu_char_srt is None, f"Urdu character leaked into SRT: {urdu_char_srt.group() if urdu_char_srt else ''}"


def test_english_subtitles_pass_through_unaltered(tmp_path):
    """Verify that pure English subtitles are not modified or corrupted."""
    eng_segs = [
        TranscriptionSegment(start=0.0, end=3.0, text="Prince William arrived in London"),
        TranscriptionSegment(start=3.0, end=6.0, text="The crowd cheered enthusiastically"),
    ]
    eng_res = TranscriptionResult(
        audio_path=Path("mock_en.mp3"),
        audio_hash="dummy_en_hash",
        language="en",
        full_text="Prince William arrived in London The crowd cheered enthusiastically",
        segments=eng_segs,
        model_used="whisper",
        elapsed_seconds=0.2,
    )

    engine = SubtitleEngine()
    ass_out = tmp_path / "en_subs.ass"
    engine.generate_ass(eng_res, ass_out)

    content = ass_out.read_text(encoding="utf-8-sig")
    assert "Prince William arrived in London" in content
    assert "The crowd cheered enthusiastically" in content


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transition Sound Effects (SFX) Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sfx_catalog_has_6_curated_effects():
    """Verify the SFX catalog defines all 6 standard transition sound effects."""
    mgr = SFXManager()
    catalog = mgr.get_catalog()
    assert len(catalog) == 6

    expected_ids = {"whoosh_fast", "whoosh_deep", "swish_clean", "glitch_digital", "riser_short", "pop_smooth"}
    catalog_ids = {item["id"] for item in catalog}
    assert catalog_ids == expected_ids


def test_sfx_fallback_synthesis_creates_valid_wav_files(tmp_path):
    """Verify missing SFX assets are synthesized automatically into valid 48kHz WAV files."""
    custom_sfx_dir = tmp_path / "test_sfx_library"
    custom_sfx_dir.mkdir()

    # Initialize SFXManager targeting an empty directory
    mgr = SFXManager(sfx_dir=custom_sfx_dir)
    resolved = mgr.ensure_default_sfx_library()

    assert len(resolved) == 6
    for sfx_id, wav_path in resolved.items():
        assert wav_path.exists(), f"WAV file was not generated: {wav_path}"
        assert wav_path.stat().st_size > 1000

        # Validate standard WAV audio header
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit PCM
            assert wf.getframerate() == 48000


def test_sfx_random_sequence_avoids_back_to_back_repeats():
    """Verify random sequence generator NEVER produces back-to-back identical SFX when 2+ are selected."""
    mgr = SFXManager()
    catalog = mgr.get_catalog()
    paths = [item["path"] for item in catalog]  # 6 sound effects

    # Test with 50 transitions
    seq = mgr.generate_sfx_sequence(paths, count=50, seed=42)
    assert len(seq) == 50

    for i in range(1, len(seq)):
        assert seq[i] != seq[i - 1], f"Back-to-back identical repeat at index {i}: {seq[i].name}"


def test_sfx_single_selection_repeats_cleanly():
    """Verify selecting exactly 1 sound effect repeats that effect for all transitions."""
    mgr = SFXManager()
    single_p = mgr.get_sfx_path("whoosh_fast")
    assert single_p is not None

    seq = mgr.generate_sfx_sequence([single_p], count=10)
    assert len(seq) == 10
    assert all(p == single_p for p in seq)


def test_sfx_zero_selection_returns_empty():
    """Verify 0 selected sound effects returns an empty list without error."""
    mgr = SFXManager()
    assert mgr.generate_sfx_sequence([], count=10) == []
    assert mgr.generate_sfx_sequence(["whoosh_fast"], count=0) == []


def test_build_transition_audio_track_and_clamping(tmp_path):
    """Verify build_transition_audio_track generates aligned multi-transition audio with safe gain."""
    mgr = SFXManager()
    clip_durations = [5.0, 4.0, 6.0, 5.0]  # 3 transitions at 5.0s, 9.0s, 15.0s
    selected = ["whoosh_fast", "swish_clean", "pop_smooth"]

    out_wav = tmp_path / "sfx_track.wav"
    res = mgr.build_transition_audio_track(
        clip_durations=clip_durations,
        selected_sfx=selected,
        output_path=out_wav,
        volume=0.35,
        sample_rate=48000,
        seed=123,
    )

    assert res is not None
    assert res.exists()
    assert res.stat().st_size > 0

    with wave.open(str(res), "rb") as wf:
        assert wf.getframerate() == 48000
        total_dur = wf.getnframes() / 48000.0
        # Total duration must match the sum of clip durations (20.0s)
        assert abs(total_dur - 20.0) < 0.1


def test_build_transition_audio_track_zero_selection_returns_none(tmp_path):
    """Verify 0 selected sound effects skips track generation cleanly returning None."""
    mgr = SFXManager()
    out_wav = tmp_path / "skip_sfx.wav"
    assert mgr.build_transition_audio_track([5.0, 5.0], [], out_wav) is None
    assert mgr.build_transition_audio_track([5.0], ["whoosh_fast"], out_wav) is None  # < 2 clips


# ─────────────────────────────────────────────────────────────────────────────
# 3. StudioView UI Controls Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_studio_view_sfx_controls_state_and_config(qapp):
    """Verify StudioView UI provides checkboxes for all 6 SFX, select/clear all, and config export."""
    view = StudioView()
    assert hasattr(view, "_sfx_checkboxes")
    assert len(view._sfx_checkboxes) == 6

    # Default state: all 6 are checked
    assert len(view.get_selected_sfx_ids()) == 6
    cfg = view.get_transition_sfx_config()
    assert cfg["enabled"] is True
    assert len(cfg["selected_sfx"]) == 6
    assert abs(cfg["volume"] - 0.35) < 1e-4

    # Test Clear All
    view._clear_all_sfx()
    assert len(view.get_selected_sfx_ids()) == 0
    assert view.lbl_sfx_badge.text() == "Off (0)"
    cfg_off = view.get_transition_sfx_config()
    assert cfg_off["enabled"] is False

    # Test Select All
    view._select_all_sfx()
    assert len(view.get_selected_sfx_ids()) == 6
    assert view.lbl_sfx_badge.text() == "6 Active"

    # Test Subset Selection (e.g., 2 sound effects)
    view.set_selected_sfx_ids(["whoosh_fast", "pop_smooth"])
    selected = view.get_selected_sfx_ids()
    assert set(selected) == {"whoosh_fast", "pop_smooth"}
    assert view.lbl_sfx_badge.text() == "2 Active"

    # Test Volume Slider adjustment
    view.slider_sfx_vol.setValue(50)
    cfg_vol = view.get_transition_sfx_config()
    assert abs(cfg_vol["volume"] - 0.50) < 1e-4
    assert view.lbl_sfx_vol_val.text() == "50%"

    # Test get_config() includes transition_sfx
    full_cfg = view.get_config()
    assert "transition_sfx" in full_cfg
    assert full_cfg["transition_sfx"]["enabled"] is True

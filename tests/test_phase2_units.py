"""
tests/test_phase2_units.py
==========================
Unit tests for Phase 2 Template 1 modules.

Run:
    pytest tests/test_phase2_units.py -v
"""

from __future__ import annotations

import pytest
from pathlib import Path

from services.script_parser import ScriptParser, ScriptBlock
from services.character_matcher import CharacterMatcher, _ImagePool
from services.subtitle_engine import SubtitleEngine, _to_ass_time, _to_srt_time
from services.transcriber import TranscriptionSegment, TranscriptionResult
from core.ffmpeg_wrapper import KenBurnsPreset, _build_zoompan_filter, _get_encoder_args


# ── ScriptParser Tests ────────────────────────────────────────────────────────

def test_script_parser_colon_syntax():
    parser = ScriptParser(known_characters=["Ali", "Zara"])
    text = "Ali: Hello there!\nZara: Hi Ali, welcome back."
    blocks = parser.parse_text(text)
    assert len(blocks) == 2
    assert blocks[0].character == "Ali"
    assert blocks[0].text == "Hello there!"
    assert blocks[1].character == "Zara"


def test_script_parser_bracket_syntax():
    parser = ScriptParser(known_characters=["Ali"])
    text = "[Ali] walks toward the window.\n(Ali) looks outside."
    blocks = parser.parse_text(text)
    assert len(blocks) == 2
    assert blocks[0].character == "Ali"
    assert blocks[1].character == "Ali"


def test_script_parser_fallback_to_narrator():
    parser = ScriptParser(known_characters=["Ali"])
    text = "The city skyline was quiet in the early morning."
    blocks = parser.parse_text(text)
    assert len(blocks) == 1
    assert blocks[0].character == "narrator"


def test_script_parser_align_to_segments():
    parser = ScriptParser(known_characters=["Ali", "Zara"])
    blocks = [
        ScriptBlock(character="Ali", text="Ali enters the room"),
        ScriptBlock(character="Zara", text="Zara is sitting by the window"),
    ]
    segments = [
        "Ali opened the door slowly",
        "Zara looked up with a smile",
        "It was raining outside",
    ]
    aligned = parser.align_to_segments(blocks, segments)
    assert len(aligned) == 3
    assert aligned[0] == "Ali"
    assert aligned[1] == "Zara"


# ── CharacterMatcher Tests ───────────────────────────────────────────────────

def test_image_pool_anti_repetition(tmp_path: Path):
    imgs = [tmp_path / f"img_{i}.jpg" for i in range(3)]
    for img in imgs:
        img.write_text("fake")

    pool = _ImagePool("test", imgs)
    picks = [pool.next_image() for _ in range(6)]

    # Consecutive picks within pool must alternate
    for i in range(len(picks) - 1):
        assert picks[i] != picks[i + 1]


def test_character_matcher_assignment_cycle(tmp_path: Path):
    char_dir = tmp_path / "ali"
    char_dir.mkdir()
    (char_dir / "ali_1.jpg").write_text("fake")
    (char_dir / "ali_2.jpg").write_text("fake")

    matcher = CharacterMatcher(images_dir=tmp_path)
    segs = [TranscriptionSegment(start=i * 2.0, end=(i + 1) * 2.0, text=f"seg {i}") for i in range(4)]
    mock_res = TranscriptionResult(
        audio_path=tmp_path / "audio.mp3",
        audio_hash="HASH",
        language="en",
        full_text="text",
        segments=segs,
        model_used="small",
        elapsed_seconds=0.0,
    )

    assignments = matcher.assign(mock_res, character_assignments=["ali"] * 4)
    assert len(assignments) == 4
    # Check no consecutive same image
    for i in range(len(assignments) - 1):
        assert assignments[i].image_path != assignments[i + 1].image_path


# ── SubtitleEngine Tests ─────────────────────────────────────────────────────

def test_subtitle_engine_time_conversion():
    assert _to_ass_time(0.0) == "0:00:00.00"
    assert _to_ass_time(65.5) == "0:01:05.50"
    assert _to_srt_time(65.5) == "00:01:05,500"


def test_subtitle_engine_ass_generation(tmp_path: Path):
    engine = SubtitleEngine(color="yellow", font_size=48)
    segs = [TranscriptionSegment(start=0.0, end=3.5, text="Welcome to AI Video Editor")]
    mock_res = TranscriptionResult(
        audio_path=tmp_path / "audio.mp3",
        audio_hash="HASH",
        language="en",
        full_text="Welcome to AI Video Editor",
        segments=segs,
        model_used="small",
        elapsed_seconds=0.0,
    )
    out_ass = tmp_path / "subs.ass"
    engine.generate_ass(mock_res, out_ass)

    content = out_ass.read_text(encoding="utf-8-sig")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "[Events]" in content
    assert "Dialogue:" in content
    assert "Welcome to AI Video Editor" in content


# ── FFmpeg Filter & Encoder Tests ────────────────────────────────────────────

def test_zoompan_filter_syntax():
    for preset in KenBurnsPreset:
        flt = _build_zoompan_filter(duration=3.0, preset=preset)
        assert "scale=" in flt
        assert "zoompan=" in flt
        assert "format=yuv420p" in flt
        # No unescaped commas inside parameters that could break FFmpeg option parsing
        assert "max(" not in flt


def test_encoder_args_returns_valid_list():
    args = _get_encoder_args()
    assert "-c:v" in args
    assert ("libx264" in args or "h264_nvenc" in args)


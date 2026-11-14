"""
Unit and integration tests for:
1. Blurred background pillarbox on portrait/vertical images.
2. SubtitleEngine advanced typography (bold, italic, font family, ASS borderstyle=3 box padding, auto-wrapping).
"""

from pathlib import Path
import pytest
from PIL import Image

from core.ffmpeg_wrapper import build_ken_burns_clip, probe_media
from services.subtitle_engine import SubtitleEngine
from services.transcriber import TranscriptionResult, TranscriptionSegment


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_portrait_blurred_pillarbox(temp_dir: Path):
    """
    Verify portrait image (e.g. 1080x1920) rendered onto 1920x1080 canvas
    triggers dual-layer blurred pillarbox without error and produces 1920x1080 output.
    """
    portrait_img = temp_dir / "portrait_photo.png"
    # Create 1080x1920 portrait test image
    img = Image.new("RGB", (1080, 1920), color=(220, 50, 50))
    img.save(portrait_img)

    out_clip = temp_dir / "portrait_pillarbox.mp4"
    build_ken_burns_clip(
        image_path=portrait_img,
        duration=1.5,
        output_path=out_clip,
        resolution="1080p",
    )

    assert out_clip.exists(), "Output clip must be generated"
    assert out_clip.stat().st_size > 0, "Output clip must not be empty"

    media_info = probe_media(out_clip)
    assert media_info.width == 1920
    assert media_info.height == 1080
    assert media_info.duration >= 1.4


def test_subtitle_engine_typography_and_box_padding(temp_dir: Path):
    """
    Verify SubtitleEngine sets ASS Bold=-1, Italic=-1, BorderStyle=3,
    and maps box_padding to Outline width.
    """
    engine = SubtitleEngine(
        text_color="#FFFF00",
        highlight_color="#00FFFF",
        has_bg_box=True,
        bg_style="custom_box",
        bg_color="#1E293B",
        bg_opacity=0.75,
        box_padding=18.0,
        font_size=60,
        font_name="Montserrat",
        bold=True,
        italic=True,
        resolution="1080p",
    )

    dummy_result = TranscriptionResult(
        audio_path=Path("dummy.mp3"),
        audio_hash="dummyhash",
        language="en",
        full_text="The unexpected royal decision shaped modern history forever.",
        segments=[
            TranscriptionSegment(
                start=0.0,
                end=3.5,
                text="The unexpected royal decision shaped modern history forever.",
            )
        ],
        model_used="base",
        elapsed_seconds=1.0,
    )

    ass_path = temp_dir / "test_styled.ass"
    engine.generate_ass(dummy_result, ass_path)

    assert ass_path.exists()
    content = ass_path.read_text(encoding="utf-8")

    # Check Style line
    # Style: Default,Montserrat,60,...
    assert "Style: Default,Montserrat,60" in content
    # Bold=-1 and Italic=-1
    assert ",-1,-1," in content
    # BorderStyle=3 (box mode)
    assert ",3,18.0," in content or ",3,18," in content


def test_subtitle_engine_balanced_multiline_wrapping():
    """
    Verify smart balanced auto-wrapping wraps long lines cleanly with \\N.
    """
    engine = SubtitleEngine(
        max_chars_per_line=38,
    )

    # 1. Short text should not be wrapped
    short_text = "Hello world"
    assert engine._wrap_text(short_text) == "Hello world"

    # 2. Text needing 2 lines should be split with \\N
    long_text = "The Princess royal made a surprising announcement at Windsor Castle today"
    wrapped = engine._wrap_text(long_text)
    assert r"\N" in wrapped
    parts = wrapped.split(r"\N")
    assert len(parts) == 2
    for p in parts:
        assert len(p) <= 45, f"Line too long: {p}"

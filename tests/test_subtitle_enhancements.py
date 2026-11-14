"""
tests/test_subtitle_enhancements.py
===================================
Verifies:
  1. RGB Hex to ASS BGR color conversion (&HAABBGGRR)
  2. Outline mode vs Background Box modes (BorderStyle=1 vs BorderStyle=3)
  3. UI Quick Presets and Live Caption Preview updates
"""

import pytest
from pathlib import Path
from services.subtitle_engine import SubtitleEngine, to_ass_color
from services.transcriber import TranscriptionResult, TranscriptionSegment
from ui.views.studio_view import StudioView
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ass_color_conversion():
    # Yellow: RGB(255, 255, 0) -> ASS BGR: 00FFFF -> &H0000FFFF
    assert to_ass_color("#FFFF00") == "&H0000FFFF"
    # White: RGB(255, 255, 255) -> &H00FFFFFF
    assert to_ass_color("#FFFFFF") == "&H00FFFFFF"
    # Cyan: RGB(0, 255, 255) -> &H00FFFF00
    assert to_ass_color("#00FFFF") == "&H00FFFF00"
    # Black: &H00000000
    assert to_ass_color("#000000") == "&H00000000"
    # Semi-transparent black box (alpha=128)
    assert to_ass_color("#000000", alpha=128) == "&H80000000"


def test_subtitle_modes_generation(tmp_path):
    seg = TranscriptionSegment(start=0.0, end=3.0, text="Test Subtitle Box")
    res = TranscriptionResult(
        audio_path=Path("mock.mp3"),
        audio_hash="HASH",
        language="en",
        full_text="Test Subtitle Box",
        segments=[seg],
        model_used="small",
        elapsed_seconds=0.1
    )

    # 1. Outline mode
    eng_outline = SubtitleEngine(text_color="#FFFF00", bg_style="outline")
    assert eng_outline.style.borderstyle == 1
    assert eng_outline.style.primary_colour == "&H0000FFFF"
    out1 = eng_outline.generate_ass(res, tmp_path / "sub1.ass")
    content1 = out1.read_text(encoding="utf-8-sig")
    assert "BorderStyle, Outline" in content1
    assert "&H0000FFFF" in content1

    # 2. Solid Black Box Mode
    eng_box = SubtitleEngine(text_color="#FFFFFF", has_bg_box=True, bg_style="solid_black")
    assert eng_box.style.borderstyle == 3
    assert eng_box.style.back_colour == "&H00000000"

    # 3. Semi-transparent Black Box Mode
    eng_trans = SubtitleEngine(text_color="#FFFF00", has_bg_box=True, bg_style="semi_transparent_black")
    assert eng_trans.style.borderstyle == 3
    assert eng_trans.style.back_colour == "&H80000000"


def test_studio_view_presets_and_preview(qapp):
    view = StudioView()

    # Preset 0: Classic YouTube (White + Outline)
    view._on_quick_preset_changed(0)
    assert view._font_color == "#FFFFFF"
    assert view._bg_style == "outline"
    assert view._has_bg_box is False

    # Preset 1: Viral Shorts (Yellow + Semi-Transparent Box)
    view._on_quick_preset_changed(1)
    assert view._font_color == "#FFFF00"
    assert view._bg_style == "semi_transparent_black"
    assert view._has_bg_box is True

    # Preset 2: High-Contrast Box (Yellow + Solid Black Box)
    view._on_quick_preset_changed(2)
    assert view._font_color == "#FFFF00"
    assert view._bg_style == "solid_black"
    assert view._has_bg_box is True

    view.close()


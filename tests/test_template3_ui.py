"""
Unit tests for Template 3 (+ Interview) UI & Conditional Visibility in StudioView.
"""
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.views.studio_view import StudioView


def test_template3_dropdown_and_conditional_visibility(qtbot, tmp_path, monkeypatch):
    """Verify Template 3 option exists and card_interview is only visible when Template 3 is selected."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.show()

    # 1. Check dropdown items
    assert view.combo_format.count() >= 3
    assert "Template 3" in view.combo_format.itemText(2)
    assert "+ Interview" in view.combo_format.itemText(2)

    # 2. Template 1 (index 0) active by default -> card_interview hidden
    view.combo_format.setCurrentIndex(0)
    assert not view.card_interview.isVisible()
    assert view.card_interview.isHidden()

    # 3. Template 2 (index 1) -> card_interview hidden
    view.combo_format.setCurrentIndex(1)
    assert not view.card_interview.isVisible()
    assert view.card_interview.isHidden()

    # 4. Template 3 (index 2) -> card_interview becomes visible
    view.combo_format.setCurrentIndex(2)
    assert view.card_interview.isVisible()
    assert not view.card_interview.isHidden()

    # 5. Back to Template 1 -> card_interview hidden again
    view.combo_format.setCurrentIndex(0)
    assert not view.card_interview.isVisible()
    assert view.card_interview.isHidden()


def test_template3_default_controls_and_toggle(qtbot, monkeypatch):
    """Verify default control values, styling, and shape toggling for Template 3."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    # Sliders and default values
    assert view.slider_interview_music_vol.value() == 15
    assert view.slider_interview_scale.value() == 70
    assert view.slider_interview_blur.value() == 0
    assert view.chk_interview_border.isChecked() is True
    assert view.slider_interview_border_thickness.value() == 4
    assert view._interview_border_color == "#EF4444"
    assert view._interview_corner_shape == "rounded"
    assert view.slider_interview_curve.value() == 24
    assert view.slider_interview_curve.isEnabled() is True

    # Toggle to Sharp corners
    view.btn_corner_sharp.click()
    assert view._interview_corner_shape == "sharp"
    assert view.slider_interview_curve.isEnabled() is False

    # Toggle back to Rounded corners
    view.btn_corner_rounded.click()
    assert view._interview_corner_shape == "rounded"
    assert view.slider_interview_curve.isEnabled() is True

    # Test border checkbox toggle
    view.chk_interview_border.setChecked(False)
    assert view.slider_interview_border_thickness.isEnabled() is False
    assert view.btn_interview_border_color.isEnabled() is False
    view.chk_interview_border.setChecked(True)
    assert view.slider_interview_border_thickness.isEnabled() is True
    assert view.btn_interview_border_color.isEnabled() is True

    # Test collapsible card toggle button
    assert not view.interview_body.isHidden()
    view.btn_toggle_interview_card.click()
    assert view.interview_body.isHidden()
    assert view.btn_toggle_interview_card.text() == "▲"
    view.btn_toggle_interview_card.click()
    assert not view.interview_body.isHidden()
    assert view.btn_toggle_interview_card.text() == "▼"


def test_template3_config_serialization(qtbot, monkeypatch):
    """Verify get_interview_config and get_render_config produce expected payloads."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    view.edit_interview_music.setText("C:/audio/bgm.mp3")
    view.slider_interview_music_vol.setValue(25)
    view.slider_interview_scale.setValue(80)
    view.slider_interview_blur.setValue(40)
    view.slider_interview_border_thickness.setValue(8)
    view.edit_youtube_link.setText("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    interview_cfg = view.get_interview_config()
    assert interview_cfg["music_path"] == "C:/audio/bgm.mp3"
    assert interview_cfg["music_volume"] == 0.25
    assert interview_cfg["clip_scale"] == 0.8
    assert interview_cfg["bg_blur"] == 0.4
    assert interview_cfg["border_enabled"] is True
    assert interview_cfg["border_thickness"] == 8
    assert interview_cfg["border_color"] == "#EF4444"
    assert interview_cfg["youtube_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    render_cfg = view.get_render_config()
    assert render_cfg["template"] == "template_3"
    assert "interview" in render_cfg
    assert render_cfg["interview"]["music_volume"] == 0.25


def test_template3_youtube_scan_signal(qtbot, monkeypatch):
    """Verify clicking Scan Clips emits request_scan_youtube signal with URL."""
    from unittest.mock import patch
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    test_url = "https://www.youtube.com/watch?v=sample123"
    view.edit_youtube_link.setText(test_url)

    with patch("ui.workers.interview_scan_worker.InterviewScanWorker.start"):
        with qtbot.waitSignal(view.request_scan_youtube, timeout=1000) as blocker:
            view.btn_scan_youtube.click()

    assert blocker.args == [test_url]

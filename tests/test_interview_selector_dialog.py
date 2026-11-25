"""
tests/test_interview_selector_dialog.py
=======================================
Unit and integration tests for Part 3:
Interactive Clip Selection, Custom Duration Trimming & Placement Timing Dialog.
"""

from unittest.mock import patch
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QDialog

from services.interview_scanner import InterviewSegment
from ui.dialogs.interview_selector_dialog import (
    InterviewSelectorDialog,
    TrimmedInterviewClip,
    format_seconds_to_mmss,
)
from ui.views.studio_view import StudioView


@pytest.fixture
def sample_segments():
    return [
        InterviewSegment(
            segment_id=1,
            start_sec=10.0,
            end_sec=35.0,
            duration_sec=25.0,
            start_time="00:10",
            end_time="00:35",
            label="Neural pipeline breakthrough",
            transcript="We developed the continuous timeline algorithm.",
            speaker="Lead Architect",
        ),
        InterviewSegment(
            segment_id=2,
            start_sec=50.0,
            end_sec=75.0,
            duration_sec=25.0,
            start_time="00:50",
            end_time="01:15",
            label="Aspect ratio framing techniques",
            transcript="Vertical and horizontal framing need smart borders.",
            speaker="Designer",
        ),
        InterviewSegment(
            segment_id=3,
            start_sec=90.0,
            end_sec=110.0,
            duration_sec=20.0,
            start_time="01:30",
            end_time="01:50",
            label="Concluding thoughts and takeaways",
            transcript="In conclusion, automated workflows save hours.",
            speaker="Host",
        ),
    ]


def test_dialog_initialization_and_table_population(qtbot, sample_segments):
    """Verify dialog populates table with all detected segments and defaults to selected."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)

    assert dlg.table.rowCount() == 3
    assert dlg.table.columnCount() == 6
    assert len(dlg._rows) == 3

    # All rows checked by default
    for r in dlg._rows:
        assert r.chk_include.isChecked() is True
        assert r.spin_start.value() > 0
        assert r.spin_end.value() > r.spin_start.value()

    assert "3 of 3 clips selected" in dlg.lbl_summary_badge.text()
    assert dlg.card_warning.isVisible() is False


def test_select_all_and_clear_all(qtbot, sample_segments):
    """Verify Select All and Clear All buttons toggle row checkboxes and summary."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)

    # Click Clear All
    dlg.btn_clear_all.click()
    for r in dlg._rows:
        assert r.chk_include.isChecked() is False
    assert "0 of 3 clips selected" in dlg.lbl_summary_badge.text()

    # Click Select All
    dlg.btn_select_all.click()
    for r in dlg._rows:
        assert r.chk_include.isChecked() is True
    assert "3 of 3 clips selected" in dlg.lbl_summary_badge.text()


def test_custom_duration_trimming_updates(qtbot, sample_segments):
    """Verify trimming start and end times dynamically recalculates duration and summary."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()

    row0 = dlg._rows[0]
    # Initially 10.0 to 35.0 (25.0s)
    assert row0.spin_start.value() == 10.0
    assert row0.spin_end.value() == 35.0
    assert "25.0s" in row0.lbl_dur.text()

    # Trim to 10.0 -> 22.0 (12.0s)
    row0.spin_end.setValue(22.0)
    assert "12.0s" in row0.lbl_dur.text()

    # If end <= start, shows invalid warning
    row0.spin_end.setValue(8.0)
    assert "Invalid" in row0.lbl_dur.text()
    assert not dlg.card_warning.isHidden()

    # Restore valid range
    row0.spin_end.setValue(25.0)
    assert "15.0s" in row0.lbl_dur.text()
    assert dlg.card_warning.isHidden()


def test_timeline_overlap_guard(qtbot, sample_segments):
    """Verify detection of overlapping target video placement intervals."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()

    # Clip 1: dur 25s (10 -> 35), target_start 10s -> occupies [10, 35]
    dlg._rows[0].spin_target.setValue(10.0)

    # Clip 2: dur 25s (50 -> 75), target_start set to 20s -> occupies [20, 45] (collides with Clip 1!)
    dlg._rows[1].spin_target.setValue(20.0)

    overlaps = dlg.check_target_overlaps()
    assert len(overlaps) >= 1
    assert (1, 2) in overlaps
    assert not dlg.card_warning.isHidden()
    assert "Timeline Overlap Detected" in dlg.lbl_warning_text.text()

    # Click confirm should trigger warning and not accept
    with patch.object(QMessageBox, "warning") as mock_warn:
        dlg._on_confirm_clicked()
        assert mock_warn.called
        assert dlg.result() != int(QDialog.DialogCode.Accepted)

    # Resolve overlap by moving Clip 2 target_start to 40s -> occupies [40, 65]
    dlg._rows[1].spin_target.setValue(40.0)
    overlaps_resolved = dlg.check_target_overlaps()
    assert len(overlaps_resolved) == 0
    assert dlg.card_warning.isHidden()


def test_confirm_empty_selection_blocked(qtbot, sample_segments):
    """Verify confirming with 0 clips selected blocks acceptance."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()
    dlg.btn_clear_all.click()

    with patch.object(QMessageBox, "warning") as mock_warn:
        dlg._on_confirm_clicked()
        assert mock_warn.called
        assert dlg.result() != int(QDialog.DialogCode.Accepted)


def test_confirm_success_and_model_output(qtbot, sample_segments):
    """Verify successful confirmation builds TrimmedInterviewClip models and serialized dicts."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)

    # Keep only Clip 1 and Clip 3
    dlg._rows[1].chk_include.setChecked(False)

    # Trim Clip 1: start 12.0, end 32.0 (dur 20s), target 5.0
    dlg._rows[0].spin_start.setValue(12.0)
    dlg._rows[0].spin_end.setValue(32.0)
    dlg._rows[0].spin_target.setValue(5.0)

    # Clip 3: target 60.0
    dlg._rows[2].spin_target.setValue(60.0)

    dlg._on_confirm_clicked()
    assert len(dlg.confirmed_clips) == 2

    clip1 = dlg.confirmed_clips[0]
    assert clip1.segment_id == 1
    assert clip1.trim_start_sec == 12.0
    assert clip1.trim_end_sec == 32.0
    assert clip1.duration_sec == 20.0
    assert clip1.target_start_sec == 5.0
    assert clip1.target_end_sec == 25.0
    assert clip1.source_video == "C:/videos/sample_interview.mp4"

    clip3 = dlg.confirmed_clips[1]
    assert clip3.segment_id == 3
    assert clip3.target_start_sec == 60.0

    # Test serialized data format
    dict_list = dlg.get_confirmed_clips_data()
    assert len(dict_list) == 2
    assert dict_list[0]["trim_start_sec"] == 12.0
    assert dict_list[0]["duration_sec"] == 20.0


def test_prepopulating_previously_confirmed_clips(qtbot, sample_segments):
    """Verify reopening dialog pre-populates previous trimmed values and checkbox states."""
    prev_confirmed = [
        TrimmedInterviewClip(
            segment_id=2,
            source_video="C:/videos/sample_interview.mp4",
            trim_start_sec=52.0,
            trim_end_sec=70.0,
            duration_sec=18.0,
            target_start_sec=30.0,
            target_end_sec=48.0,
            label="Aspect ratio framing techniques",
            speaker="Designer",
            is_selected=True,
        )
    ]

    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
        confirmed_clips=prev_confirmed,
    )
    qtbot.addWidget(dlg)

    row1 = dlg._rows[1] # Segment 2
    assert row1.chk_include.isChecked() is True
    assert row1.spin_start.value() == 52.0
    assert row1.spin_end.value() == 70.0
    assert row1.spin_target.value() == 30.0
    assert "18.0s" in row1.lbl_dur.text()


def test_studioview_integration_and_config_serialization(qtbot, sample_segments, monkeypatch):
    """Verify StudioView handles confirmed clips and includes them in get_interview_config()."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.show()
    view.combo_format.setCurrentIndex(2)

    assert hasattr(view, "btn_review_interview_clips")
    assert view.btn_review_interview_clips.isEnabled() is False

    # Simulate scan completion with sample segments
    view._scanned_interview_segments = sample_segments
    view._interview_source_video = "C:/videos/sample_interview.mp4"
    view.btn_review_interview_clips.setEnabled(True)

    # Apply 2 confirmed clips
    confirmed = [
        TrimmedInterviewClip(
            segment_id=1,
            source_video="C:/videos/sample_interview.mp4",
            trim_start_sec=10.0,
            trim_end_sec=25.0,
            duration_sec=15.0,
            target_start_sec=0.0,
            target_end_sec=15.0,
            label="Neural pipeline breakthrough",
            speaker="Lead Architect",
        ),
        TrimmedInterviewClip(
            segment_id=2,
            source_video="C:/videos/sample_interview.mp4",
            trim_start_sec=50.0,
            trim_end_sec=70.0,
            duration_sec=20.0,
            target_start_sec=30.0,
            target_end_sec=50.0,
            label="Aspect ratio framing techniques",
            speaker="Designer",
        ),
    ]

    view._apply_confirmed_interview_clips(confirmed)

    assert "✓ 2 of 3 clips selected (35.0s total)" in view.edit_interview_clips.text()
    assert "✓ 2 clips confirmed (35.0s)" in view.lbl_interview_scan_status.text()

    # Verify get_interview_config serialization
    cfg = view.get_interview_config()
    assert "confirmed_clips" in cfg
    assert len(cfg["confirmed_clips"]) == 2
    assert cfg["confirmed_clips"][0]["segment_id"] == 1
    assert cfg["confirmed_clips"][0]["trim_start_sec"] == 10.0
    assert cfg["confirmed_clips"][0]["trim_end_sec"] == 25.0
    assert cfg["confirmed_clips"][0]["duration_sec"] == 15.0
    assert cfg["confirmed_clips"][1]["target_start_sec"] == 30.0


def test_format_seconds_to_mmss_helpers():
    """Verify bidirectional and subsecond MM:SS formatting."""
    assert format_seconds_to_mmss(0.0, show_decimals=True) == "00:00.0"
    assert format_seconds_to_mmss(7.6, show_decimals=True) == "00:07.6"
    assert format_seconds_to_mmss(19.4, show_decimals=True) == "00:19.4"
    assert format_seconds_to_mmss(11.8, show_decimals=True) == "00:11.8"
    assert format_seconds_to_mmss(70.0, show_decimals=False) == "01:10"
    assert format_seconds_to_mmss(125.6, show_decimals=True) == "02:05.6"
    assert format_seconds_to_mmss(59.96, show_decimals=True) == "01:00.0"


def test_table_row_height_and_column_widths(qtbot, sample_segments):
    """Verify table row heights (>=72px) and column widths eliminate text clipping."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()

    # Check minimum row height for breathing room
    for i in range(dlg.table.rowCount()):
        assert dlg.table.rowHeight(i) >= 72

    # Verify column widths prevent horizontal truncation
    assert dlg.table.columnWidth(0) == 65
    assert dlg.table.columnWidth(2) >= 120
    assert dlg.table.columnWidth(3) >= 120
    assert dlg.table.columnWidth(4) >= 140
    assert dlg.table.columnWidth(5) >= 165

    # Verify Clip / Topic widget has word wrapping enabled
    topic_cell = dlg.table.cellWidget(0, 1)
    assert topic_cell is not None
    labels = topic_cell.findChildren(type(dlg.lbl_summary_badge))
    for lbl in labels:
        assert lbl.wordWrap() is True


def test_live_mmss_badges_and_duration_precision(qtbot, sample_segments):
    """Verify exact duration calculation (end - start) and live MM:SS badge synchronization."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/sample_interview.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()

    row0 = dlg._rows[0]
    # Set exact values from user's screenshot: 7.6s to 19.4s
    row0.spin_start.setValue(7.6)
    row0.spin_end.setValue(19.4)
    row0.spin_target.setValue(7.6)

    # Duration: 19.4 - 7.6 = 11.8s
    assert "11.8s" in row0.lbl_dur.text()
    assert "(00:11.8)" in row0.lbl_dur.text()

    # Start & End MM:SS Badges
    assert row0.lbl_start_time.text() == "⏱ 00:07.6"
    assert row0.lbl_end_time.text() == "⏱ 00:19.4"

    # Target Placement Range Badge: 00:07.6 -> 00:19.4
    assert "00:07.6" in row0.lbl_target_time.text()
    assert "00:19.4" in row0.lbl_target_time.text()
    assert "→" in row0.lbl_target_time.text()

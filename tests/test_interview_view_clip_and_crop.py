"""
tests/test_interview_view_clip_and_crop.py
==========================================
Unit and integration tests for Part 3:
- "View Clip" interactive preview player (QMediaPlayer + QVideoWidget)
- Interactive 4-side crop margins (Top, Bottom, Left, Right)
- Visual draggable crop overlay widget & two-way synchronization
- FFmpeg even-dimension crop filter generation and compositing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtMultimedia import QMediaPlayer

from core.ffmpeg_wrapper import build_crop_filter
from services.interview_scanner import InterviewSegment
from services.interview_compiler import InterviewCompiler, InterviewCompileConfig
from ui.dialogs.interview_selector_dialog import (
    InterviewSelectorDialog,
    CropOverlayWidget,
    TrimmedInterviewClip,
    format_seconds_to_mmss,
)


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
    ]


@pytest.fixture(autouse=True)
def mock_media_player_playback():
    """Avoid passing fake null-byte mp4 files into native C++ demuxer threads."""
    with patch("PySide6.QtMultimedia.QMediaPlayer.setSource"), \
         patch("PySide6.QtMultimedia.QMediaPlayer.play"), \
         patch("PySide6.QtMultimedia.QMediaPlayer.pause"), \
         patch("PySide6.QtMultimedia.QMediaPlayer.stop"):
        yield


@pytest.fixture
def mock_video_file(tmp_path):
    f = tmp_path / "mock_interview.mp4"
    f.write_bytes(b"\x00" * 4000)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# 1. FFmpeg 4-Side Crop Filter Helper Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_build_crop_filter_zero_crop_bypassed():
    """Verify 0 crop on all 4 sides returns an empty string, cleanly bypassing filter."""
    assert build_crop_filter(1920, 1080, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0) == ""


def test_build_crop_filter_even_dimensions_and_offsets():
    """Verify standard 4-side crop produces valid crop=w:h:x:y with even dimensions."""
    # 1920 - 100 - 100 = 1720 (even)
    # 1080 - 60 - 60 = 960 (even)
    filt = build_crop_filter(1920, 1080, crop_top=60, crop_bottom=60, crop_left=100, crop_right=100)
    assert filt == "crop=1720:960:100:60"

    # Odd input dimensions are rounded down to nearest even integer for libx264
    # 1920 - 101 - 100 = 1719 -> rounded to 1718
    # 1080 - 61 - 60 = 959 -> rounded to 958
    filt_odd = build_crop_filter(1920, 1080, crop_top=61, crop_bottom=60, crop_left=101, crop_right=100)
    assert filt_odd == "crop=1718:958:101:61"
    parts = filt_odd.replace("crop=", "").split(":")
    w, h = int(parts[0]), int(parts[1])
    assert w % 2 == 0
    assert h % 2 == 0


def test_build_crop_filter_overcrop_clamping():
    """Verify over-cropping is clamped so out_w >= 64 and out_h >= 64 without negative dimensions."""
    filt = build_crop_filter(1920, 1080, crop_top=600, crop_bottom=600, crop_left=1000, crop_right=1000)
    assert filt != ""
    parts = filt.replace("crop=", "").split(":")
    w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    assert w >= 64
    assert h >= 64
    assert w % 2 == 0
    assert h % 2 == 0
    assert x >= 0
    assert y >= 0
    assert x + w <= 1920
    assert y + h <= 1080


# ─────────────────────────────────────────────────────────────────────────────
# 2. CropOverlayWidget Two-Way Synchronization & Interaction Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_crop_overlay_widget_mapping_and_signals(qtbot):
    """Verify CropOverlayWidget maps source coordinates and emits cropChanged on set_crop."""
    overlay = CropOverlayWidget()
    qtbot.addWidget(overlay)
    overlay.resize(400, 300)
    overlay.set_source_dimensions(1920, 1080)

    # Check video rect calculation
    vr = overlay.get_video_rect()
    assert vr.width() > 0
    assert vr.height() > 0

    # Set crop and check screen rect
    overlay.set_crop(top=100, bottom=100, left=200, right=200)
    assert overlay.get_crop() == (100, 100, 200, 200)

    cr = overlay.get_crop_screen_rect()
    assert cr.width() > 0
    assert cr.height() > 0
    assert cr.left() >= vr.left()
    assert cr.right() <= vr.right()

    # Reset crop
    overlay.reset_crop()
    assert overlay.get_crop() == (0, 0, 0, 0)


def test_crop_overlay_drag_interaction(qtbot):
    """Verify dragging inside CropOverlayWidget shifts or resizes crop coordinates."""
    overlay = CropOverlayWidget()
    qtbot.addWidget(overlay)
    overlay.resize(400, 300)
    overlay.set_source_dimensions(1920, 1080)
    overlay.set_crop(top=60, bottom=60, left=100, right=100)

    cr = overlay.get_crop_screen_rect()
    center_pt = cr.center()

    captured_crops = []
    overlay.cropChanged.connect(lambda t, b, l, r: captured_crops.append((t, b, l, r)))

    # Simulate mouse press inside box (MOVE mode)
    with patch.object(overlay, "_hit_test", return_value="MOVE"):
        overlay._drag_mode = "MOVE"
        overlay._drag_start_pos = center_pt
        overlay._drag_start_crops = (60, 60, 100, 100)

        # Drag 20px right, 10px down
        new_pt = QPoint(center_pt.x() + 20, center_pt.y() + 10)
        overlay._handle_drag(new_pt)

    assert len(captured_crops) >= 1
    t, b, l, r = captured_crops[-1]
    # In MOVE mode, left increases and right decreases
    assert l > 100
    assert r < 100


# ─────────────────────────────────────────────────────────────────────────────
# 3. InterviewSelectorDialog View Clip Player & Crop Controls Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_dialog_view_clip_player_ui_elements(qtbot, sample_segments, mock_video_file):
    """Verify dialog contains multimedia preview player, scrub controls, and 4-side crop spinboxes."""
    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080

    with patch("ui.dialogs.interview_selector_dialog.probe_media", return_value=mock_probe):
        dlg = InterviewSelectorDialog(
            source_video=str(mock_video_file),
            segments=sample_segments,
        )
        qtbot.addWidget(dlg)

    # Preview elements
    assert hasattr(dlg, "video_widget")
    assert hasattr(dlg, "crop_overlay")
    assert hasattr(dlg, "btn_play_pause")
    assert hasattr(dlg, "slider_seek")
    assert hasattr(dlg, "lbl_player_time")
    assert hasattr(dlg, "chk_loop")
    assert dlg.chk_loop.isChecked() is True

    # 4-Side Crop elements
    assert hasattr(dlg, "spin_crop_top")
    assert hasattr(dlg, "spin_crop_bottom")
    assert hasattr(dlg, "spin_crop_left")
    assert hasattr(dlg, "spin_crop_right")
    assert hasattr(dlg, "btn_reset_crop")
    assert hasattr(dlg, "lbl_crop_summary")

    # Row 'View Clip' button
    assert len(dlg._rows) == 2
    for r in dlg._rows:
        assert hasattr(r, "btn_view_clip")
        assert r.btn_view_clip is not None
        assert "View Clip" in r.btn_view_clip.text()

    # Initial full frame readout
    assert "Full Frame" in dlg.lbl_crop_summary.text()


def test_dialog_view_clip_switch_and_crop_independence(qtbot, sample_segments, mock_video_file):
    """Verify clicking View Clip loads clip, updates ranges, and preserves independent crops per clip."""
    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080

    with patch("ui.dialogs.interview_selector_dialog.probe_media", return_value=mock_probe):
        dlg = InterviewSelectorDialog(
            source_video=str(mock_video_file),
            segments=sample_segments,
        )
        qtbot.addWidget(dlg)

    # 1. Clip 0 is initially active
    assert dlg._active_preview_row_idx == 0
    assert "Clip #1" in dlg.lbl_preview_clip_title.text()

    # Apply crop to Clip 0 (Left: 120, Right: 80, Top: 40, Bottom: 50)
    dlg.spin_crop_left.setValue(120)
    dlg.spin_crop_right.setValue(80)
    dlg.spin_crop_top.setValue(40)
    dlg.spin_crop_bottom.setValue(50)

    assert "Cropped" in dlg.lbl_crop_summary.text()
    assert dlg._rows[0].crop_left == 120

    # 2. Switch to Clip 1 via View Clip button
    dlg._rows[1].btn_view_clip.click()
    assert dlg._active_preview_row_idx == 1
    assert "Clip #2" in dlg.lbl_preview_clip_title.text()

    # Clip 1 should start at 0 crop (independent of Clip 0)
    assert dlg.spin_crop_left.value() == 0
    assert dlg.spin_crop_top.value() == 0
    assert "Full Frame" in dlg.lbl_crop_summary.text()

    # Apply different crop to Clip 1 (Top: 70, Bottom: 90)
    dlg.spin_crop_top.setValue(70)
    dlg.spin_crop_bottom.setValue(90)
    assert dlg._rows[1].crop_top == 70

    # 3. Switch back to Clip 0
    dlg._rows[0].btn_view_clip.click()
    assert dlg._active_preview_row_idx == 0
    # Restores Clip 0's independent crop values
    assert dlg.spin_crop_left.value() == 120
    assert dlg.spin_crop_right.value() == 80
    assert dlg.spin_crop_top.value() == 40
    assert dlg.spin_crop_bottom.value() == 50


def test_dialog_reset_crop_button(qtbot, sample_segments, mock_video_file):
    """Verify Reset Full Frame sets all 4 spinboxes to 0 and updates readout."""
    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080

    with patch("ui.dialogs.interview_selector_dialog.probe_media", return_value=mock_probe):
        dlg = InterviewSelectorDialog(
            source_video=str(mock_video_file),
            segments=sample_segments,
        )
        qtbot.addWidget(dlg)

    dlg.spin_crop_top.setValue(100)
    dlg.spin_crop_left.setValue(150)
    assert "Cropped" in dlg.lbl_crop_summary.text()

    dlg.btn_reset_crop.click()
    assert dlg.spin_crop_top.value() == 0
    assert dlg.spin_crop_bottom.value() == 0
    assert dlg.spin_crop_left.value() == 0
    assert dlg.spin_crop_right.value() == 0
    assert "Full Frame" in dlg.lbl_crop_summary.text()


def test_dialog_media_player_transport_and_release(qtbot, sample_segments, mock_video_file):
    """Verify play/pause toggles player and dialog release closes player cleanly."""
    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080

    with patch("ui.dialogs.interview_selector_dialog.probe_media", return_value=mock_probe):
        dlg = InterviewSelectorDialog(
            source_video=str(mock_video_file),
            segments=sample_segments,
        )
        qtbot.addWidget(dlg)

    # Play toggle
    dlg.btn_play_pause.click()
    # Seek simulation
    dlg._on_seek_moved(500)
    # Position changed callback inside range
    dlg._on_player_position_changed(20000)
    assert dlg.slider_seek.value() > 0

    # Clean release check
    dlg._release_player()
    assert dlg._player.playbackState() == QMediaPlayer.PlaybackState.StoppedState


def test_dialog_confirmed_clips_crop_payload(qtbot, sample_segments, mock_video_file):
    """Verify get_confirmed_clips and get_confirmed_clips_data return crop metadata."""
    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080

    with patch("ui.dialogs.interview_selector_dialog.probe_media", return_value=mock_probe):
        dlg = InterviewSelectorDialog(
            source_video=str(mock_video_file),
            segments=sample_segments,
        )
        qtbot.addWidget(dlg)

    # Set crop on Clip 0
    dlg.spin_crop_top.setValue(30)
    dlg.spin_crop_bottom.setValue(40)
    dlg.spin_crop_left.setValue(50)
    dlg.spin_crop_right.setValue(60)

    with patch("PySide6.QtWidgets.QDialog.accept"):
        dlg._on_confirm_clicked()

    confirmed = dlg.get_confirmed_clips()
    assert len(confirmed) == 2
    assert confirmed[0].crop_top == 30
    assert confirmed[0].crop_bottom == 40
    assert confirmed[0].crop_left == 50
    assert confirmed[0].crop_right == 60

    data = dlg.get_confirmed_clips_data()
    assert data[0]["crop_top"] == 30
    assert data[0]["crop_bottom"] == 40
    assert data[0]["crop_left"] == 50
    assert data[0]["crop_right"] == 60


# ─────────────────────────────────────────────────────────────────────────────
# 4. InterviewCompiler 4-Side Crop Filter Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_render_styled_clip_with_crop_filter(mock_video_file, tmp_path):
    """Verify _render_single_styled_interview_clip applies FFmpeg crop filter before scale."""
    cfg = InterviewCompileConfig(
        audio_path=tmp_path / "vo.mp3",
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.70,
            "border_enabled": False,
            "corner_shape": "sharp",
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_video_file),
        "trim_start_sec": 0.0,
        "trim_end_sec": 5.0,
        "duration_sec": 5.0,
        "target_start_sec": 0.0,
        "target_end_sec": 5.0,
        "label": "Cropped Clip",
        "segment_id": 1,
        "crop_top": 40,
        "crop_bottom": 40,
        "crop_left": 100,
        "crop_right": 100,
    }

    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080
    mock_probe.has_audio = True

    ffmpeg_cmds = []

    def mock_run(cmd, duration, label=None):
        ffmpeg_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x00" * 100)

    with patch("services.interview_compiler._run_with_progress", side_effect=mock_run), \
         patch("services.interview_compiler.probe_media", return_value=mock_probe):
        out_styled = tmp_path / "styled_cropped.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    assert len(ffmpeg_cmds) == 1
    cmd_str = " ".join(ffmpeg_cmds[0])

    # 1920 - 100 - 100 = 1720, 1080 - 40 - 40 = 1000
    assert "crop=1720:1000:100:40" in cmd_str
    assert "scale=" in cmd_str
    assert "overlay=" in cmd_str


def test_crop_overlay_rubberband_screen_area_selection(qtbot):
    """Verify clicking and dragging on the preview area creates an exact area-selection box."""
    overlay = CropOverlayWidget()
    qtbot.addWidget(overlay)
    overlay.resize(400, 300)
    overlay.set_source_dimensions(1920, 1080)
    overlay.reset_crop()

    vr = overlay.get_video_rect()
    assert vr.width() > 0
    assert vr.height() > 0

    # Start selection at (20% x, 20% y) inside video rect, drag to (80% x, 80% y)
    start_pt = QPoint(vr.left() + int(vr.width() * 0.20), vr.top() + int(vr.height() * 0.20))
    end_pt = QPoint(vr.left() + int(vr.width() * 0.80), vr.top() + int(vr.height() * 0.80))

    overlay._drag_mode = "SELECT"
    overlay._drag_start_pos = start_pt
    overlay._drag_start_crops = (0, 0, 0, 0)

    captured_crops = []
    overlay.cropChanged.connect(lambda t, b, l, r: captured_crops.append((t, b, l, r)))

    overlay._handle_drag(end_pt)

    assert len(captured_crops) >= 1
    top, bottom, left, right = captured_crops[-1]

    # Verify source video pixels match 20% and 80% boundaries (1920 * 0.2 = 384, 1080 * 0.2 = 216)
    assert abs(left - 384) <= 4
    assert abs(right - 384) <= 4
    assert abs(top - 216) <= 4
    assert abs(bottom - 216) <= 4

    # Verify normalized crop ratio matches [0.2, 0.2, 0.6, 0.6]
    nx, ny, nw, nh = overlay.get_normalized_crop()
    assert abs(nx - 0.2) <= 0.02
    assert abs(ny - 0.2) <= 0.02
    assert abs(nw - 0.6) <= 0.02
    assert abs(nh - 0.6) <= 0.02


def test_crop_overlay_normalized_ratio_roundtrip(qtbot):
    """Verify setting crop from normalized ratios roundtrips accurately."""
    overlay = CropOverlayWidget()
    qtbot.addWidget(overlay)
    overlay.set_source_dimensions(1920, 1080)

    # Set normalized crop (0.10, 0.15, 0.70, 0.60)
    overlay.set_crop_from_normalized(0.10, 0.15, 0.70, 0.60)
    t, b, l, r = overlay.get_crop()

    # Left = 192px, Top = 162px, Right = 1920 - 192 - 1344 = 384px, Bottom = 1080 - 162 - 648 = 270px
    assert abs(l - 192) <= 2
    assert abs(t - 162) <= 2

    nx, ny, nw, nh = overlay.get_normalized_crop()
    assert abs(nx - 0.10) <= 0.01
    assert abs(ny - 0.15) <= 0.01
    assert abs(nw - 0.70) <= 0.01
    assert abs(nh - 0.60) <= 0.01


def test_interview_compiler_person_segmentation_compositing(mock_video_file, tmp_path):
    """Verify background-removed interview foreground is composited over blurred user background."""
    bg_img = tmp_path / "custom_bg.png"
    bg_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    cfg = InterviewCompileConfig(
        audio_path=tmp_path / "vo.mp3",
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        bg_image_path=str(bg_img),
        bg_blur_amount=20.0,
        interview_config={
            "clip_scale": 0.75,
            "border_enabled": False,
            "corner_shape": "sharp",
            "bg_removal": True,
            "bg_image_path": str(bg_img),
            "bg_blur_amount": 20.0,
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_video_file),
        "trim_start_sec": 0.0,
        "trim_end_sec": 4.0,
        "duration_sec": 4.0,
        "target_start_sec": 0.0,
        "target_end_sec": 4.0,
        "label": "Interview Subject",
        "segment_id": 1,
        "crop_top": 50,
        "crop_bottom": 50,
        "crop_left": 80,
        "crop_right": 80,
    }

    mock_probe = MagicMock()
    mock_probe.width = 1920
    mock_probe.height = 1080
    mock_probe.has_audio = True

    ffmpeg_cmds = []

    def mock_run(cmd, duration, label=None):
        ffmpeg_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x00" * 100)

    # Mock person mask generator to simulate fast alpha mask output
    fake_mask_path = tmp_path / "clip_1_person_mask.mp4"
    fake_mask_path.write_bytes(b"\x00" * 500)

    with patch("services.interview_compiler._run_with_progress", side_effect=mock_run), \
         patch("services.interview_compiler.probe_media", return_value=mock_probe), \
         patch("services.interview_segmenter.generate_person_mask_video", return_value=fake_mask_path):
        out_styled = tmp_path / "styled_person.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    assert len(ffmpeg_cmds) == 1
    cmd_str = " ".join(ffmpeg_cmds[0])

    # Crop filter is applied
    assert "crop=1760:980:80:50" in cmd_str

    # Person mask input is passed to FFmpeg
    assert str(fake_mask_path) in cmd_str

    # Alpha-merging person mask onto cropped foreground
    assert "alphamerge" in cmd_str

    # Background image with boxblur
    assert "boxblur=20" in cmd_str
    assert "overlay=" in cmd_str


def test_dialog_framing_presets_and_unclipped_layout(qtbot, sample_segments):
    """Verify quick framing preset buttons and unclipped dimensions in crop studio dialog."""
    dlg = InterviewSelectorDialog(
        source_video="C:/videos/test.mp4",
        segments=sample_segments,
    )
    qtbot.addWidget(dlg)
    dlg.show()

    # Verify preset buttons exist
    assert hasattr(dlg, "btn_preset_full")
    assert hasattr(dlg, "btn_preset_left")
    assert hasattr(dlg, "btn_preset_center")
    assert hasattr(dlg, "btn_preset_right")

    # Verify generous spinbox dimensions to avoid clipping
    assert dlg.spin_crop_top.width() >= 96
    assert dlg.spin_crop_bottom.width() >= 96
    assert dlg.spin_crop_left.width() >= 96
    assert dlg.spin_crop_right.width() >= 96
    assert dlg.btn_reset_crop.minimumWidth() >= 130

    # 1. Click Left Speaker preset
    dlg.btn_preset_left.click()
    t, b, l, r = dlg.crop_overlay.get_crop()
    # Left speaker preset should keep left side and crop right side
    assert l == 0
    assert r > 0
    assert dlg.spin_crop_left.value() == 0
    assert dlg.spin_crop_right.value() > 0

    # 2. Click Right Speaker preset
    dlg.btn_preset_right.click()
    t, b, l, r = dlg.crop_overlay.get_crop()
    assert l > 0
    assert r == 0
    assert dlg.spin_crop_left.value() > 0
    assert dlg.spin_crop_right.value() == 0

    # 3. Click Center Focus preset
    dlg.btn_preset_center.click()
    t, b, l, r = dlg.crop_overlay.get_crop()
    assert l > 0 and r > 0
    assert dlg.spin_crop_left.value() > 0
    assert dlg.spin_crop_right.value() > 0

    # 4. Click Full Frame preset
    dlg.btn_preset_full.click()
    t, b, l, r = dlg.crop_overlay.get_crop()
    assert (t, b, l, r) == (0, 0, 0, 0)
    assert dlg.spin_crop_top.value() == 0
    assert dlg.spin_crop_bottom.value() == 0
    assert dlg.spin_crop_left.value() == 0
    assert dlg.spin_crop_right.value() == 0



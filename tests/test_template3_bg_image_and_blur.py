"""
tests/test_template3_bg_image_and_blur.py
==========================================
Unit and integration tests for Part 2:
Template 3 UI Background Image & Blur controls and Interview Segment
background replacement compositing in InterviewCompiler.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from core.ffmpeg_wrapper import build_background_filter
from services.interview_compiler import (
    InterviewCompiler,
    InterviewCompileConfig,
    InterviewCompileResult,
)
from ui.views.studio_view import StudioView
from ui.workers.video_worker import VideoRenderWorker


@pytest.fixture
def mock_audio(tmp_path):
    f = tmp_path / "voiceover.mp3"
    f.write_bytes(b"\x00" * 2000)
    return f


@pytest.fixture
def mock_source_video(tmp_path):
    f = tmp_path / "interview_source.mp4"
    f.write_bytes(b"\x00" * 4000)
    return f


@pytest.fixture
def sample_bg_image(tmp_path):
    """Creates a valid sample background image for testing."""
    img_path = tmp_path / "custom_bg.png"
    img = Image.new("RGB", (1920, 1080), color=(30, 60, 90))
    img.save(img_path)
    return img_path


# ─────────────────────────────────────────────────────────────────────────────
# 1. UI Controls & Serialization Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_template3_bg_image_and_blur_ui_defaults(qtbot, monkeypatch):
    """Verify default UI states for background image and blur controls."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    assert hasattr(view, "edit_interview_bg_image")
    assert hasattr(view, "btn_browse_interview_bg")
    assert hasattr(view, "btn_clear_interview_bg")
    assert hasattr(view, "lbl_interview_bg_preview")
    assert hasattr(view, "slider_interview_blur")
    assert hasattr(view, "lbl_interview_blur")

    assert view.edit_interview_bg_image.text() == ""
    assert view.slider_interview_blur.value() == 0
    assert view.lbl_interview_blur.text() == "0%"
    assert view.lbl_interview_bg_preview.text() == "No Image"


def test_template3_bg_image_selection_and_thumbnail(qtbot, sample_bg_image, monkeypatch):
    """Verify selecting a background image updates the path, thumbnail preview, and config."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    # Set background image path using helper
    view.set_interview_bg_image(str(sample_bg_image))
    assert view.edit_interview_bg_image.text() == str(sample_bg_image)
    # Thumbnail should now display a pixmap, clearing the text
    assert view.lbl_interview_bg_preview.text() == ""
    assert not view.lbl_interview_bg_preview.pixmap().isNull()

    # Change blur slider
    view.set_interview_blur(35)
    assert view.slider_interview_blur.value() == 35
    assert view.lbl_interview_blur.text() == "35%"

    # Check serialization
    cfg = view.get_interview_config()
    assert cfg["bg_image_path"] == str(sample_bg_image)
    assert cfg["bg_blur_amount"] == 35
    assert cfg["bg_blur"] == 0.35

    render_cfg = view.get_render_config()
    assert render_cfg["template"] == "template_3"
    assert render_cfg["interview"]["bg_image_path"] == str(sample_bg_image)
    assert render_cfg["interview"]["bg_blur_amount"] == 35


def test_template3_bg_image_clear_action(qtbot, sample_bg_image, monkeypatch):
    """Verify Clear button resets background image path and resets thumbnail to 'No Image'."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    # First set image
    view.set_interview_bg_image(str(sample_bg_image))
    assert view.edit_interview_bg_image.text() == str(sample_bg_image)

    # Click Clear
    view.btn_clear_interview_bg.click()
    assert view.edit_interview_bg_image.text() == ""
    assert view.lbl_interview_bg_preview.text() == "No Image"

    cfg = view.get_interview_config()
    assert cfg["bg_image_path"] == ""


def test_template3_bg_image_browse_dialog(qtbot, sample_bg_image, monkeypatch):
    """Verify browse dialog sets selected background image."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDummyKeyForTestingPurposes12345")
    view = StudioView()
    qtbot.addWidget(view)
    view.combo_format.setCurrentIndex(2)

    with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(sample_bg_image), "Images")):
        view.btn_browse_interview_bg.click()

    assert view.edit_interview_bg_image.text() == str(sample_bg_image)
    assert not view.lbl_interview_bg_preview.pixmap().isNull()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FFmpeg Filtergraph & Compositing Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_build_background_filter_helper():
    """Verify core background filter helper for sharp and blurred modes."""
    # Sharp background (blur == 0)
    filt_sharp = build_background_filter("1:v", 1920, 1080, blur_amount=0, output_label="bg")
    assert filt_sharp == "[1:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080[bg]"
    assert "boxblur" not in filt_sharp

    # Blurred background (blur == 25)
    filt_blur = build_background_filter("1:v", 1080, 1920, blur_amount=25, output_label="bg")
    assert "[1:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25[bg]" == filt_blur

    # Even dimension protection
    filt_odd = build_background_filter("0:v", 1081, 1919, blur_amount=10)
    assert "scale=1080:1918" in filt_odd


def test_render_styled_interview_clip_with_custom_blurred_bg(
    mock_audio, mock_source_video, sample_bg_image, tmp_path
):
    """
    Verify interview clip replaces original background with user's custom background image
    and applies configured blur filter.
    """
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.70,
            "bg_image_path": str(sample_bg_image),
            "bg_blur_amount": 28,
            "border_enabled": True,
            "border_color": "#EF4444",
            "border_thickness": 4,
            "corner_shape": "rounded",
            "corner_curve": 24,
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_source_video),
        "trim_start_sec": 5.0,
        "trim_end_sec": 12.0,
        "duration_sec": 7.0,
        "target_start_sec": 10.0,
        "target_end_sec": 17.0,
        "label": "Test Clip",
        "segment_id": 1,
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
        out_styled = tmp_path / "styled_bg_blur.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    assert len(ffmpeg_cmds) == 1
    cmd = ffmpeg_cmds[0]
    cmd_str = " ".join(cmd)

    # 1. Background image input should be passed via -loop 1 -framerate 30 -i <bg_image>
    assert str(sample_bg_image) in cmd
    assert "-loop" in cmd
    bg_img_idx = cmd.index(str(sample_bg_image))
    assert cmd[bg_img_idx - 1] == "-i"
    assert cmd[bg_img_idx - 5] == "-loop"
    assert cmd[bg_img_idx - 4] == "1"
    assert cmd[bg_img_idx - 3] == "-framerate"

    # 2. Filtergraph should scale, crop, and apply boxblur to custom bg input [1:v]
    assert "[1:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=28[bg]" in cmd_str

    # 3. Alpha mask and border should composite foreground [fg] over [bg]
    assert "alphamerge[fg_masked]" in cmd_str
    assert "[bg][fg_masked]overlay=(W-w)/2:(H-h)/2" in cmd_str
    assert "-map [vout]" in cmd_str


def test_render_styled_interview_clip_with_custom_sharp_bg(
    mock_audio, mock_source_video, sample_bg_image, tmp_path
):
    """
    Verify interview clip with custom background image and blur == 0 renders sharp
    without boxblur in the filtergraph.
    """
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.75,
            "bg_image_path": str(sample_bg_image),
            "bg_blur_amount": 0,
            "border_enabled": False,
            "corner_shape": "sharp",
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_source_video),
        "trim_start_sec": 0.0,
        "trim_end_sec": 5.0,
        "duration_sec": 5.0,
        "target_start_sec": 0.0,
        "target_end_sec": 5.0,
        "label": "Sharp BG Clip",
        "segment_id": 2,
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
        out_styled = tmp_path / "styled_sharp_bg.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    cmd_str = " ".join(ffmpeg_cmds[0])

    # Custom background image present
    assert str(sample_bg_image) in cmd_str
    # Filter scales and crops without boxblur
    assert "[1:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080[bg]" in cmd_str
    assert "boxblur" not in cmd_str


def test_render_styled_interview_clip_missing_bg_fallback(
    mock_audio, mock_source_video, tmp_path
):
    """
    Verify that if bg_image_path is missing or does not exist, it falls back gracefully
    to blurred source video or default background without crashing.
    """
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.70,
            "bg_image_path": str(tmp_path / "non_existent_bg.jpg"),
            "bg_blur_amount": 20,
            "border_enabled": False,
            "corner_shape": "sharp",
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_source_video),
        "trim_start_sec": 0.0,
        "trim_end_sec": 4.0,
        "duration_sec": 4.0,
        "target_start_sec": 0.0,
        "target_end_sec": 4.0,
        "label": "Fallback Clip",
        "segment_id": 3,
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
        out_styled = tmp_path / "styled_fallback.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    cmd_str = " ".join(ffmpeg_cmds[0])

    # Non-existent image was NOT passed as input
    assert "non_existent_bg.jpg" not in cmd_str
    # Falls back to source video [0:v] as blurred background
    assert "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20[bg]" in cmd_str


def test_video_render_worker_payload_passing(mock_audio, sample_bg_image, tmp_path):
    """Verify VideoRenderWorker propagates bg_image_path and bg_blur_amount to InterviewCompiler."""
    out_video = tmp_path / "out_worker_test.mp4"
    worker = VideoRenderWorker(
        audio_path=mock_audio,
        images_dir=tmp_path,
        videos_path=tmp_path,
        output_path=out_video,
        template_type="template_3",
        bg_image_path=str(sample_bg_image),
        bg_blur_amount=30,
        interview_config={
            "confirmed_clips": [],
            "music_volume": 0.20,
        },
    )

    assert worker.bg_image_path == str(sample_bg_image)
    assert worker.bg_blur_amount == 30

    mock_res = InterviewCompileResult(
        output_path=out_video,
        duration_sec=10.0,
        confirmed_clips_count=0,
        elapsed_seconds=1.0,
    )

    captured_config = []

    def fake_init(self, config, **kwargs):
        captured_config.append(config)

    with patch.object(InterviewCompiler, "__init__", fake_init), \
         patch.object(InterviewCompiler, "compile", return_value=mock_res), \
         patch.object(worker, "_emit_success"):
        worker.run()

    assert len(captured_config) == 1
    cfg: InterviewCompileConfig = captured_config[0]
    assert cfg.bg_image_path == str(sample_bg_image)
    assert cfg.bg_blur_amount == 30
    assert cfg.interview_config["bg_image_path"] == str(sample_bg_image)
    assert cfg.interview_config["bg_blur_amount"] == 30

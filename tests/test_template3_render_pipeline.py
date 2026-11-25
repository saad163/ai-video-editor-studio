"""
tests/test_template3_render_pipeline.py
=======================================
Unit and integration tests for Part 4:
Final Render Pipeline for Template 3 (+ Interview).
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from services.interview_compiler import (
    InterviewCompiler,
    InterviewCompileConfig,
    InterviewCompileResult,
)
from services.transcriber import TranscriptionResult, TranscriptionSegment
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
def mock_bgm(tmp_path):
    f = tmp_path / "interview_bgm.mp3"
    f.write_bytes(b"\x00" * 2000)
    return f


def test_clean_confirmed_clips_sorting_and_clamping(mock_audio, mock_source_video, tmp_path):
    """Verify raw confirmed clips are validated, clamped, and sorted chronologically."""
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        interview_config={
            "confirmed_clips": [
                {
                    "source_video": str(mock_source_video),
                    "trim_start_sec": 50.0,
                    "trim_end_sec": 70.0,
                    "target_start_sec": 40.0,
                    "label": "Clip 2",
                },
                {
                    "source_video": str(mock_source_video),
                    "trim_start_sec": 10.0,
                    "trim_end_sec": 25.0,
                    "target_start_sec": 5.0,
                    "label": "Clip 1",
                },
                {
                    "source_video": "non_existent_video.mp4",
                    "trim_start_sec": 0.0,
                    "trim_end_sec": 10.0,
                },
            ]
        },
    )
    compiler = InterviewCompiler(config=cfg)
    cleaned = compiler._clean_confirmed_clips(cfg.interview_config["confirmed_clips"], total_duration=100.0)

    # Missing video skipped
    assert len(cleaned) == 2
    # Sorted by target_start_sec
    assert cleaned[0]["label"] == "Clip 1"
    assert cleaned[0]["target_start_sec"] == 5.0
    assert cleaned[0]["duration_sec"] == 15.0
    assert cleaned[0]["target_end_sec"] == 20.0

    assert cleaned[1]["label"] == "Clip 2"
    assert cleaned[1]["target_start_sec"] == 40.0
    assert cleaned[1]["duration_sec"] == 20.0
    assert cleaned[1]["target_end_sec"] == 60.0


def test_filter_subtitles_for_interviews(mock_audio, tmp_path):
    """Verify voiceover subtitles overlapping active interview windows are suppressed."""
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
    )
    compiler = InterviewCompiler(config=cfg)

    transcription = TranscriptionResult(
        audio_path=mock_audio,
        audio_hash="dummy_hash",
        language="en",
        full_text="Segment 1. Segment 2. Segment 3.",
        segments=[
            TranscriptionSegment(start=0.0, end=8.0, text="Intro segment."),
            TranscriptionSegment(start=12.0, end=25.0, text="Overlapped with interview."),
            TranscriptionSegment(start=32.0, end=45.0, text="Outro segment."),
        ],
        model_used="tiny",
        elapsed_seconds=1.0,
    )

    confirmed_clips = [
        {"target_start_sec": 10.0, "target_end_sec": 30.0}
    ]

    filtered = compiler._filter_subtitles_for_interviews(transcription, confirmed_clips)
    assert len(filtered.segments) == 2
    assert filtered.segments[0].text == "Intro segment."
    assert filtered.segments[1].text == "Outro segment."


def test_master_audio_assembly_with_ducking_and_bgm(mock_audio, mock_source_video, mock_bgm, tmp_path):
    """Verify master audio partitions intervals into voiceover gaps and interview audio + BGM."""
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        interview_config={
            "music_path": str(mock_bgm),
            "music_volume": 0.15,
        },
    )
    compiler = InterviewCompiler(config=cfg)

    confirmed_clips = [
        {
            "source_video": str(mock_source_video),
            "trim_start_sec": 10.0,
            "trim_end_sec": 25.0,
            "duration_sec": 15.0,
            "target_start_sec": 10.0,
            "target_end_sec": 25.0,
            "label": "Interview Clip 1",
        }
    ]

    ffmpeg_calls = []

    def mock_run(cmd, duration, label=None, progress_cb=None):
        ffmpeg_calls.append((cmd, label))
        # Create destination file so exists check succeeds
        out_f = Path(cmd[-1])
        out_f.write_bytes(b"\x00" * 100)

    mock_probe = MagicMock()
    mock_probe.has_audio = True

    with patch("services.interview_compiler._run_with_progress", side_effect=mock_run), \
         patch("services.interview_compiler.probe_media", return_value=mock_probe):
        out_audio = tmp_path / "master_audio.wav"
        compiler._assemble_master_audio(
            confirmed_clips=confirmed_clips,
            total_duration=50.0,
            output_audio_path=out_audio,
            temp_dir=tmp_path,
        )

    # 1. Voiceover gap 0.0s -> 10.0s
    # 2. Interview audio 10.0s -> 25.0s with BGM mix (amix)
    # 3. Voiceover tail 25.0s -> 50.0s
    # 4. Merge concat
    assert len(ffmpeg_calls) >= 4

    labels = [l for _, l in ffmpeg_calls if l]
    assert any("Extract VO gap" in l for l in labels)
    assert any("Mix interview audio" in l for l in labels)
    assert any("Extract VO tail" in l for l in labels)
    assert any("Merge master audio" in l for l in labels)

    # Check that interview audio mixed BGM
    int_cmd = [cmd for cmd, l in ffmpeg_calls if l and "Mix interview audio" in l][0]
    int_cmd_str = " ".join(int_cmd)
    assert "amix=inputs=2" in int_cmd_str
    assert "volume=0.150" in int_cmd_str


def test_styled_interview_clip_framing_generation(mock_audio, mock_source_video, tmp_path):
    """Verify scale, background blur, rounded corner mask, and border overlay generation."""
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.70,
            "bg_blur": 0.30,
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
        "trim_end_sec": 20.0,
        "duration_sec": 15.0,
        "target_start_sec": 10.0,
        "target_end_sec": 25.0,
        "label": "Test Clip",
        "segment_id": 1,
    }

    mock_probe = MagicMock()
    mock_probe.width = 1080
    mock_probe.height = 1920
    mock_probe.has_audio = True

    ffmpeg_cmds = []

    def mock_run(cmd, duration, label=None):
        ffmpeg_cmds.append(cmd)
        Path(cmd[-1]).write_bytes(b"\x00" * 100)

    with patch("services.interview_compiler._run_with_progress", side_effect=mock_run), \
         patch("services.interview_compiler.probe_media", return_value=mock_probe):
        out_styled = tmp_path / "styled.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    # Verify Pillow mask and border files were generated
    mask_file = tmp_path / "clip_1_mask.png"
    border_file = tmp_path / "clip_1_border.png"
    assert mask_file.exists()
    assert border_file.exists()

    # Verify filter complex contains scale, boxblur, alphamerge, and overlay
    cmd_str = " ".join(ffmpeg_cmds[0])
    assert "boxblur=" in cmd_str
    assert "alphamerge" in cmd_str
    assert "overlay=" in cmd_str


def test_styled_interview_clip_sharp_corners_no_mask(mock_audio, mock_source_video, tmp_path):
    """Verify sharp corner setting generates sharp border without alphamerge mask."""
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=tmp_path / "out.mp4",
        resolution="1080p",
        interview_config={
            "clip_scale": 0.80,
            "bg_blur": 0.0,
            "border_enabled": True,
            "border_color": "#3B82F6",
            "border_thickness": 6,
            "corner_shape": "sharp",
            "corner_curve": 0,
        },
    )
    compiler = InterviewCompiler(config=cfg)

    clip = {
        "source_video": str(mock_source_video),
        "trim_start_sec": 0.0,
        "trim_end_sec": 10.0,
        "duration_sec": 10.0,
        "target_start_sec": 0.0,
        "target_end_sec": 10.0,
        "label": "Sharp Clip",
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
        out_styled = tmp_path / "sharp_styled.mp4"
        compiler._render_single_styled_interview_clip(clip, out_styled, tmp_path)

    # Mask should not be created for sharp corners
    mask_file = tmp_path / "clip_2_mask.png"
    assert not mask_file.exists()

    # Border file exists
    border_file = tmp_path / "clip_2_border.png"
    assert border_file.exists()


def test_zero_confirmed_clips_graceful_render(mock_audio, tmp_path):
    """Verify zero confirmed clips runs base timeline without errors."""
    out_video = tmp_path / "master.mp4"
    cfg = InterviewCompileConfig(
        audio_path=mock_audio,
        footage_path=tmp_path,
        output_path=out_video,
        interview_config={"confirmed_clips": []},
    )
    compiler = InterviewCompiler(config=cfg)

    mock_audio_probe = MagicMock()
    mock_audio_probe.duration = 15.0

    mock_transcription = TranscriptionResult(
        audio_path=mock_audio,
        audio_hash="h1",
        language="en",
        full_text="Base audio.",
        segments=[TranscriptionSegment(start=0.0, end=15.0, text="Base audio.")],
        model_used="tiny",
        elapsed_seconds=1.0,
    )

    with patch("services.interview_compiler.probe_media", return_value=mock_audio_probe), \
         patch("services.transcriber.TranscriptionService.transcribe", return_value=mock_transcription), \
         patch.object(compiler, "_render_base_timeline", return_value=[tmp_path / "dummy_base.mp4"]), \
         patch.object(compiler, "_assemble_master_audio", return_value=tmp_path / "master_audio.wav"), \
         patch("services.interview_compiler.concatenate_clips") as mock_concat:
        res = compiler.compile()

    assert res is not None
    assert res.confirmed_clips_count == 0
    assert mock_concat.called


def test_video_render_worker_template3_routing(mock_audio, tmp_path):
    """Verify VideoRenderWorker routes template_type='template_3' to InterviewCompiler."""
    out_video = tmp_path / "out_t3.mp4"
    worker = VideoRenderWorker(
        audio_path=mock_audio,
        images_dir=tmp_path,
        videos_path=tmp_path,
        output_path=out_video,
        template_type="template_3",
        interview_config={
            "confirmed_clips": [],
            "music_volume": 0.20,
        },
    )

    mock_res = InterviewCompileResult(
        output_path=out_video,
        duration_sec=25.0,
        confirmed_clips_count=1,
        elapsed_seconds=5.0,
    )

    with patch("services.interview_compiler.InterviewCompiler.compile", return_value=mock_res) as mock_compile, \
         patch.object(worker, "_emit_success") as mock_success:
        worker.run()

    assert mock_compile.called
    assert mock_success.called

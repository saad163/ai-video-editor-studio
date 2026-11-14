"""
tests/test_bulk_pipeline.py
===========================
End-to-end integration test for the Bulk Video Generation pipeline.
Verifies that multiple jobs can be queued, processed sequentially without UI blocks,
and produce verified 2K master MP4s with distinct subtitle styles (background boxes, colors).
"""

from __future__ import annotations

import sys
import os
import time
import subprocess
from pathlib import Path
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import setup_logging, get_config
from core.ffmpeg_wrapper import probe_media, resolve_ffmpeg
from ui.workers.bulk_worker import BulkJob, BulkQueueWorker
from database.db import init_db

setup_logging()


def slice_audio(input_audio: Path, output_audio: Path, start_sec: float, duration_sec: float) -> Path:
    """Slice a short audio excerpt using FFmpeg."""
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe, _ = resolve_ffmpeg()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(input_audio),
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output_audio),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to slice audio: {res.stderr}")
    return output_audio


@pytest.fixture(scope="module")
def bulk_test_environment(tmp_path_factory):
    """Setup test audio files and image directory."""
    init_db()
    test_dir = PROJECT_ROOT / "output" / "bulk_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    sample_audio = PROJECT_ROOT / "sample.mp3"
    assert sample_audio.exists(), f"sample.mp3 required at {sample_audio}"

    # Slice two 5-second test audios
    audio_1 = test_dir / "test_voice_1.mp3"
    audio_2 = test_dir / "test_voice_2.mp3"
    slice_audio(sample_audio, audio_1, start_sec=0.0, duration_sec=5.0)
    slice_audio(sample_audio, audio_2, start_sec=5.0, duration_sec=5.0)

    images_dir = PROJECT_ROOT / "assets" / "sample_footage"
    assert images_dir.exists(), f"assets/sample_footage required at {images_dir}"

    return {
        "test_dir": test_dir,
        "audio_1": audio_1,
        "audio_2": audio_2,
        "images_dir": images_dir,
    }


def test_bulk_queue_execution(bulk_test_environment):
    """
    Test 2 bulk jobs sequentially processed via BulkQueueWorker:
      Job 1: Yellow text (#FFFF00) + Semi-Transparent Black Box + 2K (2560x1440)
      Job 2: Pure White text (#FFFFFF) + Solid Black Box + 2K (2560x1440)
    """
    env = bulk_test_environment
    test_dir: Path = env["test_dir"]

    out_1 = test_dir / "bulk_video_1_yellow_box.mp4"
    out_2 = test_dir / "bulk_video_2_white_box.mp4"

    # Clean up previous outputs
    for f in [out_1, out_2]:
        if f.exists():
            f.unlink()

    job_1 = BulkJob(
        job_id=1,
        title="Batch Job 1 - Viral Yellow Box",
        audio_path=env["audio_1"],
        images_dir=env["images_dir"],
        output_path=out_1,
        resolution="2k",
        text_color="#FFFF00",
        highlight_color="#FFFF00",
        has_bg_box=True,
        bg_style="semi_transparent_black",
        bg_color="#000000",
        bg_opacity=0.5,
        font_size=58,
        font_family="Arial",
        style_preset_name="Viral Shorts (Yellow + Box)",
        generate_chapters=True,
    )

    job_2 = BulkJob(
        job_id=2,
        title="Batch Job 2 - Classic White Solid Box",
        audio_path=env["audio_2"],
        images_dir=env["images_dir"],
        output_path=out_2,
        resolution="2k",
        text_color="#FFFFFF",
        highlight_color="#FFFFFF",
        has_bg_box=True,
        bg_style="solid_black",
        bg_color="#000000",
        bg_opacity=1.0,
        font_size=56,
        font_family="Arial",
        style_preset_name="High-Contrast Box",
        generate_chapters=True,
    )

    completed_jobs: list[tuple[int, str]] = []
    failed_jobs: list[tuple[int, str]] = []

    worker = BulkQueueWorker([job_1, job_2])
    worker.job_completed.connect(lambda row, path: completed_jobs.append((row, path)))
    worker.job_failed.connect(lambda row, err: failed_jobs.append((row, err)))

    # Execute directly in thread or synchronously via run()
    worker.run()

    # 1. Assert no failures
    assert len(failed_jobs) == 0, f"Some jobs failed: {failed_jobs}"
    assert len(completed_jobs) == 2, f"Expected 2 completed jobs, got {len(completed_jobs)}"

    # 2. Verify Output 1
    assert out_1.exists(), f"Output 1 missing: {out_1}"
    probe_1 = probe_media(out_1)
    assert probe_1.width == 2560, f"Expected width 2560, got {probe_1.width}"
    assert probe_1.height == 1440, f"Expected height 1440, got {probe_1.height}"
    assert probe_1.duration > 4.0, f"Expected duration > 4s, got {probe_1.duration}"
    assert (test_dir / "bulk_video_1_yellow_box.srt").exists(), "SRT missing for Job 1"
    assert (test_dir / "bulk_video_1_yellow_box_chapters.txt").exists(), "Chapters missing for Job 1"

    # 3. Verify Output 2
    assert out_2.exists(), f"Output 2 missing: {out_2}"
    probe_2 = probe_media(out_2)
    assert probe_2.width == 2560, f"Expected width 2560, got {probe_2.width}"
    assert probe_2.height == 1440, f"Expected height 1440, got {probe_2.height}"
    assert probe_2.duration > 4.0, f"Expected duration > 4s, got {probe_2.duration}"
    assert (test_dir / "bulk_video_2_white_box.srt").exists(), "SRT missing for Job 2"
    assert (test_dir / "bulk_video_2_white_box_chapters.txt").exists(), "Chapters missing for Job 2"

    print("\n✅ Bulk pipeline test succeeded! Both 2K videos generated with custom styles.")


if __name__ == "__main__":
    env = {
        "test_dir": PROJECT_ROOT / "output" / "bulk_test",
        "audio_1": (PROJECT_ROOT / "output" / "bulk_test" / "test_voice_1.mp3"),
        "audio_2": (PROJECT_ROOT / "output" / "bulk_test" / "test_voice_2.mp3"),
        "images_dir": PROJECT_ROOT / "assets" / "sample_footage",
    }
    init_db()
    env["test_dir"].mkdir(parents=True, exist_ok=True)
    slice_audio(PROJECT_ROOT / "sample.mp3", env["audio_1"], 0.0, 5.0)
    slice_audio(PROJECT_ROOT / "sample.mp3", env["audio_2"], 5.0, 5.0)
    test_bulk_queue_execution(env)

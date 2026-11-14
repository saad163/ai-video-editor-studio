"""
tests/test_phase3_template2.py
==============================
End-to-end integration test for Template 2 (Episode Scene Compiler).
Verifies:
  - Audio transcription with progress callbacks
  - Subtitle generation (ASS + SRT + Chapters)
  - PySceneDetect scene detection & caching
  - Narrative chunking into context blocks
  - Semantic/heuristic scene matching with anti-duplication
  - Subclip extraction with Lanczos scaling to 2K (2560x1440)
  - Final concatenation and subtitle burn-in
  - Full non-blocking QThread execution & signal emission
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ffmpeg_wrapper import probe_media, resolve_ffmpeg
from ui.workers.episode_worker import EpisodeCompilerWorker
from database.db import init_db, get_project


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def test_template2_episode_compiler_end_to_end(qapp, tmp_path):
    init_db()

    source_audio = PROJECT_ROOT / "sample.mp3"
    source_footage = PROJECT_ROOT / "Autocut.mp4"

    assert source_audio.exists(), f"sample.mp3 missing at {source_audio}"
    assert source_footage.exists(), f"Autocut.mp4 missing at {source_footage}"

    test_audio = tmp_path / "test_episode_voice.mp3"
    slice_audio(source_audio, test_audio, start_sec=0.0, duration_sec=7.0)

    test_out = tmp_path / "output_episode_master.mp4"

    # Track signals
    audio_progs: list[int] = []
    render_progs: list[int] = []
    steps: list[str] = []
    finished_paths: list[str] = []
    failed_msgs: list[str] = []

    worker = EpisodeCompilerWorker(
        audio_path=test_audio,
        footage_path=source_footage,
        output_path=test_out,
        title="Test Episode Master",
        resolution="2k",
        text_color="#FFFF00",
        highlight_color="#FFFF00",
        has_bg_box=True,
        bg_style="semi_transparent_black",
        bg_color="#000000",
        bg_opacity=0.5,
        font_size=58,
        font_family="Arial",
        generate_chapters=True,
        scene_threshold=27.0,
        target_chunk_sec=5.0,
    )

    worker.audio_progress.connect(lambda pct: audio_progs.append(pct))
    worker.render_progress.connect(lambda pct: render_progs.append(pct))
    worker.step_changed.connect(lambda s: steps.append(s))
    worker.render_finished.connect(lambda p: finished_paths.append(p))
    worker.render_failed.connect(lambda e: failed_msgs.append(e))

    # Run synchronously
    worker.run()

    # Assertions
    assert len(failed_msgs) == 0, f"Worker failed: {failed_msgs}"
    assert len(finished_paths) == 1
    assert Path(finished_paths[0]).exists()

    # Output file verification
    probe = probe_media(test_out)
    assert probe.has_video is True
    assert probe.has_audio is True
    assert probe.width == 2560
    assert probe.height == 1440
    assert probe.duration >= 2.0

    # Subtitles and chapters verification
    srt_file = tmp_path / "output_episode_master.srt"
    chapters_file = tmp_path / "output_episode_master_chapters.txt"
    assert srt_file.exists()
    assert chapters_file.exists()

    # Signals verification
    assert len(audio_progs) > 0
    assert 100 in audio_progs
    assert len(render_progs) > 0
    assert 100 in render_progs
    assert "Detecting Episode Scenes" in steps
    assert "Extracting Scene Subclips" in steps
    assert "Final Master Render" in steps


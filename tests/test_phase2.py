"""
tests/test_phase2.py
====================
Phase 2 Integration Test — Template 1 Core Engine

This script runs a complete end-to-end pipeline test without requiring any
external media files:

  1. Generates 3 test images with Pillow (coloured backgrounds + text labels).
  2. Loads the transcription of sample.mp3 from the SQLite cache (instant — no
     Whisper re-run needed after Phase 1).
  3. Builds styled ASS subtitles from the transcription.
  4. Assigns images to segments with the CharacterMatcher (anti-repetition).
  5. Renders each segment as a Ken Burns 1080p clip with FFmpegWrapper.
  6. Concatenates all clips with the voiceover audio and burned subtitles.
  7. Outputs a real 1080p MP4 to output/test_phase2_output.mp4.

Run from the project root (with .venv activated):
    python tests/test_phase2.py

Requirements:
  • FFmpeg must be installed and on PATH, or FFMPEG_PATH set in .env.
  • sample.mp3 must exist in the project root (Phase 1 test audio).
  • Phase 1 must have been run at least once so the transcription is cached.

Tips:
  • The Ken Burns zoompan filter is CPU-intensive. On the dev laptop (i5-1145G7),
    expect ~3–8× realtime per clip (a 5s clip may take 15–40 s to render).
  • To speed up the test, reduce MAX_TEST_SEGMENTS below.
  • On the RTX 2070 with NVENC, the same render completes in ~0.5× realtime.
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import setup_logging, get_config
from database.db import init_db, create_project

setup_logging()
logger = logging.getLogger("test_phase2")

# ─────────────────────────────────────────────────────────────────────────────
# Test configuration
# ─────────────────────────────────────────────────────────────────────────────

# Limit the number of segments to process (keeps test fast on CPU)
# Set to None to process ALL segments in the transcription.
MAX_TEST_SEGMENTS: int | None = 3

# Max duration per clip during test (in seconds) to avoid long renders on CPU
MAX_CLIP_DURATION_SEC: float | None = 4.0

# Which Ken Burns preset to use for the first clip (others rotate)
FIRST_PRESET_INDEX = 0  # 0 = ZOOM_IN_CENTER

# Output file name
OUTPUT_FILENAME = "test_phase2_output.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate test images
# ─────────────────────────────────────────────────────────────────────────────

def create_test_images(output_dir: Path, count: int = 3) -> list[Path]:
    """
    Generate *count* simple 1920×1080 test images using Pillow.

    Each image has a distinct background colour and a text label so it's easy
    to verify which image appears in each segment of the output video.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for test image generation.\n"
            "Install it with: pip install Pillow"
        )

    # Colour palette for test images (vivid, easy to distinguish)
    palettes = [
        ("#1A1A2E", "#E94560", "Scene 01 — Intro"),
        ("#16213E", "#0F3460", "Scene 02 — Rising Action"),
        ("#0F3460", "#533483", "Scene 03 — Climax"),
        ("#E94560", "#1A1A2E", "Scene 04 — Resolution"),
        ("#533483", "#16213E", "Scene 05 — Epilogue"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for i in range(min(count, len(palettes))):
        bg_color, accent, label = palettes[i]
        img = Image.new("RGB", (1920, 1080), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Accent stripe at top
        draw.rectangle([0, 0, 1920, 8], fill=accent)
        draw.rectangle([0, 1072, 1920, 1080], fill=accent)

        # Main label (large)
        try:
            # Try to use a system font for better visuals
            font_large = ImageFont.truetype("arial.ttf", 96)
            font_small = ImageFont.truetype("arial.ttf", 36)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_small = font_large

        draw.text(
            (960, 480), label,
            fill=accent, font=font_large, anchor="mm"
        )
        draw.text(
            (960, 620),
            f"AI Video Editor Studio — Test Image {i + 1}",
            fill="#CCCCCC", font=font_small, anchor="mm",
        )

        out_path = output_dir / f"test_scene_{i + 1:02}.jpg"
        img.save(out_path, quality=95)
        created.append(out_path)
        logger.info("Created test image: %s", out_path.name)

    return created


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Load cached transcription
# ─────────────────────────────────────────────────────────────────────────────

def load_cached_transcription(audio_path: Path):
    """
    Load transcription from SQLite cache (Phase 1 must have been run).
    Falls back to running Whisper if no cache entry exists.
    """
    from services.transcriber import TranscriptionService
    svc = TranscriptionService()

    # Create a test project for this run
    init_db()
    project_id = create_project("Phase2_Test", template_type="template_1")

    logger.info("Loading transcription for: %s", audio_path.name)
    result = svc.transcribe(audio_path, project_id=project_id, use_cache=True)

    if result.from_cache:
        logger.info("✓ Loaded from cache — %d segments", result.segment_count)
    else:
        logger.info("✓ Transcribed — %d segments in %.1f s", result.segment_count, result.elapsed_seconds)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2_pipeline(
    audio_path: Path,
    images_dir: Path,
    output_path: Path,
) -> None:
    """
    Execute the complete Phase 2 Template 1 pipeline.
    """
    cfg = get_config()
    t_pipeline_start = time.perf_counter()

    # ── Imports ────────────────────────────────────────────────────────────────
    from core.ffmpeg_wrapper import (
        resolve_ffmpeg, build_ken_burns_clip, concatenate_clips,
        _nvenc_available, _PRESET_CYCLE,
    )
    from services.character_matcher import CharacterMatcher
    from services.subtitle_engine import SubtitleEngine

    # ── Verify FFmpeg ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2 PIPELINE — Template 1 Core Engine")
    print("=" * 65)

    try:
        ff, ffp = resolve_ffmpeg()
        nvenc = _nvenc_available()
        print(f"  FFmpeg   : {ff}")
        print(f"  FFprobe  : {ffp}")
        print(f"  NVENC    : {'YES (GPU encode)' if nvenc else 'NO (CPU libx264)'}")
    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

    # ── Step 1: Test images ────────────────────────────────────────────────────
    print("\n[1/6] Generating test images …")
    test_images = create_test_images(images_dir, count=5)
    print(f"  Created {len(test_images)} test image(s) in {images_dir}")

    # ── Step 2: Load transcription ─────────────────────────────────────────────
    print("\n[2/6] Loading transcription …")
    transcription = load_cached_transcription(audio_path)
    segments = transcription.segments

    # Limit for test run
    if MAX_TEST_SEGMENTS is not None and len(segments) > MAX_TEST_SEGMENTS:
        segments = segments[:MAX_TEST_SEGMENTS]
        print(f"  Using first {MAX_TEST_SEGMENTS} of {transcription.segment_count} segment(s)")
    else:
        print(f"  Using all {len(segments)} segment(s)")

    # Adjust durations for test run if MAX_CLIP_DURATION_SEC is configured
    from services.transcriber import TranscriptionSegment
    if MAX_CLIP_DURATION_SEC is not None:
        clamped_segments = []
        curr_t = 0.0
        for s in segments:
            dur = min(s.duration, MAX_CLIP_DURATION_SEC)
            dur = max(dur, 1.5)
            clamped_segments.append(
                TranscriptionSegment(
                    start=curr_t,
                    end=curr_t + dur,
                    text=s.text,
                    words=s.words,
                )
            )
            curr_t += dur
        segments = clamped_segments

    # Reconstruct a trimmed TranscriptionResult for subtitle generation
    from services.transcriber import TranscriptionResult
    trimmed_result = TranscriptionResult(
        audio_path=transcription.audio_path,
        audio_hash=transcription.audio_hash,
        language=transcription.language,
        full_text=" ".join(s.text for s in segments),
        segments=segments,
        model_used=transcription.model_used,
        elapsed_seconds=0.0,
        from_cache=True,
    )

    # ── Step 3: Generate subtitles ─────────────────────────────────────────────
    print("\n[3/6] Generating ASS subtitles …")
    engine = SubtitleEngine(
        color="yellow",
        word_highlight=False,   # set True if word timestamps were captured
        font_size=52,
    )
    sub_path = cfg.temp_cache_dir / "phase2_subs.ass"
    engine.generate_ass(trimmed_result, sub_path)
    print(f"  ASS file → {sub_path.name}")

    # Also write SRT as a human-readable reference
    srt_path = output_path.parent / (output_path.stem + ".srt")
    engine.generate_srt(trimmed_result, srt_path)
    print(f"  SRT file → {srt_path.name}")

    # ── Step 4: Assign images to segments ─────────────────────────────────────
    print("\n[4/6] Assigning images to segments …")
    matcher = CharacterMatcher(images_dir=images_dir)
    assignments = matcher.assign(trimmed_result)

    print(f"  Assignments ({len(assignments)}):")
    for a in assignments:
        print(
            f"    [{a.clip_index:02d}] {a.segment.start:6.2f}s→{a.segment.end:6.2f}s "
            f"({a.segment.duration:.1f}s) | {a.preset.name:20s} | {a.image_path.name}"
        )

    # ── Step 5: Render Ken Burns clips ────────────────────────────────────────
    print(f"\n[5/6] Rendering {len(assignments)} Ken Burns clip(s) …")
    print("      (CPU zoompan is compute-intensive — please wait)")
    clips_dir = cfg.temp_chunks_dir / "phase2_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []
    total_video_duration = 0.0

    for a in assignments:
        clip_path = clips_dir / f"clip_{a.clip_index:03d}.mp4"
        t0 = time.perf_counter()

        def _progress(pct: float, idx=a.clip_index) -> None:
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(f"\r    Clip {idx:02d}: [{bar}] {pct*100:.0f}%", end="", flush=True)

        build_ken_burns_clip(
            image_path=a.image_path,
            duration=a.segment.duration,
            output_path=clip_path,
            preset=a.preset,
            progress_cb=_progress,
        )
        elapsed = time.perf_counter() - t0
        print(f"\r    Clip {a.clip_index:02d}: ✓ {a.segment.duration:.1f}s rendered in {elapsed:.1f}s")
        clip_paths.append(clip_path)
        total_video_duration += a.segment.duration

    # ── Step 6: Concatenate + audio + subtitles ────────────────────────────────
    print(f"\n[6/6] Final render: {len(clip_paths)} clips + audio + subtitles …")

    def _final_progress(pct: float) -> None:
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r  Progress: [{bar}] {pct*100:.0f}%", end="", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concatenate_clips(
        clip_paths=clip_paths,
        audio_path=audio_path,
        output_path=output_path,
        subtitle_path=sub_path,
        progress_cb=_final_progress,
        total_duration=total_video_duration,
    )
    print()

    # ── Results summary ────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_pipeline_start
    out_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("✓  PHASE 2 PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Output     : {output_path}")
    print(f"  Size       : {out_size_mb:.1f} MB")
    print(f"  Duration   : {total_video_duration:.1f} s of video")
    print(f"  Elapsed    : {total_elapsed:.1f} s total pipeline time")
    print(f"  SRT (ref)  : {srt_path}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_config()

    # ── Locate sample audio ────────────────────────────────────────────────────
    audio_candidates = list(cfg.project_root.glob("*.mp3")) + list(cfg.project_root.glob("*.wav"))
    if not audio_candidates:
        print("[ERROR] No audio file found in project root.")
        print("        Place a .mp3 or .wav file there and run Phase 1 first.")
        sys.exit(1)

    audio_path = audio_candidates[0]
    print(f"Audio source: {audio_path.name}")

    # Directories
    images_dir = cfg.temp_cache_dir / "phase2_test_images"
    output_path = cfg.output_dir / OUTPUT_FILENAME

    run_phase2_pipeline(
        audio_path=audio_path,
        images_dir=images_dir,
        output_path=output_path,
    )


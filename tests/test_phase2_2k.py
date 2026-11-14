"""
tests/test_phase2_2k.py
=======================
Ultra-Crisp 2K (2560x1440) Pipeline Runner with Clean Subtitle Styling.
Addresses all user-observed issues:
  1. Enforces clean English transcription (language="en").
  2. Breaks down speech into bite-sized 4-6 word chunks to prevent clutter.
  3. Uses single high-impact YouTube subtitle styling (58pt, white/yellow, 3.5px black outline, 90px bottom margin).
  4. Generates clean 2K visual assets with NO canvas text watermarks.
  5. Renders 2K video via Lanczos pre-scaling (2880x1620 -> 2560x1440) and high-quality CRF 19 encoding.
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import setup_logging, get_config
from database.db import init_db, create_project

setup_logging()
logger = logging.getLogger("test_phase2_2k")


def create_clean_2k_visuals(output_dir: Path, count: int = 4) -> list[Path]:
    """
    Generate clean, aesthetic 2560x1440 background visuals.
    CRITICAL: Does NOT draw any text or watermarks on the image canvas
    so only the burned subtitles appear on screen.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow")

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # Modern cinematic dark background tones (Slate, Teal, Indigo, Obsidian)
    color_schemes = [
        ("#0f172a", "#1e293b"),  # Dark Slate Obsidian
        ("#022c22", "#064e3b"),  # Deep Emerald Forest
        ("#172554", "#1e3a8a"),  # Midnight Sapphire
        ("#18181b", "#27272a"),  # Deep Carbon Zinc
    ]

    for i in range(min(count, len(color_schemes))):
        c1, c2 = color_schemes[i]
        # Create 2560x1440 image with smooth vertical gradient
        img = Image.new("RGB", (2560, 1440), color=c1)
        draw = ImageDraw.Draw(img)

        # Draw subtle gradient bands for depth
        for y in range(0, 1440, 2):
            factor = y / 1440.0
            # Interpolate between c1 and c2
            r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
            r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
            r = int(r1 + (r2 - r1) * factor)
            g = int(g1 + (g2 - g1) * factor)
            b = int(b1 + (b2 - b1) * factor)
            draw.line([(0, y), (2560, y)], fill=(r, g, b), width=2)

        out_path = output_dir / f"clean_scene_{i + 1:02}.jpg"
        img.save(out_path, quality=95)
        created.append(out_path)
        logger.info("Generated clean 2K visual: %s", out_path.name)

    return created


def run_clean_2k_pipeline(audio_path: Path, output_path: Path) -> Path:
    cfg = get_config()
    t_start = time.perf_counter()

    from core.ffmpeg_wrapper import (
        resolve_ffmpeg, build_ken_burns_clip, concatenate_clips,
        _nvenc_available, probe_media, KenBurnsPreset
    )
    from services.transcriber import TranscriptionService, TranscriptionResult, TranscriptionSegment
    from services.character_matcher import CharacterMatcher
    from services.subtitle_engine import SubtitleEngine

    print("\n" + "=" * 65)
    print("AI VIDEO EDITOR STUDIO — 2K ULTRA-SHARP CLEAN EXPORT")
    print("=" * 65)

    ff, ffp = resolve_ffmpeg()
    nvenc = _nvenc_available()
    print(f"  FFmpeg   : {ff}")
    print(f"  Target   : 2560x1440 (2K QHD)")
    print(f"  Encoder  : {'NVENC p5 (RTX GPU)' if nvenc else 'libx264 medium crf=19 (CPU)'}")

    # ── 1. Create clean 2K visuals (no canvas text) ───────────────────────────
    print("\n[1/5] Generating pristine 2K background visuals (no canvas text)...")
    images_dir = cfg.temp_cache_dir / "clean_2k_images"
    images = create_clean_2k_visuals(images_dir, count=4)

    # ── 2. Transcribe voiceover with strict English constraint ────────────────
    print("\n[2/5] Transcribing audio with strict English constraint...")
    svc = TranscriptionService()
    init_db()
    pid = create_project("Clean_2K_Test", template_type="template_1")

    # Force English and chunk to 5-6 words per event
    result: TranscriptionResult = svc.transcribe(
        audio_path,
        project_id=pid,
        language="en",
        task="transcribe",
        chunk_subtitles=True,
        max_words_per_chunk=6,
        force_retranscribe=False,
    )
    print(f"  Detected language : {result.language}")
    print(f"  Total chunks      : {result.segment_count}")

    # For fast testing on CPU, use the first 4 bite-sized segments (clamped to 3.5s max each)
    test_segments = result.segments[:4]
    clamped_segments: list[TranscriptionSegment] = []
    curr_t = 0.0
    for s in test_segments:
        dur = min(max(s.duration, 2.0), 3.5)
        clamped_segments.append(
            TranscriptionSegment(
                start=curr_t,
                end=curr_t + dur,
                text=s.text.strip(),
                words=s.words,
            )
        )
        curr_t += dur

    trimmed_result = TranscriptionResult(
        audio_path=result.audio_path,
        audio_hash=result.audio_hash,
        language="en",
        full_text=" ".join(s.text for s in clamped_segments),
        segments=clamped_segments,
        model_used=result.model_used,
        elapsed_seconds=0.0,
        from_cache=True,
    )

    print("\n  Subtitle Events (Bite-Sized Chunks):")
    for idx, s in enumerate(clamped_segments, 1):
        print(f"    [{idx:02d}] {s.start:5.2f}s -> {s.end:5.2f}s ({s.duration:.1f}s) | \"{s.text}\"")

    # ── 3. Generate 2K Styled Subtitles ───────────────────────────────────────
    print("\n[3/5] Generating High-Impact YouTube 2K Subtitles...")
    engine = SubtitleEngine(
        color="white",             # Pure White
        highlight_color="yellow",  # Vibrant Yellow active highlight
        word_highlight=False,
        font_size=58,              # 58pt for 2K
        font_name="Arial",
        max_chars_per_line=36,
    )
    # Calibrated 2K styling
    engine.style.outline_width = 3.5
    engine.style.margin_v = 90     # Safe bottom margin

    ass_path = cfg.temp_cache_dir / "clean_2k_subs.ass"
    engine.generate_ass(trimmed_result, ass_path)
    print(f"  ASS file -> {ass_path.name}")

    srt_path = output_path.parent / (output_path.stem + ".srt")
    engine.generate_srt(trimmed_result, srt_path)
    print(f"  SRT file -> {srt_path.name}")

    # ── 4. Assign images and render 2K Ken Burns clips ────────────────────────
    print("\n[4/5] Rendering 2K Ken Burns camera clips (2880x1620 -> 2560x1440 Lanczos)...")
    matcher = CharacterMatcher(images_dir)
    assignments = matcher.assign(trimmed_result)

    clips_dir = cfg.temp_chunks_dir / "clean_2k_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    total_dur = 0.0

    for a in assignments:
        clip_file = clips_dir / f"clip_2k_{a.clip_index:03d}.mp4"
        print(f"    Clip {a.clip_index+1}/{len(assignments)}: {a.preset.name} ({a.duration:.1f}s)...", end="", flush=True)
        t0 = time.perf_counter()
        build_ken_burns_clip(
            image_path=a.image_path,
            duration=a.duration,
            output_path=clip_file,
            preset=a.preset,
            resolution="2k",
        )
        print(f" done in {time.perf_counter()-t0:.1f}s")
        clip_paths.append(clip_file)
        total_dur += a.duration

    # ── 5. Assemble final 2K video + Subtitle Burn-In ──────────────────────────
    print(f"\n[5/5] Final 2K assembly + audio mix + subtitle burn-in...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concatenate_clips(
        clip_paths=clip_paths,
        audio_path=audio_path,
        output_path=output_path,
        subtitle_path=ass_path,
        total_duration=total_dur,
    )

    elapsed = time.perf_counter() - t_start
    info = probe_media(output_path)
    sz_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("✓  2K CLEAN VIDEO EXPORT COMPLETE")
    print("=" * 65)
    print(f"  Output File : {output_path}")
    print(f"  Resolution  : {info.resolution_label} (2K QHD)")
    print(f"  Duration    : {info.duration:.2f} s")
    print(f"  Framerate   : {info.fps:.1f} FPS")
    print(f"  Video Codec : {info.video_codec} (CRF 19 High Quality)")
    print(f"  File Size   : {sz_mb:.2f} MB")
    print(f"  Total Time  : {elapsed:.1f} s")
    print("=" * 65)
    return output_path


if __name__ == "__main__":
    cfg = get_config()
    audio = cfg.project_root / "sample.mp3"
    target_output = cfg.output_dir / "test_phase2_2k_clean.mp4"
    run_clean_2k_pipeline(audio_path=audio, output_path=target_output)


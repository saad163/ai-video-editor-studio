"""
tests/test_smooth_ken_burns.py
==============================
Verification tests for:
1. Butter-Smooth Ken Burns motion with 2.5x Lanczos upscale (sub-pixel precision).
2. All 6 multi-directional presets (ZOOM_IN_CENTER, ZOOM_OUT_CENTER, PAN_LEFT_TO_RIGHT,
   PAN_RIGHT_TO_LEFT, PAN_TOP_TO_BOTTOM, PAN_BOTTOM_TO_TOP).
3. Non-repeating consecutive preset selection.
4. Dual-layer blurred pillarbox with smooth Ken Burns motion on portrait images.
5. End-to-end rendering of 3 consecutive images with distinct presets.
"""

from pathlib import Path
import random
import pytest
from PIL import Image

from core.ffmpeg_wrapper import (
    KenBurnsPreset,
    KEN_BURNS_PRESETS,
    get_random_ken_burns_preset,
    generate_ken_burns_preset_sequence,
    _build_zoompan_filter,
    _build_portrait_blurred_filter,
    build_ken_burns_clip,
    probe_media,
)


def test_ken_burns_preset_definitions():
    """Verify all 6 core presets are present in KEN_BURNS_PRESETS."""
    expected_names = [
        "ZOOM_IN_CENTER",
        "ZOOM_OUT_CENTER",
        "PAN_LEFT_TO_RIGHT",
        "PAN_RIGHT_TO_LEFT",
        "PAN_TOP_TO_BOTTOM",
        "PAN_BOTTOM_TO_TOP",
    ]
    actual_names = [p.name for p in KEN_BURNS_PRESETS]
    for name in expected_names:
        assert name in actual_names, f"Preset {name} missing from KEN_BURNS_PRESETS"
    assert len(KEN_BURNS_PRESETS) == 6


def test_non_repeating_preset_selection():
    """Verify get_random_ken_burns_preset never returns the same preset consecutively."""
    rng = random.Random(42)
    last_preset = None
    for _ in range(50):
        preset = get_random_ken_burns_preset(previous_preset=last_preset, rng=rng)
        assert preset != last_preset, f"Consecutive duplicate preset detected: {preset.name}"
        assert preset in KEN_BURNS_PRESETS
        last_preset = preset


def test_preset_sequence_generator():
    """Verify generate_ken_burns_preset_sequence generates balanced sequence with zero consecutive duplicates."""
    rng = random.Random(123)
    seq = generate_ken_burns_preset_sequence(30, rng=rng)
    assert len(seq) == 30
    for i in range(len(seq) - 1):
        assert seq[i] != seq[i + 1], f"Duplicate at index {i} and {i+1}: {seq[i].name}"

    # Verify all 6 presets are represented in a 30-item sequence
    unique_in_seq = set(seq)
    assert len(unique_in_seq) == 6, "All 6 presets must be used across sequence"


def test_high_density_prescale_zoompan_filter():
    """Verify filtergraph contains >= 3840 prescale and smooth zoompan expressions."""
    for preset in KEN_BURNS_PRESETS:
        flt = _build_zoompan_filter(duration=3.0, preset=preset, resolution="1080p")
        # High density prescale check (e.g. 4800:2700 for 1080p)
        assert "scale=4800:2700" in flt or "scale=3840:2160" in flt
        assert "flags=lanczos" in flt
        assert "zoompan=" in flt
        assert "d=90" in flt          # 3.0s * 30fps
        assert "s=1920x1080" in flt
        assert "format=yuv420p" in flt


def test_portrait_blurred_filter_with_all_presets():
    """Verify portrait blurred filter has static background and foreground zoompan for all presets."""
    for preset in KEN_BURNS_PRESETS:
        flt = _build_portrait_blurred_filter(duration=2.0, preset=preset, resolution="1080p")
        assert "split=2[bg_src][fg_src]" in flt
        assert "boxblur=25:10" in flt
        assert "scale=1920:1080" in flt
        assert "overlay=(W-w)/2:(H-h)/2" in flt
        assert "zoompan=" in flt
        assert "format=yuv420p" in flt


def test_render_3_consecutive_images_distinct_presets(tmp_path: Path):
    """
    Render 3 consecutive images with distinct, non-repeating smooth Ken Burns presets.
    Verifies valid playable MP4s and zero errors.
    """
    colors = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
    img_paths: list[Path] = []
    for i, col in enumerate(colors):
        p = tmp_path / f"test_img_{i}.jpg"
        im = Image.new("RGB", (1280, 720), color=col)
        im.save(p)
        img_paths.append(p)

    rng = random.Random(999)
    last_p = None
    rendered_clips: list[Path] = []

    for i, img_p in enumerate(img_paths):
        chosen_preset = get_random_ken_burns_preset(previous_preset=last_p, rng=rng)
        assert chosen_preset != last_p
        last_p = chosen_preset

        out_clip = tmp_path / f"clip_{i}.mp4"
        build_ken_burns_clip(
            image_path=img_p,
            duration=1.2,
            output_path=out_clip,
            preset=chosen_preset,
            resolution="720p",
        )
        assert out_clip.exists()
        assert out_clip.stat().st_size > 0
        info = probe_media(out_clip)
        assert info.width == 1280
        assert info.height == 720
        assert info.duration >= 1.1
        rendered_clips.append(out_clip)

    assert len(rendered_clips) == 3


"""
tests/test_hybrid_mixer.py
==========================
Tests for Part 2:
  1. Timeline slot builder with configurable hold duration (e.g. 5s slots for 30s audio).
  2. Hybrid media ratio allocation (e.g. 30% images / 70% video) and controlled interleaving.
  3. Long-video slicing pool generation.
  4. End-to-end hybrid render test (stills + video subclips + voiceover audio).
"""

from pathlib import Path
import pytest

from core.ffmpeg_wrapper import probe_media, extract_subclip, build_ken_burns_clip, concatenate_clips
from services.hybrid_mixer import (
    build_timeline_slots,
    slice_long_videos,
    HybridMediaMixer,
    VisualSlot,
)


def test_timeline_slots_partition():
    """Verify 5s hold duration splits a 30s timeline into 6 exact 5s slots."""
    slots = build_timeline_slots(total_duration=30.0, clip_duration=5.0)
    assert len(slots) == 6
    for start, dur in slots:
        assert dur == 5.0

    # Total covered duration equals 30.0s
    total_dur = sum(dur for _, dur in slots)
    assert abs(total_dur - 30.0) < 0.001

    # Check timestamps are contiguous
    for i in range(len(slots) - 1):
        assert abs((slots[i][0] + slots[i][1]) - slots[i + 1][0]) < 0.001


def test_timeline_slots_tail_handling():
    """Verify tail remainder handling on uneven durations (e.g. 62s at 5s slots)."""
    slots = build_timeline_slots(total_duration=62.0, clip_duration=5.0)
    total_dur = sum(dur for _, dur in slots)
    assert abs(total_dur - 62.0) < 0.001
    assert slots[-1][0] + slots[-1][1] == pytest.approx(62.0, 0.01)


def test_hybrid_ratio_distribution_and_interleaving(tmp_path: Path):
    """Verify 30% image / 70% video ratio produces exact slot distribution and interleaved types."""
    # Create dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "char1.jpg").write_text("fake")
    (img_dir / "char2.jpg").write_text("fake")

    # Create dummy video paths
    vid_file = tmp_path / "episode.mp4"
    vid_file.write_text("fake")

    mixer = HybridMediaMixer(
        images_dir=img_dir,
        videos_path=vid_file,
        clip_duration=5.0,
        image_ratio=0.30,  # 30% images, 70% videos
        seed=42,
    )
    # Manually populate dummy video files for unit testing without full ffprobe
    mixer._video_files = [vid_file]

    # Test interleaving logic on 10 slots
    types = mixer._interleave_types(num_images=3, num_videos=7)
    assert len(types) == 10
    assert types.count("image") == 3
    assert types.count("video") == 7

    # Check that images are not all clustered at the start or end
    # Neither first 3 are all images nor last 3 are all images
    assert not (types[:3] == ["image", "image", "image"])
    assert not (types[-3:] == ["image", "image", "image"])


def test_slice_long_videos_with_real_footage():
    """Verify slicing candidate subclips from source video (skipping first and last 2s)."""
    video_path = Path("Autocut.mp4")
    if not video_path.exists():
        pytest.skip("Autocut.mp4 not found in workspace")

    info = probe_media(video_path)
    dur = info.duration
    assert dur > 10.0

    pool = slice_long_videos([video_path], clip_duration=5.0, skip_edges=2.0)
    assert len(pool) >= 2

    # Verify each subclip is 5s and starts after edge skip (>= 2.0)
    for vpath, start_s, end_s in pool:
        assert vpath == video_path.resolve()
        assert start_s >= 2.0
        assert end_s <= dur
        assert abs((end_s - start_s) - 5.0) < 0.01


def test_hybrid_mixer_end_to_end_render(tmp_path: Path):
    """End-to-end integration test: mix 1 real image and 1 real video subclip into a final MP4."""
    video_src = Path("Autocut.mp4")
    audio_src = Path("sample.mp3")
    if not video_src.exists() or not audio_src.exists():
        pytest.skip("Sample media not present for integration test")

    # Create 1 real still image
    from PIL import Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_file = img_dir / "sample_still.jpg"
    im = Image.new("RGB", (1280, 720), color=(240, 110, 30))
    im.save(img_file)

    mixer = HybridMediaMixer(
        images_dir=img_dir,
        videos_path=video_src,
        clip_duration=3.0,
        image_ratio=0.5,
        seed=10,
    )

    # 6 seconds total duration -> two 3-second slots (1 image, 1 video)
    slots = mixer.create_mix(total_duration=6.0)
    assert len(slots) == 2

    # Render each slot
    clip_paths = []
    for idx, slot in enumerate(slots):
        out_clip = tmp_path / f"slot_{idx}.mp4"
        if slot.media_type == "image":
            build_ken_burns_clip(
                image_path=slot.source_path,
                duration=slot.duration,
                output_path=out_clip,
                preset=slot.preset,
                resolution="720p",
            )
        else:
            extract_subclip(
                video_path=slot.source_path,
                start_sec=slot.subclip_start,
                end_sec=slot.subclip_end,
                output_path=out_clip,
                resolution="720p",
                target_duration=slot.duration,
                keep_audio=False,
            )
        assert out_clip.exists()
        clip_paths.append(out_clip)

    # Concatenate final video
    final_output = tmp_path / "hybrid_final_render.mp4"
    concatenate_clips(
        clip_paths=clip_paths,
        audio_path=audio_src,
        output_path=final_output,
        resolution="720p",
        total_duration=6.0,
    )

    assert final_output.exists()
    final_info = probe_media(final_output)
    assert final_info.duration >= 5.5
    assert final_info.width == 1280
    assert final_info.height == 720


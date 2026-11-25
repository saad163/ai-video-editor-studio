"""
services/hybrid_mixer.py
========================
Phase 3 — Hybrid Media Mixer & Long-Video Slicing Engine.

Features:
  1. Timeline Block Builder: Decouples visual clip hold durations (1s to 20s)
     from short Whisper subtitles while preserving exact audio sync.
  2. Long-Video Slicing: Probes 6–7 min episode footage and slices candidate
     subclip pools of length `clip_duration_sec` (skipping edge intro/credits).
  3. Controlled Shuffle & Media Mixing: Interleaves still character images and
     raw video subclips according to the user-selected ratio (e.g., 30% images / 70% video),
     distributing them naturally across the entire timeline with anti-repetition.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from core.ffmpeg_wrapper import (
    KenBurnsPreset, _PRESET_CYCLE, probe_media, get_random_ken_burns_preset
)
from services.character_matcher import CharacterMatcher, _ImagePool
from services.transcriber import TranscriptionResult
from utils.file_helpers import SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS, is_image_file, is_video_file
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisualSlot:
    """
    Represents a single continuous visual segment on the final timeline.

    Attributes
    ----------
    slot_index     : Zero-based sequential position on output timeline.
    media_type     : 'image' (Ken Burns still) or 'video' (cut from episode footage).
    start_time     : Start timestamp on final video timeline (seconds).
    duration       : Duration of this visual block (seconds).
    source_path    : Path to still image or raw episode video file.
    subclip_start  : Start offset in raw video file (if media_type == 'video').
    subclip_end    : End offset in raw video file (if media_type == 'video').
    preset         : Ken Burns preset (if media_type == 'image').
    character      : Character attribution (if applicable).
    """
    slot_index: int
    media_type: str
    start_time: float
    duration: float
    source_path: Path
    subclip_start: float = 0.0
    subclip_end: float = 0.0
    preset: KenBurnsPreset = KenBurnsPreset.ZOOM_IN_CENTER
    character: str = "general"

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


# ─────────────────────────────────────────────────────────────────────────────
# Timeline Slots Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline_slots(total_duration: float, clip_duration: float = 5.0) -> list[tuple[float, float]]:
    """
    Partition `total_duration` (0.0 to total_duration) into uniform intervals.

    Parameters
    ----------
    total_duration : Full voiceover audio duration (seconds).
    clip_duration  : Target duration per visual block (seconds, clamped 1.0 to 20.0).

    Returns
    -------
    List of (start_sec, duration_sec) covering 100% of total_duration continuously.
    """
    if total_duration <= 0.0:
        return []

    clip_duration = max(1.0, min(float(clip_duration), 20.0))

    # If audio is shorter than or equal to clip_duration, return single slot
    if total_duration <= clip_duration:
        return [(0.0, round(total_duration, 3))]

    num_slots = int(total_duration // clip_duration)
    rem = total_duration - (num_slots * clip_duration)

    slots: list[tuple[float, float]] = []
    cur = 0.0

    for _ in range(num_slots):
        slots.append((round(cur, 3), round(clip_duration, 3)))
        cur += clip_duration

    # Handle remaining tail
    if rem >= 1.5:
        # Meaningful tail: add a dedicated final slot
        slots.append((round(cur, 3), round(rem, 3)))
    elif rem > 0.0:
        # Small tail (< 1.5s): extend previous slot to avoid flash
        prev_start, prev_dur = slots[-1]
        slots[-1] = (prev_start, round(prev_dur + rem, 3))

    return slots


# ─────────────────────────────────────────────────────────────────────────────
# Long-Video Slicing Pool
# ─────────────────────────────────────────────────────────────────────────────

def slice_long_videos(
    video_paths: Sequence[Path],
    clip_duration: float = 5.0,
    skip_edges: float = 2.0,
) -> list[tuple[Path, float, float]]:
    """
    Probe long episode videos and divide them into a candidate pool of subclip slices.

    Parameters
    ----------
    video_paths   : List of raw episode video files.
    clip_duration : Target duration for each subclip slice (seconds).
    skip_edges    : Seconds to skip at beginning and end of each file (avoids black/credits).

    Returns
    -------
    List of (video_path, subclip_start, subclip_end) tuples.
    """
    pool: list[tuple[Path, float, float]] = []

    for vpath in video_paths:
        vpath = Path(vpath).resolve()
        if not vpath.exists():
            continue

        try:
            info = probe_media(vpath)
            dur = info.duration
        except Exception as exc:
            logger.warning("Could not probe video %s: %s", vpath.name, exc)
            continue

        if dur <= (skip_edges * 2 + 1.0):
            # Video is very short; take whatever is available
            pool.append((vpath, 0.0, dur))
            continue

        start = skip_edges
        safe_end = dur - skip_edges

        while start + clip_duration <= safe_end:
            pool.append((vpath, round(start, 3), round(start + clip_duration, 3)))
            start += clip_duration

    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Media Mixer
# ─────────────────────────────────────────────────────────────────────────────

class HybridMediaMixer:
    """
    Coordinates character images and raw episode video footage into a cohesive,
    dynamically interleaved visual timeline.
    """

    def __init__(
        self,
        images_dir: Optional[Path] = None,
        videos_path: Optional[Path] = None,
        clip_duration: float = 5.0,
        image_ratio: float = 0.5,
        seed: Optional[int] = None,
        image_semantics: Optional[dict[Path, Any]] = None,
    ) -> None:
        self.images_dir = Path(images_dir).resolve() if images_dir else None
        self.videos_path = Path(videos_path).resolve() if videos_path else None
        self.clip_duration = max(1.0, min(float(clip_duration), 20.0))
        self.image_ratio = max(0.0, min(float(image_ratio), 1.0))
        self.rng = random.Random(seed)
        self.image_semantics = image_semantics

        self._image_files: list[Path] = []
        self._video_files: list[Path] = []
        self._char_matcher: Optional[CharacterMatcher] = None

        self._scan_sources()

    def _scan_sources(self) -> None:
        """Scan input paths for image and video assets."""
        # 1. Scan images
        if self.images_dir and self.images_dir.exists():
            if self.images_dir.is_dir():
                try:
                    self._char_matcher = CharacterMatcher(self.images_dir, image_semantics=self.image_semantics)
                    self._image_files = list(self._char_matcher._all_images)
                except Exception:
                    self._image_files = [
                        p for p in self.images_dir.rglob("*")
                        if p.is_file() and is_image_file(p)
                    ]
            elif self.images_dir.is_file() and is_image_file(self.images_dir):
                self._image_files = [self.images_dir]

        # 2. Scan videos
        target_video_candidates: list[Path] = []
        if self.videos_path and self.videos_path.exists():
            if self.videos_path.is_file() and is_video_file(self.videos_path):
                target_video_candidates.append(self.videos_path)
            elif self.videos_path.is_dir():
                target_video_candidates.extend([
                    p for p in self.videos_path.rglob("*")
                    if p.is_file() and is_video_file(p)
                ])

        # Also check if images_dir itself contains video files (hybrid single folder)
        if self.images_dir and self.images_dir.is_dir():
            for p in self.images_dir.iterdir():
                if p.is_file() and is_video_file(p) and p not in target_video_candidates:
                    target_video_candidates.append(p)

        self._video_files = target_video_candidates

        logger.info(
            "HybridMediaMixer discovered %d image(s) and %d video file(s)",
            len(self._image_files), len(self._video_files),
        )

    @property
    def has_images(self) -> bool:
        return bool(self._image_files)

    @property
    def has_videos(self) -> bool:
        return bool(self._video_files)

    def create_mix(
        self,
        total_duration: float,
        transcription: Optional[TranscriptionResult] = None,
        character_assignments: Optional[list[str]] = None,
        used_images: Optional[set[Path]] = None,
    ) -> list[VisualSlot]:
        """
        Build an interleaved timeline of image and video slots matching `total_duration`.

        Parameters
        ----------
        total_duration : Full voiceover audio duration (seconds).
        transcription  : Optional transcription result for speech alignment.
        character_assignments : Optional character name mappings per segment.
        used_images    : Optional mutable set of image paths already used (for cross-part continuity).

        Returns
        -------
        List of VisualSlot covering 100% of the timeline.
        """
        raw_intervals = build_timeline_slots(total_duration, self.clip_duration)
        if not raw_intervals:
            return []

        num_slots = len(raw_intervals)

        # ── Determine counts for each media type ──────────────────────────────
        if not self.has_videos and not self.has_images:
            raise ValueError("No valid images or videos found in provided media paths.")

        if not self.has_videos:
            slot_types = ["image"] * num_slots
        elif not self.has_images:
            slot_types = ["video"] * num_slots
        else:
            # Both media types available: calculate target ratio
            num_images = int(round(num_slots * self.image_ratio))
            if 0.05 < self.image_ratio < 0.95 and num_slots >= 2:
                num_images = max(1, min(num_slots - 1, num_images))
            num_videos = num_slots - num_images

            # Controlled Interleaving: distribute images evenly across timeline
            slot_types = self._interleave_types(num_images, num_videos)

        # ── Prepare video candidate subclips pool ─────────────────────────────
        video_subclips: list[tuple[Path, float, float]] = []
        if "video" in slot_types and self._video_files:
            video_subclips = slice_long_videos(self._video_files, self.clip_duration)
            if video_subclips:
                self.rng.shuffle(video_subclips)

        # ── Prepare character matcher / image pool ────────────────────────────
        if self._char_matcher is None and self.images_dir:
            self._char_matcher = CharacterMatcher(self.images_dir, image_semantics=self.image_semantics)

        assignments_image: list = []
        if "image" in slot_types and self._char_matcher and transcription:
            try:
                assignments_image = self._char_matcher.assign(
                    transcription, character_assignments=character_assignments, total_duration=total_duration
                )
            except Exception:
                assignments_image = []

        # ── Build final VisualSlots ───────────────────────────────────────────
        slots: list[VisualSlot] = []
        img_idx = 0
        vid_idx = 0
        last_image: Optional[Path] = None
        last_preset: Optional[KenBurnsPreset] = None
        if used_images is None:
            used_images = set()

        for idx, (slot_start, slot_dur) in enumerate(raw_intervals):
            stype = slot_types[idx]

            if stype == "image" or not video_subclips:
                # Pick image
                chosen_img: Optional[Path] = None
                char_name = "general"

                # Extract spoken narration for this slot
                slot_narration = ""
                if transcription and transcription.segments:
                    slot_end = slot_start + slot_dur
                    overlapping = [
                        seg.text for seg in transcription.segments
                        if seg.start < slot_end and seg.end > slot_start
                    ]
                    if overlapping:
                        slot_narration = " ".join(overlapping)

                if assignments_image and img_idx < len(assignments_image):
                    char_name = assignments_image[img_idx % len(assignments_image)].character

                if self._char_matcher:
                    chosen_img = self._char_matcher.select_best_image_for_text(
                        character=char_name,
                        narration_text=slot_narration,
                        used_images=used_images,
                        last_image=last_image,
                    )
                elif assignments_image and img_idx < len(assignments_image):
                    chosen_img = assignments_image[img_idx % len(assignments_image)].image_path
                elif self._image_files:
                    candidates = [p for p in self._image_files if p != last_image]
                    chosen_img = self.rng.choice(candidates if candidates else self._image_files)

                if chosen_img is None:
                    # Fallback to video if image unavailable
                    stype = "video"
                else:
                    preset = get_random_ken_burns_preset(last_preset, rng=self.rng)
                    last_preset = preset
                    slots.append(VisualSlot(
                        slot_index=idx,
                        media_type="image",
                        start_time=slot_start,
                        duration=slot_dur,
                        source_path=chosen_img,
                        preset=preset,
                        character=char_name,
                    ))
                    last_image = chosen_img
                    used_images.add(chosen_img)
                    img_idx += 1
                    continue

            # Video subclip slot
            if video_subclips:
                subclip = video_subclips[vid_idx % len(video_subclips)]
                vpath, s_start, s_end = subclip
                # If slot_dur differs from default slice, adjust s_end cleanly
                actual_end = min(s_start + slot_dur, s_end + slot_dur)

                slots.append(VisualSlot(
                    slot_index=idx,
                    media_type="video",
                    start_time=slot_start,
                    duration=slot_dur,
                    source_path=vpath,
                    subclip_start=s_start,
                    subclip_end=actual_end,
                ))
                vid_idx += 1
            else:
                # Absolute fallback: if no video subclips, use image
                img_path = self._image_files[idx % len(self._image_files)] if self._image_files else Path("fallback.jpg")
                preset = get_random_ken_burns_preset(last_preset, rng=self.rng)
                last_preset = preset
                slots.append(VisualSlot(
                    slot_index=idx,
                    media_type="image",
                    start_time=slot_start,
                    duration=slot_dur,
                    source_path=img_path,
                    preset=preset,
                ))

        logger.info(
            "Created %d visual slot(s): %d images (%.0f%%), %d videos (%.0f%%), total duration: %.1f s",
            len(slots),
            sum(1 for s in slots if s.media_type == "image"),
            (sum(1 for s in slots if s.media_type == "image") / len(slots) * 100) if slots else 0,
            sum(1 for s in slots if s.media_type == "video"),
            (sum(1 for s in slots if s.media_type == "video") / len(slots) * 100) if slots else 0,
            sum(s.duration for s in slots),
        )
        return slots

    def _interleave_types(self, num_images: int, num_videos: int) -> list[str]:
        """
        Interleave images and video slots evenly across the timeline.
        Prevents clustering of images or videos at the beginning or end.
        """
        total = num_images + num_videos
        if total == 0:
            return []
        if num_images == 0:
            return ["video"] * num_videos
        if num_videos == 0:
            return ["image"] * num_images

        # Proportional distribution (Bresenham-style interleaving)
        result: list[str] = []
        img_count = 0
        vid_count = 0

        for i in range(total):
            # Target image count up to this point
            ideal_images = int(round((i + 1) * (num_images / total)))
            if img_count < ideal_images and img_count < num_images:
                result.append("image")
                img_count += 1
            elif vid_count < num_videos:
                result.append("video")
                vid_count += 1
            else:
                result.append("image")
                img_count += 1

        # Apply slight randomized swapping between adjacent elements to feel natural while preserving ratio
        for i in range(len(result) - 1):
            if self.rng.random() < 0.20:
                result[i], result[i + 1] = result[i + 1], result[i]

        return result


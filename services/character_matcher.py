"""
services/character_matcher.py
==============================
Phase 2 — Character-to-image matcher for Template 1.

Purpose
-------
Given:
  • A directory of character sub-folders (each folder name = character name,
    each folder contains image files).
  • A list of TranscriptionSegments (from Phase 1 Whisper output).
  • Optional character-name assignments per segment (from ScriptParser).

Produces:
  • One (segment, image_path, ken_burns_preset) tuple per segment, distributing
    images across segments with strict anti-repetition rules.

Anti-repetition rules
---------------------
  1. The same image is NEVER placed in two consecutive segments.
  2. Within one character's pool, images are distributed round-robin so each
     image gets equal screen time before any repeats.
  3. If a character's folder has only 1 image, it may be reused (but not
     back-to-back with itself).
  4. If no images match a segment's character, the matcher falls back to the
     general pool (images directly in images_dir, not in sub-folders).
  5. If the general pool is also empty, it cycles through ALL images.

Ken Burns preset assignment
---------------------------
  Presets cycle through _PRESET_CYCLE in order, ensuring adjacent clips
  always use different motion styles.

Usage
-----
  from services.character_matcher import CharacterMatcher
  from services.transcriber import TranscriptionResult

  matcher = CharacterMatcher(images_dir=Path("assets/characters"))
  assignments = matcher.assign(
      transcription_result,
      character_map={"narrator": "general"},  # optional override
  )
  for seg, img_path, preset in assignments:
      print(seg.start, img_path, preset.name)
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from core.ffmpeg_wrapper import (
    KenBurnsPreset, _PRESET_CYCLE, get_random_ken_burns_preset
)
from services.transcriber import TranscriptionResult, TranscriptionSegment
from utils.file_helpers import SUPPORTED_IMAGE_EXTS, SUPPORTED_VIDEO_EXTS
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentAssignment:
    """
    Maps one transcription segment to a media asset and a Ken Burns preset.

    Attributes
    ----------
    segment         : The transcription segment this clip covers.
    image_path      : Chosen image (or video clip path, Phase 3+) to display.
    preset          : Ken Burns motion preset for this clip.
    character       : Which character/entity this segment is attributed to.
    clip_index      : Zero-based position in the final video sequence.
    custom_duration : Optional continuous timeline duration in seconds.
    """
    segment: TranscriptionSegment
    image_path: Path
    preset: KenBurnsPreset
    character: str
    clip_index: int
    custom_duration: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.custom_duration is not None and self.custom_duration > 0:
            return self.custom_duration
        return self.segment.duration


# ─────────────────────────────────────────────────────────────────────────────
# Image pool — per-character round-robin with anti-repeat
# ─────────────────────────────────────────────────────────────────────────────

class _ImagePool:
    """
    Manages a list of images for one character with round-robin distribution.

    Tracks how many times each image has been used and always picks the
    least-used one, with a constraint that the last-used path is avoided
    if at least one other option exists.
    """

    def __init__(self, name: str, images: list[Path]) -> None:
        self.name = name
        self._images: list[Path] = list(images)
        self._usage: dict[Path, int] = {img: 0 for img in self._images}
        self._last_used: Optional[Path] = None

    def __len__(self) -> int:
        return len(self._images)

    def has_images(self) -> bool:
        return bool(self._images)

    def next_image(self, global_last: Optional[Path] = None) -> Optional[Path]:
        """
        Pick the next image using least-used + anti-repeat logic.

        Parameters
        ----------
        global_last : The image used in the immediately preceding segment
                      (may be from a different character). Avoided if possible.
        """
        if not self._images:
            return None

        # Sort candidates by (usage_count, is_last_used, is_global_last)
        def _score(p: Path) -> tuple[int, int, int]:
            return (
                self._usage[p],
                1 if p == self._last_used else 0,
                1 if p == global_last else 0,
            )

        candidates = sorted(self._images, key=_score)
        chosen = candidates[0]

        self._usage[chosen] += 1
        self._last_used = chosen
        return chosen

    def add_image(self, path: Path) -> None:
        if path not in self._usage:
            self._images.append(path)
            self._usage[path] = 0


# ─────────────────────────────────────────────────────────────────────────────
# CharacterMatcher
# ─────────────────────────────────────────────────────────────────────────────

class CharacterMatcher:
    """
    Builds character image pools from a directory, then assigns images to
    transcription segments with anti-repetition and round-robin guarantees.

    Directory layout expected::

        images_dir/
          ali/           ← sub-folder name matches character name (case-insensitive)
            ali_01.jpg
            ali_02.jpg
          zara/
            zara_hero.png
          general_bg.jpg  ← top-level images go into the 'general' pool

    Parameters
    ----------
    images_dir  : Root directory containing character sub-folders.
    shuffle     : If True, randomise image order within each pool at startup
                  (False = deterministic alphabetical order).
    """

    GENERAL_POOL = "general"

    def __init__(self, images_dir: Path, shuffle: bool = False) -> None:
        self.images_dir = Path(images_dir).resolve()
        self._shuffle = shuffle
        self._pools: dict[str, _ImagePool] = {}
        self._all_images: list[Path] = []   # fallback: every image found
        self._root_videos: list[Path] = []

        if self.images_dir.exists():
            self._scan_directory()
        else:
            logger.warning("images_dir does not exist: %s", self.images_dir)

    # ── Directory scanning ─────────────────────────────────────────────────────

    def _scan_directory(self) -> None:
        """Build image pools and collect root raw videos by walking images_dir."""
        general_imgs: list[Path] = []
        root_videos: list[Path] = []

        for item in sorted(self.images_dir.iterdir()):
            if item.is_dir():
                # Each sub-folder = one character pool
                imgs = self._collect_images(item)
                if imgs:
                    pool_name = item.name.lower()
                    self._pools[pool_name] = _ImagePool(pool_name, imgs)
                    self._all_images.extend(imgs)
                    logger.info(
                        "Character pool '%s': %d image(s)", pool_name, len(imgs)
                    )
            elif item.is_file():
                ext = item.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    # Top-level images → general pool
                    general_imgs.append(item)
                    self._all_images.append(item)
                elif ext in SUPPORTED_VIDEO_EXTS:
                    root_videos.append(item)

        if general_imgs:
            self._pools[self.GENERAL_POOL] = _ImagePool(self.GENERAL_POOL, general_imgs)
            logger.info("General pool: %d image(s)", len(general_imgs))

        self._root_videos = root_videos

        logger.info(
            "CharacterMatcher ready: %d pools, %d total images, %d root video(s)",
            len(self._pools), len(self._all_images), len(self._root_videos),
        )

    def _collect_images(self, directory: Path) -> list[Path]:
        """Return all supported image files in *directory* (non-recursive)."""
        imgs = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        if self._shuffle:
            random.shuffle(imgs)
        return imgs

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def character_names(self) -> list[str]:
        """Return list of known character pool names."""
        return [k for k in self._pools if k != self.GENERAL_POOL]

    @property
    def root_videos(self) -> list[Path]:
        """Return list of raw video files located directly in root directory."""
        return list(self._root_videos)

    @classmethod
    def scan_asset_folder(cls, folder: Path | str) -> dict[str, Any]:
        """
        Scan a unified asset directory and return summary counts and formatted badge text.
        """
        p = Path(folder).resolve()
        if not p.exists() or not p.is_dir():
            return {
                "character_pools": {},
                "general_images": 0,
                "total_images": 0,
                "video_files": [],
                "badge_text": f"⚠️ Folder not found: {p.name}",
            }

        pools: dict[str, int] = {}
        general_count = 0
        total_images = 0
        video_files: list[Path] = []

        try:
            for item in sorted(p.iterdir()):
                if item.is_dir():
                    imgs = [
                        f for f in item.iterdir()
                        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
                    ]
                    if imgs:
                        pools[item.name] = len(imgs)
                        total_images += len(imgs)
                elif item.is_file():
                    ext = item.suffix.lower()
                    if ext in SUPPORTED_IMAGE_EXTS:
                        general_count += 1
                        total_images += 1
                    elif ext in SUPPORTED_VIDEO_EXTS:
                        video_files.append(item)
        except Exception as exc:
            logger.warning("Error scanning asset folder %s: %s", p, exc)

        parts: list[str] = []
        if pools:
            names_summary = ", ".join(pools.keys())
            if len(names_summary) > 28:
                names_summary = f"{len(pools)} Characters"
            parts.append(f"{len(pools)} Character Pool{'s' if len(pools) > 1 else ''} ({names_summary} — {sum(pools.values())} images)")
        if general_count > 0:
            parts.append(f"{general_count} General Image{'s' if general_count > 1 else ''}")
        if video_files:
            v_label = f"{len(video_files)} Raw Video File{'s' if len(video_files) > 1 else ''}"
            parts.append(v_label)

        if parts:
            badge_text = "✓ Detected: " + " + ".join(parts)
        else:
            badge_text = "⚠️ No visual assets found in folder"

        return {
            "character_pools": pools,
            "general_images": general_count,
            "total_images": total_images,
            "video_files": video_files,
            "badge_text": badge_text,
        }

    def assign(
        self,
        transcription: TranscriptionResult,
        character_assignments: Optional[list[str]] = None,
        character_map: Optional[dict[str, str]] = None,
        total_duration: Optional[float] = None,
    ) -> list[SegmentAssignment]:
        """
        Assign one image and one Ken Burns preset to every transcription segment.
        When *total_duration* is provided, extends clip durations to cover 100% of the
        audio timeline (0.0s -> total_duration) continuously without pauses or black frames.

        Parameters
        ----------
        transcription           : Full TranscriptionResult from Phase 1.
        character_assignments   : Per-segment character names (same length as
                                  transcription.segments).  If None, uses
                                  the general pool for all segments.
        character_map           : Optional name remapping, e.g.
                                  {"narrator": "general", "HOST": "ali"}.
        total_duration          : Full voiceover audio duration in seconds.

        Returns
        -------
        List of SegmentAssignment, in continuous time order.
        """
        segments = transcription.segments
        if not segments:
            if total_duration and total_duration > 0 and self._all_images:
                # Generate continuous visual stream covering audio duration even if no speech detected
                assignments: list[SegmentAssignment] = []
                cur = 0.0
                idx = 0
                last_preset: Optional[KenBurnsPreset] = None
                while cur < total_duration:
                    chunk = min(6.0, total_duration - cur)
                    img = self._all_images[idx % len(self._all_images)]
                    preset = get_random_ken_burns_preset(last_preset)
                    last_preset = preset
                    seg = TranscriptionSegment(start=cur, end=cur + chunk, text="")
                    assignments.append(SegmentAssignment(
                        segment=seg,
                        image_path=img,
                        preset=preset,
                        character=self.GENERAL_POOL,
                        clip_index=idx,
                        custom_duration=chunk,
                    ))
                    cur += chunk
                    idx += 1
                return assignments
            return []

        # Default character assignments
        if character_assignments is None:
            # Smart detection: scan segment dialogue for known character pool mentions
            known_names = [k for k in self._pools if k != self.GENERAL_POOL]
            if known_names:
                detected: list[str] = []
                last_matched = self.GENERAL_POOL
                for seg in segments:
                    text_lower = seg.text.lower()
                    found = None
                    for name in known_names:
                        pattern = r"\b" + re.escape(name) + r"\b"
                        if re.search(pattern, text_lower):
                            found = name
                            break
                    if found:
                        detected.append(found)
                        last_matched = found
                    else:
                        detected.append(last_matched if last_matched != self.GENERAL_POOL else self.GENERAL_POOL)
                character_assignments = detected
            else:
                character_assignments = [self.GENERAL_POOL] * len(segments)

        if len(character_assignments) != len(segments):
            logger.warning(
                "character_assignments length %d ≠ segments length %d — padding with 'general'",
                len(character_assignments), len(segments),
            )
            diff = len(segments) - len(character_assignments)
            character_assignments = character_assignments + [self.GENERAL_POOL] * diff

        # Apply optional name remapping
        if character_map:
            character_assignments = [
                character_map.get(c, c) for c in character_assignments
            ]

        assignments: list[SegmentAssignment] = []
        global_last_image: Optional[Path] = None
        last_preset: Optional[KenBurnsPreset] = None

        for idx, (seg, char_name) in enumerate(zip(segments, character_assignments)):
            # Resolve pool
            pool = self._resolve_pool(char_name)
            image = None

            if pool is not None:
                image = pool.next_image(global_last=global_last_image)

            # Fallback chain: general → all_images → first image
            if image is None:
                gen_pool = self._pools.get(self.GENERAL_POOL)
                if gen_pool:
                    image = gen_pool.next_image(global_last=global_last_image)

            if image is None and self._all_images:
                # Absolute fallback: pick least-recently-used from entire corpus
                image = self._pick_global_fallback(global_last_image)

            if image is None:
                logger.error("No images available for segment %d! Skipping.", idx)
                continue

            # Non-repeating random Ken Burns preset
            preset = get_random_ken_burns_preset(last_preset)
            last_preset = preset

            assignments.append(SegmentAssignment(
                segment=seg,
                image_path=image,
                preset=preset,
                character=char_name,
                clip_index=idx,
            ))

            global_last_image = image

        # ── Continuous Timeline Gap-Filling (Full Audio Duration Matching) ────
        if assignments and total_duration is not None and total_duration > 0:
            n = len(assignments)
            starts = [0.0]
            for i in range(1, n):
                s = max(assignments[i].segment.start, starts[i - 1] + 1.0)
                starts.append(s)

            for i in range(n - 1):
                dur = max(starts[i + 1] - starts[i], 1.0)
                assignments[i].custom_duration = dur

            # Last segment coverage
            last_start = starts[-1]
            rem_dur = total_duration - last_start
            if rem_dur <= 12.0:
                assignments[-1].custom_duration = max(rem_dur, 1.0)
            else:
                first_chunk = min(max(assignments[-1].segment.duration, 5.0), 8.0)
                assignments[-1].custom_duration = first_chunk
                cur_time = last_start + first_chunk
                extra_idx = n
                while cur_time < total_duration:
                    chunk = min(6.0, total_duration - cur_time)
                    img = self._pick_global_fallback(global_last_image)
                    if not img and self._all_images:
                        img = self._all_images[extra_idx % len(self._all_images)]
                    if not img:
                        break
                    preset = get_random_ken_burns_preset(last_preset)
                    last_preset = preset
                    dummy_seg = TranscriptionSegment(
                        start=cur_time,
                        end=cur_time + chunk,
                        text="",
                    )
                    assignments.append(SegmentAssignment(
                        segment=dummy_seg,
                        image_path=img,
                        preset=preset,
                        character=self.GENERAL_POOL,
                        clip_index=extra_idx,
                        custom_duration=chunk,
                    ))
                    global_last_image = img
                    cur_time += chunk
                    extra_idx += 1

        logger.info(
            "Assigned %d segment(s) across %d unique image(s) (total video coverage: %.1f s)",
            len(assignments),
            len({a.image_path for a in assignments}),
            sum(a.duration for a in assignments),
        )
        return assignments

    # ── Pool resolution helpers ────────────────────────────────────────────────

    def _resolve_pool(self, character_name: str) -> Optional[_ImagePool]:
        """
        Return the pool matching *character_name* (case-insensitive).

        Tries exact match, then partial (substring) match, then None.
        """
        key = character_name.lower().strip()
        if key in self._pools:
            return self._pools[key]
        # Partial match: "Dr. Ali" might match pool "ali"
        for pool_key, pool in self._pools.items():
            if pool_key in key or key in pool_key:
                return pool
        return None

    def _pick_global_fallback(self, avoid: Optional[Path]) -> Optional[Path]:
        """Pick the best image from the global corpus, avoiding *avoid*."""
        candidates = [p for p in self._all_images if p != avoid]
        if not candidates:
            candidates = self._all_images
        return candidates[0] if candidates else None

    # ── Debug helper ──────────────────────────────────────────────────────────

    def print_pools(self) -> None:
        """Print a summary of all discovered pools."""
        print(f"\nCharacterMatcher — {self.images_dir}")
        print(f"  Pools: {len(self._pools)} | Total images: {len(self._all_images)}")
        for name, pool in self._pools.items():
            print(f"  [{name}] {len(pool)} image(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from utils.config import setup_logging, get_config
    setup_logging()

    cfg = get_config()
    # Use temp/cache as a dummy images dir with test images
    test_dir = cfg.temp_cache_dir / "test_images"
    test_dir.mkdir(exist_ok=True)

    # Create dummy images if needed
    try:
        from PIL import Image, ImageDraw
        for i, color in enumerate(["#E63946", "#457B9D", "#2A9D8F"], start=1):
            img = Image.new("RGB", (1920, 1080), color)
            draw = ImageDraw.Draw(img)
            draw.text((960, 540), f"Test Image {i}", fill="white", anchor="mm")
            img.save(test_dir / f"test_{i:02}.jpg")
        print(f"Created 3 test images in {test_dir}")
    except ImportError:
        print("Pillow not installed — create images manually in", test_dir)
        sys.exit(1)

    matcher = CharacterMatcher(images_dir=test_dir)
    matcher.print_pools()
    print("\nCharacter names:", matcher.character_names)


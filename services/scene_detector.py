"""
services/scene_detector.py
==========================
Phase 3 — PySceneDetect Scene Splitter & Keyframe Extractor.

Responsibilities:
  1. Detect visual scene cuts in source video episodes using PySceneDetect (ContentDetector).
  2. Extract 1 middle keyframe per scene for visual context and card previews.
  3. Store scene metadata (timestamps, duration, keyframe path) in SQLite (`scenes` table).
  4. Cache lookup: If the video has already been analyzed (matching SHA-256 hash),
     instantly retrieve scenes from SQLite and skip re-processing.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from utils.config import get_config
from utils.file_helpers import compute_file_hash, is_video_file
from utils.logger import get_logger
from database.db import (
    init_db,
    insert_scene,
    get_scenes_by_video_hash,
    insert_media_asset,
    asset_hash_exists,
)

logger = get_logger(__name__)


@dataclass
class Scene:
    """Represents a single detected scene from a source video."""
    scene_id: Optional[int]
    video_hash: str
    video_path: Path
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    keyframe_path: Optional[Path] = None
    description: str = ""
    used_count: int = 0

    @property
    def label(self) -> str:
        return f"Scene {self.scene_index:03d} [{self.start_sec:.1f}s - {self.end_sec:.1f}s ({self.duration_sec:.1f}s)]"


class SceneDetectorService:
    """
    Automates scene boundary detection and keyframe sampling for episode videos.
    """

    def __init__(self, threshold: float = 27.0, min_scene_len_sec: float = 1.5) -> None:
        """
        Parameters
        ----------
        threshold          : PySceneDetect ContentDetector threshold (27.0 is optimal for anime/drama/film).
        min_scene_len_sec  : Minimum duration in seconds for a detected scene.
        """
        self.threshold = threshold
        self.min_scene_len_sec = min_scene_len_sec
        init_db()

    def detect_scenes(
        self,
        video_path: Path,
        project_id: Optional[int] = None,
        extract_keyframes: bool = True,
        use_cache: bool = True,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> list[Scene]:
        """
        Detect scenes in *video_path*. If already processed, returns cached scenes from SQLite.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not is_video_file(video_path):
            raise ValueError(f"Unsupported video format: {video_path.suffix}")

        cfg = get_config()
        video_hash = compute_file_hash(video_path)

        # ── 1. Cache Check ────────────────────────────────────────────────────
        if use_cache:
            cached_rows = get_scenes_by_video_hash(video_hash)
            if cached_rows:
                logger.info("Cache hit for video '%s' — %d scenes loaded from SQLite.", video_path.name, len(cached_rows))
                return [
                    Scene(
                        scene_id=row["id"],
                        video_hash=row["video_hash"],
                        video_path=Path(row["video_path"]),
                        scene_index=row["scene_index"],
                        start_sec=row["start_sec"],
                        end_sec=row["end_sec"],
                        duration_sec=row["duration_sec"],
                        keyframe_path=Path(row["keyframe_path"]) if row["keyframe_path"] else None,
                        description=row["scene_description"] or "",
                        used_count=row["used_count"],
                    )
                    for row in cached_rows
                ]

        # ── 2. Run PySceneDetect ──────────────────────────────────────────────
        logger.info("Running PySceneDetect on '%s' (threshold=%.1f)...", video_path.name, self.threshold)
        t0 = time.perf_counter()

        video = open_video(str(video_path))
        fps = video.frame_rate
        min_scene_len_frames = int(self.min_scene_len_sec * fps)

        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=self.threshold, min_scene_len=min_scene_len_frames)
        )

        # Detect scenes across video
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()

        # If no cuts were detected, treat the entire video as one scene
        if not scene_list:
            total_frames = video.duration.get_frames()
            total_sec = video.duration.get_seconds()
            scene_list = [(video.duration.new(0), video.duration.new(total_frames))]

        logger.info("Detected %d scene(s) in %.2fs.", len(scene_list), time.perf_counter() - t0)

        # ── 3. Extract Keyframes & Persist to SQLite ───────────────────────────
        scenes: list[Scene] = []
        cap = cv2.VideoCapture(str(video_path)) if extract_keyframes else None

        frames_dir = cfg.temp_frames_dir / f"ep_{video_hash[:10]}"
        if extract_keyframes:
            frames_dir.mkdir(parents=True, exist_ok=True)

        for idx, (start_time, end_time) in enumerate(scene_list, start=1):
            start_s = start_time.get_seconds()
            end_s = end_time.get_seconds()
            dur_s = end_s - start_s

            keyframe_path: Optional[Path] = None

            if extract_keyframes and cap and cap.isOpened():
                # Target middle frame of the scene for best representation
                mid_sec = start_s + (dur_s / 2.0)
                mid_frame = int(mid_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if ret and frame is not None:
                    kf_file = frames_dir / f"scene_{idx:03d}_{mid_sec:.1f}s.jpg"
                    # Save keyframe at standard 720p/1080p preview resolution
                    cv2.imwrite(str(kf_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    keyframe_path = kf_file

            # Save to SQLite
            db_scene_id = insert_scene(
                video_hash=video_hash,
                video_path=str(video_path),
                scene_index=idx,
                start_sec=start_s,
                end_sec=end_s,
                duration_sec=dur_s,
                keyframe_path=str(keyframe_path) if keyframe_path else None,
                scene_description=f"Episode scene {idx} ({dur_s:.1f}s)",
                project_id=project_id,
            )

            scenes.append(
                Scene(
                    scene_id=db_scene_id,
                    video_hash=video_hash,
                    video_path=video_path,
                    scene_index=idx,
                    start_sec=start_s,
                    end_sec=end_s,
                    duration_sec=dur_s,
                    keyframe_path=keyframe_path,
                    description=f"Episode scene {idx} ({dur_s:.1f}s)",
                    used_count=0,
                )
            )

            if progress_cb:
                progress_cb((idx / len(scene_list)))

        if cap:
            cap.release()

        # Record media asset in DB
        if project_id and not asset_hash_exists(video_hash):
            insert_media_asset(
                project_id=project_id,
                file_path=str(video_path),
                file_hash=video_hash,
                media_type="video",
                duration_sec=video.duration.get_seconds(),
            )

        logger.info("Persisted %d scenes with keyframes to database.", len(scenes))
        return scenes


if __name__ == "__main__":
    import sys
    from utils.config import setup_logging
    setup_logging()

    cfg = get_config()
    test_video = cfg.output_dir / "test_phase2_2k_clean.mp4"
    if not test_video.exists():
        print(f"[ERROR] Test video not found at: {test_video}")
        sys.exit(1)

    detector = SceneDetectorService(threshold=20.0)
    scenes = detector.detect_scenes(test_video, extract_keyframes=True)
    print(f"\nDetected {len(scenes)} scenes:")
    for s in scenes:
        kf = s.keyframe_path.name if s.keyframe_path else "None"
        print(f"  {s.label} | Keyframe: {kf}")

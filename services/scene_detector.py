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
    is_interview: bool = False
    interview_score: float = 0.0

    @property
    def label(self) -> str:
        return f"Scene {self.scene_index:03d} [{self.start_sec:.1f}s - {self.end_sec:.1f}s ({self.duration_sec:.1f}s)]"


def evaluate_interview_framing(frame_bgr: "cv2.typing.MatLike | np.ndarray", rembg_session: Any = None) -> dict[str, Any]:
    """
    Evaluate human / face presence and interview-style framing in a video frame.

    Interview framing criteria:
      - Visible human face / upper body in upper-middle framing (1-person or 2-person host/guest).
      - Moderate human area coverage (typically 6% to 75% of screen area).
      - Distinguishes human subjects from distant scenery, landscape, or non-human cuts.

    Returns dict with keys:
      - is_interview (bool): True if frame matches interview framing criteria.
      - score (float): 0.0 to 1.0 confidence score.
      - face_count (int): Number of detected face/head candidates.
      - face_coverage (float): Ratio of face area to total frame area.
      - human_coverage (float): Estimated human body coverage.
    """
    if frame_bgr is None or not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        return {"is_interview": False, "score": 0.0, "face_count": 0, "face_coverage": 0.0, "human_coverage": 0.0}

    # Resize to standardized evaluation resolution for high speed
    eval_w, eval_h = 320, 180
    small = cv2.resize(frame_bgr, (eval_w, eval_h))
    total_px = eval_w * eval_h

    # 1. Skin-tone detection in YCrCb color space
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    # 2. Contour analysis for face/head bounding boxes
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= (total_px * 0.005):  # At least 0.5% of frame
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1, bw)
            bbox_area = bw * bh

            # Faces / upper heads typically have aspect ratio 0.5 to 3.2 and are not entire screen
            if 0.5 <= aspect <= 3.2 and bbox_area < (total_px * 0.70):
                cy = y + (bh // 2)
                # Face center usually in upper 85% of frame (not solely feet/floor)
                if cy <= int(eval_h * 0.85):
                    face_candidates.append({
                        "bbox": (x, y, bw, bh),
                        "area": area,
                        "aspect": aspect,
                        "center": (x + bw // 2, cy),
                    })

    face_count = len(face_candidates)
    total_face_area = sum(f["area"] for f in face_candidates)
    face_coverage = total_face_area / total_px

    # 3. Neural person matting verification if rembg_session provided
    human_coverage = 0.0
    if rembg_session is not None:
        try:
            from PIL import Image
            import rembg
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mask = rembg.remove(Image.fromarray(rgb), session=rembg_session, only_mask=True)
            m_arr = np.array(mask)
            human_coverage = float(np.count_nonzero(m_arr > 50) / total_px)
        except Exception as exc:
            logger.debug("Neural matting fallback in interview evaluation: %s", exc)
            human_coverage = min(0.65, face_coverage * 3.5)
    else:
        human_coverage = min(0.65, face_coverage * 3.5)

    # 4. Composite Interview Framing Score
    score = 0.0
    is_interview = False

    if face_count >= 1:
        # 1-person close-up/medium shot or 2-person host/guest setup
        face_score = min(1.0, face_coverage / 0.03) * 0.45
        pos_score = 0.25 if all(f["center"][1] <= int(eval_h * 0.75) for f in face_candidates) else 0.10
        body_score = min(1.0, human_coverage / 0.15) * 0.30 if human_coverage > 0 else 0.15
        score = face_score + pos_score + body_score
        is_interview = bool(score >= 0.38 and face_coverage >= 0.007)
    elif human_coverage >= 0.12:
        # Person visible (e.g. turned head or side profile)
        score = min(0.65, human_coverage * 2.0)
        is_interview = bool(score >= 0.42)

    return {
        "is_interview": is_interview,
        "score": round(float(score), 3),
        "face_count": face_count,
        "face_coverage": round(float(face_coverage), 4),
        "human_coverage": round(float(human_coverage), 4),
    }


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

    def filter_interview_scenes(
        self,
        scenes: list[Scene],
        min_score: float = 0.38,
        relax_if_empty: bool = True,
    ) -> list[Scene]:
        """
        Filter a list of detected scenes down to interview-style candidate scenes
        (person / talking / host-guest framing).

        If relax_if_empty is True and no scenes pass the threshold, returns all scenes
        gracefully so the pipeline never crashes.
        """
        if not scenes:
            return []

        interview_scenes: list[Scene] = []
        for s in scenes:
            # If scene already evaluated
            if s.is_interview:
                interview_scenes.append(s)
                continue

            # Evaluate keyframe if available
            if s.keyframe_path and Path(s.keyframe_path).is_file():
                try:
                    frame = cv2.imread(str(s.keyframe_path))
                    eval_res = evaluate_interview_framing(frame)
                    s.is_interview = eval_res["is_interview"]
                    s.interview_score = eval_res["score"]
                    if eval_res["is_interview"] or eval_res["score"] >= min_score:
                        interview_scenes.append(s)
                except Exception as exc:
                    logger.debug("Failed evaluating scene %d keyframe for interview framing: %s", s.scene_index, exc)
            else:
                # If no keyframe, treat as neutral
                if not s.is_interview and s.interview_score >= min_score:
                    interview_scenes.append(s)

        if not interview_scenes and relax_if_empty:
            logger.info("No interview scenes met threshold %.2f; relaxing filter to return all %d scene(s).", min_score, len(scenes))
            return list(scenes)

        return interview_scenes


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

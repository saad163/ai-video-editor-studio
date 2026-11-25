"""
services/interview_segmenter.py
===============================
Foreground Subject Isolation & Background Removal for Interview Clips in Template 3.
Uses neural person matting (rembg u2net_human_seg) with edge feathering and morphological
smoothing to generate alpha mask streams for seamless compositing over custom blurred backgrounds.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from PIL import Image

from utils.logger import get_logger
from core.ffmpeg_wrapper import _ff

logger = get_logger(__name__)

# Ensure u2net models are stored and loaded from the project temp directory
_TEMP_MODELS_DIR = Path(__file__).resolve().parent.parent / "temp" / "models"
_TEMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("U2NET_HOME", str(_TEMP_MODELS_DIR))

_REMBG_SESSION = None


def get_rembg_session():
    """Lazily initialize and cache the rembg u2net_human_seg session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION

    try:
        import rembg
        logger.info("Initializing neural person segmentation session (u2net_human_seg)...")
        _REMBG_SESSION = rembg.new_session("u2net_human_seg")
        return _REMBG_SESSION
    except Exception as exc:
        logger.warning("Could not initialize rembg session: %s", exc)
        return None


def generate_person_mask_video(
    source_video: Path,
    start_sec: float,
    duration_sec: float,
    output_mask_path: Path,
    crop_filter: Optional[str] = None,
    target_width: int = 640,
    target_height: int = 360,
    fps: int = 25,
    sample_step: int = 1,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Optional[Path]:
    """
    Extract frames from a cropped/trimmed interview clip, run foreground person matting,
    apply spatial feathering, and encode a synchronized black-and-white alpha mask video.

    Returns the path to output_mask_path on success, or None on failure/fallback.
    """
    session = get_rembg_session()
    if session is None:
        logger.warning("rembg session unavailable; skipping person segmentation.")
        return None

    try:
        import cv2
        import rembg
    except ImportError:
        logger.warning("cv2 or rembg not installed; skipping person segmentation.")
        return None

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Start FFmpeg reader process piping raw RGB24 frames
    vf_parts = []
    if crop_filter:
        vf_parts.append(crop_filter)
    vf_parts.append(f"scale={target_width}:{target_height}")
    vf_parts.append(f"fps={fps}")
    vf_chain = ",".join(vf_parts)

    read_cmd = [
        _ff(),
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(source_video),
        "-t", f"{duration_sec:.3f}",
        "-vf", vf_chain,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-an",
        "-"
    ]

    # 2. Start FFmpeg writer process encoding mask frames into MP4
    write_cmd = [
        _ff(),
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{target_width}x{target_height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "18",
        str(output_mask_path)
    ]

    frame_size = target_width * target_height * 3
    total_frames = max(1, int(round(duration_sec * fps)))

    try:
        read_proc = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        write_proc = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as exc:
        logger.error("Failed to start FFmpeg pipes for segmentation: %s", exc)
        return None

    frame_idx = 0
    last_mask_bytes: Optional[bytes] = None

    try:
        while True:
            raw_frame = read_proc.stdout.read(frame_size)
            if not raw_frame or len(raw_frame) < frame_size:
                break

            if sample_step > 1 and (frame_idx % sample_step != 0) and last_mask_bytes is not None:
                write_proc.stdin.write(last_mask_bytes)
                frame_idx += 1
                continue

            # Convert raw bytes to PIL Image
            frame_img = Image.frombytes("RGB", (target_width, target_height), raw_frame)

            # Generate alpha mask using neural segmentation
            mask_img = rembg.remove(frame_img, session=session, only_mask=True)
            mask_arr = np.array(mask_img)

            # Apply morphological edge feathering / Gaussian blur for natural hair/shoulder blending
            if mask_arr.ndim == 2:
                feathered = cv2.GaussianBlur(mask_arr, (5, 5), 0)
            else:
                feathered = mask_arr

            mask_bytes = feathered.tobytes()
            last_mask_bytes = mask_bytes
            write_proc.stdin.write(mask_bytes)
            frame_idx += 1

            if progress_cb and frame_idx % 10 == 0:
                progress_cb(min(1.0, frame_idx / total_frames))

    except Exception as exc:
        logger.error("Error during person segmentation stream: %s", exc)
        try:
            read_proc.kill()
            write_proc.kill()
        except Exception:
            pass
        return None
    finally:
        if read_proc.stdout:
            read_proc.stdout.close()
        read_proc.wait()

        if write_proc.stdin:
            write_proc.stdin.close()
        write_proc.wait()

    if output_mask_path.exists() and output_mask_path.stat().st_size > 500:
        logger.info(
            "Person segmentation completed (%d frames): %s (%.1f KB)",
            frame_idx,
            output_mask_path.name,
            output_mask_path.stat().st_size / 1024,
        )
        return output_mask_path
    else:
        logger.warning("Segmentation output mask was empty or missing: %s", output_mask_path)
        return None

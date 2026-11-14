"""
utils/file_helpers.py
=====================
Shared file-system utilities: hashing, extension checks, temp-dir cleanup.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".jfif"}


def compute_file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Return hex digest of a file using the specified algorithm."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_AUDIO_EXTS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTS


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTS


def get_media_type(path: Path) -> str:
    """Return 'audio', 'video', 'image', or 'unknown'."""
    ext = path.suffix.lower()
    if ext in SUPPORTED_AUDIO_EXTS:
        return "audio"
    if ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    return "unknown"


def clean_directory(directory: Path, pattern: str = "*") -> int:
    """
    Delete all files matching *pattern* inside *directory*.

    Returns the number of files deleted.  Does NOT recurse into sub-dirs.
    """
    deleted = 0
    for item in directory.glob(pattern):
        if item.is_file():
            try:
                item.unlink()
                deleted += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", item, exc)
    logger.info("Cleaned %d file(s) from %s", deleted, directory)
    return deleted


def get_directory_size_gb(directory: Path) -> float:
    """Return total size of *directory* (recursive) in gigabytes."""
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def safe_copy(src: Path, dst: Path) -> Path:
    """Copy *src* to *dst*, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(src, dst))


def ensure_sample_assets() -> Path:
    """
    Ensure required sample visual assets and directories exist inside `assets/`.
    Organizes sample footage in `assets/sample_footage/` and reference UI in `assets/reference_ui/`.
    Returns the Path to `assets/sample_footage/`.
    """
    root = Path(__file__).resolve().parent.parent
    sample_dir = root / "assets" / "sample_footage"
    ref_dir = root / "assets" / "reference_ui"

    sample_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


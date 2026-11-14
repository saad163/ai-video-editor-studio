"""
utils/config.py
===============
Centralised configuration loader for AI Video Editor Studio.

Reads from .env (via python-dotenv), validates values with Pydantic, and
exposes a single `get_config()` call that returns a frozen AppConfig object.
All path resolution happens here so the rest of the codebase never constructs
raw os.path strings.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ConfigDict

# ── Resolve project root (the directory that contains main.py) ────────────────
# This file lives at <root>/utils/config.py  →  two levels up = project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from project root (silently does nothing if the file is absent)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Settings Model
# ─────────────────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Immutable application configuration resolved at startup."""

    # ── Google Gemini ──────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Whisper / Transcription ────────────────────────────────────────────────
    whisper_model_size: str = Field(
        default="small",
        description="Model size: tiny | base | small | medium | large-v3",
    )
    compute_device: str = Field(
        default="auto",
        description="Execution device: auto | cpu | cuda",
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root: Path = Field(default=PROJECT_ROOT)
    temp_dir: Path = Field(default=PROJECT_ROOT / "temp")
    output_dir: Path = Field(default=PROJECT_ROOT / "output")
    db_path: Path = Field(default=PROJECT_ROOT / "database" / "app.db")

    # ── Runtime Limits ─────────────────────────────────────────────────────────
    max_temp_size_gb: float = Field(default=5.0)
    log_level: str = Field(default="INFO")

    # ── Computed sub-paths (populated by validator) ────────────────────────────
    temp_chunks_dir: Path = Field(default=PROJECT_ROOT / "temp" / "chunks")
    temp_frames_dir: Path = Field(default=PROJECT_ROOT / "temp" / "frames")
    temp_cache_dir: Path = Field(default=PROJECT_ROOT / "temp" / "cache")

    model_config = ConfigDict(frozen=True)

    # ── Field validators ───────────────────────────────────────────────────────

    @field_validator("whisper_model_size")
    @classmethod
    def validate_model_size(cls, v: str) -> str:
        allowed = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
        if v not in allowed:
            raise ValueError(f"whisper_model_size must be one of {allowed}, got '{v}'")
        return v

    @field_validator("compute_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda"}
        if v not in allowed:
            raise ValueError(f"compute_device must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log level: {v}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Factory — reads environment and resolves paths
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Build and cache the application configuration.

    Env vars override defaults; custom paths in .env override computed defaults.
    All directories are created on first access here so callers can rely on them
    already existing.
    """
    raw_temp = os.getenv("TEMP_DIR", "").strip()
    raw_output = os.getenv("OUTPUT_DIR", "").strip()

    temp_dir = Path(raw_temp) if raw_temp else PROJECT_ROOT / "temp"
    output_dir = Path(raw_output) if raw_output else PROJECT_ROOT / "output"

    cfg = AppConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        compute_device=os.getenv("COMPUTE_DEVICE", "auto"),
        temp_dir=temp_dir,
        output_dir=output_dir,
        db_path=PROJECT_ROOT / "database" / "app.db",
        max_temp_size_gb=float(os.getenv("MAX_TEMP_SIZE_GB", "5")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        temp_chunks_dir=temp_dir / "chunks",
        temp_frames_dir=temp_dir / "frames",
        temp_cache_dir=temp_dir / "cache",
    )

    # Ensure all required directories exist at startup
    for directory in (
        cfg.temp_dir,
        cfg.temp_chunks_dir,
        cfg.temp_frames_dir,
        cfg.temp_cache_dir,
        cfg.output_dir,
        cfg.db_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (call once from main.py)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger based on config LOG_LEVEL."""
    cfg = get_config()
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),                                    # Console
            logging.FileHandler(log_dir / "app.log", encoding="utf-8"),  # File
        ],
    )
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    cfg = get_config()
    print("=" * 60)
    print("AppConfig loaded successfully")
    print(f"  Project root   : {cfg.project_root}")
    print(f"  Temp dir       : {cfg.temp_dir}")
    print(f"  Output dir     : {cfg.output_dir}")
    print(f"  DB path        : {cfg.db_path}")
    print(f"  Whisper model  : {cfg.whisper_model_size}")
    print(f"  Compute device : {cfg.compute_device}")
    print(f"  Gemini key set : {'YES' if cfg.gemini_api_key else 'NO (add to .env)'}")
    print("=" * 60)


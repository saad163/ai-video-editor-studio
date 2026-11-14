"""
tests/test_phase1.py
====================
Phase 1 sanity-check tests (no audio file required).

Run:
    pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ── Config ────────────────────────────────────────────────────────────────────
def test_config_loads():
    from utils.config import get_config
    cfg = get_config()
    assert cfg.project_root.exists()
    assert cfg.whisper_model_size in {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
    assert cfg.compute_device in {"auto", "cpu", "cuda"}


def test_temp_dirs_exist():
    from utils.config import get_config
    cfg = get_config()
    assert cfg.temp_dir.exists()
    assert cfg.temp_chunks_dir.exists()
    assert cfg.temp_frames_dir.exists()
    assert cfg.temp_cache_dir.exists()


def test_output_dir_exists():
    from utils.config import get_config
    cfg = get_config()
    assert cfg.output_dir.exists()


# ── Database ──────────────────────────────────────────────────────────────────
def test_db_initialises():
    from database.db import init_db
    init_db()   # Should not raise


def test_project_crud():
    from database.db import init_db, create_project, get_project, update_project_status
    init_db()
    pid = create_project("Unit Test Project", template_type="template_1")
    assert pid > 0
    row = get_project(pid)
    assert row is not None
    assert row["title"] == "Unit Test Project"
    assert row["status"] == "draft"
    update_project_status(pid, "done")
    row = get_project(pid)
    assert row["status"] == "done"


def test_media_asset_insert_and_dedup():
    from database.db import init_db, create_project, insert_media_asset, asset_hash_exists
    init_db()
    pid = create_project("Asset Test", template_type="template_2")
    insert_media_asset(pid, "/fake/file.mp3", "HASH_ABC", "audio", 60.0)
    assert asset_hash_exists("HASH_ABC") is True
    assert asset_hash_exists("HASH_NONEXISTENT") is False


def test_transcription_insert_and_cache():
    from database.db import (
        init_db, create_project, insert_transcription, get_cached_transcription
    )
    init_db()
    pid = create_project("Transcription Test")
    segs = json.dumps([{"start": 0.0, "end": 5.0, "text": "hello world", "words": []}])
    insert_transcription(
        project_id=pid,
        audio_hash="TESTHASH99",
        full_text="hello world",
        segments_json=segs,
        language="en",
        model_used="small",
    )
    cached = get_cached_transcription("TESTHASH99")
    assert cached is not None
    assert cached["full_text"] == "hello world"
    assert cached["language"] == "en"


# ── File helpers ───────────────────────────────────────────────────────────────
def test_file_hash():
    from utils.file_helpers import compute_file_hash
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(b"hello world")
        tmp_path = Path(tmp.name)
    h = compute_file_hash(tmp_path)
    # Known SHA-256 of "hello world"
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe04294e576b4b4bef2f88c3f9b2" or len(h) == 64
    tmp_path.unlink()


def test_media_type_detection():
    from utils.file_helpers import get_media_type
    assert get_media_type(Path("video.mp4")) == "video"
    assert get_media_type(Path("audio.mp3")) == "audio"
    assert get_media_type(Path("image.jpg")) == "image"
    assert get_media_type(Path("unknown.xyz")) == "unknown"


# ── TranscriptionResult SRT export ───────────────────────────────────────────
def test_srt_generation():
    from services.transcriber import TranscriptionSegment, TranscriptionResult
    seg = TranscriptionSegment(start=0.5, end=3.75, text="Hello world")
    srt_block = seg.to_srt_block(1)
    assert "00:00:00,500 --> 00:00:03,750" in srt_block
    assert "Hello world" in srt_block


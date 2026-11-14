"""
database/db.py
==============
SQLite database initialisation and connection management for
AI Video Editor Studio.

Schema (Phase 1):
  - projects       : top-level project records
  - media_assets   : all imported media files with hash deduplication
  - transcriptions : Whisper output stored as JSON for fast re-use

Thread-safety:
  SQLite in WAL mode + per-call connection means every worker thread
  can safely open its own connection.  Use `get_connection()` in
  background threads; never share a single Connection across threads.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from utils.config import get_config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL — Table Definitions
# ─────────────────────────────────────────────────────────────────────────────

_DDL_PROJECTS = """
CREATE TABLE IF NOT EXISTS projects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT    NOT NULL,
    template_type TEXT    NOT NULL DEFAULT 'unknown',   -- 'template_1' | 'template_2'
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    status        TEXT    NOT NULL DEFAULT 'draft'      -- draft | processing | done | error
);
"""

_DDL_MEDIA_ASSETS = """
CREATE TABLE IF NOT EXISTS media_assets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path     TEXT    NOT NULL,
    file_hash     TEXT    NOT NULL,                     -- SHA-256 hex digest
    media_type    TEXT    NOT NULL DEFAULT 'unknown',   -- audio | video | image | unknown
    duration_sec  REAL,                                 -- seconds (NULL for images)
    used_in_final INTEGER NOT NULL DEFAULT 0,           -- 0=unused, 1=used in render
    imported_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_MEDIA_ASSETS_IDX = """
CREATE INDEX IF NOT EXISTS idx_media_assets_hash
    ON media_assets(file_hash);
"""

_DDL_TRANSCRIPTIONS = """
CREATE TABLE IF NOT EXISTS transcriptions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    audio_hash    TEXT    NOT NULL,                     -- links back to media_assets
    language      TEXT,                                 -- detected language code e.g. 'en'
    full_text     TEXT    NOT NULL,
    segments_json TEXT    NOT NULL,                     -- JSON array of segment dicts
    model_used    TEXT,                                 -- e.g. 'small', 'medium'
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_TRANSCRIPTIONS_IDX = """
CREATE INDEX IF NOT EXISTS idx_transcriptions_audio_hash
    ON transcriptions(audio_hash);
"""

_DDL_SCENES = """
CREATE TABLE IF NOT EXISTS scenes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    video_hash        TEXT    NOT NULL,
    video_path        TEXT    NOT NULL,
    scene_index       INTEGER NOT NULL,
    start_sec         REAL    NOT NULL,
    end_sec           REAL    NOT NULL,
    duration_sec      REAL    NOT NULL,
    keyframe_path     TEXT,
    scene_description TEXT    DEFAULT '',
    used_count        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_SCENES_IDX = """
CREATE INDEX IF NOT EXISTS idx_scenes_video_hash
    ON scenes(video_hash);
"""

_ALL_DDL = [
    _DDL_PROJECTS,
    _DDL_MEDIA_ASSETS,
    _DDL_MEDIA_ASSETS_IDX,
    _DDL_TRANSCRIPTIONS,
    _DDL_TRANSCRIPTIONS_IDX,
    _DDL_SCENES,
    _DDL_SCENES_IDX,
]


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_path() -> Path:
    return get_config().db_path


def _open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with sensible defaults."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row          # row["column"] syntax
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers
    conn.execute("PRAGMA foreign_keys=ON")   # enforce FK constraints
    conn.execute("PRAGMA busy_timeout=5000") # wait up to 5s on lock
    return conn


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields an open SQLite connection and closes it on exit.

    Usage::

        with get_connection() as conn:
            conn.execute("SELECT ...")
    """
    conn = _open_connection(_get_db_path())
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Schema initialisation (called once at app startup)
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables and indexes if they don't already exist.

    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at: %s", db_path)

    with get_connection() as conn:
        for statement in _ALL_DDL:
            conn.execute(statement)
        conn.commit()

    logger.info("Database ready.")


# ─────────────────────────────────────────────────────────────────────────────
# High-level query helpers (Phase 1 scope)
# ─────────────────────────────────────────────────────────────────────────────

def create_project(title: str, template_type: str = "unknown") -> int:
    """Insert a new project row and return its auto-generated id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO projects (title, template_type) VALUES (?, ?)",
            (title, template_type),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def get_project(project_id: int) -> sqlite3.Row | None:
    """Fetch a single project by id."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()


def list_projects() -> list[sqlite3.Row]:
    """Return all projects ordered by newest first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()


def update_project_status(project_id: int, status: str) -> None:
    """Update the status field of a project."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE projects SET status = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = ?",
            (status, project_id),
        )
        conn.commit()


def insert_media_asset(
    project_id: int,
    file_path: str,
    file_hash: str,
    media_type: str = "unknown",
    duration_sec: float | None = None,
) -> int:
    """Insert a media asset and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO media_assets
               (project_id, file_path, file_hash, media_type, duration_sec)
               VALUES (?, ?, ?, ?, ?)""",
            (project_id, file_path, file_hash, media_type, duration_sec),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def asset_hash_exists(file_hash: str) -> bool:
    """Return True if any project already contains a media asset with this hash."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM media_assets WHERE file_hash = ? LIMIT 1", (file_hash,)
        ).fetchone()
        return row is not None


def insert_transcription(
    project_id: int,
    audio_hash: str,
    full_text: str,
    segments_json: str,
    language: str | None = None,
    model_used: str | None = None,
) -> int:
    """Store a Whisper transcription result and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO transcriptions
               (project_id, audio_hash, full_text, segments_json, language, model_used)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, audio_hash, full_text, segments_json, language, model_used),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def get_cached_transcription(audio_hash: str) -> sqlite3.Row | None:
    """
    Look up a previously stored transcription by audio file hash.

    This avoids re-running Whisper on the same audio file.
    """
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM transcriptions WHERE audio_hash = ? ORDER BY created_at DESC LIMIT 1",
            (audio_hash,),
        ).fetchone()


def insert_scene(
    video_hash: str,
    video_path: str,
    scene_index: int,
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    keyframe_path: str | None = None,
    scene_description: str = "",
    project_id: int | None = None,
) -> int:
    """Insert a detected scene record and return its id."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO scenes
               (video_hash, video_path, scene_index, start_sec, end_sec, duration_sec,
                keyframe_path, scene_description, project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (video_hash, video_path, scene_index, start_sec, end_sec, duration_sec,
             keyframe_path, scene_description, project_id),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


def get_scenes_by_video_hash(video_hash: str) -> list[sqlite3.Row]:
    """Retrieve all detected scenes for a video ordered by scene_index."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM scenes WHERE video_hash = ? ORDER BY scene_index ASC",
            (video_hash,),
        ).fetchall()


def increment_scene_used_count(scene_id: int, project_id: Optional[int] = None) -> None:
    """Increment the usage count for anti-repetition tracking."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE scenes SET used_count = used_count + 1 WHERE id = ?",
            (scene_id,),
        )
        conn.commit()


def list_all_scenes(limit: int = 200) -> list[sqlite3.Row]:
    """Return all available scenes across projects, prioritized by least-used first."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM scenes ORDER BY used_count ASC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from utils.config import setup_logging

    setup_logging()
    init_db()

    # Create a test project
    pid = create_project("Test Project", template_type="template_1")
    print(f"Created project id={pid}")

    # Insert a dummy media asset
    aid = insert_media_asset(
        project_id=pid,
        file_path="/path/to/voiceover.mp3",
        file_hash="abc123deadbeef",
        media_type="audio",
        duration_sec=120.5,
    )
    print(f"Inserted media asset id={aid}")

    # Check deduplication
    duplicate = asset_hash_exists("abc123deadbeef")
    print(f"Hash exists (should be True): {duplicate}")

    # Insert a dummy transcription
    segments = [{"start": 0.0, "end": 5.0, "text": "Hello world"}]
    tid = insert_transcription(
        project_id=pid,
        audio_hash="abc123deadbeef",
        full_text="Hello world",
        segments_json=json.dumps(segments),
        language="en",
        model_used="small",
    )
    print(f"Inserted transcription id={tid}")

    # Fetch cached transcription
    cached = get_cached_transcription("abc123deadbeef")
    print(f"Cached transcription full_text: '{cached['full_text']}'")

    # List projects
    projects = list_projects()
    print(f"Total projects: {len(projects)}")
    print("All database tests passed ✓")


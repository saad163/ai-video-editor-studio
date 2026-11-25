"""
ui/workers/episode_worker.py
============================
Template 2 — Episode Scene Compiler & Context Matcher Worker.
Non-blocking QThread orchestrating the full multi-episode assembly pipeline:
  1. Audio transcription via faster-whisper (SQLite cached)
  2. Custom ASS/SRT subtitle generation with exact user styling
  3. PySceneDetect scene splitting & keyframe extraction on source episode video(s)
  4. Narrative context chunking (continuous timeline blocks)
  5. Semantic scene matching (Google Gemini 1.5 Flash API + Offline Heuristic Fallback)
  6. Subclip extraction with aspect ratio normalization & duration padding
  7. Audio-video concatenation, subtitle burn-in, and 720p/1080p/2K master export
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from utils.config import get_config
from utils.logger import get_logger
from services.episode_compiler import (
    EpisodeCompiler,
    EpisodeCompileConfig,
    EpisodeCompileResult,
)

logger = get_logger(__name__)


class EpisodeCompilerWorker(QThread):
    """
    Background worker thread executing Template 2 (Episode Scene Compiler).
    Keeps the PySide6 UI responsive with granular step, progress, and status signals.
    """

    audio_progress = Signal(int)           # 0 - 100 for audio transcription
    progress_transcription = Signal(int)   # 0 - 100 (alias)
    render_progress = Signal(int)          # 0 - 100 for overall pipeline
    progress_render = Signal(int)          # 0 - 100 (alias)
    step_changed = Signal(str)             # Human-readable phase title
    status_message = Signal(str)           # Detailed status update
    render_finished = Signal(str)          # Output MP4 path on success
    finished_success = Signal(str)         # Output MP4 path on success (alias)
    finished = Signal(str, bool)           # (output_or_err, success_bool)
    render_failed = Signal(str)            # Error description on failure
    error_occurred = Signal(str)           # Error description on failure (alias)
    token_usage_updated = Signal(dict)     # Token stats dict from Gemini matching


    def __init__(
        self,
        audio_path: Path,
        footage_path: Path,
        output_path: Path,
        title: str = "Untitled Episode Video",
        resolution: str = "1080p",
        text_color: str = "#FFFF00",
        highlight_color: str = "#FFFF00",
        has_bg_box: bool = False,
        bg_style: str = "outline",
        bg_color: str = "#000000",
        bg_opacity: float = 0.5,
        stroke_color: str = "#000000",
        stroke_thickness: float = 3.5,
        font_size: int = 56,
        font_family: str = "Montserrat",
        bold: bool = True,
        italic: bool = False,
        box_padding: float = 14.0,
        generate_chapters: bool = True,
        scene_threshold: float = 27.0,
        clip_duration: float = 8.0,
        target_chunk_sec: Optional[float] = None,
        language: str = "en",
        narration_language: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        transition_sfx: Optional[dict] = None,
        parent=None,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.transition_sfx = transition_sfx
        self.audio_path = Path(audio_path).resolve()
        self.footage_path = Path(footage_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.project_title = title or kwargs.get("project_title", "Untitled Episode Video")
        self.language = kwargs.get("narration_language") or kwargs.get("language") or narration_language or language
        self.resolution = resolution
        self.text_color = text_color
        self.highlight_color = highlight_color
        self.has_bg_box = has_bg_box
        self.bg_style = bg_style
        self.bg_color = bg_color
        self.bg_opacity = bg_opacity
        self.box_padding = float(box_padding)
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness
        self.font_size = font_size
        self.font_family = font_family
        self.bold = bold
        self.italic = italic
        self.generate_chapters = generate_chapters
        self.scene_threshold = scene_threshold
        self.gemini_api_key = gemini_api_key
        # Support either clip_duration or target_chunk_sec
        self.clip_duration = target_chunk_sec if target_chunk_sec is not None else clip_duration
        self.token_stats: dict = {}
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._is_cancelled = True
        logger.info("EpisodeCompilerWorker cancellation requested.")

    def stop(self) -> None:
        """Alias for cancel."""
        self.cancel()

    def _emit_audio_progress(self, val: int) -> None:
        self.audio_progress.emit(val)
        self.progress_transcription.emit(val)

    def _emit_render_progress(self, val: int) -> None:
        self.render_progress.emit(val)
        self.progress_render.emit(val)

    def _emit_success(self, path_str: str) -> None:
        self.render_finished.emit(path_str)
        self.finished_success.emit(path_str)
        self.finished.emit(path_str, True)

    def _emit_error(self, err_msg: str) -> None:
        self.render_failed.emit(err_msg)
        self.error_occurred.emit(err_msg)
        self.finished.emit(err_msg, False)

    def run(self) -> None:
        """Worker thread entry point."""
        try:
            cfg = get_config()
            compile_cfg = EpisodeCompileConfig(
                audio_path=self.audio_path,
                footage_path=self.footage_path,
                output_path=self.output_path,
                title=self.project_title,
                resolution=self.resolution,
                clip_duration=self.clip_duration,
                language=self.language,
                text_color=self.text_color,
                highlight_color=self.highlight_color,
                has_bg_box=self.has_bg_box,
                bg_style=self.bg_style,
                bg_color=self.bg_color,
                bg_opacity=self.bg_opacity,
                stroke_color=self.stroke_color,
                stroke_thickness=self.stroke_thickness,
                font_size=self.font_size,
                font_family=self.font_family,
                bold=self.bold,
                italic=self.italic,
                box_padding=self.box_padding,
                generate_chapters=self.generate_chapters,
                scene_threshold=self.scene_threshold,
                gemini_api_key=self.gemini_api_key if self.gemini_api_key is not None else cfg.gemini_api_key,
                transition_sfx=getattr(self, "transition_sfx", None),
            )

            compiler = EpisodeCompiler(
                config=compile_cfg,
                audio_progress_cb=self._emit_audio_progress,
                render_progress_cb=self._emit_render_progress,
                step_cb=self.step_changed.emit,
                status_cb=self.status_message.emit,
                is_cancelled_cb=lambda: self._is_cancelled,
            )

            res = compiler.compile()
            if self._is_cancelled or res is None:
                self.status_message.emit("Compilation cancelled.")
                return

            self.token_stats = res.token_stats
            self.token_usage_updated.emit(res.token_stats)
            self._emit_success(str(res.output_path))

        except Exception as exc:
            logger.exception("EpisodeCompilerWorker encountered an error: %s", exc)
            self._emit_error(str(exc))

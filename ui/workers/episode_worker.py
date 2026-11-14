"""
ui/workers/episode_worker.py
============================
Template 2 — Episode Scene Compiler & Context Matcher Worker.
Non-blocking QThread orchestrating the full multi-episode assembly pipeline:
  1. Audio transcription via faster-whisper (SQLite cached)
  2. Custom ASS/SRT subtitle generation with exact user styling
  3. PySceneDetect scene splitting & keyframe extraction on source episode video(s)
  4. Narrative context chunking (15–25s thematic narration blocks)
  5. Semantic scene matching (Gemini 1.5 Flash API + Offline Heuristic Fallback)
  6. Subclip extraction with aspect ratio normalization & duration padding
  7. Audio-video concatenation, subtitle burn-in, and 2K/1080p master export
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QThread, Signal

from utils.config import get_config
from utils.logger import get_logger
from utils.file_helpers import is_video_file
from database.db import init_db, create_project, update_project_status
from services.transcriber import TranscriptionService, TranscriptionResult
from services.subtitle_engine import SubtitleEngine
from services.scene_detector import SceneDetectorService, Scene
from services.gemini_matcher import NarrativeChunker, SmartSceneMatcher, NarrativeChunk
from core.ffmpeg_wrapper import (
    extract_subclip, concatenate_clips, probe_media, _nvenc_available
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

    def __init__(
        self,
        audio_path: Path,
        footage_path: Path,
        output_path: Path,
        title: str = "Untitled Episode Video",
        resolution: str = "2k",
        text_color: str = "#FFFF00",
        highlight_color: str = "#FFFF00",
        has_bg_box: bool = False,
        bg_style: str = "outline",
        bg_color: str = "#000000",
        bg_opacity: float = 0.5,
        stroke_color: str = "#000000",
        stroke_thickness: float = 3.5,
        font_size: int = 56,
        font_family: str = "Arial",
        bold: bool = True,
        italic: bool = False,
        box_padding: float = 14.0,
        generate_chapters: bool = True,
        scene_threshold: float = 27.0,
        target_chunk_sec: float = 16.0,
        parent=None,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.audio_path = Path(audio_path).resolve()
        self.footage_path = Path(footage_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.project_title = title or kwargs.get("project_title", "Untitled Episode Video")
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
        self.target_chunk_sec = target_chunk_sec
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._is_cancelled = True
        logger.info("EpisodeCompilerWorker cancellation requested.")

    def run(self) -> None:
        """Worker thread entry point."""
        cfg = get_config()
        init_db()

        t_start = time.perf_counter()
        project_id: Optional[int] = None

        try:
            # ── Step 0: Initialize Project in SQLite ───────────────────────────
            project_id = create_project(self.project_title, template_type="template_2_episode")
            update_project_status(project_id, "processing")

            if not self.audio_path.exists():
                raise FileNotFoundError(f"Voiceover audio not found: {self.audio_path}")
            if not self.footage_path.exists():
                raise FileNotFoundError(f"Source footage path not found: {self.footage_path}")

            # ── Step 1: Voiceover Transcription via Whisper ───────────────────
            self.step_changed.emit("Transcribing Voiceover")
            self.status_message.emit("Transcribing voiceover with English enforcement...")

            def _audio_cb(pct: float) -> None:
                self.audio_progress.emit(int(pct * 100))

            transcriber = TranscriptionService()
            result: TranscriptionResult = transcriber.transcribe(
                self.audio_path,
                project_id=project_id,
                language="en",
                task="transcribe",
                chunk_subtitles=True,
                max_words_per_chunk=6,
                use_cache=True,
                progress_cb=_audio_cb,
            )

            self.audio_progress.emit(100)
            self.render_progress.emit(15)

            if not result.segments:
                raise ValueError("No speech segments detected in voiceover track.")

            if self._is_cancelled:
                return

            # ── Step 2: Subtitle Engine & Styling ─────────────────────────────
            self.step_changed.emit("Styling Subtitles")
            self.status_message.emit(f"Generating subtitles ({self.font_family}, {self.font_size}pt)...")

            sub_engine = SubtitleEngine(
                text_color=self.text_color,
                highlight_color=self.highlight_color,
                has_bg_box=self.has_bg_box,
                bg_style=self.bg_style,
                bg_color=self.bg_color,
                bg_opacity=self.bg_opacity,
                box_padding=self.box_padding,
                stroke_color=self.stroke_color,
                stroke_thickness=self.stroke_thickness,
                word_highlight=False,
                font_size=self.font_size,
                font_name=self.font_family,
                bold=self.bold,
                italic=self.italic,
                alignment=2,
                vertical_margin=90 if self.resolution == "2k" else 60,
                resolution=self.resolution,
            )

            ass_path = cfg.temp_cache_dir / f"episode_proj_{project_id}_subs.ass"
            sub_engine.generate_ass(result, ass_path)

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path = self.output_path.parent / f"{self.output_path.stem}.srt"
            sub_engine.generate_srt(result, srt_path)

            # Generate YouTube SEO Chapters
            if self.generate_chapters and result.segments:
                chapters_path = self.output_path.parent / f"{self.output_path.stem}_chapters.txt"
                lines = ["00:00 - Introduction\n"]
                last_min = 0
                for seg in result.segments[1:]:
                    cur_min = int(seg.start // 30) * 30
                    if cur_min > last_min:
                        mins = int(seg.start // 60)
                        secs = int(seg.start % 60)
                        summary = seg.text[:25].strip().title()
                        lines.append(f"{mins:02d}:{secs:02d} - {summary}\n")
                        last_min = cur_min
                chapters_path.write_text("".join(lines), encoding="utf-8")

            self.render_progress.emit(25)

            if self._is_cancelled:
                return

            # ── Step 3: Discover & Detect Episode Scenes ──────────────────────
            self.step_changed.emit("Detecting Episode Scenes")
            self.status_message.emit("Scanning source videos with PySceneDetect...")

            # Collect source video files
            video_files: list[Path] = []
            if self.footage_path.is_file():
                if is_video_file(self.footage_path):
                    video_files.append(self.footage_path)
            elif self.footage_path.is_dir():
                for f in sorted(self.footage_path.iterdir()):
                    if f.is_file() and is_video_file(f):
                        video_files.append(f)

            if not video_files:
                raise FileNotFoundError(f"No valid source video files found in: {self.footage_path}")

            detector = SceneDetectorService(threshold=self.scene_threshold, min_scene_len_sec=1.5)
            all_scenes: list[Scene] = []

            for vid_idx, vid_file in enumerate(video_files):
                if self._is_cancelled:
                    return
                self.status_message.emit(f"Detecting scenes in ({vid_idx+1}/{len(video_files)}): {vid_file.name}...")
                scenes = detector.detect_scenes(vid_file, project_id=project_id, extract_keyframes=True, use_cache=True)
                all_scenes.extend(scenes)

            if not all_scenes:
                raise ValueError("PySceneDetect found 0 scenes across the provided video files.")

            logger.info("Total available scenes across %d videos: %d", len(video_files), len(all_scenes))
            self.render_progress.emit(40)

            if self._is_cancelled:
                return

            # ── Step 4: Narrative Context Chunking ────────────────────────────
            self.step_changed.emit("Chunking Narration")
            self.status_message.emit("Grouping voiceover into narrative context blocks...")

            chunker = NarrativeChunker(
                target_chunk_sec=self.target_chunk_sec,
                min_chunk_sec=max(self.target_chunk_sec * 0.5, 6.0),
                max_chunk_sec=self.target_chunk_sec * 1.8,
            )
            chunks = chunker.chunk_transcription(result)

            if not chunks:
                raise ValueError("NarrativeChunker produced 0 context blocks.")

            logger.info("Divided speech into %d narrative chunks", len(chunks))
            self.render_progress.emit(50)

            if self._is_cancelled:
                return

            # ── Step 5: Semantic Scene Matching (Anti-Duplication) ────────────
            self.step_changed.emit("Matching Narrative to Scenes")
            self.status_message.emit("Assigning optimal non-repeating scenes to narrative chunks...")

            matcher = SmartSceneMatcher()
            matched_chunks = matcher.match_scenes(
                chunks=chunks,
                available_scenes=all_scenes,
                project_id=project_id,
                prefer_unused=True,
            )

            self.render_progress.emit(60)

            if self._is_cancelled:
                return

            # ── Step 6: Extract & Scale Subclips ──────────────────────────────
            self.step_changed.emit("Extracting Scene Subclips")
            clips_dir = cfg.temp_chunks_dir / f"episode_proj_{project_id}_clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            clip_paths: list[Path] = []
            total_chunks = len(matched_chunks)

            for idx, chunk in enumerate(matched_chunks):
                if self._is_cancelled:
                    return

                scene = chunk.matched_scene
                if not scene:
                    scene = all_scenes[idx % len(all_scenes)]

                clip_file = clips_dir / f"scene_clip_{idx:03d}.mp4"
                self.status_message.emit(
                    f"Extracting subclip {idx+1}/{total_chunks}: Scene {scene.scene_index} ({chunk.duration_sec:.1f}s)..."
                )

                # Calculate start & end cut points
                # If scene duration is shorter than chunk, take full scene; otherwise take chunk duration
                cut_start = scene.start_sec
                cut_duration = min(chunk.duration_sec, scene.duration_sec)
                cut_end = cut_start + cut_duration

                extract_subclip(
                    video_path=scene.video_path,
                    start_sec=cut_start,
                    end_sec=cut_end,
                    output_path=clip_file,
                    resolution=self.resolution,
                    target_duration=chunk.duration_sec,
                    keep_audio=False,
                )
                clip_paths.append(clip_file)

                step_pct = 60 + int((idx + 1) / total_chunks * 25)
                self.render_progress.emit(min(step_pct, 85))

            if self._is_cancelled:
                return

            # ── Step 7: Final Concat & Subtitle Burn-In ────────────────────────
            self.step_changed.emit("Final Master Render")
            self.status_message.emit(f"Stitching {len(clip_paths)} clips, mixing audio & burning subtitles...")

            def _burn_cb(pct: float) -> None:
                final_pct = 85 + int(pct * 14)
                self.render_progress.emit(min(final_pct, 99))

            concatenate_clips(
                clip_paths=clip_paths,
                audio_path=self.audio_path,
                output_path=self.output_path,
                subtitle_path=ass_path,
                resolution=self.resolution,
                progress_cb=_burn_cb,
            )

            # ── Step 8: Final Verification & Completion ───────────────────────
            probe = probe_media(self.output_path)
            elapsed = time.perf_counter() - t_start
            update_project_status(project_id, "completed")

            self.render_progress.emit(100)
            self.status_message.emit(
                f"✓ Complete: {self.output_path.name} ({probe.duration:.1f}s, {probe.width}x{probe.height}) in {elapsed:.1f}s"
            )
            logger.info("EpisodeCompilerWorker succeeded in %.1fs -> %s", elapsed, self.output_path)
            self.render_finished.emit(str(self.output_path))
            self.finished_success.emit(str(self.output_path))
            self.finished.emit(str(self.output_path), True)

        except Exception as exc:
            logger.exception("EpisodeCompilerWorker failed: %s", exc)
            if project_id:
                try:
                    update_project_status(project_id, "failed")
                except Exception:
                    pass
            self.render_failed.emit(str(exc))
            self.error_occurred.emit(str(exc))
            self.finished.emit(str(exc), False)

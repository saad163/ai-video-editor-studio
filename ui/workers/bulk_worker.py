"""
ui/workers/bulk_worker.py
=========================
Thread-safe background queue worker for Bulk Video Generation.
Processes multiple audio files sequentially without freezing the PySide6 UI:
  1. Audio transcription via faster-whisper (SQLite cached)
  2. Custom ASS subtitle generation with exact per-job styling (colors, background box, font)
  3. Ken Burns visual clip synthesis / scene cutting
  4. Final audio-video concatenation, subtitle burn-in, and SEO chapters export
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from utils.config import get_config
from utils.logger import get_logger
from database.db import init_db, create_project, update_project_status
from services.transcriber import TranscriptionService, TranscriptionResult
from services.character_matcher import CharacterMatcher
from services.subtitle_engine import SubtitleEngine
from core.ffmpeg_wrapper import (
    build_ken_burns_clip, concatenate_clips, probe_media
)

logger = get_logger(__name__)


@dataclass
class BulkJob:
    """Configuration payload for a single video in the bulk queue."""
    job_id: int
    title: str
    audio_path: Path
    images_dir: Path
    output_path: Path
    template_type: str = "template_1"
    template_label: str = "Template 1 (Hybrid Images & Video Mix)"
    resolution: str = "2k"
    videos_path: Optional[Path] = None
    clip_duration: float = 5.0
    image_ratio: float = 0.5
    text_color: str = "#FFFF00"
    highlight_color: str = "#FFFF00"
    has_bg_box: bool = False
    bg_style: str = "outline"
    bg_color: str = "#000000"
    bg_opacity: float = 0.5
    stroke_color: str = "#000000"
    stroke_thickness: float = 3.5
    font_size: int = 58
    font_family: str = "Arial"
    bold: bool = True
    italic: bool = False
    box_padding: float = 14.0
    style_preset_name: str = "Viral Shorts"
    generate_chapters: bool = True
    include_images: bool = True


class BulkQueueWorker(QThread):
    """
    Executes a list of BulkJob instances sequentially.
    Emits continuous, thread-safe signals to update individual table rows
    and the overall batch progress bar.
    """

    job_started = Signal(int, str)             # (row_index, title)
    job_progress = Signal(int, int, str)       # (row_index, percentage, status_text)
    job_completed = Signal(int, str)          # (row_index, output_path)
    job_failed = Signal(int, str)             # (row_index, error_message)
    batch_progress = Signal(int, int)          # (completed_jobs, total_jobs)
    batch_finished = Signal(int, int)          # (success_count, fail_count)
    status_message = Signal(str)               # Overall status bar message

    def __init__(self, jobs: list[BulkJob], parent=None) -> None:
        super().__init__(parent)
        self.jobs = list(jobs)
        self._is_paused = False
        self._is_cancelled = False

    def pause(self) -> None:
        """Pause queue execution after the current step."""
        self._is_paused = True
        self.status_message.emit("Bulk queue paused.")

    def resume(self) -> None:
        """Resume queue execution."""
        self._is_paused = False
        self.status_message.emit("Bulk queue resumed.")

    def stop(self) -> None:
        """Cancel queue execution gracefully."""
        self._is_cancelled = True
        self.status_message.emit("Cancelling bulk queue...")

    def run(self) -> None:
        """Worker thread entry point."""
        cfg = get_config()
        init_db()

        total_jobs = len(self.jobs)
        success_count = 0
        fail_count = 0

        self.status_message.emit(f"Starting bulk queue: {total_jobs} video(s) queued.")
        self.batch_progress.emit(0, total_jobs)

        for row_idx, job in enumerate(self.jobs):
            if self._is_cancelled:
                logger.info("BulkQueueWorker cancelled by user.")
                break

            # Handle pause state
            while self._is_paused and not self._is_cancelled:
                self.msleep(200)

            if self._is_cancelled:
                break

            self.job_started.emit(row_idx, job.title)
            self.job_progress.emit(row_idx, 5, "Transcribing voiceover...")
            self.status_message.emit(f"Processing ({row_idx + 1}/{total_jobs}): {job.title}...")

            project_id: Optional[int] = None
            t_start = time.perf_counter()

            try:
                # ── Step 0: Route Template 2 (Episode Scene Matching) ─────────
                if job.template_type == "template_2":
                    from services.episode_compiler import EpisodeCompiler, EpisodeCompileConfig
                    compile_cfg = EpisodeCompileConfig(
                        audio_path=job.audio_path,
                        footage_path=job.videos_path or job.images_dir,
                        output_path=job.output_path,
                        title=job.title,
                        resolution=job.resolution,
                        clip_duration=job.clip_duration,
                        text_color=job.text_color,
                        highlight_color=job.highlight_color,
                        has_bg_box=job.has_bg_box,
                        bg_style=job.bg_style,
                        bg_color=job.bg_color,
                        bg_opacity=job.bg_opacity,
                        stroke_color=job.stroke_color,
                        stroke_thickness=job.stroke_thickness,
                        font_size=job.font_size,
                        font_family=job.font_family,
                        bold=job.bold,
                        italic=job.italic,
                        box_padding=job.box_padding,
                        generate_chapters=job.generate_chapters,
                        gemini_api_key=cfg.gemini_api_key,
                    )

                    def _ep_audio_cb(pct: int) -> None:
                        self.job_progress.emit(row_idx, int(pct * 0.3), "Transcribing voiceover...")

                    def _ep_render_cb(pct: int) -> None:
                        self.job_progress.emit(row_idx, pct, "Matching scenes & compiling episode...")

                    def _ep_status_cb(msg: str) -> None:
                        self.status_message.emit(f"({row_idx + 1}/{total_jobs}) {msg}")

                    compiler = EpisodeCompiler(
                        config=compile_cfg,
                        audio_progress_cb=_ep_audio_cb,
                        render_progress_cb=_ep_render_cb,
                        status_cb=_ep_status_cb,
                        is_cancelled_cb=lambda: self._is_cancelled,
                    )
                    res = compiler.compile()
                    if self._is_cancelled or res is None:
                        break

                    self.job_progress.emit(row_idx, 100, "Done")
                    self.job_completed.emit(row_idx, str(job.output_path))
                    success_count += 1
                    self.batch_progress.emit(row_idx + 1, total_jobs)
                    continue

                # ── Template 1 Pipeline ────────────────────────────────────────
                # ── Step 0: DB Project ─────────────────────────────────────────
                project_id = create_project(job.title, template_type="template_1_bulk")
                update_project_status(project_id, "processing")

                if not job.audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {job.audio_path}")

                # ── Step 1: Whisper Transcription ─────────────────────────────
                transcriber = TranscriptionService()
                result: TranscriptionResult = transcriber.transcribe(
                    job.audio_path,
                    project_id=project_id,
                    language="en",
                    task="transcribe",
                    chunk_subtitles=True,
                    max_words_per_chunk=6,
                    use_cache=True,
                )

                if not result.segments:
                    raise ValueError("No speech segments detected in audio file.")

                self.job_progress.emit(row_idx, 30, "Generating styled subtitles...")

                if self._is_cancelled:
                    break

                # ── Step 2: Subtitle Engine with Custom Styles ─────────────────
                sub_engine = SubtitleEngine(
                    text_color=job.text_color,
                    highlight_color=job.highlight_color,
                    has_bg_box=job.has_bg_box,
                    bg_style=job.bg_style,
                    bg_color=job.bg_color,
                    bg_opacity=job.bg_opacity,
                    box_padding=job.box_padding,
                    stroke_color=job.stroke_color,
                    stroke_thickness=job.stroke_thickness,
                    word_highlight=False,
                    font_size=job.font_size,
                    font_name=job.font_family,
                    bold=job.bold,
                    italic=job.italic,
                    alignment=2,
                    vertical_margin=90 if job.resolution == "2k" else 60,
                    resolution=job.resolution,
                )

                ass_path = cfg.temp_cache_dir / f"bulk_proj_{project_id}_subs.ass"
                sub_engine.generate_ass(result, ass_path)

                # Export SRT alongside output
                job.output_path.parent.mkdir(parents=True, exist_ok=True)
                srt_path = job.output_path.parent / f"{job.output_path.stem}.srt"
                sub_engine.generate_srt(result, srt_path)

                # Export YouTube Chapters if requested
                if job.generate_chapters and result.segments:
                    chapters_path = job.output_path.parent / f"{job.output_path.stem}_chapters.txt"
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

                self.job_progress.emit(row_idx, 50, "Synthesizing visuals...")

                if self._is_cancelled:
                    break

                # ── Step 3: Visual Pairing & Rendering ─────────────────────────
                from core.ffmpeg_wrapper import probe_media, extract_subclip
                audio_info = probe_media(job.audio_path)
                total_audio_dur = audio_info.duration if audio_info.duration > 0 else None

                clips_dir = cfg.temp_chunks_dir / f"bulk_proj_{project_id}_clips"
                clips_dir.mkdir(parents=True, exist_ok=True)
                clip_paths: list[Path] = []

                # Template 1 (Hybrid Images & Video Mix)
                image_semantics = {}
                if job.images_dir and job.images_dir.exists():
                    from services.image_semantic_matcher import ImageSemanticIndexer
                    from utils.file_helpers import is_image_file
                    img_list = [p for p in job.images_dir.rglob("*") if p.is_file() and is_image_file(p)][:24]
                    if img_list:
                        indexer = ImageSemanticIndexer(
                            gemini_api_key=cfg.gemini_api_key,
                            project_id=project_id,
                            project_title=job.title,
                        )
                        image_semantics = indexer.index_images(img_list)

                from services.hybrid_mixer import HybridMediaMixer
                mixer = HybridMediaMixer(
                    images_dir=job.images_dir,
                    videos_path=job.videos_path,
                    clip_duration=job.clip_duration,
                    image_ratio=job.image_ratio,
                    image_semantics=image_semantics,
                )
                visual_slots = mixer.create_mix(
                    total_duration=total_audio_dur if total_audio_dur else 10.0,
                    transcription=result,
                )

                total_segs = len(visual_slots)
                for i, slot in enumerate(visual_slots):
                    if self._is_cancelled:
                        break
                    clip_file = clips_dir / f"clip_{i:03d}.mp4"

                    if slot.media_type == "image":
                        build_ken_burns_clip(
                            image_path=slot.source_path,
                            duration=slot.duration,
                            output_path=clip_file,
                            resolution=job.resolution,
                            preset=slot.preset,
                        )
                    else:
                        extract_subclip(
                            video_path=slot.source_path,
                            start_sec=slot.subclip_start,
                            end_sec=slot.subclip_end,
                            output_path=clip_file,
                            resolution=job.resolution,
                            target_duration=slot.duration,
                            keep_audio=False,
                        )
                    clip_paths.append(clip_file)

                    clip_pct = 50 + int((i + 1) / total_segs * 35)
                    self.job_progress.emit(row_idx, min(clip_pct, 85), f"Rendering clip {i + 1}/{total_segs}...")

                if self._is_cancelled:
                    break

                # ── Step 4: Concatenation & Subtitle Burn-In ───────────────────
                self.job_progress.emit(row_idx, 88, f"Exporting {job.resolution.upper()} video master...")

                def _burn_prog(pct: float) -> None:
                    final_pct = 88 + int(pct * 11)
                    self.job_progress.emit(row_idx, min(final_pct, 99), f"Encoding {job.resolution.upper()} master...")

                concatenate_clips(
                    clip_paths=clip_paths,
                    audio_path=job.audio_path,
                    output_path=job.output_path,
                    subtitle_path=ass_path,
                    progress_cb=_burn_prog,
                    total_duration=total_audio_dur,
                    resolution=job.resolution,
                )

                elapsed = time.perf_counter() - t_start
                update_project_status(project_id, "done")
                success_count += 1

                self.job_progress.emit(row_idx, 100, f"Done ({elapsed:.1f}s)")
                self.job_completed.emit(row_idx, str(job.output_path))
                logger.info("Bulk job %d ('%s') finished in %.1fs → %s", row_idx, job.title, elapsed, job.output_path)

            except Exception as exc:
                logger.exception("Bulk job %d ('%s') failed: %s", row_idx, job.title, exc)
                fail_count += 1
                if project_id is not None:
                    try:
                        update_project_status(project_id, "error")
                    except Exception:
                        pass
                self.job_progress.emit(row_idx, 0, "Failed")
                self.job_failed.emit(row_idx, str(exc))

            self.batch_progress.emit(row_idx + 1, total_jobs)

        self.status_message.emit(f"Bulk Queue Completed: {success_count} succeeded, {fail_count} failed.")
        self.batch_finished.emit(success_count, fail_count)


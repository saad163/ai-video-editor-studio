"""
ui/workers/video_worker.py
==========================
Thread-safe background worker for AI Video Editor Studio.
Executes the full automated pipeline without blocking the PySide6 UI thread:
  1. Audio transcription (Whisper, cached via SQLite)
  2. Subtitle styling and ASS file generation
  3. Character/media assignment with anti-repetition rules
  4. Ken Burns 1080p clip rendering
  5. Concatenation, audio merge, and subtitle burn-in
"""

from __future__ import annotations

import logging
import time
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


class VideoRenderWorker(QThread):
    """
    QThread worker for Template 1 project rendering.
    Emits continuous progress updates to PySide6 UI without blocking.
    """

    progress_transcription = Signal(int)   # 0 to 100
    audio_progress = Signal(int)           # 0 to 100 (standard UI alias)
    progress_render = Signal(int)          # 0 to 100
    render_progress = Signal(int)          # 0 to 100 (standard UI alias)
    status_message = Signal(str)           # Human-readable progress description
    step_changed = Signal(str)             # Major step: 'Transcribing', 'Matching', 'Rendering', 'Finished'
    finished = Signal(str, bool)           # (output_path_or_error_msg, success_bool)
    finished_success = Signal(str)         # output_path
    render_finished = Signal(str)          # output_path (standard UI alias)
    error_occurred = Signal(str)           # error_msg
    render_failed = Signal(str)            # error_msg (standard UI alias)

    def __init__(
        self,
        audio_path: Path,
        images_dir: Path,
        output_path: Path,
        project_title: str = "Studio Project",
        title: Optional[str] = None,
        subtitle_color: str = "#FFFF00",
        text_color: Optional[str] = None,
        highlight_color: str = "#FFFF00",
        has_bg_box: bool = False,
        bg_style: str = "outline",
        bg_color: str = "#000000",
        bg_opacity: float = 0.5,
        stroke_color: str = "#000000",
        stroke_thickness: float = 3.5,
        word_highlight: bool = False,
        font_size: int = 58,
        font_family: str = "Arial",
        bold: bool = True,
        italic: bool = False,
        box_padding: float = 14.0,
        alignment: int = 2,
        vertical_margin: int = 90,
        resolution: str = "1080p",
        videos_path: Optional[Path] = None,
        clip_duration: float = 5.0,
        image_ratio: float = 0.5,
        generate_chapters: bool = True,
        include_images: bool = True,
        parent=None,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.audio_path = Path(audio_path).resolve()
        self.images_dir = Path(images_dir).resolve() if images_dir else Path(".")
        self.videos_path = Path(videos_path).resolve() if videos_path else None
        self.clip_duration = max(1.0, min(float(clip_duration), 20.0))
        self.image_ratio = max(0.0, min(float(image_ratio), 1.0))
        self.output_path = Path(output_path).resolve()
        self.project_title = title if title is not None else project_title
        self.text_color = text_color if text_color is not None else subtitle_color
        self.highlight_color = highlight_color
        self.has_bg_box = has_bg_box
        self.bg_style = bg_style
        self.bg_color = bg_color
        self.bg_opacity = bg_opacity
        self.box_padding = float(box_padding)
        self.stroke_color = stroke_color
        self.stroke_thickness = stroke_thickness
        self.word_highlight = word_highlight
        self.font_size = font_size
        self.font_family = font_family
        self.bold = bold
        self.italic = italic
        self.alignment = alignment
        self.vertical_margin = vertical_margin
        self.resolution = resolution
        self.generate_chapters = generate_chapters
        self.include_images = include_images
        self._is_cancelled = False

    def _emit_audio_progress(self, val: int) -> None:
        self.progress_transcription.emit(val)
        self.audio_progress.emit(val)

    def _emit_render_progress(self, val: int) -> None:
        self.progress_render.emit(val)
        self.render_progress.emit(val)

    def _emit_success(self, path: str) -> None:
        self.finished.emit(path, True)
        self.finished_success.emit(path)
        self.render_finished.emit(path)

    def _emit_error(self, err_msg: str) -> None:
        self.finished.emit(err_msg, False)
        self.error_occurred.emit(err_msg)
        self.render_failed.emit(err_msg)

    def stop(self) -> None:
        """Request cancellation."""
        self._is_cancelled = True
        self.status_message.emit("Cancelling background process...")

    def run(self) -> None:
        """Worker thread entry point."""
        cfg = get_config()
        t_start = time.perf_counter()
        project_id: Optional[int] = None

        try:
            # ── Step 0: Initialise DB project ─────────────────────────────────
            init_db()
            project_id = create_project(self.project_title, template_type="template_1")
            update_project_status(project_id, "processing")

            if not self.audio_path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {self.audio_path}")

            # ── Step 1: Voiceover Transcription ───────────────────────────────
            self.step_changed.emit("Transcribing Voiceover")
            self.status_message.emit("Scanning audio & loading Whisper model...")
            self._emit_audio_progress(10)

            if self._is_cancelled:
                return

            def _audio_cb(pct: float) -> None:
                self._emit_audio_progress(int(pct * 100))

            transcriber = TranscriptionService()
            self._emit_audio_progress(25)
            self.status_message.emit("Transcribing audio into timestamped segments (English)...")

            result: TranscriptionResult = transcriber.transcribe(
                self.audio_path,
                project_id=project_id,
                language="en",
                task="transcribe",
                word_timestamps=self.word_highlight,
                chunk_subtitles=True,
                max_words_per_chunk=6,
                use_cache=True,
                progress_cb=_audio_cb,
            )

            self._emit_audio_progress(100)
            cache_note = " (from cache)" if result.from_cache else ""
            self.status_message.emit(f"Transcription complete: {result.segment_count} segments{cache_note}")

            if not result.segments:
                raise ValueError("No speech segments detected in audio file.")

            if self._is_cancelled:
                return

            # Probe audio duration early for continuous timeline matching
            from core.ffmpeg_wrapper import probe_media
            audio_info = probe_media(self.audio_path)
            total_audio_duration = audio_info.duration if audio_info.duration > 0 else 0.0

            # ── Step 2: Subtitle Engine ───────────────────────────────────────
            self.step_changed.emit("Generating Subtitles")
            self.status_message.emit("Compiling ASS subtitle styling...")

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
                word_highlight=self.word_highlight,
                font_size=self.font_size,
                font_name=self.font_family,
                bold=self.bold,
                italic=self.italic,
                alignment=self.alignment,
                vertical_margin=self.vertical_margin,
                resolution=self.resolution,
            )
            ass_path = cfg.temp_cache_dir / f"project_{project_id}_subs.ass"
            sub_engine.generate_ass(result, ass_path)

            # Also generate SRT alongside output for client reference
            srt_path = self.output_path.parent / f"{self.output_path.stem}.srt"
            sub_engine.generate_srt(result, srt_path)

            if self._is_cancelled:
                return

            # ── Step 3: Media Assignment & Hybrid Mixing ───────────────────────
            self.step_changed.emit("Synthesizing Visual Timeline")
            self.status_message.emit("Mixing visual assets & slicing footage...")

            from services.hybrid_mixer import HybridMediaMixer
            from core.ffmpeg_wrapper import extract_subclip

            mixer = HybridMediaMixer(
                images_dir=self.images_dir,
                videos_path=self.videos_path,
                clip_duration=self.clip_duration,
                image_ratio=self.image_ratio,
            )
            visual_slots = mixer.create_mix(
                total_duration=total_audio_duration,
                transcription=result,
            )

            if not visual_slots:
                # Fallback to direct CharacterMatcher if hybrid mixer finds nothing
                matcher = CharacterMatcher(self.images_dir)
                assignments = matcher.assign(result, total_duration=total_audio_duration)
                if not assignments:
                    raise ValueError(f"No valid visual assets found in: {self.images_dir} or {self.videos_path}")
                # Convert assignments to visual slots
                from services.hybrid_mixer import VisualSlot
                visual_slots = [
                    VisualSlot(
                        slot_index=a.clip_index,
                        media_type="image",
                        start_time=a.segment.start,
                        duration=a.duration,
                        source_path=a.image_path,
                        preset=a.preset,
                        character=a.character,
                    )
                    for a in assignments
                ]

            if self._is_cancelled:
                return

            # ── Step 4: Render Video Clips (Ken Burns & Subclips) ─────────────
            self.step_changed.emit("Rendering Video Clips")
            total_clips = len(visual_slots)
            clips_dir = cfg.temp_chunks_dir / f"project_{project_id}_clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            clip_paths: list[Path] = []
            total_video_duration = 0.0

            for idx, slot in enumerate(visual_slots):
                if self._is_cancelled:
                    return

                clip_file = clips_dir / f"clip_{idx:03d}.mp4"
                desc = slot.preset.name if slot.media_type == "image" else f"Subclip {slot.subclip_start:.1f}s"
                self.status_message.emit(
                    f"Rendering clip {idx+1}/{total_clips} [{slot.media_type.upper()}]: {desc} ({slot.duration:.1f}s)..."
                )

                def _clip_prog(pct: float, current_idx=idx):
                    global_pct = int(((current_idx + pct) / (total_clips + 1)) * 90)
                    self._emit_render_progress(global_pct)

                if slot.media_type == "image":
                    build_ken_burns_clip(
                        image_path=slot.source_path,
                        duration=slot.duration,
                        output_path=clip_file,
                        preset=slot.preset,
                        resolution=self.resolution,
                        progress_cb=_clip_prog,
                    )
                else:
                    extract_subclip(
                        video_path=slot.source_path,
                        start_sec=slot.subclip_start,
                        end_sec=slot.subclip_end,
                        output_path=clip_file,
                        resolution=self.resolution,
                        target_duration=slot.duration,
                        keep_audio=False,
                        progress_cb=_clip_prog,
                    )

                clip_paths.append(clip_file)
                total_video_duration += slot.duration

            if self._is_cancelled:
                return

            # ── Step 5: Final Concatenation & Subtitle Burn-In ─────────────────
            self.step_changed.emit(f"Exporting {self.resolution.upper()} Master")
            self.status_message.emit("Joining clips, mixing voiceover audio, and burning subtitles...")
            self._emit_render_progress(92)

            def _final_prog(pct: float):
                render_pct = 92 + int(pct * 8)
                self._emit_render_progress(min(render_pct, 99))

            final_total_dur = total_audio_duration if total_audio_duration > 0 else total_video_duration
            concatenate_clips(
                clip_paths=clip_paths,
                audio_path=self.audio_path,
                output_path=self.output_path,
                subtitle_path=ass_path,
                progress_cb=_final_prog,
                total_duration=final_total_dur,
                resolution=self.resolution,
            )

            self._emit_render_progress(100)
            elapsed = time.perf_counter() - t_start
            update_project_status(project_id, "done")

            # Write YouTube chapters text file if requested
            if self.generate_chapters and result.segments:
                try:
                    chapters_path = self.output_path.parent / f"{self.output_path.stem}_chapters.txt"
                    lines = ["00:00 - Introduction\n"]
                    last_min = 0
                    for seg in result.segments[1:]:
                        cur_min = int(seg.start // 30) * 30  # Every 30s chunk
                        if cur_min > last_min:
                            mins = int(seg.start // 60)
                            secs = int(seg.start % 60)
                            summary = seg.text[:25].strip().title()
                            lines.append(f"{mins:02d}:{secs:02d} - {summary}\n")
                            last_min = cur_min
                    chapters_path.write_text("".join(lines), encoding="utf-8")
                    logger.info("Generated YouTube SEO chapters: %s", chapters_path)
                except Exception as chap_err:
                    logger.warning("Could not generate chapters: %s", chap_err)

            self.step_changed.emit("Export Completed")
            self.status_message.emit(f"Ready: Render finished in {elapsed:.1f}s → {self.output_path.name}")
            self._emit_success(str(self.output_path))

        except Exception as exc:
            logger.exception("VideoRenderWorker encountered an error:")
            if project_id is not None:
                try:
                    update_project_status(project_id, "error")
                except Exception:
                    pass
            self.status_message.emit(f"Error: {exc}")
            self._emit_error(str(exc))


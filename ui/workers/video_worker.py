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
from typing import Optional, Sequence

from PySide6.QtCore import QThread, Signal

from utils.config import get_config
from utils.logger import get_logger
from database.db import init_db, create_project, update_project_status
from services.transcriber import (
    TranscriptionService, TranscriptionResult, TranscriptionSegment, WordTimestamp
)
from services.character_matcher import CharacterMatcher
from services.subtitle_engine import SubtitleEngine
from core.ffmpeg_wrapper import (
    build_ken_burns_clip, extract_subclip, concatenate_clips, probe_media, lossless_concat_videos
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
    token_usage_updated = Signal(dict)     # Token stats dict (optional)
    error_occurred = Signal(str)           # error_msg
    render_failed = Signal(str)            # error_msg (standard UI alias)


    def __init__(
        self,
        audio_path: Optional[Path | str] = None,
        images_dir: Path = Path("."),
        output_path: Path = Path("output.mp4"),
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
        format_mode: str = "Template 1 (Hybrid Images & Video Mix)",
        template_type: str = "template_1",
        gemini_api_key: Optional[str] = None,
        audio_paths: Optional[Sequence[Path | str]] = None,
        language: str = "en",
        narration_language: Optional[str] = None,
        transition_sfx: Optional[dict] = None,
        parent=None,
        **kwargs,
    ) -> None:
        super().__init__(parent)
        self.transition_sfx = transition_sfx
        self.format_mode = kwargs.get("format_mode", format_mode)
        self.template_type = kwargs.get("template_type", template_type)
        self.gemini_api_key = kwargs.get("gemini_api_key", gemini_api_key)
        self.language = kwargs.get("narration_language") or kwargs.get("language") or narration_language or language
        self.interview_config = kwargs.get("interview_config", {})
        self.bg_image_path = kwargs.get("bg_image_path", self.interview_config.get("bg_image_path", ""))
        self.bg_blur_amount = kwargs.get("bg_blur_amount", self.interview_config.get("bg_blur_amount", 0))

        if audio_paths:
            self.audio_paths = [Path(p).resolve() for p in audio_paths if str(p).strip()]
            self.audio_path = self.audio_paths[0] if self.audio_paths else Path(".")
        elif audio_path is not None:
            self.audio_path = Path(audio_path).resolve()
            self.audio_paths = [self.audio_path]
        else:
            self.audio_paths = []
            self.audio_path = Path(".")

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
        self.token_stats: dict = {}
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
            # ── Route Template 2 (Episode Scene Matching) ─────────────────────
            is_template_2 = (
                "template_2" in self.template_type.lower()
                or "template 2" in str(self.format_mode).lower()
            )
            if is_template_2:
                from services.episode_compiler import EpisodeCompiler, EpisodeCompileConfig
                compile_cfg = EpisodeCompileConfig(
                    audio_path=self.audio_path,
                    footage_path=self.videos_path or self.images_dir,
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
                    gemini_api_key=self.gemini_api_key if self.gemini_api_key is not None else cfg.gemini_api_key,
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
                    return
                self.token_stats = res.token_stats
                self.token_usage_updated.emit(res.token_stats)
                self._emit_success(str(res.output_path))
                return

            # ── Route Template 3 (+ Interview) ───────────────────────────────
            is_template_3 = (
                "template_3" in self.template_type.lower()
                or "template 3" in str(self.format_mode).lower()
            )
            if is_template_3:
                from services.interview_compiler import InterviewCompiler, InterviewCompileConfig
                interview_cfg_dict = getattr(self, "interview_config", {}) or {}
                compile_cfg = InterviewCompileConfig(
                    audio_path=self.audio_path,
                    footage_path=self.videos_path or self.images_dir,
                    output_path=self.output_path,
                    title=self.project_title,
                    resolution=self.resolution,
                    clip_duration=self.clip_duration,
                    image_ratio=self.image_ratio,
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
                    include_images=self.include_images,
                    interview_config=interview_cfg_dict,
                    bg_image_path=getattr(self, "bg_image_path", interview_cfg_dict.get("bg_image_path")),
                    bg_blur_amount=getattr(self, "bg_blur_amount", interview_cfg_dict.get("bg_blur_amount", 0)),
                )
                compiler = InterviewCompiler(
                    config=compile_cfg,
                    audio_progress_cb=self._emit_audio_progress,
                    render_progress_cb=self._emit_render_progress,
                    step_cb=self.step_changed.emit,
                    status_cb=self.status_message.emit,
                    is_cancelled_cb=lambda: self._is_cancelled,
                )
                res = compiler.compile()
                if self._is_cancelled or res is None:
                    return
                self._emit_success(str(res.output_path))
                return

            # ── Step 0: Initialise DB project ─────────────────────────────────
            init_db()
            project_id = create_project(self.project_title, template_type="template_1")
            update_project_status(project_id, "processing")

            # ── Route Multi-Part Audio Mode ───────────────────────────────────
            if len(self.audio_paths) > 1:
                self._run_multipart(cfg, project_id, t_start)
                return

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
                gemini_api_key=self.gemini_api_key,
                ensure_english=True,
            )
            ass_path = cfg.temp_cache_dir / f"project_{project_id}_subs.ass"
            sub_engine.generate_ass(result, ass_path, ensure_english=True)

            # Also generate SRT alongside output for client reference
            srt_path = self.output_path.parent / f"{self.output_path.stem}.srt"
            sub_engine.generate_srt(result, srt_path)

            if self._is_cancelled:
                return

            # ── Step 2.5: Image Semantic Vision Indexing ──────────────────────
            image_semantics = {}
            if self.images_dir and self.images_dir.exists():
                self.step_changed.emit("Analyzing Visual Semantics")
                self.status_message.emit("Analyzing image scenes with Gemini Vision...")
                from services.image_semantic_matcher import ImageSemanticIndexer
                from utils.file_helpers import is_image_file
                img_list = [p for p in self.images_dir.rglob("*") if p.is_file() and is_image_file(p)][:24]
                if img_list:
                    indexer = ImageSemanticIndexer(
                        gemini_api_key=cfg.gemini_api_key,
                        project_id=project_id,
                        project_title=self.project_title,
                    )
                    image_semantics = indexer.index_images(
                        img_list,
                        progress_cb=lambda pct, msg: self.status_message.emit(msg),
                    )
                    usage = indexer.get_usage_summary()
                    if usage.get("total_tokens", 0) > 0:
                        self.token_stats = usage
                        self.token_usage_updated.emit(usage)

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
                image_semantics=image_semantics,
            )
            visual_slots = mixer.create_mix(
                total_duration=total_audio_duration,
                transcription=result,
            )

            if not visual_slots:
                # Fallback to direct CharacterMatcher if hybrid mixer finds nothing
                matcher = CharacterMatcher(self.images_dir, image_semantics=image_semantics)
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
            effective_audio = self.audio_path
            sfx_track_tmp: Optional[Path] = None
            mixed_audio_tmp: Optional[Path] = None
            sfx_cfg = self.transition_sfx or {}

            if sfx_cfg.get("enabled", True) and sfx_cfg.get("selected_sfx") and len(clip_paths) > 1:
                try:
                    from services.sfx_manager import SFXManager
                    sfx_mgr = SFXManager()
                    clip_durs = [probe_media(p).duration for p in clip_paths if p.exists()]
                    sfx_track_tmp = cfg.temp_cache_dir / f"proj_{project_id}_sfx_track.wav"
                    built = sfx_mgr.build_transition_audio_track(
                        clip_durations=clip_durs,
                        selected_sfx=sfx_cfg["selected_sfx"],
                        output_path=sfx_track_tmp,
                        volume=sfx_cfg.get("volume", 0.35),
                    )
                    if built and built.exists():
                        mixed_audio_tmp = cfg.temp_cache_dir / f"proj_{project_id}_mixed_audio.wav"
                        sfx_mgr.mix_voiceover_with_sfx(self.audio_path, built, mixed_audio_tmp)
                        if mixed_audio_tmp.exists() and mixed_audio_tmp.stat().st_size > 0:
                            effective_audio = mixed_audio_tmp
                            logger.info("Transition SFX mixed successfully into master audio.")
                except Exception as sfx_err:
                    logger.warning("Could not apply transition SFX: %s; proceeding with clean audio.", sfx_err)

            concatenate_clips(
                clip_paths=clip_paths,
                audio_path=effective_audio,
                output_path=self.output_path,
                subtitle_path=ass_path,
                progress_cb=_final_prog,
                total_duration=final_total_dur,
                resolution=self.resolution,
            )

            # Clean up temporary sfx audio files
            if sfx_track_tmp and sfx_track_tmp.exists():
                sfx_track_tmp.unlink(missing_ok=True)
            if mixed_audio_tmp and mixed_audio_tmp.exists():
                mixed_audio_tmp.unlink(missing_ok=True)

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

    def _run_multipart(self, cfg, project_id: int, t_start: float) -> None:
        """
        Render multi-part audio project by processing each audio segment sequentially,
        maintaining visual variety/anti-repetition across parts, and losslessly
        concatenating them via FFmpeg demuxer (-c copy) into one master video.
        """
        for ap in self.audio_paths:
            if not ap.exists():
                raise FileNotFoundError(f"Multi-part audio file does not exist: {ap}")

        # Step 2.5: Image Semantic Vision Indexing (once for the whole project)
        image_semantics = {}
        if self.images_dir and self.images_dir.exists():
            self.step_changed.emit("Analyzing Visual Semantics")
            self.status_message.emit("Analyzing image scenes with Gemini Vision...")
            from services.image_semantic_matcher import ImageSemanticIndexer
            from utils.file_helpers import is_image_file
            img_list = [p for p in self.images_dir.rglob("*") if p.is_file() and is_image_file(p)][:24]
            if img_list:
                indexer = ImageSemanticIndexer(
                    gemini_api_key=cfg.gemini_api_key,
                    project_id=project_id,
                    project_title=self.project_title,
                )
                image_semantics = indexer.index_images(
                    img_list,
                    progress_cb=lambda pct, msg: self.status_message.emit(msg),
                )
                usage = indexer.get_usage_summary()
                if usage.get("total_tokens", 0) > 0:
                    self.token_stats = usage
                    self.token_usage_updated.emit(usage)

        if self._is_cancelled:
            return

        total_parts = len(self.audio_paths)
        parts_dir = cfg.temp_chunks_dir / f"project_{project_id}_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        shared_used_images: set[Path] = set()
        part_video_paths: list[Path] = []
        all_segments: list[TranscriptionSegment] = []
        current_time_offset: float = 0.0

        transcriber = TranscriptionService()
        from services.hybrid_mixer import HybridMediaMixer, VisualSlot

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

        for part_idx, part_audio in enumerate(self.audio_paths):
            if self._is_cancelled:
                return

            part_num = part_idx + 1
            part_name = part_audio.name
            self.step_changed.emit(f"Part {part_num}/{total_parts}: Transcribing")
            self.status_message.emit(f"Rendering Part {part_num} of {total_parts}: {part_name}...")

            def _part_audio_cb(pct: float, current_idx=part_idx):
                global_audio_pct = int(((current_idx + pct) / total_parts) * 100)
                self._emit_audio_progress(min(global_audio_pct, 100))

            # Transcribe this audio part
            result: TranscriptionResult = transcriber.transcribe(
                part_audio,
                project_id=project_id,
                language="en",
                task="transcribe",
                word_timestamps=self.word_highlight,
                chunk_subtitles=True,
                max_words_per_chunk=6,
                use_cache=True,
                progress_cb=_part_audio_cb,
            )

            if not result.segments:
                raise ValueError(f"No speech segments detected in audio part {part_num}: {part_name}")

            audio_info = probe_media(part_audio)
            part_audio_dur = audio_info.duration if audio_info.duration > 0 else 0.0

            # Accumulate global segments with time offset
            for seg in result.segments:
                offset_words = [
                    WordTimestamp(
                        word=w.word,
                        start=round(w.start + current_time_offset, 3),
                        end=round(w.end + current_time_offset, 3),
                        probability=w.probability,
                    )
                    for w in seg.words
                ] if seg.words else []
                all_segments.append(
                    TranscriptionSegment(
                        start=round(seg.start + current_time_offset, 3),
                        end=round(seg.end + current_time_offset, 3),
                        text=seg.text,
                        words=offset_words,
                    )
                )

            # Generate subtitles for this part
            ass_path = cfg.temp_cache_dir / f"project_{project_id}_part_{part_num}_subs.ass"
            sub_engine.generate_ass(result, ass_path)

            if self._is_cancelled:
                return

            # Visual timeline for this part (passing shared_used_images so images aren't repeated)
            self.step_changed.emit(f"Part {part_num}/{total_parts}: Timeline Mix")
            mixer = HybridMediaMixer(
                images_dir=self.images_dir,
                videos_path=self.videos_path,
                clip_duration=self.clip_duration,
                image_ratio=self.image_ratio,
                image_semantics=image_semantics,
            )
            visual_slots = mixer.create_mix(
                total_duration=part_audio_dur,
                transcription=result,
                used_images=shared_used_images,
            )

            if not visual_slots:
                matcher = CharacterMatcher(self.images_dir, image_semantics=image_semantics)
                assignments = matcher.assign(result, total_duration=part_audio_dur)
                if not assignments:
                    raise ValueError(f"No valid visual assets found for part {part_num}")
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

            # Render video clips for this part
            part_clips_dir = cfg.temp_chunks_dir / f"project_{project_id}_part_{part_num}_clips"
            part_clips_dir.mkdir(parents=True, exist_ok=True)
            part_clip_paths: list[Path] = []
            total_clips = len(visual_slots)
            part_video_dur = 0.0

            for c_idx, slot in enumerate(visual_slots):
                if self._is_cancelled:
                    return

                clip_file = part_clips_dir / f"clip_{c_idx:03d}.mp4"
                desc = slot.preset.name if slot.media_type == "image" else f"Subclip {slot.subclip_start:.1f}s"
                self.status_message.emit(
                    f"Part {part_num}/{total_parts} - Clip {c_idx+1}/{total_clips} [{slot.media_type.upper()}]: {desc} ({slot.duration:.1f}s)..."
                )

                def _clip_prog(pct: float, current_c=c_idx, current_p=part_idx):
                    part_prog = (current_c + pct) / (total_clips + 1)
                    global_pct = int(((current_p + part_prog) / total_parts) * 90)
                    self._emit_render_progress(min(global_pct, 90))

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

                part_clip_paths.append(clip_file)
                part_video_dur += slot.duration

            if self._is_cancelled:
                return

            # Assemble this part video
            part_out_mp4 = parts_dir / f"part_{part_num:03d}.mp4"
            self.step_changed.emit(f"Part {part_num}/{total_parts}: Assembling")
            self.status_message.emit(f"Joining {len(part_clip_paths)} clips for Part {part_num}...")

            def _part_concat_prog(pct: float, current_p=part_idx):
                part_prog = 0.9 + (pct * 0.1)
                global_pct = int(((current_p + part_prog) / total_parts) * 90)
                self._emit_render_progress(min(global_pct, 90))

            part_final_dur = part_audio_dur if part_audio_dur > 0 else part_video_dur
            concatenate_clips(
                clip_paths=part_clip_paths,
                audio_path=part_audio,
                output_path=part_out_mp4,
                subtitle_path=ass_path,
                progress_cb=_part_concat_prog,
                total_duration=part_final_dur,
                resolution=self.resolution,
            )

            part_video_paths.append(part_out_mp4)
            current_time_offset += part_final_dur

            # Clean up intermediate clip files for this part
            try:
                import shutil
                shutil.rmtree(part_clips_dir, ignore_errors=True)
            except Exception:
                pass

        if self._is_cancelled:
            return

        # ── Step 5: Lossless Concat Demuxer Merge ──────────────────────────────
        self.step_changed.emit("Merging Audio Parts")
        self.status_message.emit(f"Losslessly merging {len(part_video_paths)} video parts into master video...")
        self._emit_render_progress(92)

        def _merge_prog(pct: float):
            render_pct = 92 + int(pct * 7)
            self._emit_render_progress(min(render_pct, 99))

        lossless_concat_videos(
            video_paths=part_video_paths,
            output_path=self.output_path,
            progress_cb=_merge_prog,
        )

        self._emit_render_progress(100)
        elapsed = time.perf_counter() - t_start
        update_project_status(project_id, "done")

        # Clean up temporary part MP4s after successful merge
        try:
            import shutil
            shutil.rmtree(parts_dir, ignore_errors=True)
        except Exception:
            pass

        # Write combined SRT file
        if all_segments:
            try:
                combined_result = TranscriptionResult(
                    audio_path=self.audio_paths[0],
                    audio_hash="",
                    language="en",
                    full_text=" ".join(s.text for s in all_segments),
                    segments=all_segments,
                    model_used="whisper",
                    elapsed_seconds=elapsed,
                )
                srt_path = self.output_path.parent / f"{self.output_path.stem}.srt"
                sub_engine.generate_srt(combined_result, srt_path)
            except Exception as srt_err:
                logger.warning("Could not generate merged SRT: %s", srt_err)

        # Write YouTube chapters text file if requested
        if self.generate_chapters and all_segments:
            try:
                chapters_path = self.output_path.parent / f"{self.output_path.stem}_chapters.txt"
                lines = ["00:00 - Introduction\n"]
                last_min = 0
                for seg in all_segments[1:]:
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
        self.status_message.emit(f"Ready: Multi-part render finished in {elapsed:.1f}s → {self.output_path.name}")
        self._emit_success(str(self.output_path))


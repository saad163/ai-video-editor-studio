"""
services/episode_compiler.py
============================
Phase 3 — End-to-End Episode Scene Matching & Compilation Service.

Orchestrates the complete Template 2 (Episode Scene Matching) pipeline:
  1. Audio duration probing & Whisper speech-to-text (SQLite cached).
  2. Subtitle styling, ASS/SRT file generation, and optional SEO chapters.
  3. Episode video discovery & PySceneDetect scene splitting with SQLite caching.
  4. Narrative context chunking with continuous [0.0s, total_audio_duration] coverage.
  5. Semantic scene matching (Google Gemini 1.5 Flash API + offline heuristic fallback + anti-repetition).
  6. Subclip extraction with resolution normalization (720p/1080p/2k, 30fps, -an).
  7. Audio-video concatenation, subtitle burn-in, and master MP4 verification.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from utils.config import get_config
from utils.logger import get_logger
from utils.file_helpers import is_video_file, compute_file_hash
from database.db import init_db, create_project, update_project_status, record_api_usage
from services.transcriber import TranscriptionService, TranscriptionResult
from services.subtitle_engine import SubtitleEngine
from services.scene_detector import SceneDetectorService, Scene
from services.gemini_matcher import NarrativeChunker, SmartSceneMatcher, NarrativeChunk
from core.ffmpeg_wrapper import extract_subclip, concatenate_clips, probe_media

logger = get_logger(__name__)


@dataclass
class EpisodeCompileConfig:
    """Configuration options for Template 2 episode compilation."""
    audio_path: Path
    footage_path: Path
    output_path: Path
    title: str = "Untitled Episode Video"
    resolution: str = "1080p"
    clip_duration: float = 8.0
    language: str = "en"
    narration_language: Optional[str] = None
    text_color: str = "#FFFF00"
    highlight_color: str = "#FFFF00"
    has_bg_box: bool = False
    bg_style: str = "outline"
    bg_color: str = "#000000"
    bg_opacity: float = 0.5
    stroke_color: str = "#000000"
    stroke_thickness: float = 3.5
    font_size: int = 56
    font_family: str = "Montserrat"
    bold: bool = True
    italic: bool = False
    box_padding: float = 14.0
    generate_chapters: bool = True
    scene_threshold: float = 27.0
    gemini_api_key: Optional[str] = None
    require_gemini: Optional[bool] = None
    project_id: Optional[int] = None
    transition_sfx: Optional[dict] = None
    bg_image_path: Optional[str] = None
    bg_blur_amount: float = 0.0


@dataclass
class EpisodeCompileResult:
    """Outcome payload from a successful episode compilation."""
    output_path: Path
    project_id: Optional[int]
    total_clips: int
    duration_sec: float
    width: int
    height: int
    elapsed_sec: float
    matched_chunks: list[NarrativeChunk] = field(default_factory=list)
    token_stats: dict = field(default_factory=dict)



class EpisodeCompiler:
    """
    Pure Python & FFmpeg orchestration service for Template 2.
    Decoupled from PySide6 QThread so it can be called headless, in tests,
    or within QThread workers (EpisodeCompilerWorker, VideoRenderWorker, BulkQueueWorker).
    """

    def __init__(
        self,
        config: EpisodeCompileConfig,
        audio_progress_cb: Optional[Callable[[int], None]] = None,
        render_progress_cb: Optional[Callable[[int], None]] = None,
        step_cb: Optional[Callable[[str], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        is_cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config
        self.audio_progress_cb = audio_progress_cb
        self.render_progress_cb = render_progress_cb
        self.step_cb = step_cb
        self.status_cb = status_cb
        self.is_cancelled_cb = is_cancelled_cb

    def _check_cancelled(self) -> bool:
        if self.is_cancelled_cb and self.is_cancelled_cb():
            logger.info("EpisodeCompiler execution cancelled by caller.")
            return True
        return False

    def _emit_step(self, step: str) -> None:
        if self.step_cb:
            self.step_cb(step)

    def _emit_status(self, msg: str) -> None:
        if self.status_cb:
            self.status_cb(msg)

    def _emit_audio_progress(self, pct: int) -> None:
        if self.audio_progress_cb:
            self.audio_progress_cb(max(0, min(pct, 100)))

    def _emit_render_progress(self, pct: int) -> None:
        if self.render_progress_cb:
            self.render_progress_cb(max(0, min(pct, 100)))

    def compile(self) -> Optional[EpisodeCompileResult]:
        """
        Execute the full Template 2 pipeline.
        Returns EpisodeCompileResult on success, or None if cancelled.
        Raises exceptions on operational failures.
        """
        cfg = get_config()
        init_db()
        t_start = time.perf_counter()

        audio_path = Path(self.config.audio_path).resolve()
        footage_path = Path(self.config.footage_path).resolve()
        output_path = Path(self.config.output_path).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Voiceover audio track not found: {audio_path}")
        if not footage_path.exists():
            raise FileNotFoundError(f"Source footage path not found: {footage_path}")

        # ── Step 0: DB Project ────────────────────────────────────────────────
        project_id = self.config.project_id
        if project_id is None:
            project_id = create_project(self.config.title, template_type="template_2_episode")
        update_project_status(project_id, "processing")

        try:
            # ── Step 1: Probe Audio Duration & Transcribe ─────────────────────
            self._emit_step("Transcribing Voiceover")
            self._emit_status("Probing audio timeline & transcribing voiceover...")
            self._emit_audio_progress(5)

            if self._check_cancelled():
                return None

            audio_info = probe_media(audio_path)
            total_audio_duration = audio_info.duration if audio_info.duration > 0 else 0.0
            if total_audio_duration <= 0.0:
                raise ValueError(f"Could not determine valid duration for audio: {audio_path}")

            def _audio_progress_bridge(pct: float) -> None:
                self._emit_audio_progress(int(pct * 100))

            raw_lang = self.config.narration_language or self.config.language or "ur"
            lang_code = "ur" if ("ur" in str(raw_lang).lower() or "اردو" in str(raw_lang)) else "en"

            transcriber = TranscriptionService()
            result: TranscriptionResult = transcriber.transcribe(
                audio_path,
                project_id=project_id,
                language=lang_code,
                task="transcribe",
                chunk_subtitles=True,
                max_words_per_chunk=6,
                use_cache=True,
                progress_cb=_audio_progress_bridge,
            )

            self._emit_audio_progress(100)
            self._emit_render_progress(15)

            if not result.segments:
                raise ValueError("No speech segments detected in voiceover track.")

            if self._check_cancelled():
                return None

            # ── Step 2: Subtitle Engine & Chapters ────────────────────────────
            self._emit_step("Styling Subtitles")
            self._emit_status(f"Generating styled subtitles ({self.config.font_family}, {self.config.font_size}pt)...")

            sub_engine = SubtitleEngine(
                text_color=self.config.text_color,
                highlight_color=self.config.highlight_color,
                has_bg_box=self.config.has_bg_box,
                bg_style=self.config.bg_style,
                bg_color=self.config.bg_color,
                bg_opacity=self.config.bg_opacity,
                box_padding=self.config.box_padding,
                stroke_color=self.config.stroke_color,
                stroke_thickness=self.config.stroke_thickness,
                word_highlight=False,
                font_size=self.config.font_size,
                font_name=self.config.font_family,
                bold=self.config.bold,
                italic=self.config.italic,
                alignment=2,
                vertical_margin=90 if self.config.resolution == "2k" else 60,
                resolution=self.config.resolution,
                gemini_api_key=self.config.gemini_api_key,
                ensure_english=True,
            )

            ass_path = cfg.temp_cache_dir / f"episode_proj_{project_id}_subs.ass"
            sub_engine.generate_ass(result, ass_path, ensure_english=True)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            srt_path = output_path.parent / f"{output_path.stem}.srt"
            sub_engine.generate_srt(result, srt_path, ensure_english=True)

            # Optional YouTube SEO Chapters
            if self.config.generate_chapters and result.segments:
                chapters_path = output_path.parent / f"{output_path.stem}_chapters.txt"
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

            self._emit_render_progress(25)

            if self._check_cancelled():
                return None

            # ── Step 3: Discover Video Files & Detect Scenes ──────────────────
            self._emit_step("Detecting Episode Scenes")
            self._emit_status("Scanning source video files for scene boundaries...")

            video_files: list[Path] = []
            if footage_path.is_file() and is_video_file(footage_path):
                video_files.append(footage_path)
            elif footage_path.is_dir():
                for f in sorted(footage_path.iterdir()):
                    if f.is_file() and is_video_file(f):
                        video_files.append(f)
                # If no video files at root, check immediate subdirectories
                if not video_files:
                    for sub in sorted(footage_path.iterdir()):
                        if sub.is_dir():
                            for f in sorted(sub.iterdir()):
                                if f.is_file() and is_video_file(f):
                                    video_files.append(f)

            if not video_files:
                raise FileNotFoundError(f"No source video files found in: {footage_path}")

            detector = SceneDetectorService(threshold=self.config.scene_threshold, min_scene_len_sec=1.5)
            all_scenes: list[Scene] = []

            for vid_idx, vid_file in enumerate(video_files):
                if self._check_cancelled():
                    return None
                self._emit_status(f"Detecting scenes ({vid_idx+1}/{len(video_files)}): {vid_file.name}...")
                scenes = detector.detect_scenes(vid_file, project_id=project_id, extract_keyframes=True, use_cache=True)
                if scenes:
                    all_scenes.extend(scenes)
                else:
                    # Fallback single scene for video if PySceneDetect detects 0 cuts
                    vf_info = probe_media(vid_file)
                    vf_dur = vf_info.duration if vf_info.duration > 0 else 30.0
                    fallback_scene = Scene(
                        scene_id=None,
                        video_hash=compute_file_hash(vid_file),
                        video_path=vid_file,
                        scene_index=1,
                        start_sec=0.0,
                        end_sec=vf_dur,
                        duration_sec=vf_dur,
                        description=f"Full clip: {vid_file.stem}",
                    )
                    all_scenes.append(fallback_scene)

            if not all_scenes:
                raise ValueError("PySceneDetect found 0 scenes across the provided video files.")

            logger.info("Total available scenes across %d videos: %d", len(video_files), len(all_scenes))
            self._emit_render_progress(40)

            if self._check_cancelled():
                return None

            # ── Step 4: Narrative Context Chunking ────────────────────────────
            self._emit_step("Chunking Narration")
            self._emit_status("Grouping voiceover into continuous narrative context blocks...")

            chunk_len = max(2.0, float(self.config.clip_duration))
            chunker = NarrativeChunker(
                target_chunk_sec=chunk_len,
                min_chunk_sec=max(chunk_len * 0.5, 2.0),
                max_chunk_sec=max(chunk_len * 1.8, 6.0),
            )
            chunks = chunker.chunk_transcription(result)

            if not chunks:
                raise ValueError("NarrativeChunker produced 0 context blocks.")

            # Ensure continuous timeline coverage [0.0, total_audio_duration]
            if chunks[0].start_sec > 0.0:
                chunks[0].duration_sec += chunks[0].start_sec
                chunks[0].start_sec = 0.0

            for i in range(len(chunks) - 1):
                if chunks[i].end_sec != chunks[i + 1].start_sec:
                    chunks[i].end_sec = chunks[i + 1].start_sec
                    chunks[i].duration_sec = chunks[i].end_sec - chunks[i].start_sec

            if chunks[-1].end_sec < total_audio_duration:
                # Extend the last chunk to cover the entire voiceover tail
                chunks[-1].end_sec = total_audio_duration
                chunks[-1].duration_sec = total_audio_duration - chunks[-1].start_sec
            elif chunks[-1].end_sec > total_audio_duration:
                chunks[-1].end_sec = total_audio_duration
                chunks[-1].duration_sec = max(0.5, total_audio_duration - chunks[-1].start_sec)

            logger.info("Divided speech into %d narrative chunks covering %.1fs", len(chunks), total_audio_duration)
            self._emit_render_progress(50)

            if self._check_cancelled():
                return None

            # ── Step 5: Semantic Scene Matching (Anti-Repetition) ────────────
            self._emit_step("Matching Narrative to Scenes")
            self._emit_status("Assigning optimal non-repeating scenes via SmartSceneMatcher...")

            is_urdu = ("ur" in lang_code.lower() or "اردو" in lang_code)
            req_gemini = getattr(self.config, "require_gemini", None)
            if req_gemini is not None:
                strict_gemini = req_gemini
            elif is_urdu:
                strict_gemini = True
            elif self.config.gemini_api_key == "":
                strict_gemini = False
            else:
                strict_gemini = True

            matcher = SmartSceneMatcher(
                gemini_api_key=self.config.gemini_api_key,
                strict_mode=strict_gemini,
            )
            vo_hash = getattr(result, "audio_hash", None)
            vid_hash = all_scenes[0].video_hash if all_scenes else None

            matched_chunks = matcher.match_scenes(
                chunks=chunks,
                available_scenes=all_scenes,
                project_id=project_id,
                prefer_unused=True,
                language=lang_code,
                strict_mode=strict_gemini,
                voiceover_hash=vo_hash,
                video_hash=vid_hash,
                template_mode="template_2",
            )
            usage_stats = matcher.get_usage_summary()
            logger.info("EpisodeCompiler: Gemini API Session Usage: %s", usage_stats)
            if usage_stats.get("total_tokens", 0) > 0:
                try:
                    record_api_usage(
                        project_id=project_id,
                        project_title=self.config.title,
                        prompt_tokens=usage_stats["prompt_tokens"],
                        output_tokens=usage_stats["output_tokens"],
                        total_tokens=usage_stats["total_tokens"],
                        estimated_cost_usd=usage_stats["estimated_cost_usd"],
                    )
                except Exception as db_err:
                    logger.warning("Failed to record API usage in database: %s", db_err)


            # Strict consecutive anti-repetition guard
            for i in range(1, len(matched_chunks)):
                prev_s = matched_chunks[i - 1].matched_scene
                curr_s = matched_chunks[i].matched_scene
                if prev_s and curr_s and (prev_s.video_path == curr_s.video_path and prev_s.start_sec == curr_s.start_sec):
                    # Pick an alternate scene from pool
                    alt_candidates = [s for s in all_scenes if s.start_sec != prev_s.start_sec or s.video_path != prev_s.video_path]
                    if alt_candidates:
                        matched_chunks[i].matched_scene = alt_candidates[i % len(alt_candidates)]

            self._emit_render_progress(60)

            if self._check_cancelled():
                return None

            # ── Step 6: Extract & Scale Subclips ──────────────────────────────
            self._emit_step("Extracting Scene Subclips")
            clips_dir = cfg.temp_chunks_dir / f"episode_proj_{project_id}_clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            clip_paths: list[Path] = []
            total_chunks = len(matched_chunks)

            # Cache video durations to avoid re-probing on every chunk
            video_durations: dict[Path, float] = {}
            for sc in all_scenes:
                if sc.video_path not in video_durations:
                    try:
                        v_info = probe_media(sc.video_path)
                        video_durations[sc.video_path] = v_info.duration if v_info.duration > 0 else 3600.0
                    except Exception:
                        video_durations[sc.video_path] = 3600.0

            for idx, chunk in enumerate(matched_chunks):
                if self._check_cancelled():
                    return None

                scene = chunk.matched_scene or all_scenes[idx % len(all_scenes)]
                clip_file = clips_dir / f"scene_clip_{idx:03d}.mp4"

                self._emit_status(
                    f"Extracting subclip {idx+1}/{total_chunks}: Scene {scene.scene_index} ({chunk.duration_sec:.1f}s)..."
                )

                # Precision timestamp slicing:
                # Subclip duration must match chunk.duration_sec exactly
                # so cuts align precisely with the narration timeline [chunk.start_sec, chunk.end_sec].
                target_dur = chunk.duration_sec
                vid_dur = video_durations.get(scene.video_path, 3600.0)

                cut_start = max(0.0, scene.start_sec)
                # If scene start is too close to end of video, shift start backwards
                if cut_start + target_dur > vid_dur:
                    cut_start = max(0.0, vid_dur - target_dur)
                cut_end = min(vid_dur, cut_start + target_dur)

                extract_subclip(
                    video_path=scene.video_path,
                    start_sec=cut_start,
                    end_sec=cut_end,
                    output_path=clip_file,
                    resolution=self.config.resolution,
                    target_duration=target_dur,
                    keep_audio=False,
                    strip_source_audio=True,
                )
                clip_paths.append(clip_file)

                step_pct = 60 + int((idx + 1) / total_chunks * 25)
                self._emit_render_progress(min(step_pct, 85))

            if self._check_cancelled():
                return None

            # ── Step 7: Final Concat, Transition SFX & Subtitle Burn-In ───────
            self._emit_step("Final Master Render")
            self._emit_status(f"Concatenating {len(clip_paths)} clips, mixing audio & burning subtitles...")

            def _burn_cb(pct: float) -> None:
                final_pct = 85 + int(pct * 14)
                self._emit_render_progress(min(final_pct, 99))

            # Apply transition sound effects across scene boundaries if enabled
            effective_audio = audio_path
            sfx_track_tmp: Optional[Path] = None
            mixed_audio_tmp: Optional[Path] = None
            sfx_cfg = self.config.transition_sfx or {}

            if sfx_cfg.get("enabled", True) and sfx_cfg.get("selected_sfx") and len(clip_paths) > 1:
                try:
                    from services.sfx_manager import SFXManager
                    sfx_mgr = SFXManager()
                    clip_durs = [probe_media(p).duration for p in clip_paths if p.exists()]
                    sfx_track_tmp = cfg.temp_cache_dir / f"ep_{project_id}_sfx_track.wav"
                    built = sfx_mgr.build_transition_audio_track(
                        clip_durations=clip_durs,
                        selected_sfx=sfx_cfg["selected_sfx"],
                        output_path=sfx_track_tmp,
                        volume=sfx_cfg.get("volume", 0.35),
                    )
                    if built and built.exists():
                        mixed_audio_tmp = cfg.temp_cache_dir / f"ep_{project_id}_mixed_audio.wav"
                        sfx_mgr.mix_voiceover_with_sfx(audio_path, built, mixed_audio_tmp)
                        if mixed_audio_tmp.exists() and mixed_audio_tmp.stat().st_size > 0:
                            effective_audio = mixed_audio_tmp
                            logger.info("Transition SFX mixed successfully into master audio.")
                except Exception as sfx_err:
                    logger.warning("Could not apply transition SFX: %s; proceeding with clean audio.", sfx_err)

            concatenate_clips(
                clip_paths=clip_paths,
                audio_path=effective_audio,
                output_path=output_path,
                subtitle_path=ass_path,
                resolution=self.config.resolution,
                progress_cb=_burn_cb,
            )

            # ── Step 8: Final Verification & Storage Cleanup ──────────────────
            probe = probe_media(output_path)
            elapsed = time.perf_counter() - t_start
            update_project_status(project_id, "completed")

            # Clean up intermediate render chunks to keep disk space minimal
            try:
                import shutil
                if clips_dir.exists():
                    shutil.rmtree(clips_dir, ignore_errors=True)
                    logger.info("Cleaned up intermediate subclip chunks: %s", clips_dir)
                if ass_path.exists():
                    ass_path.unlink(missing_ok=True)
                if sfx_track_tmp and sfx_track_tmp.exists():
                    sfx_track_tmp.unlink(missing_ok=True)
                if mixed_audio_tmp and mixed_audio_tmp.exists():
                    mixed_audio_tmp.unlink(missing_ok=True)
            except Exception as clean_err:
                logger.warning("Could not clean up intermediate render files: %s", clean_err)

            self._emit_render_progress(100)
            self._emit_status(
                f"✓ Complete: {output_path.name} ({probe.duration:.1f}s, {probe.width}x{probe.height}) in {elapsed:.1f}s"
            )

            return EpisodeCompileResult(
                output_path=output_path,
                project_id=project_id,
                total_clips=len(clip_paths),
                duration_sec=probe.duration,
                width=probe.width,
                height=probe.height,
                elapsed_sec=elapsed,
                matched_chunks=matched_chunks,
                token_stats=usage_stats,
            )

        except Exception as exc:
            logger.exception("EpisodeCompiler failed: %s", exc)
            if project_id:
                try:
                    update_project_status(project_id, "failed")
                except Exception:
                    pass
            raise exc


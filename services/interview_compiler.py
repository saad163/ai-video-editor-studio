"""
services/interview_compiler.py
==============================
Template 3 (+ Interview) Final Render & Compositing Pipeline.

Composites confirmed, trimmed interview clips onto the visual timeline with:
  1. Exact timeline placement and duration trimming.
  2. Audio ducking/switching: mutes main voiceover strictly during interview windows
     and plays interview dialogue.
  3. Interview-only background music mixing at user volume (e.g. 15%).
  4. Mobile/vertical clip framing:
     - Scale (e.g. 70%)
     - Background blur backdrop
     - Rounded corners (px radius) or sharp edges
     - Frame border (color + line thickness)
  5. Fallback safety: handles zero clips, missing audio, or edge-case timestamps cleanly.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PIL import Image, ImageDraw

from utils.config import get_config
from utils.logger import get_logger
from utils.file_helpers import is_video_file, is_image_file
from database.db import init_db, create_project, update_project_status
from core.ffmpeg_wrapper import (
    _ff,
    _get_encoder_args,
    _run_with_progress,
    probe_media,
    extract_subclip,
    build_ken_burns_clip,
    concatenate_clips,
    build_crop_filter,
    RESOLUTIONS,
    DEFAULT_RESOLUTION,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_FPS,
)
from services.transcriber import TranscriptionService, TranscriptionResult
from services.subtitle_engine import SubtitleEngine
from services.hybrid_mixer import HybridMediaMixer, VisualSlot

logger = get_logger(__name__)


@dataclass
class InterviewCompileConfig:
    """Configuration options for Template 3 (+ Interview) video compilation."""
    audio_path: Path
    footage_path: Path
    output_path: Path
    title: str = "Untitled Interview Project"
    resolution: str = "1080p"
    clip_duration: float = 5.0
    image_ratio: float = 0.5
    language: str = "en"
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
    generate_chapters: bool = False
    include_images: bool = True

    # Template 3 specific interview framing & audio configuration
    interview_config: dict[str, Any] = field(default_factory=dict)
    bg_image_path: Optional[str] = None
    bg_blur_amount: float = 0.0

    def __post_init__(self) -> None:
        if self.bg_image_path and "bg_image_path" not in self.interview_config:
            self.interview_config["bg_image_path"] = self.bg_image_path
        if self.bg_blur_amount and "bg_blur_amount" not in self.interview_config:
            self.interview_config["bg_blur_amount"] = self.bg_blur_amount


@dataclass
class InterviewCompileResult:
    """Execution statistics returned upon successful Template 3 compilation."""
    output_path: Path
    duration_sec: float
    confirmed_clips_count: int
    elapsed_seconds: float


class InterviewCompiler:
    """
    Orchestrates the complete Template 3 (+ Interview) compilation pipeline:
      1. Voiceover transcription via faster-whisper.
      2. Timeline interval partitioning: separates voiceover base intervals from interview windows.
      3. Styled interview clip rendering (scale, background blur, rounded corners, border, and BGM mix).
      4. Base footage rendering (Ken Burns images + video subclips) for non-interview intervals.
      5. Master audio assembly: mutes voiceover during interview clips, plays interview audio + BGM.
      6. Final concatenation & ASS subtitle burn-in.
    """

    def __init__(
        self,
        config: InterviewCompileConfig,
        audio_progress_cb: Optional[Callable[[int], None]] = None,
        render_progress_cb: Optional[Callable[[int], None]] = None,
        step_cb: Optional[Callable[[str], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        is_cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = config
        self.audio_progress_cb = audio_progress_cb
        self.render_progress_cb = render_progress_cb
        self.step_cb = step_cb
        self.status_cb = status_cb
        self.is_cancelled_cb = is_cancelled_cb or (lambda: False)

    def _step(self, msg: str) -> None:
        if self.step_cb:
            self.step_cb(msg)

    def _status(self, msg: str) -> None:
        if self.status_cb:
            self.status_cb(msg)

    def _audio_progress(self, pct: int) -> None:
        if self.audio_progress_cb:
            self.audio_progress_cb(max(0, min(100, pct)))

    def _render_progress(self, pct: int) -> None:
        if self.render_progress_cb:
            self.render_progress_cb(max(0, min(100, pct)))

    def compile(self) -> Optional[InterviewCompileResult]:
        """Execute the end-to-end Template 3 compilation."""
        t_start = time.perf_counter()
        cfg = get_config()
        self.cfg.output_path = Path(self.cfg.output_path).resolve()
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.cfg.audio_path.exists():
            raise FileNotFoundError(f"Voiceover audio not found: {self.cfg.audio_path}")

        init_db()
        project_id = create_project(self.cfg.title, template_type="template_3")
        update_project_status(project_id, "processing")

        # ── Step 1: Probe Audio Duration & Transcribe ─────────────────────────
        self._step("Transcribing Voiceover")
        self._status("Probing voiceover duration...")
        self._audio_progress(10)

        if self.is_cancelled_cb():
            return None

        audio_info = probe_media(self.cfg.audio_path)
        total_audio_duration = audio_info.duration if audio_info.duration > 0 else 10.0

        transcriber = TranscriptionService()
        self._audio_progress(30)
        self._status("Transcribing voiceover audio with Whisper...")

        def _whisp_cb(pct: float) -> None:
            self._audio_progress(30 + int(pct * 60))

        transcription_res: TranscriptionResult = transcriber.transcribe(
            self.cfg.audio_path,
            project_id=project_id,
            language=self.cfg.language,
            task="transcribe",
            progress_cb=_whisp_cb,
        )
        self._audio_progress(100)
        self._status(f"Transcription complete: {transcription_res.segment_count} segments")

        if self.is_cancelled_cb():
            return None

        # ── Step 2: Read & Clean Confirmed Interview Clips ───────────────────
        interview_config = self.cfg.interview_config or {}
        raw_confirmed = interview_config.get("confirmed_clips", [])
        confirmed_clips = self._clean_confirmed_clips(raw_confirmed, total_audio_duration)

        logger.info(
            "Template 3 compilation: %d confirmed interview clip(s) on timeline (Total audio: %.1fs)",
            len(confirmed_clips),
            total_audio_duration,
        )

        # ── Step 3: Subtitle Generation (Filtered for Interview Windows) ─────
        self._step("Compiling Subtitles")
        self._status("Generating ASS subtitles...")
        self._render_progress(5)

        # Filter out subtitle events falling strictly inside interview windows to avoid voiceover text clashing
        filtered_transcription = self._filter_subtitles_for_interviews(transcription_res, confirmed_clips)

        sub_engine = SubtitleEngine(
            text_color=self.cfg.text_color,
            highlight_color=self.cfg.highlight_color,
            has_bg_box=self.cfg.has_bg_box,
            bg_style=self.cfg.bg_style,
            bg_color=self.cfg.bg_color,
            bg_opacity=self.cfg.bg_opacity,
            box_padding=self.cfg.box_padding,
            stroke_color=self.cfg.stroke_color,
            stroke_thickness=self.cfg.stroke_thickness,
            font_size=self.cfg.font_size,
            font_name=self.cfg.font_family,
            bold=self.cfg.bold,
            italic=self.cfg.italic,
            resolution=self.cfg.resolution,
        )
        ass_path = cfg.temp_cache_dir / f"project_{project_id}_interview_subs.ass"
        sub_engine.generate_ass(filtered_transcription, ass_path)

        srt_path = self.cfg.output_path.parent / f"{self.cfg.output_path.stem}.srt"
        sub_engine.generate_srt(filtered_transcription, srt_path)

        if self.is_cancelled_cb():
            return None

        # ── Step 4: Assemble Timeline Video Segments ──────────────────────────
        self._step("Rendering Interview & Visual Timeline")
        temp_dir = cfg.temp_chunks_dir / f"project_{project_id}_interview_render"
        temp_dir.mkdir(parents=True, exist_ok=True)

        timeline_clips: list[Path] = []

        if not confirmed_clips:
            # No interview clips confirmed: fall back to standard visual mixer
            logger.info("No confirmed interview clips; rendering base visual timeline.")
            timeline_clips = self._render_base_timeline(
                start_t=0.0,
                duration=total_audio_duration,
                out_dir=temp_dir,
                transcription=filtered_transcription,
                progress_offset=10,
                progress_span=75,
            )
        else:
            # Build timeline slices with interview clips and interleaved base visual footage
            timeline_clips = self._render_composited_interview_timeline(
                confirmed_clips=confirmed_clips,
                total_duration=total_audio_duration,
                temp_dir=temp_dir,
                transcription=filtered_transcription,
            )

        if self.is_cancelled_cb():
            return None

        # ── Step 5: Assemble Master Audio (Voiceover Ducked + Interview Audio + BGM) ─
        self._step("Mixing Audio Tracks")
        self._status("Ducking voiceover and blending interview audio with BGM...")
        self._render_progress(88)

        master_audio_path = temp_dir / "master_audio_track.wav"
        self._assemble_master_audio(
            confirmed_clips=confirmed_clips,
            total_duration=total_audio_duration,
            output_audio_path=master_audio_path,
            temp_dir=temp_dir,
        )

        if self.is_cancelled_cb():
            return None

        # ── Step 6: Final Concatenation & Subtitle Burn-In ────────────────────
        self._step(f"Exporting {self.cfg.resolution.upper()} Master Video")
        self._status("Concatenating clips and encoding final output...")
        self._render_progress(92)

        def _final_prog(pct: float) -> None:
            self._render_progress(92 + int(pct * 7))

        final_dur = total_audio_duration
        if confirmed_clips:
            max_end = max(c["target_end_sec"] for c in confirmed_clips)
            final_dur = max(final_dur, max_end)

        concatenate_clips(
            clip_paths=timeline_clips,
            audio_path=master_audio_path,
            output_path=self.cfg.output_path,
            subtitle_path=ass_path,
            progress_cb=_final_prog,
            total_duration=final_dur,
            resolution=self.cfg.resolution,
        )

        self._render_progress(100)
        elapsed = time.perf_counter() - t_start
        update_project_status(project_id, "done")
        self._status(f"✓ Master video rendered successfully ({final_dur:.1f}s)")

        return InterviewCompileResult(
            output_path=self.cfg.output_path,
            duration_sec=round(final_dur, 2),
            confirmed_clips_count=len(confirmed_clips),
            elapsed_seconds=round(elapsed, 2),
        )

    def _clean_confirmed_clips(self, raw_clips: list[dict[str, Any]], total_duration: float) -> list[dict[str, Any]]:
        """Validate, sort, and clamp confirmed interview clips by target start time."""
        cleaned = []
        for c in raw_clips:
            source = c.get("source_video", "")
            if not source or not Path(source).exists():
                logger.warning("Skipping interview clip with missing source video: %s", source)
                continue

            trim_start = max(0.0, float(c.get("trim_start_sec", 0.0)))
            trim_end = max(trim_start + 0.5, float(c.get("trim_end_sec", trim_start + 1.0)))
            dur = round(trim_end - trim_start, 2)
            target_start = max(0.0, float(c.get("target_start_sec", trim_start)))
            target_end = round(target_start + dur, 2)

            cleaned.append({
                "source_video": str(Path(source).resolve()),
                "trim_start_sec": trim_start,
                "trim_end_sec": trim_end,
                "duration_sec": dur,
                "target_start_sec": target_start,
                "target_end_sec": target_end,
                "label": str(c.get("label", "Interview Clip")),
                "crop_top": max(0, int(c.get("crop_top", 0))),
                "crop_bottom": max(0, int(c.get("crop_bottom", 0))),
                "crop_left": max(0, int(c.get("crop_left", 0))),
                "crop_right": max(0, int(c.get("crop_right", 0))),
                "norm_x": float(c.get("norm_x", 0.0)),
                "norm_y": float(c.get("norm_y", 0.0)),
                "norm_w": float(c.get("norm_w", 1.0)),
                "norm_h": float(c.get("norm_h", 1.0)),
            })

        # Sort chronologically by target_start_sec
        cleaned.sort(key=lambda x: x["target_start_sec"])
        return cleaned

    def _filter_subtitles_for_interviews(
        self,
        transcription: TranscriptionResult,
        confirmed_clips: list[dict[str, Any]],
    ) -> TranscriptionResult:
        """Suppress main voiceover subtitles during active interview windows to avoid text clash."""
        if not confirmed_clips:
            return transcription

        new_segments = []
        for seg in transcription.segments:
            # Check if this subtitle segment falls within any interview clip window
            is_inside_interview = False
            for c in confirmed_clips:
                t_start = c["target_start_sec"]
                t_end = c["target_end_sec"]
                # If there is substantial overlap between subtitle segment and interview window
                if max(seg.start, t_start) < min(seg.end, t_end):
                    is_inside_interview = True
                    break

            if not is_inside_interview:
                new_segments.append(seg)

        return TranscriptionResult(
            audio_path=transcription.audio_path,
            audio_hash=transcription.audio_hash,
            language=transcription.language,
            full_text=transcription.full_text,
            segments=new_segments,
            model_used=transcription.model_used,
            elapsed_seconds=transcription.elapsed_seconds,
            from_cache=transcription.from_cache,
        )

    def _render_composited_interview_timeline(
        self,
        confirmed_clips: list[dict[str, Any]],
        total_duration: float,
        temp_dir: Path,
        transcription: TranscriptionResult,
    ) -> list[Path]:
        """
        Builds the video timeline by alternating between base footage and styled interview clips.
        """
        timeline_clips: list[Path] = []
        current_time = 0.0
        total_items = len(confirmed_clips) * 2 + 1

        for idx, clip in enumerate(confirmed_clips):
            t_start = clip["target_start_sec"]
            t_end = clip["target_end_sec"]
            dur = clip["duration_sec"]

            # 1. Base visual footage gap before this interview clip
            if t_start > current_time:
                gap_dur = round(t_start - current_time, 2)
                if gap_dur >= 0.1:
                    logger.info("Rendering base visual gap [%.2fs - %.2fs (%.2fs)]", current_time, t_start, gap_dur)
                    gap_clips = self._render_base_timeline(
                        start_t=current_time,
                        duration=gap_dur,
                        out_dir=temp_dir / f"gap_{idx:02d}",
                        transcription=transcription,
                        progress_offset=10 + int((idx / total_items) * 75),
                        progress_span=int((1 / total_items) * 75),
                    )
                    timeline_clips.extend(gap_clips)

            # 2. Render styled interview clip
            self._status(f"Styling interview clip {idx+1}/{len(confirmed_clips)}: {clip['label']} ({dur:.1f}s)...")
            styled_clip_path = temp_dir / f"styled_interview_{idx:02d}.mp4"
            self._render_single_styled_interview_clip(clip, styled_clip_path, temp_dir)
            timeline_clips.append(styled_clip_path)

            current_time = t_end

        # 3. Trailing base visual footage after last interview clip
        if total_duration > current_time:
            tail_dur = round(total_duration - current_time, 2)
            if tail_dur >= 0.1:
                logger.info("Rendering base visual tail [%.2fs - %.2fs (%.2fs)]", current_time, total_duration, tail_dur)
                tail_clips = self._render_base_timeline(
                    start_t=current_time,
                    duration=tail_dur,
                    out_dir=temp_dir / "tail_gap",
                    transcription=transcription,
                    progress_offset=80,
                    progress_span=8,
                )
                timeline_clips.extend(tail_clips)

        return timeline_clips

    def _render_single_styled_interview_clip(
        self,
        clip: dict[str, Any],
        output_path: Path,
        temp_dir: Path,
    ) -> Path:
        """
        Renders a single interview clip with:
          - Scale (e.g. 70%)
          - Background blur layer
          - Rounded corners (alpha mask) or sharp edges
          - Frame border (color + line thickness)
        """
        cfg_interview = self.cfg.interview_config or {}
        clip_scale = max(0.1, min(1.0, float(cfg_interview.get("clip_scale", 0.70))))

        # Background image & blur configuration
        bg_image_raw = cfg_interview.get("bg_image_path") or getattr(self.cfg, "bg_image_path", None)
        bg_image_path: Optional[Path] = Path(str(bg_image_raw).strip()) if bg_image_raw and str(bg_image_raw).strip() else None
        has_bg_image = bg_image_path is not None and bg_image_path.is_file()

        bg_blur_val = cfg_interview.get("bg_blur_amount")
        if bg_blur_val is None:
            bg_blur_val = cfg_interview.get("bg_blur", getattr(self.cfg, "bg_blur_amount", 0.0))
        try:
            bg_blur_float = float(bg_blur_val)
        except (ValueError, TypeError):
            bg_blur_float = 0.0

        if bg_blur_float <= 0.0:
            blur_amount = 0
        elif bg_blur_float <= 1.0:
            blur_amount = max(1, min(50, int(round(bg_blur_float * 40))))
        else:
            blur_amount = max(1, min(50, int(round(bg_blur_float))))

        border_enabled = bool(cfg_interview.get("border_enabled", True))
        border_color = str(cfg_interview.get("border_color", "#EF4444")).strip()
        border_thickness = max(0, min(30, int(cfg_interview.get("border_thickness", 4))))
        corner_shape = str(cfg_interview.get("corner_shape", "rounded")).lower()
        corner_curve = max(0, min(120, int(cfg_interview.get("corner_curve", 24))))

        w, h = RESOLUTIONS.get(self.cfg.resolution, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        source_vid = Path(clip["source_video"])
        start_sec = clip["trim_start_sec"]
        end_sec = clip["trim_end_sec"]
        dur = clip["duration_sec"]

        # Probe source video aspect ratio
        try:
            info = probe_media(source_vid)
            sw = info.width if info.width > 0 else 1080
            sh = info.height if info.height > 0 else 1920
        except Exception:
            sw, sh = 1080, 1920

        # 4-Side Screen Area / Crop Margins
        crop_top = max(0, int(clip.get("crop_top", 0)))
        crop_bottom = max(0, int(clip.get("crop_bottom", 0)))
        crop_left = max(0, int(clip.get("crop_left", 0)))
        crop_right = max(0, int(clip.get("crop_right", 0)))

        # Fallback to normalized ratio coordinates if pixel margins are 0 but norm_w < 1.0
        if crop_top == 0 and crop_bottom == 0 and crop_left == 0 and crop_right == 0 and float(clip.get("norm_w", 1.0)) < 0.999:
            nx = float(clip.get("norm_x", 0.0))
            ny = float(clip.get("norm_y", 0.0))
            nw = float(clip.get("norm_w", 1.0))
            nh = float(clip.get("norm_h", 1.0))
            crop_left = int(round(nx * sw))
            crop_top = int(round(ny * sh))
            crop_right = max(0, sw - crop_left - int(round(nw * sw)))
            crop_bottom = max(0, sh - crop_top - int(round(nh * sh)))

        crop_filt = build_crop_filter(
            source_width=sw,
            source_height=sh,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            crop_right=crop_right,
        )

        if crop_filt:
            logger.info("Applying exact interview clip crop filter: %s (source: %dx%d)", crop_filt, sw, sh)
            parts = crop_filt.replace("crop=", "").split(":")
            cw = int(parts[0])
            ch = int(parts[1])
            ar = cw / ch
        else:
            ar = sw / sh

        # Calculate foreground dimensions to preserve aspect ratio within scaled bounding box
        max_w = w * clip_scale
        max_h = h * clip_scale
        if ar < (max_w / max_h):
            fh = int(round(max_h))
            fw = int(round(fh * ar))
        else:
            fw = int(round(max_w))
            fh = int(round(fw / ar))

        fw = fw if fw % 2 == 0 else fw - 1
        fh = fh if fh % 2 == 0 else fh - 1
        fw = max(fw, 16)
        fh = max(fh, 16)

        # Generate Alpha Mask (for rounded corners) and Border Overlay via Pillow
        mask_path: Optional[Path] = None
        border_path: Optional[Path] = None
        has_rounded = (corner_shape == "rounded" and corner_curve > 0)
        radius = min(corner_curve, min(fw, fh) // 2) if has_rounded else 0

        prefix = f"clip_{clip.get('segment_id', 0)}"
        if has_rounded:
            mask_img = Image.new("L", (fw, fh), 0)
            d_mask = ImageDraw.Draw(mask_img)
            d_mask.rounded_rectangle([(0, 0), (fw - 1, fh - 1)], radius=radius, fill=255)
            mask_path = temp_dir / f"{prefix}_mask.png"
            mask_img.save(mask_path)

        if border_enabled and border_thickness > 0:
            border_img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            d_border = ImageDraw.Draw(border_img)
            b_color = border_color if border_color.startswith("#") else f"#{border_color}"
            if has_rounded:
                d_border.rounded_rectangle([(0, 0), (fw - 1, fh - 1)], radius=radius, outline=b_color, width=border_thickness)
            else:
                d_border.rectangle([(0, 0), (fw - 1, fh - 1)], outline=b_color, width=border_thickness)
            border_path = temp_dir / f"{prefix}_border.png"
            border_img.save(border_path)

        # Real Person Segmentation / Foreground Subject Isolation:
        # If a custom background image is provided (or bg_removal is explicitly enabled),
        # remove original background from interview speaker using neural segmentation
        bg_removal = bool(cfg_interview.get("bg_removal", True))
        person_mask_path: Optional[Path] = None
        if has_bg_image and bg_removal:
            try:
                from services.interview_segmenter import generate_person_mask_video
                p_mask = temp_dir / f"{prefix}_person_mask.mp4"
                res_mask = generate_person_mask_video(
                    source_video=source_vid,
                    start_sec=start_sec,
                    duration_sec=dur,
                    output_mask_path=p_mask,
                    crop_filter=crop_filt,
                    target_width=fw,
                    target_height=fh,
                    fps=OUTPUT_FPS,
                )
                if res_mask and p_mask.exists():
                    person_mask_path = p_mask
            except Exception as exc:
                logger.warning("Person segmentation could not be computed for %s, falling back: %s", prefix, exc)
                person_mask_path = None

        # Input tracking
        extra_inputs = []
        next_input_idx = 1

        bg_idx: Optional[int] = None
        if has_bg_image and bg_image_path is not None:
            extra_inputs.extend(["-loop", "1", "-framerate", str(OUTPUT_FPS), "-i", str(bg_image_path)])
            bg_idx = next_input_idx
            next_input_idx += 1

        person_mask_idx: Optional[int] = None
        if person_mask_path and person_mask_path.exists():
            extra_inputs.extend(["-i", str(person_mask_path)])
            person_mask_idx = next_input_idx
            next_input_idx += 1

        mask_idx = None
        if mask_path and mask_path.exists():
            extra_inputs.extend(["-i", str(mask_path)])
            mask_idx = next_input_idx
            next_input_idx += 1

        border_idx = None
        if border_path and border_path.exists():
            extra_inputs.extend(["-i", str(border_path)])
            border_idx = next_input_idx
            next_input_idx += 1

        # Build FFmpeg Filter Complex
        # Background layer:
        if bg_idx is not None:
            # User custom background image
            if blur_amount > 0:
                bg_filt = f"[{bg_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur={blur_amount}[bg]"
            else:
                bg_filt = f"[{bg_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}[bg]"
        else:
            # Fallback when no custom background image is provided or file is missing
            if blur_amount > 0:
                bg_filt = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur={blur_amount}[bg]"
            else:
                bg_filt = f"color=c=#121216:s={w}x{h}:r={OUTPUT_FPS}:d={dur:.3f}[bg]"

        # Foreground & compositing:
        filt_parts = [bg_filt]
        fg_crop_prefix = f"{crop_filt}," if crop_filt else ""

        if person_mask_idx is not None:
            logger.info("Compositing background-removed interview foreground subject onto blurred background")
            if mask_idx is not None:
                filt_parts.append(
                    f"[0:v]{fg_crop_prefix}scale={fw}:{fh}:flags=lanczos[fg];"
                    f"[{person_mask_idx}:v]scale={fw}:{fh}[fg_pmask];"
                    f"[{mask_idx}:v]scale={fw}:{fh}[fg_rmask];"
                    f"[fg_pmask][fg_rmask]blend=all_mode=multiply[fg_comb_mask];"
                    f"[fg][fg_comb_mask]alphamerge[fg_alpha];"
                    f"[bg][fg_alpha]overlay=(W-w)/2:(H-h)/2[layer1]"
                )
            else:
                filt_parts.append(
                    f"[0:v]{fg_crop_prefix}scale={fw}:{fh}:flags=lanczos[fg];"
                    f"[{person_mask_idx}:v]scale={fw}:{fh}[fg_mask];"
                    f"[fg][fg_mask]alphamerge[fg_alpha];"
                    f"[bg][fg_alpha]overlay=(W-w)/2:(H-h)/2[layer1]"
                )
            base_layer = "[layer1]"
        elif mask_idx is not None:
            filt_parts.append(
                f"[0:v]{fg_crop_prefix}scale={fw}:{fh}:flags=lanczos[fg];"
                f"[fg][{mask_idx}:v]alphamerge[fg_masked];"
                f"[bg][fg_masked]overlay=(W-w)/2:(H-h)/2[layer1]"
            )
            base_layer = "[layer1]"
        else:
            filt_parts.append(
                f"[0:v]{fg_crop_prefix}scale={fw}:{fh}:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[layer1]"
            )
            base_layer = "[layer1]"

        if border_idx is not None:
            filt_parts.append(f"{base_layer}[{border_idx}:v]overlay=(W-w)/2:(H-h)/2[vout]")
        else:
            filt_parts[-1] = filt_parts[-1].replace("[layer1]", "[vout]")

        filter_complex = ";".join(filt_parts)
        enc = _get_encoder_args()

        cmd = [
            _ff(),
            "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(source_vid),
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-t", f"{dur:.3f}",
            *enc,
            "-r", str(OUTPUT_FPS),
            "-an",  # Audio will be combined cleanly in the master audio assembly pass
            str(output_path),
        ]

        logger.debug("Compositing interview clip #%s: %s", clip.get("segment_id", 0), " ".join(cmd[:12]))
        _run_with_progress(cmd, dur, label=f"Style clip {output_path.stem}")
        return output_path

    def _render_base_timeline(
        self,
        start_t: float,
        duration: float,
        out_dir: Path,
        transcription: TranscriptionResult,
        progress_offset: int = 10,
        progress_span: int = 70,
    ) -> list[Path]:
        """
        Renders Ken Burns image clips and video b-roll from the assets directory to fill a gap of `duration`.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        mixer = HybridMediaMixer(
            images_dir=self.cfg.footage_path,
            videos_path=self.cfg.footage_path,
            clip_duration=self.cfg.clip_duration,
            image_ratio=self.cfg.image_ratio,
        )
        visual_slots = mixer.create_mix(
            total_duration=duration,
            transcription=transcription,
        )

        clips: list[Path] = []
        total_slots = len(visual_slots)

        for i, slot in enumerate(visual_slots):
            clip_file = out_dir / f"base_clip_{i:03d}.mp4"
            if slot.media_type == "image":
                build_ken_burns_clip(
                    image_path=slot.source_path,
                    duration=slot.duration,
                    output_path=clip_file,
                    preset=slot.preset,
                    resolution=self.cfg.resolution,
                )
            else:
                extract_subclip(
                    video_path=slot.source_path,
                    start_sec=slot.subclip_start,
                    end_sec=slot.subclip_end,
                    output_path=clip_file,
                    resolution=self.cfg.resolution,
                    target_duration=slot.duration,
                    keep_audio=False,
                )
            clips.append(clip_file)

        return clips

    def _assemble_master_audio(
        self,
        confirmed_clips: list[dict[str, Any]],
        total_duration: float,
        output_audio_path: Path,
        temp_dir: Path,
    ) -> Path:
        """
        Builds the unified master audio track:
          - Outside interview clips: plays main voiceover audio cleanly.
          - During interview clips: mutes main voiceover, plays interview audio.
          - If interview BGM is configured: mixes BGM strictly during interview windows at music_volume.
        """
        if not confirmed_clips:
            # No interview clips: copy or return main voiceover audio
            cmd = [_ff(), "-y", "-i", str(self.cfg.audio_path), "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", str(output_audio_path)]
            _run_with_progress(cmd, total_duration, label="Export voiceover audio")
            return output_audio_path

        audio_parts: list[Path] = []
        current_time = 0.0

        interview_config = self.cfg.interview_config or {}
        bgm_path_str = interview_config.get("music_path", "")
        bgm_volume = float(interview_config.get("music_volume", 0.15))
        has_bgm = bool(bgm_path_str and Path(bgm_path_str).exists() and bgm_volume > 0)
        bgm_file = Path(bgm_path_str) if has_bgm else None

        for idx, clip in enumerate(confirmed_clips):
            t_start = clip["target_start_sec"]
            t_end = clip["target_end_sec"]
            dur = clip["duration_sec"]

            # 1. Main voiceover gap before interview clip
            if t_start > current_time:
                gap_dur = round(t_start - current_time, 2)
                if gap_dur >= 0.05:
                    gap_audio = temp_dir / f"audio_vo_gap_{idx:02d}.wav"
                    cmd_vo = [
                        _ff(), "-y",
                        "-ss", f"{current_time:.3f}",
                        "-t", f"{gap_dur:.3f}",
                        "-i", str(self.cfg.audio_path),
                        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                        str(gap_audio),
                    ]
                    _run_with_progress(cmd_vo, gap_dur, label=f"Extract VO gap {idx}")
                    audio_parts.append(gap_audio)

            # 2. Extract interview clip audio (+ optional ducked BGM)
            interview_audio = temp_dir / f"audio_interview_{idx:02d}.wav"
            source_vid = Path(clip["source_video"])
            trim_start = clip["trim_start_sec"]

            try:
                info = probe_media(source_vid)
                has_native_audio = info.has_audio
            except Exception:
                has_native_audio = False

            if has_native_audio and has_bgm and bgm_file:
                # Mix interview audio with ducked BGM
                filt = (
                    f"[0:a]volume=1.0[a0];"
                    f"[1:a]volume={bgm_volume:.3f}[a1];"
                    f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
                cmd_int = [
                    _ff(), "-y",
                    "-ss", f"{trim_start:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", str(source_vid),
                    "-stream_loop", "-1",
                    "-i", str(bgm_file),
                    "-filter_complex", filt,
                    "-map", "[aout]",
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    str(interview_audio),
                ]
            elif has_native_audio:
                # Native interview audio only
                cmd_int = [
                    _ff(), "-y",
                    "-ss", f"{trim_start:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", str(source_vid),
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    str(interview_audio),
                ]
            elif has_bgm and bgm_file:
                # No native audio stream, but BGM is present
                cmd_int = [
                    _ff(), "-y",
                    "-stream_loop", "-1",
                    "-i", str(bgm_file),
                    "-t", f"{dur:.3f}",
                    "-filter_complex", f"[0:a]volume={bgm_volume:.3f}[aout]",
                    "-map", "[aout]",
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    str(interview_audio),
                ]
            else:
                # Generate clean silence for the duration
                cmd_int = [
                    _ff(), "-y",
                    "-f", "lavfi",
                    "-i", f"anullsrc=r=44100:cl=stereo:d={dur:.3f}",
                    "-t", f"{dur:.3f}",
                    "-c:a", "pcm_s16le",
                    str(interview_audio),
                ]

            _run_with_progress(cmd_int, dur, label=f"Mix interview audio {idx}")
            audio_parts.append(interview_audio)
            current_time = t_end

        # 3. Trailing voiceover after last interview clip
        if total_duration > current_time:
            tail_dur = round(total_duration - current_time, 2)
            if tail_dur >= 0.05:
                tail_audio = temp_dir / "audio_vo_tail.wav"
                cmd_tail = [
                    _ff(), "-y",
                    "-ss", f"{current_time:.3f}",
                    "-t", f"{tail_dur:.3f}",
                    "-i", str(self.cfg.audio_path),
                    "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    str(tail_audio),
                ]
                _run_with_progress(cmd_tail, tail_dur, label="Extract VO tail")
                audio_parts.append(tail_audio)

        # Concatenate audio parts together
        concat_txt = temp_dir / "audio_concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in audio_parts:
                f.write(f"file '{p.resolve()}'\n")

        cmd_merge = [
            _ff(), "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_audio_path),
        ]
        _run_with_progress(cmd_merge, total_duration, label="Merge master audio")
        return output_audio_path

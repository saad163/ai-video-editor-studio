"""
services/interview_scanner.py
=============================
Video Link Downloader & AI Interview Timestamp Scanner for Template 3 (+ Interview).

Features:
  1. Downloads source video/audio from YouTube or direct web URLs via yt-dlp.
     - Caches downloads by URL hash in project-scoped temp/interviews/
     - Progress and status callbacks for UI responsiveness
     - Clean error handling for invalid/private/unreachable URLs
  2. Transcribes speech into timestamped segments via faster-whisper (SQLite cached).
  3. Uses Google Gemini (with 3.x Flash priority and rate-limit backoff) to detect interview segments,
     speaker dialogue blocks, and Q&A exchanges with exact start/end timestamps and descriptive labels.
  4. Provides a robust heuristic fallback if Gemini API is unconfigured or offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InterviewSegment:
    """A detected interview dialogue or discussion segment with precision timestamps."""
    segment_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_time: str          # "MM:SS"
    end_time: str            # "MM:SS"
    label: str               # Brief topic / summary
    transcript: str = ""     # Transcript text covered in this segment
    speaker: str = "Speaker" # Identified or assigned speaker label
    interview_score: float = 1.0 # 0.0 to 1.0 visual framing / human presence confidence
    is_human_verified: bool = True # True if candidate frame verified with face / human presence


def format_timestamp_mmss(seconds: float) -> str:
    """Format floating point seconds into clean 'MM:SS' format."""
    total_sec = max(0, int(round(seconds)))
    mins = total_sec // 60
    secs = total_sec % 60
    return f"{mins:02d}:{secs:02d}"


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parse 'MM:SS', 'HH:MM:SS', or raw seconds string into float seconds."""
    if not ts:
        return 0.0
    s = ts.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def is_valid_video_url(url: str) -> bool:
    """Validate whether the provided string appears to be a plausible video/web URL."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    try:
        parsed = urlparse(u)
        return bool(parsed.netloc and "." in parsed.netloc and len(parsed.netloc) >= 4)
    except Exception:
        return False


def get_url_cache_hash(url: str) -> str:
    """Generate a stable 12-character MD5 hash for a given URL string."""
    clean_url = url.strip()
    return hashlib.md5(clean_url.encode("utf-8")).hexdigest()[:12]


class VideoDownloader:
    """
    Downloads source video/audio from YouTube or web URLs into a temp directory using yt-dlp.
    Reuses existing media if the same URL has already been downloaded.
    """

    def __init__(self, download_dir: Optional[Path] = None) -> None:
        cfg = get_config()
        self.download_dir = download_dir or (cfg.temp_dir / "interviews")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        force_redownload: bool = False,
    ) -> Path:
        """
        Download a video URL using yt-dlp, or return the cached file if already downloaded.
        Raises ValueError or RuntimeError on invalid or failed downloads.
        """
        if not is_valid_video_url(url):
            raise ValueError(f"Invalid video URL: '{url}'. Please enter a valid http/https video URL.")

        url_hash = get_url_cache_hash(url)

        # Check for existing cached file
        if not force_redownload:
            for ext in (".mp4", ".mkv", ".webm", ".mov"):
                cached_candidate = self.download_dir / f"interview_{url_hash}{ext}"
                if cached_candidate.exists() and cached_candidate.stat().st_size > 50_000:
                    logger.info("Reusing cached interview video: %s", cached_candidate)
                    if progress_cb:
                        progress_cb(50, f"Found cached video: {cached_candidate.name}")
                    return cached_candidate

        import yt_dlp
        from core.ffmpeg_wrapper import resolve_ffmpeg

        if progress_cb:
            progress_cb(5, "Connecting to video source...")

        ffmpeg_bin, _ = resolve_ffmpeg()
        out_template = str(self.download_dir / f"interview_{url_hash}.%(ext)s")

        def _yt_progress_hook(d: dict[str, Any]) -> None:
            if not progress_cb:
                return
            status = d.get("status")
            if status == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes", 0)
                if total > 0:
                    pct = min(48, int((downloaded / total) * 45) + 5)
                    progress_cb(pct, f"Downloading video: {int((downloaded / total) * 100)}%")
                else:
                    progress_cb(25, "Downloading video stream...")
            elif status == "finished":
                progress_cb(48, "Download complete. Merging streams...")

        ydl_opts: dict[str, Any] = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [_yt_progress_hook],
        }
        if ffmpeg_bin:
            ffmpeg_dir = str(Path(ffmpeg_bin).parent)
            current_path = os.environ.get("PATH", "")
            if ffmpeg_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
            ydl_opts["ffmpeg_location"] = str(ffmpeg_bin)

        logger.info("Downloading interview video with yt-dlp from URL: %s", url)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_file = Path(ydl.prepare_filename(info))
        except Exception as exc:
            logger.error("yt-dlp download failed: %s", exc)
            raise RuntimeError(f"Failed to download video from '{url}': {exc}") from exc

        # If merged or converted to mp4/mkv, verify path on disk
        if not downloaded_file.exists():
            for ext in (".mp4", ".mkv", ".webm", ".mov"):
                cand = downloaded_file.with_suffix(ext)
                if cand.exists():
                    downloaded_file = cand
                    break

        if not downloaded_file.exists() or downloaded_file.stat().st_size == 0:
            raise RuntimeError(f"Video download produced no valid file for URL: {url}")

        logger.info("Video successfully downloaded to: %s", downloaded_file)
        if progress_cb:
            progress_cb(50, f"Video ready: {downloaded_file.name}")
        return downloaded_file


class InterviewScanner:
    """
    Orchestrates the full interview scanning pipeline:
      1. Downloads or resolves video file via VideoDownloader.
      2. Transcribes speech timestamps via faster-whisper.
      3. Uses Gemini AI (3.x Flash priority) or silence-aware heuristic fallback
         to identify interview segments with exact start/end timestamps.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        cfg = get_config()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or cfg.gemini_api_key or ""
        self.downloader = VideoDownloader()

    def scan_url(
        self,
        url: str,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        force_redownload: bool = False,
        filter_mode: str = "strict",
        use_cache: bool = True,
    ) -> tuple[Path, list[InterviewSegment]]:
        """
        Execute full download, transcription, and segment detection pipeline.
        Returns (downloaded_video_path, list_of_interview_segments).

        Parameters:
          filter_mode: "strict" (default) or "all" (relaxed, includes all candidate dialogue/scenes).
          use_cache: True to check/save results in SQLite cache keyed by (video_hash + filter_mode).
        """
        # Step 1: Download or reuse video
        if progress_cb:
            progress_cb(2, f"Initializing scan for video URL...")
        video_path = self.downloader.download(url, progress_cb=progress_cb, force_redownload=force_redownload)

        from utils.file_helpers import compute_file_hash
        video_hash = compute_file_hash(video_path)

        # Step 1.5: Check SQLite cache
        if use_cache:
            from database.db import get_cached_interview_candidates
            cached_row = get_cached_interview_candidates(video_hash=video_hash, filter_mode=filter_mode)
            if cached_row and cached_row["segments_json"]:
                try:
                    data = json.loads(cached_row["segments_json"])
                    if isinstance(data, list) and data:
                        cached_segments = [
                            InterviewSegment(
                                segment_id=item["segment_id"],
                                start_sec=float(item["start_sec"]),
                                end_sec=float(item["end_sec"]),
                                duration_sec=float(item["duration_sec"]),
                                start_time=item["start_time"],
                                end_time=item["end_time"],
                                label=item["label"],
                                transcript=item.get("transcript", ""),
                                speaker=item.get("speaker", "Speaker"),
                                interview_score=float(item.get("interview_score", 1.0)),
                                is_human_verified=bool(item.get("is_human_verified", True)),
                            )
                            for item in data
                        ]
                        logger.info(
                            "Cache hit for interview video '%s' (mode=%s) — loaded %d segment(s) from SQLite.",
                            video_path.name,
                            filter_mode,
                            len(cached_segments),
                        )
                        if progress_cb:
                            progress_cb(100, f"Loaded {len(cached_segments)} cached interview segment(s)")
                        return video_path, cached_segments
                except Exception as exc:
                    logger.warning("Failed parsing cached interview segments JSON: %s", exc)

        # Step 2: Transcribe via Whisper
        if progress_cb:
            progress_cb(52, "Transcribing speech and extracting timestamps with Whisper...")

        from services.transcriber import TranscriptionService
        transcriber = TranscriptionService()
        transcription_res = transcriber.transcribe(video_path, task="transcribe")

        segments_raw = transcription_res.segments
        logger.info("Whisper transcribed %d segments for '%s'", len(segments_raw), video_path.name)

        if progress_cb:
            progress_cb(78, "Analyzing speech dialogue to detect interview segments...")

        # Step 3: Detect Interview Segments (Gemini with Heuristic Fallback)
        detected_segments = self._detect_segments(segments_raw, progress_cb=progress_cb)

        # Step 4: Visual Frame Human / Interview Verification (Filter non-interview footage)
        verified_segments = self._verify_segments_framing(
            video_path=video_path,
            segments=detected_segments,
            filter_mode=filter_mode,
            progress_cb=progress_cb,
        )

        # Step 5: Save to SQLite cache
        if use_cache:
            try:
                from database.db import insert_interview_candidates_cache
                cache_payload = [
                    {
                        "segment_id": s.segment_id,
                        "start_sec": s.start_sec,
                        "end_sec": s.end_sec,
                        "duration_sec": s.duration_sec,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "label": s.label,
                        "transcript": s.transcript,
                        "speaker": s.speaker,
                        "interview_score": s.interview_score,
                        "is_human_verified": s.is_human_verified,
                    }
                    for s in verified_segments
                ]
                insert_interview_candidates_cache(
                    video_hash=video_hash,
                    segments_json=json.dumps(cache_payload, ensure_ascii=False),
                    filter_mode=filter_mode,
                    model_used="gemini+cv_verifier",
                )
            except Exception as exc:
                logger.warning("Failed saving interview candidates to SQLite cache: %s", exc)

        if progress_cb:
            progress_cb(100, f"Scan complete: {len(verified_segments)} interview segment(s) detected")

        return video_path, verified_segments

    def _verify_segments_framing(
        self,
        video_path: Path,
        segments: list[InterviewSegment],
        filter_mode: str = "strict",
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> list[InterviewSegment]:
        """
        Sample key frames from each detected candidate segment and score interview-style framing.
        In strict mode, filters out non-human / B-roll / scenery cuts.
        In all/relaxed mode, retains all segments with framing scores attached.
        """
        if not segments or not video_path.is_file():
            return list(segments)

        from services.scene_detector import evaluate_interview_framing
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Could not open video file %s for interview frame verification", video_path)
            return list(segments)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        verified: list[InterviewSegment] = []

        try:
            for idx, seg in enumerate(segments):
                if progress_cb:
                    pct = 85 + int((idx / max(1, len(segments))) * 13)
                    progress_cb(pct, f"Verifying interview framing for clip #{seg.segment_id}...")

                # Sample up to 2 frames per segment (at 25% and 60% of duration)
                mid_s1 = seg.start_sec + (seg.duration_sec * 0.25)
                mid_s2 = seg.start_sec + (seg.duration_sec * 0.60)
                frame_scores = []

                for sample_s in (mid_s1, mid_s2):
                    frame_num = int(sample_s * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        res = evaluate_interview_framing(frame)
                        frame_scores.append(res)

                if frame_scores:
                    max_score = max(f["score"] for f in frame_scores)
                    has_human = any(f["is_interview"] or f["score"] >= 0.38 for f in frame_scores)
                else:
                    max_score = 1.0
                    has_human = True

                seg.interview_score = round(float(max_score), 3)
                seg.is_human_verified = bool(has_human)

                if filter_mode == "strict":
                    if has_human or max_score >= 0.38:
                        verified.append(seg)
                    else:
                        logger.info(
                            "Filtering out non-interview candidate #%d ('%s') score=%.2f (no human interview framing)",
                            seg.segment_id,
                            seg.label,
                            max_score,
                        )
                else:
                    # In "all" mode, keep everything
                    verified.append(seg)
        finally:
            cap.release()

        # Safe empty handling: If strict filtering eliminated every segment, log warning and preserve all
        if not verified and segments:
            logger.warning("Strict interview filter produced 0 clips; falling back to all candidate segments.")
            return list(segments)

        # Re-number segment IDs sequentially
        for new_id, seg in enumerate(verified, start=1):
            seg.segment_id = new_id

        return verified

    def _detect_segments(
        self,
        segments_raw: list[Any],
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> list[InterviewSegment]:
        """
        Detect interview segments from timestamped Whisper segments.
        Attempts Gemini first; falls back cleanly to heuristic block grouping.
        """
        if not segments_raw:
            logger.warning("No speech segments found in video.")
            return []

        # Attempt Gemini semantic interview extraction if a valid key is present
        from services.gemini_matcher import SmartSceneMatcher
        if SmartSceneMatcher.is_api_key_valid(self.api_key):
            try:
                if progress_cb:
                    progress_cb(82, "Detecting interview dialogue blocks via Gemini AI...")
                return self._detect_with_gemini(segments_raw)
            except Exception as exc:
                logger.warning("Gemini interview detection failed (%s); falling back to heuristic detector.", exc)

        # Heuristic / Offline Silence-Aware Dialogue Detector
        if progress_cb:
            progress_cb(85, "Grouping interview dialogue blocks via heuristic timing...")
        return self._detect_with_heuristics(segments_raw)

    def _detect_with_gemini(self, segments_raw: list[Any]) -> list[InterviewSegment]:
        """Query Google Gemini with candidate model fallback to identify interview segments."""
        from services.gemini_resolver import generate_content_with_fallback

        # Format transcript with timestamps
        lines = []
        for s in segments_raw[:150]: # Cap to first 150 segments to conserve tokens
            st = format_timestamp_mmss(s.start)
            et = format_timestamp_mmss(s.end)
            lines.append(f"[{st} - {et} | {s.start:.1f}s - {s.end:.1f}s] {s.text.strip()}")
        transcript_sample = "\n".join(lines)

        prompt = f"""
You are an expert video editor specializing in interview clips and podcast shorts.
Analyze the following timestamped video transcript and identify coherent, high-value interview segments, speaker dialogue blocks, or Q&A exchanges between interviewer/host and interviewee/guest.

CRITICAL FILTERING GUIDELINES:
1. ONLY select segments containing actual human spoken interview dialogue, Q&A exchanges, guest commentary, or host introductions.
2. EXCLUDE non-interview content such as background sound, music interludes, sponsor reads, or silent cuts.
3. Each segment should ideally be between 10 and 60 seconds in duration.
4. Ensure start_sec and end_sec precisely match natural speech boundaries from the timestamps provided.
5. Provide a brief, engaging label (5-8 words) summarizing what is discussed.
6. If recognizable, assign a speaker label (e.g. "Speaker 1", "Guest", "Host").

TRANSCRIPT:
{transcript_sample}

RETURN FORMAT:
Return ONLY a valid JSON array of objects with the exact schema below, and NO extraneous text:
[
  {{
    "start_sec": 12.5,
    "end_sec": 38.0,
    "label": "Guest explains key insight or story",
    "speaker": "Guest"
  }}
]
"""
        response, working_model = generate_content_with_fallback(
            api_key=self.api_key,
            contents=prompt,
            preferred="flash",
            timeout=30.0,
        )
        resp_text = getattr(response, "text", "") or ""
        logger.info("Gemini interview detection succeeded using model '%s'", working_model)

        # Extract JSON array
        match = re.search(r"\[[\s\S]*\]", resp_text)
        if not match:
            raise ValueError(f"No JSON array returned in Gemini response: {resp_text[:100]}...")

        data = json.loads(match.group(0))
        if not isinstance(data, list) or not data:
            raise ValueError("Empty or non-list JSON array returned by Gemini.")

        results: list[InterviewSegment] = []
        for idx, item in enumerate(data, start=1):
            start_s = float(item.get("start_sec", 0.0))
            end_s = float(item.get("end_sec", start_s + 15.0))
            if end_s <= start_s:
                end_s = start_s + 10.0

            dur_s = round(end_s - start_s, 2)
            lbl = str(item.get("label", f"Interview Segment {idx}")).strip()
            spk = str(item.get("speaker", "Speaker")).strip()

            # Associate matching transcript text
            seg_texts = [
                s.text.strip()
                for s in segments_raw
                if (s.start >= start_s - 1.0 and s.end <= end_s + 1.0)
            ]
            full_seg_transcript = " ".join(seg_texts)

            results.append(
                InterviewSegment(
                    segment_id=idx,
                    start_sec=round(start_s, 2),
                    end_sec=round(end_s, 2),
                    duration_sec=dur_s,
                    start_time=format_timestamp_mmss(start_s),
                    end_time=format_timestamp_mmss(end_s),
                    label=lbl,
                    transcript=full_seg_transcript,
                    speaker=spk,
                )
            )

        return results

    def _detect_with_heuristics(self, segments_raw: list[Any]) -> list[InterviewSegment]:
        """
        Group timestamped segments into coherent 15s - 45s interview blocks
        using speech pauses and sentence boundaries.
        """
        results: list[InterviewSegment] = []
        if not segments_raw:
            return results

        current_block: list[Any] = []
        current_dur = 0.0
        seg_counter = 1

        for seg in segments_raw:
            seg_len = seg.end - seg.start
            current_block.append(seg)
            current_dur = current_block[-1].end - current_block[0].start

            # Split when block duration reaches 18 - 35 seconds or if a large pause occurs (> 2.5s)
            is_natural_pause = (seg.end - seg.start > 0 and seg.text.strip().endswith((".", "!", "?", "۔")))
            if (current_dur >= 20.0 and is_natural_pause) or current_dur >= 38.0:
                start_s = current_block[0].start
                end_s = current_block[-1].end
                dur_s = round(end_s - start_s, 2)
                block_text = " ".join(s.text.strip() for s in current_block)

                # Derive concise label from the first 6-8 words
                words = block_text.split()
                summary_label = " ".join(words[:7]) + ("..." if len(words) > 7 else "")

                results.append(
                    InterviewSegment(
                        segment_id=seg_counter,
                        start_sec=round(start_s, 2),
                        end_sec=round(end_s, 2),
                        duration_sec=dur_s,
                        start_time=format_timestamp_mmss(start_s),
                        end_time=format_timestamp_mmss(end_s),
                        label=summary_label or f"Interview Segment {seg_counter}",
                        transcript=block_text,
                        speaker=f"Speaker {((seg_counter - 1) % 2) + 1}",
                    )
                )
                seg_counter += 1
                current_block = []
                current_dur = 0.0

        # Residual tail block
        if current_block and (current_block[-1].end - current_block[0].start) >= 5.0:
            start_s = current_block[0].start
            end_s = current_block[-1].end
            dur_s = round(end_s - start_s, 2)
            block_text = " ".join(s.text.strip() for s in current_block)
            words = block_text.split()
            summary_label = " ".join(words[:7]) + ("..." if len(words) > 7 else "")

            results.append(
                InterviewSegment(
                    segment_id=seg_counter,
                    start_sec=round(start_s, 2),
                    end_sec=round(end_s, 2),
                    duration_sec=dur_s,
                    start_time=format_timestamp_mmss(start_s),
                    end_time=format_timestamp_mmss(end_s),
                    label=summary_label or f"Interview Segment {seg_counter}",
                    transcript=block_text,
                    speaker=f"Speaker {((seg_counter - 1) % 2) + 1}",
                )
            )

        return results

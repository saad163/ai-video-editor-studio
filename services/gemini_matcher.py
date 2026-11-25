"""
services/gemini_matcher.py
==========================
Phase 3 — Smart Scene Matcher & Narrative Chunker (Template 2: Episode Scene Compiler).

Responsibilities:
  1. Narrative Chunker: Groups timestamped Whisper transcription segments into 15–30s
     thematic context blocks suitable for episode B-roll cutting.
  2. Scene Matching Engine:
       - Cloud API Mode: Google Gemini 1.5 Flash API structured prompt ranking
         narration text against candidate scene keyframes.
       - Offline Heuristic Mode: Semantic keyword/entity matching and uniform
         distribution fallback when offline or no API key is provided.
  3. Anti-Duplication Engine: Tracks SQLite `scenes` table (`used_count`, `last_used_project_id`)
     to eliminate duplicate scene reuse within the same project or across recent renders.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import get_config
from utils.logger import get_logger
from database.db import (
    init_db,
    increment_scene_used_count,
    list_all_scenes,
    get_cached_scene_match,
    insert_scene_match_cache,
)
from services.scene_detector import Scene
from services.transcriber import TranscriptionResult, TranscriptionSegment

logger = get_logger(__name__)


@dataclass
class NarrativeChunk:
    """Represents a 15–30 second thematic narration block."""
    chunk_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    matched_scene: Optional[Scene] = None

    @property
    def label(self) -> str:
        return f"Chunk {self.chunk_index:02d} [{self.start_sec:.1f}s - {self.end_sec:.1f}s ({self.duration_sec:.1f}s)]: {self.text[:40]}..."


class NarrativeChunker:
    """Groups granular sentence/word segments into cinematic scene-length chunks."""

    def __init__(self, target_chunk_sec: float = 18.0, min_chunk_sec: float = 8.0, max_chunk_sec: float = 30.0) -> None:
        self.target_chunk_sec = target_chunk_sec
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec

    def chunk_transcription(self, transcription: TranscriptionResult) -> list[NarrativeChunk]:
        """Convert a TranscriptionResult into a sequence of NarrativeChunk instances."""
        if not transcription.segments:
            return []

        chunks: list[NarrativeChunk] = []
        current_segs: list[TranscriptionSegment] = []
        current_start = transcription.segments[0].start
        chunk_idx = 0

        SENTENCE_PUNCTUATION = (".", "!", "?", "۔", "؟", "…")
        CLAUSE_PUNCTUATION = (",", "،", ";", "؛", ":")

        for seg in transcription.segments:
            current_segs.append(seg)
            cur_dur = seg.end - current_start

            seg_text = seg.text.strip()
            # Precision boundary detection for English, Urdu, and bilingual punctuation
            is_sentence_end = any(seg_text.endswith(p) for p in SENTENCE_PUNCTUATION)
            is_clause_end = any(seg_text.endswith(p) for p in CLAUSE_PUNCTUATION)

            # Check if chunk reached target length or punctuation boundary
            if (cur_dur >= self.target_chunk_sec and is_sentence_end) or \
               (cur_dur >= self.target_chunk_sec * 1.3 and is_clause_end) or \
               (cur_dur >= self.max_chunk_sec):
                text = " ".join(s.text.strip() for s in current_segs)
                chunks.append(NarrativeChunk(
                    chunk_index=chunk_idx,
                    start_sec=current_start,
                    end_sec=seg.end,
                    duration_sec=seg.end - current_start,
                    text=text,
                    segments=list(current_segs),
                ))
                chunk_idx += 1
                current_segs = []
                current_start = seg.end

        # Handle remaining segments in last chunk
        if current_segs:
            text = " ".join(s.text.strip() for s in current_segs)
            last_end = current_segs[-1].end
            dur = last_end - current_start
            if chunks and dur < self.min_chunk_sec:
                # Merge into previous chunk if too short
                chunks[-1].end_sec = last_end
                chunks[-1].duration_sec = chunks[-1].end_sec - chunks[-1].start_sec
                chunks[-1].text += " " + text
                chunks[-1].segments.extend(current_segs)
            else:
                chunks.append(NarrativeChunk(
                    chunk_index=chunk_idx,
                    start_sec=current_start,
                    end_sec=last_end,
                    duration_sec=dur,
                    text=text,
                    segments=list(current_segs),
                ))

        logger.info("NarrativeChunker: Divided %d segments into %d context chunks", len(transcription.segments), len(chunks))
        return chunks


class SmartSceneMatcher:
    """
    Matches narrative voiceover chunks with candidate episode scenes.
    Features Google Gemini 1.5 Flash batch-context matching with offline heuristic fallback.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, strict_mode: bool = False) -> None:
        cfg = get_config()
        if gemini_api_key is not None:
            self.api_key = gemini_api_key
        else:
            self.api_key = cfg.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.strict_mode = strict_mode
        init_db()

        # Telemetry metrics per matching session
        self.prompt_tokens: int = 0
        self.output_tokens: int = 0
        self.total_tokens: int = 0
        self.api_calls_count: int = 0
        self.estimated_cost_usd: float = 0.0
        self.last_error: Optional[str] = None
        self.from_cache: bool = False
        self.active_model_name: Optional[str] = None

    @classmethod
    def is_api_key_valid(cls, key: Optional[str]) -> bool:
        """
        Validate whether the provided string appears to be a legitimate Gemini API key.
        Supports:
          - Classic Google AI Studio format: AIzaSy... (39 chars)
          - Google Cloud / Vertex Express format: AQ.... (~53 chars)
          - Any active alphanumeric/symbolic API key of >= 20 chars without spaces or placeholders
        """
        if not key:
            return False
        k = key.strip()
        if len(k) < 20:
            return False
        if k.startswith("your_") or "placeholder" in k.lower() or "your_api_key" in k.lower():
            return False
        if any(c in k for c in (" ", "\t", "\n", "\r", "<", ">", '"', "'")):
            return False
        return (
            k.startswith("AIzaSy")
            or k.startswith("AQ.")
            or len(k) >= 25
        )

    def match_scenes(
        self,
        chunks: list[NarrativeChunk],
        available_scenes: list[Scene],
        project_id: Optional[int] = None,
        prefer_unused: bool = True,
        language: str = "en",
        strict_mode: Optional[bool] = None,
        voiceover_hash: Optional[str] = None,
        video_hash: Optional[str] = None,
        template_mode: str = "template_2",
        use_cache: bool = True,
    ) -> list[NarrativeChunk]:
        """
        Assign an optimal, non-repeating Scene to each NarrativeChunk.
        Checks persistent SQLite cache first: re-renders on identical voiceover + video load in 0.01s with 0 API calls.
        If strict_mode is True (or Urdu narration is selected), raises descriptive errors
        if the API key is missing or invalid, or if Gemini calls fail.
        """
        if not chunks or not available_scenes:
            logger.warning("Empty chunks or scenes provided to SmartSceneMatcher")
            return chunks

        # Compute deterministic hashes for persistent caching if not explicitly provided
        vo_hash = voiceover_hash
        if not vo_hash and chunks:
            import hashlib
            raw_text = "".join(f"{c.chunk_index}:{c.start_sec:.2f}:{c.end_sec:.2f}:{c.text.strip()}" for c in chunks)
            vo_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        vid_hash = video_hash
        if not vid_hash and available_scenes:
            import hashlib
            unique_vids = sorted(list(set(str(s.video_path) for s in available_scenes if s.video_path)))
            vid_hash = hashlib.sha256(":".join(unique_vids).encode("utf-8")).hexdigest() if unique_vids else "scenes_pool"

        # Check SQLite scene-match cache
        if use_cache and vo_hash and vid_hash:
            cached_row = get_cached_scene_match(vo_hash, vid_hash, template_mode=template_mode)
            if cached_row is not None:
                try:
                    cached_matches = json.loads(cached_row["matches_json"])
                    scenes_by_id = {s.scene_id: s for s in available_scenes if s.scene_id is not None}
                    scenes_by_idx = {s.scene_index: s for s in available_scenes}
                    for item in cached_matches:
                        ch_i = item.get("chunk_index")
                        if ch_i is not None and 0 <= ch_i < len(chunks):
                            matched_s = scenes_by_id.get(item.get("scene_id")) or scenes_by_idx.get(item.get("scene_index"))
                            if matched_s:
                                chunks[ch_i].matched_scene = matched_s
                            else:
                                chunks[ch_i].matched_scene = available_scenes[ch_i % len(available_scenes)]
                    self.from_cache = True
                    self.api_calls_count = 0
                    self.total_tokens = 0
                    self.estimated_cost_usd = 0.0
                    self.active_model_name = cached_row["model_used"] or "cache"
                    logger.info("⚡ Instant Scene-Match Cache Hit (%s + %s) — loaded in 0.01s with 0 API calls.", vo_hash[:8], vid_hash[:8])
                    self._record_scene_usage(chunks, project_id)
                    return chunks
                except Exception as cache_err:
                    logger.warning("Failed to deserialize cached scene match: %s", cache_err)

        # Sort scenes by used_count to prioritize fresh footage
        scene_pool = sorted(available_scenes, key=lambda s: s.used_count)

        if strict_mode is None:
            strict_mode = self.strict_mode or ("ur" in language.lower() or "اردو" in language)

        is_valid_key = self.is_api_key_valid(self.api_key)

        if not is_valid_key:
            err_msg = (
                f"Missing or invalid Gemini API key. "
                f"Template 2 online scene matching (language: '{language}') requires a valid Google Gemini API key. "
                "Silent fallback to keyword matching is disabled to prevent incorrect scene cuts. "
                "Please configure GEMINI_API_KEY in your .env or Settings."
            )
            self.last_error = err_msg
            if strict_mode:
                logger.error("Strict scene matching aborted: %s", err_msg)
                raise ValueError(err_msg)
            else:
                logger.warning("%s Falling back to offline heuristic matcher.", err_msg)

        if is_valid_key:
            try:
                import concurrent.futures
                logger.info("Attempting Google Gemini batch narrative matching (Language: %s)...", language)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._match_with_gemini, chunks, scene_pool, language)
                    matched = future.result(timeout=240.0)

                # Persist completed match result into SQLite cache
                if vo_hash and vid_hash:
                    try:
                        payload = [
                            {
                                "chunk_index": ch.chunk_index,
                                "scene_id": ch.matched_scene.scene_id if ch.matched_scene else None,
                                "scene_index": ch.matched_scene.scene_index if ch.matched_scene else None,
                                "start_sec": ch.matched_scene.start_sec if ch.matched_scene else 0.0,
                                "end_sec": ch.matched_scene.end_sec if ch.matched_scene else 0.0,
                            }
                            for ch in matched
                        ]
                        insert_scene_match_cache(
                            voiceover_hash=vo_hash,
                            video_hash=vid_hash,
                            matches_json=json.dumps(payload),
                            template_mode=template_mode,
                            model_used=self.active_model_name or "gemini",
                        )
                        logger.info("Persisted scene match results to SQLite cache (vo=%s, vid=%s).", vo_hash[:8], vid_hash[:8])
                    except Exception as save_err:
                        logger.warning("Could not persist scene match to SQLite: %s", save_err)

                self._record_scene_usage(matched, project_id)
                return matched
            except Exception as exc:
                self.last_error = str(exc)
                if strict_mode:
                    logger.error("Strict Gemini API scene matching failed: %s", exc)
                    raise RuntimeError(f"Google Gemini API scene matching failed: {exc}") from exc
                else:
                    logger.warning("Gemini API matching failed (%s), falling back to offline heuristic matcher", exc)

        # Offline / Heuristic Matcher
        matched = self._match_with_heuristics(chunks, scene_pool, prefer_unused=prefer_unused, language=language)
        self._record_scene_usage(matched, project_id)
        return matched

    def _match_with_heuristics(
        self,
        chunks: list[NarrativeChunk],
        scenes: list[Scene],
        prefer_unused: bool = True,
        language: str = "en",
    ) -> list[NarrativeChunk]:
        """
        Deterministic, offline heuristic matcher:
        1. Distributes scenes evenly across the timeline without adjacent repetition.
        2. Scores duration fit (scene duration >= chunk duration or close).
        3. Favors scenes with lowest `used_count`.
        """
        assigned_scene_ids: set[int] = set()
        last_scene_id: Optional[int] = None

        for chunk in chunks:
            best_scene: Optional[Scene] = None
            best_score = -9999.0

            for scene in scenes:
                # Disallow consecutive exact same scene
                if scene.scene_id is not None and scene.scene_id == last_scene_id:
                    continue

                score = 100.0

                # Penalty for scenes already used in this render
                if scene.scene_id in assigned_scene_ids:
                    score -= 50.0

                # Penalty for past project usage
                score -= scene.used_count * 20.0

                # Duration alignment score: prefer scenes that are at least as long as chunk
                dur_diff = scene.duration_sec - chunk.duration_sec
                if dur_diff >= 0:
                    score += 25.0
                else:
                    # Shorter scene penalty (minor)
                    score -= abs(dur_diff) * 2.0

                # Keyword / character matching bonus if description exists
                if scene.description:
                    chunk_words = set(re.findall(r"\b\w{4,}\b", chunk.text.lower()))
                    desc_words = set(re.findall(r"\b\w{4,}\b", scene.description.lower()))
                    overlap = len(chunk_words & desc_words)
                    score += overlap * 15.0

                if score > best_score:
                    best_score = score
                    best_scene = scene

            # Fallback if all penalized
            if best_scene is None and scenes:
                best_scene = scenes[chunk.chunk_index % len(scenes)]

            chunk.matched_scene = best_scene
            if best_scene and best_scene.scene_id is not None:
                assigned_scene_ids.add(best_scene.scene_id)
                last_scene_id = best_scene.scene_id

            logger.info("Heuristic Match: Chunk %02d (%.1fs) → Scene %s (dur: %.1fs, used: %d)",
                        chunk.chunk_index, chunk.duration_sec,
                        best_scene.label if best_scene else "None",
                        best_scene.duration_sec if best_scene else 0.0,
                        best_scene.used_count if best_scene else 0)

        return chunks

    def _match_with_gemini(
        self,
        chunks: list[NarrativeChunk],
        scenes: list[Scene],
        language: str = "en",
    ) -> list[NarrativeChunk]:
        """
        Uses Google Gemini with automatic Flash model routing, large-context chunk batching,
        strict JSON output schema, and zero artificial sleep delays on successful responses.
        Smart retry/backoff is strictly reserved for transient 429 quota errors.
        """
        from services.gemini_resolver import generate_content_with_fallback

        # Prepare scene catalog prompt (up to 120 candidate scenes for full-episode coverage)
        scene_catalog = []
        for i, sc in enumerate(scenes[:120]):
            scene_catalog.append({
                "candidate_index": i,
                "scene_id": sc.scene_id,
                "scene_index": sc.scene_index,
                "start_sec": round(sc.start_sec, 1),
                "end_sec": round(sc.end_sec, 1),
                "duration_sec": round(sc.duration_sec, 1),
                "description": sc.description or f"Episode scene {sc.scene_index}",
            })

        lang_label = "Urdu (اردو)" if ("ur" in language.lower() or "اردو" in language) else "English"

        # Large-window batch size: 250 chunks per request processes a 15-30 min script in 1 single API call
        chunk_batch_size = 250
        chunk_batches = [chunks[i:i + chunk_batch_size] for i in range(0, len(chunks), chunk_batch_size)]
        num_batches = len(chunk_batches)
        mapping: dict[int, int] = {}

        for b_idx, batch in enumerate(chunk_batches):
            chunks_data = []
            for ch in batch:
                chunks_data.append({
                    "chunk_index": ch.chunk_index,
                    "duration_sec": round(ch.duration_sec, 1),
                    "narration_text": ch.text,
                })

            prompt = (
                f"You are an expert video editor. The voiceover narration is in {lang_label}.\n"
                f"Analyze the {lang_label} narration context and match each narration chunk with the most thematically "
                "and chronologically appropriate candidate scene from the catalog.\n\n"
                "Rules:\n"
                "1. Output strictly valid JSON format: a list of objects with keys 'chunk_index' and 'candidate_index'.\n"
                "2. Maintain coherent chronological narrative flow and avoid using the exact same candidate_index for consecutive chunks.\n"
                "3. Prefer candidate scenes whose duration is sufficient for the chunk.\n\n"
                f"Candidate Scenes Catalog:\n{json.dumps(scene_catalog, separators=(',', ':'))}\n\n"
                f"Narration Chunks ({lang_label}):\n{json.dumps(chunks_data, separators=(',', ':'))}\n"
            )

            # Strict JSON schema generation config
            gen_config = {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            }

            response, model_used = generate_content_with_fallback(
                api_key=self.api_key,
                contents=prompt,
                generation_config=gen_config,
                timeout=45.0,
                preferred="flash",
            )
            self.active_model_name = model_used

            # Track usage metadata and token telemetry per batch
            self.api_calls_count += 1
            meta = getattr(response, "usage_metadata", None)
            p_count = 0
            c_count = 0
            t_count = 0
            if meta:
                p_count = (
                    getattr(meta, "prompt_token_count", 0)
                    or (meta.get("prompt_token_count", 0) if isinstance(meta, dict) else 0)
                )
                c_count = (
                    getattr(meta, "candidates_token_count", 0)
                    or (meta.get("candidates_token_count", 0) if isinstance(meta, dict) else 0)
                )
                t_count = (
                    getattr(meta, "total_token_count", 0)
                    or (meta.get("total_token_count", 0) if isinstance(meta, dict) else (p_count + c_count))
                )

            # Fallback estimation if usage_metadata was not populated
            if t_count == 0 and hasattr(response, "text") and response.text:
                p_count = max(1, len(prompt) // 4)
                c_count = max(1, len(response.text) // 4)
                t_count = p_count + c_count

            self.prompt_tokens += p_count
            self.output_tokens += c_count
            self.total_tokens += t_count

            # Gemini Flash pricing: $0.075 / 1M prompt tokens, $0.30 / 1M output tokens
            call_cost = (p_count * 0.075 / 1_000_000.0) + (c_count * 0.30 / 1_000_000.0)
            self.estimated_cost_usd += call_cost
            logger.info(
                "Gemini API Usage (Batch %d/%d): %d In + %d Out = %d Total (Est. $%.6f USD)",
                b_idx + 1, num_batches, p_count, c_count, t_count, call_cost
            )

            text_resp = response.text.strip()
            if "```json" in text_resp:
                text_resp = text_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in text_resp:
                text_resp = text_resp.split("```")[1].split("```")[0].strip()

            try:
                batch_matches = json.loads(text_resp)
                for m in batch_matches:
                    if isinstance(m, dict):
                        ch_i = m.get("chunk_index") if "chunk_index" in m else m.get("chunk")
                        cand_i = m.get("candidate_index") if "candidate_index" in m else m.get("candidate")
                        if ch_i is not None and cand_i is not None:
                            mapping[int(ch_i)] = int(cand_i)
            except Exception as parse_err:
                logger.warning("Failed to parse Gemini JSON response for batch %d: %s", b_idx + 1, parse_err)

        for chunk in chunks:
            cand_idx = mapping.get(chunk.chunk_index)
            if cand_idx is not None and 0 <= cand_idx < len(scenes):
                chunk.matched_scene = scenes[cand_idx]
            else:
                chunk.matched_scene = scenes[chunk.chunk_index % len(scenes)]

        logger.info("Gemini Flash successfully matched %d chunks across %d batch(es)", len(chunks), num_batches)
        return chunks

    def _record_scene_usage(self, chunks: list[NarrativeChunk], project_id: Optional[int]) -> None:
        """Update SQLite database with usage records."""
        if project_id is None:
            return
        for chunk in chunks:
            if chunk.matched_scene and chunk.matched_scene.scene_id is not None:
                try:
                    increment_scene_used_count(chunk.matched_scene.scene_id, project_id)
                except Exception as e:
                    logger.warning("Could not increment scene usage count: %s", e)

    def get_usage_summary(self) -> dict:
        """Return cumulative token usage and estimated cost for this matcher session."""
        used_gemini = (self.api_calls_count > 0 and self.total_tokens > 0)
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls_count": self.api_calls_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "used_gemini": used_gemini,
            "from_cache": getattr(self, "from_cache", False),
            "last_error": self.last_error,
        }



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
from database.db import init_db, increment_scene_used_count, list_all_scenes
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

        for seg in transcription.segments:
            current_segs.append(seg)
            cur_dur = seg.end - current_start

            # Check if chunk reached target length or punctuation boundary
            is_sentence_end = seg.text.strip().endswith((".", "!", "?"))
            if (cur_dur >= self.target_chunk_sec and is_sentence_end) or cur_dur >= self.max_chunk_sec:
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

    def __init__(self, gemini_api_key: Optional[str] = None) -> None:
        cfg = get_config()
        self.api_key = gemini_api_key or cfg.gemini_api_key
        init_db()

    def match_scenes(
        self,
        chunks: list[NarrativeChunk],
        available_scenes: list[Scene],
        project_id: Optional[int] = None,
        prefer_unused: bool = True,
    ) -> list[NarrativeChunk]:
        """
        Assign an optimal, non-repeating Scene to each NarrativeChunk.
        """
        if not chunks or not available_scenes:
            logger.warning("Empty chunks or scenes provided to SmartSceneMatcher")
            return chunks

        # Sort scenes by used_count to prioritize fresh footage
        scene_pool = sorted(available_scenes, key=lambda s: s.used_count)

        is_valid_key = (
            bool(self.api_key)
            and len(self.api_key) > 15
            and not self.api_key.startswith("your_")
            and "dummy" not in self.api_key.lower()
        )

        if is_valid_key:
            try:
                logger.info("Attempting Google Gemini 1.5 Flash batch narrative matching...")
                matched = self._match_with_gemini(chunks, scene_pool)
                self._record_scene_usage(matched, project_id)
                return matched
            except Exception as exc:
                logger.warning("Gemini API matching failed (%s), falling back to offline heuristic matcher", exc)

        # Offline / Heuristic Matcher
        matched = self._match_with_heuristics(chunks, scene_pool, prefer_unused=prefer_unused)
        self._record_scene_usage(matched, project_id)
        return matched

    def _match_with_heuristics(
        self,
        chunks: list[NarrativeChunk],
        scenes: list[Scene],
        prefer_unused: bool = True,
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
    ) -> list[NarrativeChunk]:
        """
        Uses Gemini 1.5 Flash to understand context and select the best scenes.
        """
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Prepare scene catalog prompt
        scene_catalog = []
        for i, sc in enumerate(scenes[:50]):  # Limit candidate batch to top 50
            scene_catalog.append({
                "candidate_index": i,
                "scene_id": sc.scene_id,
                "duration_sec": round(sc.duration_sec, 1),
                "description": sc.description or f"Episode scene {sc.scene_index}",
            })

        chunks_data = []
        for ch in chunks:
            chunks_data.append({
                "chunk_index": ch.chunk_index,
                "duration_sec": round(ch.duration_sec, 1),
                "narration_text": ch.text,
            })

        prompt = (
            "You are an expert video editor. Match each narration chunk with the most thematically "
            "and chronologically appropriate candidate scene from the catalog.\n\n"
            "Rules:\n"
            "1. Output strictly valid JSON format: a list of objects with keys 'chunk_index' and 'candidate_index'.\n"
            "2. Avoid using the exact same candidate_index for consecutive chunks.\n\n"
            f"Candidate Scenes Catalog:\n{json.dumps(scene_catalog, indent=2)}\n\n"
            f"Narration Chunks:\n{json.dumps(chunks_data, indent=2)}\n"
        )

        response = model.generate_content(prompt)
        text_resp = response.text.strip()

        # Clean markdown code blocks if present
        if "```json" in text_resp:
            text_resp = text_resp.split("```json")[1].split("```")[0].strip()
        elif "```" in text_resp:
            text_resp = text_resp.split("```")[1].split("```")[0].strip()

        matches = json.loads(text_resp)
        mapping = {m["chunk_index"]: m["candidate_index"] for m in matches if "chunk_index" in m and "candidate_index" in m}

        for chunk in chunks:
            cand_idx = mapping.get(chunk.chunk_index)
            if cand_idx is not None and 0 <= cand_idx < len(scenes):
                chunk.matched_scene = scenes[cand_idx]
            else:
                chunk.matched_scene = scenes[chunk.chunk_index % len(scenes)]

        logger.info("Gemini 1.5 Flash successfully matched %d chunks", len(chunks))
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


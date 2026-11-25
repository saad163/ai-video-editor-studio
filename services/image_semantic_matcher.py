"""
services/image_semantic_matcher.py
==================================
Gemini Vision Semantic Image Indexer & Context Matcher for Template 1.

Key Capabilities:
  1. Multimodal Vision Analysis: Sends new, uncached images to Google Gemini 1.5 Flash
     in micro-batches (up to 8 images per call) to extract action tags and visual descriptions.
  2. Instant SQLite Caching: Stores image hashes (SHA-256) in `image_semantic_cache` table.
     Subsequent runs incur 0 API calls and 0 tokens.
  3. Bilingual Context Matching: Scores spoken narration (English & Roman Urdu/Hindi)
     against visual scene tags using bilingual synonym expansion (e.g. taaj <-> crown,
     shaadi <-> wedding, gaari <-> car, mahal <-> palace).
  4. Automatic Offline Fallback: When offline or unconfigured, extracts keywords from
     filenames and folder structures with zero cost.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from utils.config import get_config
from utils.logger import get_logger
from utils.file_helpers import compute_file_hash, is_image_file
from database.db import (
    get_cached_image_semantics,
    save_image_semantics,
    record_api_usage,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bilingual Synonyms Dictionary (Roman Urdu / Hindi <-> English)
# ─────────────────────────────────────────────────────────────────────────────

BILINGUAL_SYNONYMS: dict[str, list[str]] = {
    # Royal & Ceremony
    "taaj": ["crown", "coronation", "tiara"],
    "crown": ["taaj", "coronation", "tiara"],
    "coronation": ["taaj", "crown", "ceremony"],
    "throne": ["takht"],
    "takht": ["throne"],
    "badshah": ["king", "monarch"],
    "king": ["badshah", "monarch"],
    "malika": ["queen"],
    "queen": ["malika"],
    "shahzada": ["prince"],
    "prince": ["shahzada"],
    "shahzadi": ["princess"],
    "princess": ["shahzadi"],

    # Wedding & Relationship
    "shaadi": ["wedding", "marriage", "bride", "groom", "ceremony"],
    "wedding": ["shaadi", "marriage", "bride", "groom", "ceremony"],
    "nikah": ["wedding", "marriage", "ceremony"],
    "dulha": ["groom", "wedding"],
    "dulhan": ["bride", "wedding", "veil"],
    "bride": ["dulhan", "wedding", "veil"],
    "groom": ["dulha", "wedding"],

    # Architecture & Places
    "mahal": ["palace", "castle", "mansion", "estate"],
    "palace": ["mahal", "castle", "mansion", "estate"],
    "castle": ["mahal", "palace", "fort", "qila"],
    "qila": ["fort", "castle", "palace"],
    "church": ["cathedral", "chapel", "wedding"],
    "cathedral": ["church", "chapel"],

    # Vehicles & Travel
    "gaari": ["car", "vehicle", "automobile", "motorcade"],
    "car": ["gaari", "vehicle", "automobile", "motorcade"],
    "vehicle": ["car", "gaari", "automobile", "motorcade"],
    "motorcade": ["car", "vehicle", "gaari"],
    "jahaz": ["airplane", "plane", "flight", "airport"],
    "plane": ["airplane", "jahaz", "flight", "airport"],
    "airplane": ["jahaz", "plane", "flight", "airport"],

    # Speech, Podium & Public
    "taqreer": ["speech", "podium", "speaking", "address", "microphone", "stage"],
    "khitab": ["speech", "podium", "speaking", "address", "microphone", "stage"],
    "speech": ["taqreer", "khitab", "podium", "speaking", "address", "microphone"],
    "podium": ["speech", "taqreer", "microphone", "speaking", "stage"],
    "microphone": ["speech", "speaking", "press", "conference"],
    "press": ["interview", "conference", "microphone", "reporters", "camera"],
    "crowd": ["hujoom", "people", "audience", "public", "gathering", "fans"],
    "hujoom": ["crowd", "people", "audience", "public"],

    # Medical & Hospital
    "aspatal": ["hospital", "clinic", "doctor", "medical", "patient", "ambulance"],
    "hospital": ["aspatal", "clinic", "doctor", "medical", "patient", "nurse"],
    "bimar": ["sick", "ill", "hospital", "patient", "medical", "unwell"],
    "sick": ["bimar", "ill", "hospital", "patient", "medical"],
    "doctor": ["aspatal", "hospital", "medical", "clinic", "stethoscope"],

    # Sadness, Death & Grief
    "mout": ["death", "funeral", "grave", "mourning", "sad", "coffin"],
    "inteqal": ["death", "funeral", "grave", "mourning", "sad", "coffin"],
    "death": ["mout", "inteqal", "funeral", "grave", "mourning", "sad"],
    "funeral": ["janaza", "grave", "cemetery", "mourning", "coffin", "death"],
    "janaza": ["funeral", "grave", "cemetery", "mourning", "coffin", "death"],
    "rona": ["crying", "tears", "sad", "weeping", "grief", "emotional"],
    "crying": ["rona", "tears", "sad", "weeping", "grief", "emotional"],
    "tears": ["rona", "aansoo", "crying", "sad"],
    "aansoo": ["tears", "crying", "sad"],
    "udaas": ["sad", "gloomy", "unhappy", "somber", "crying"],
    "sad": ["udaas", "gloomy", "unhappy", "somber", "crying", "grief"],

    # Happiness & Family
    "khush": ["happy", "smiling", "laughing", "joy", "cheerful"],
    "happy": ["khush", "smiling", "laughing", "joy", "cheerful"],
    "smiling": ["khush", "happy", "smile", "laughing"],
    "bacha": ["baby", "child", "infant", "newborn", "son", "daughter"],
    "baby": ["bacha", "child", "infant", "newborn"],
    "family": ["khandan", "parents", "children", "together"],
    "khandan": ["family", "royal family", "relatives"],

    # Attire
    "wardi": ["uniform", "military", "suit", "attire", "dress"],
    "uniform": ["wardi", "military", "army", "navy", "medals", "suit"],
    "suit": ["tuxedo", "formal", "attire", "jacket", "tie"],
}

STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "was", "are", "were", "been", "being", "have", "has",
    "had", "it", "its", "this", "that", "these", "those",
    # Roman Urdu common particles
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "pe", "hai", "hain",
    "tha", "thi", "the", "kya", "kyun", "kab", "kahan", "yeh", "woh", "un",
    "unhon", "unke", "unki", "uska", "uski", "uske", "raha", "rahi", "rahe",
}


@dataclass
class ImageSemantics:
    """Semantic representation of a single visual asset."""
    file_hash: str
    file_path: Path
    character_pool: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    is_cached: bool = False


def extract_keywords_from_filename(path: Path) -> list[str]:
    """Offline heuristic extractor for image semantics based on filename and directory."""
    clean = path.stem.lower().replace("_", " ").replace("-", " ")
    tokens = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", clean) if t not in STOP_WORDS]
    parent_name = path.parent.name.lower()
    if parent_name and parent_name not in tokens and parent_name not in STOP_WORDS:
        tokens.insert(0, parent_name)
    return list(dict.fromkeys(tokens))


def score_image_semantics(
    image_semantics: ImageSemantics,
    narration_text: str,
    character_hint: Optional[str] = None,
) -> float:
    """
    Score relevance of an image against spoken voiceover segment.
    Uses token matching, phrase overlap, and bilingual synonym expansion.
    """
    if not narration_text:
        return 0.0

    text_lower = narration_text.lower()
    text_words = [w for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", text_lower) if w not in STOP_WORDS]
    if not text_words:
        return 0.0

    score = 0.0

    # Character pool matching bonus
    if character_hint:
        char_lower = character_hint.lower()
        if char_lower in text_lower and char_lower in image_semantics.character_pool.lower():
            score += 5.0

    tags_lower = [t.lower().strip() for t in image_semantics.tags]
    desc_lower = image_semantics.description.lower()
    desc_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", desc_lower))
    stem_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", image_semantics.file_path.stem.lower()))

    # 1. Word matching with best synonym
    for word in text_words:
        word_best = 0.0

        # Direct tag match
        if word in tags_lower:
            word_best = max(word_best, 18.0)
        elif any(word in t for t in tags_lower):
            word_best = max(word_best, 10.0)

        # Description word match
        if word in desc_words:
            word_best = max(word_best, 8.0)

        # Filename match
        if word in stem_words:
            word_best = max(word_best, 6.0)

        # Bilingual synonym expansion
        synonyms = BILINGUAL_SYNONYMS.get(word, [])
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower in tags_lower:
                word_best = max(word_best, 15.0)
            elif any(syn_lower in t for t in tags_lower):
                word_best = max(word_best, 9.0)

            if syn_lower in desc_words:
                word_best = max(word_best, 7.0)

            if syn_lower in stem_words:
                word_best = max(word_best, 5.0)

        score += word_best

    # 2. Exact multi-word tag bonus
    for tag in tags_lower:
        if " " in tag and tag in text_lower:
            score += 25.0

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Image Indexer
# ─────────────────────────────────────────────────────────────────────────────

class ImageSemanticIndexer:
    """
    Multimodal image indexer combining Gemini 1.5 Flash Vision and SQLite caching.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        project_id: Optional[int] = None,
        project_title: str = "Studio Project",
    ) -> None:
        cfg = get_config()
        self.api_key = gemini_api_key if gemini_api_key is not None else cfg.gemini_api_key
        self.project_id = project_id
        self.project_title = project_title

        self.prompt_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.api_calls_count = 0
        self.estimated_cost_usd = 0.0

    @property
    def has_valid_api_key(self) -> bool:
        from services.gemini_matcher import SmartSceneMatcher
        return SmartSceneMatcher.is_api_key_valid(self.api_key)

    def get_usage_summary(self) -> dict:
        """Return cumulative token usage and estimated cost for this indexing session."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls_count": self.api_calls_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }

    def index_images(
        self,
        images: list[Path],
        character_pools: Optional[dict[Path, str]] = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        batch_size: int = 8,
    ) -> dict[Path, ImageSemantics]:
        """
        Index all provided image files:
          - Instant retrieval from SQLite for cached hashes.
          - Batch Gemini Vision analysis for uncached images.
          - Offline keyword fallback when API is unavailable.
        """
        indexed: dict[Path, ImageSemantics] = {}
        uncached: list[tuple[Path, str, str]] = []  # (path, file_hash, pool_name)

        total_images = len(images)
        if total_images == 0:
            return indexed

        if progress_cb:
            progress_cb(5, f"Checking cache for {total_images} images...")

        # Step 1: Query SQLite cache first
        for idx, img_path in enumerate(images):
            if not img_path.exists() or not is_image_file(img_path):
                continue

            pool_name = "general"
            if character_pools and img_path in character_pools:
                pool_name = character_pools[img_path]
            elif img_path.parent.name.lower() not in {"sample_footage", "assets", "footage"}:
                pool_name = img_path.parent.name.lower()

            file_hash = compute_file_hash(img_path)
            cached_data = get_cached_image_semantics(file_hash)

            if cached_data:
                indexed[img_path] = ImageSemantics(
                    file_hash=file_hash,
                    file_path=img_path,
                    character_pool=pool_name,
                    tags=cached_data.get("tags", []),
                    description=cached_data.get("description", ""),
                    is_cached=True,
                )
            else:
                uncached.append((img_path, file_hash, pool_name))

        logger.info(
            "Semantic Image Indexer: %d cached, %d need indexing",
            len(indexed), len(uncached),
        )

        if not uncached:
            if progress_cb:
                progress_cb(100, f"All {total_images} images loaded from cache.")
            return indexed

        # Step 2: Process uncached images
        # Group uncached images into batches
        batches = [uncached[i:i + batch_size] for i in range(0, len(uncached), batch_size)]
        num_batches = len(batches)

        for b_idx, batch in enumerate(batches):
            pct = 10 + int(((b_idx + 1) / num_batches) * 85)
            if progress_cb:
                progress_cb(pct, f"Analyzing images ({b_idx + 1}/{num_batches}) with Gemini Vision...")

            if self.has_valid_api_key:
                try:
                    batch_results = self._analyze_batch_with_gemini(batch)
                    for (img_path, file_hash, pool_name), res in zip(batch, batch_results):
                        tags = res.get("tags", [])
                        desc = res.get("description", "")
                        save_image_semantics(
                            file_hash=file_hash,
                            file_path=str(img_path),
                            character_pool=pool_name,
                            tags=tags,
                            description=desc,
                        )
                        indexed[img_path] = ImageSemantics(
                            file_hash=file_hash,
                            file_path=img_path,
                            character_pool=pool_name,
                            tags=tags,
                            description=desc,
                            is_cached=False,
                        )
                    continue
                except Exception as exc:
                    logger.warning("Gemini Vision indexing failed (%s), falling back to offline heuristics", exc)

            # Offline heuristic fallback
            for img_path, file_hash, pool_name in batch:
                fallback_tags = extract_keywords_from_filename(img_path)
                fallback_desc = f"Image of {pool_name}: {img_path.stem.replace('_', ' ')}"
                save_image_semantics(
                    file_hash=file_hash,
                    file_path=str(img_path),
                    character_pool=pool_name,
                    tags=fallback_tags,
                    description=fallback_desc,
                )
                indexed[img_path] = ImageSemantics(
                    file_hash=file_hash,
                    file_path=img_path,
                    character_pool=pool_name,
                    tags=fallback_tags,
                    description=fallback_desc,
                    is_cached=False,
                )

        if progress_cb:
            progress_cb(100, f"Successfully indexed {total_images} images.")

        return indexed

    def _analyze_batch_with_gemini(self, batch: list[tuple[Path, str, str]]) -> list[dict]:
        """Send image batch to Gemini Flash Vision for semantic tagging (with auto model fallback)."""
        from PIL import Image
        from services.gemini_resolver import generate_content_with_fallback

        content_parts = []
        prompt_lines = [
            "You are an expert video editor and visual annotator.",
            f"Analyze the {len(batch)} provided images in sequential order (Index 0 to {len(batch)-1}).",
            "For each image, provide:",
            "1. 'tags': A list of 6-12 concise, highly specific visual tags describing what is shown or happening (e.g. actions, clothing, objects, settings, mood). Include common bilingual English and Roman Urdu/Hindi keywords where applicable (e.g. crown/taaj, wedding/shaadi, palace/mahal, speech/taqreer, car/gaari, crying/rona, hospital/aspatal, uniform/wardi).",
            "2. 'description': A clear 1-sentence description of the visual scene.",
            "Respond ONLY with a valid JSON array of objects with keys 'index', 'tags', and 'description'.",
            "Example:",
            '[{"index": 0, "tags": ["crown", "taaj", "royal", "coronation", "throne"], "description": "Prince wearing a golden royal crown during ceremony"}]',
        ]
        content_parts.append("\n".join(prompt_lines))

        # Attach images
        for idx, (img_path, _, pool_name) in enumerate(batch):
            content_parts.append(f"\n--- Image Index {idx} (Character: {pool_name}) ---")
            try:
                pil_img = Image.open(img_path)
                content_parts.append(pil_img)
            except Exception as e:
                logger.warning("Could not open image %s with PIL: %s", img_path.name, e)

        response, model_used = generate_content_with_fallback(
            api_key=self.api_key,
            contents=content_parts,
            timeout=25.0,
            preferred="flash",
        )
        self.api_calls_count += 1

        # Token telemetry tracking
        meta = getattr(response, "usage_metadata", None)
        if meta:
            p_count = getattr(meta, "prompt_token_count", 0) or (meta.get("prompt_token_count", 0) if isinstance(meta, dict) else 0)
            c_count = getattr(meta, "candidates_token_count", 0) or (meta.get("candidates_token_count", 0) if isinstance(meta, dict) else 0)
            t_count = getattr(meta, "total_token_count", 0) or (meta.get("total_token_count", 0) if isinstance(meta, dict) else (p_count + c_count))
            self.prompt_tokens += p_count
            self.output_tokens += c_count
            self.total_tokens += t_count

            # Gemini 1.5 Flash rates: $0.075 / 1M prompt, $0.30 / 1M output
            cost = (p_count * 0.075 / 1_000_000.0) + (c_count * 0.30 / 1_000_000.0)
            self.estimated_cost_usd += cost

            if self.project_id or self.project_title:
                try:
                    record_api_usage(
                        project_id=self.project_id,
                        project_title=self.project_title,
                        prompt_tokens=p_count,
                        output_tokens=c_count,
                        total_tokens=t_count,
                        estimated_cost_usd=cost,
                    )
                except Exception as err:
                    logger.warning("Failed to record vision API usage: %s", err)

        resp_text = response.text.strip()
        if "```json" in resp_text:
            resp_text = resp_text.split("```json")[1].split("```")[0].strip()
        elif "```" in resp_text:
            resp_text = resp_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(resp_text)
        results_by_idx = {item.get("index", i): item for i, item in enumerate(parsed) if isinstance(item, dict)}

        results = []
        for i in range(len(batch)):
            res = results_by_idx.get(i, {})
            results.append({
                "tags": res.get("tags", []),
                "description": res.get("description", ""),
            })

        return results

"""
services/script_parser.py
=========================
Phase 2 — Script text parser for Template 1 (Script & Character-Based Assembler).

Purpose
-------
Reads a plain-text or Markdown script file and identifies "blocks" that belong
to named characters/entities.  These blocks are then aligned with Whisper
transcription segments by CharacterMatcher so the correct character image is
shown while that text is being narrated.

Supported script formats
------------------------
  1. Colon-delimited   : "ALI: This is Ali's dialogue"
  2. Bracket-prefixed  : "[Ali] walks into the room"
  3. All-caps names    : Lines that start with an UPPERCASE WORD followed by text
  4. Free narration    : Any line that doesn't match the above is tagged 'narrator'

The parser is intentionally lenient — it uses multiple detection strategies
and falls back gracefully so it never discards content.

Usage
-----
  from services.script_parser import ScriptParser
  parser = ScriptParser(known_characters=["Ali", "Zara", "Hassan"])
  blocks = parser.parse_file(Path("script.txt"))
  # → [ScriptBlock(character='Ali', text='...', line_no=3), ...]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScriptBlock:
    """
    One attributed block of text from the script.

    Attributes
    ----------
    character   : Detected character/entity name, or 'narrator' if unattributed.
    text        : The narration / dialogue text for this block.
    line_no     : Source line number in the script file (1-based).
    confidence  : How confident the parser is in the character attribution
                  (1.0 = explicit colon label, 0.5 = heuristic, 0.0 = fallback).
    """
    character: str
    text: str
    line_no: int = 0
    confidence: float = 1.0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_name_normalized(self) -> str:
        """Lowercase, strip whitespace — for fuzzy matching."""
        return self.character.strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
# Pattern registry
# ─────────────────────────────────────────────────────────────────────────────

# "ALI:" or "Ali:" or "NARRATOR:" at line start
_RE_COLON_LABEL = re.compile(r"^([A-Za-z][A-Za-z\s]{0,24}):\s+(.+)$")

# "[Ali]" or "(Ali)" at line start
_RE_BRACKET_LABEL = re.compile(r"^[\[\(]([A-Za-z][A-Za-z\s]{0,24})[\]\)]\s*(.*)$")

# All-caps word at start: "ALI walks …"  (at least 2 uppercase chars)
_RE_ALLCAPS_START = re.compile(r"^([A-Z]{2,}(?:\s[A-Z]{2,})?)\s+(.+)$")


# ─────────────────────────────────────────────────────────────────────────────
# ScriptParser
# ─────────────────────────────────────────────────────────────────────────────

class ScriptParser:
    """
    Parses a voiceover script file into a list of attributed ScriptBlock objects.

    Parameters
    ----------
    known_characters : Optional list of character names to prefer/validate.
                       If provided, only names that fuzzy-match this list get
                       attributed; others fall through to 'narrator'.
    merge_narrator   : If True, consecutive 'narrator' lines are merged into
                       one block (reduces total block count for short segments).
    """

    def __init__(
        self,
        known_characters: Optional[list[str]] = None,
        merge_narrator: bool = True,
    ) -> None:
        self.known_characters: list[str] = [c.strip() for c in (known_characters or [])]
        self._known_lower: set[str] = {c.lower() for c in self.known_characters}
        self.merge_narrator = merge_narrator

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_file(self, script_path: Path) -> list[ScriptBlock]:
        """Read *script_path* and return attributed ScriptBlocks."""
        script_path = Path(script_path).resolve()
        if not script_path.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")

        text = script_path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> list[ScriptBlock]:
        """
        Parse *text* directly (without reading from disk).

        Returns a list of ScriptBlocks in document order.
        """
        raw_lines = text.splitlines()
        blocks: list[ScriptBlock] = []

        for lineno, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue    # skip blanks and markdown headings / comments

            block = self._classify_line(line, lineno)
            if block is not None:
                blocks.append(block)

        # Post-process: merge consecutive narrator blocks
        if self.merge_narrator:
            blocks = self._merge_narrator_blocks(blocks)

        logger.info(
            "Script parsed: %d blocks, %d unique characters",
            len(blocks),
            len({b.character for b in blocks}),
        )
        return blocks

    # ── Character-to-segment alignment ────────────────────────────────────────

    def align_to_segments(
        self,
        blocks: list[ScriptBlock],
        segment_texts: list[str],
    ) -> list[str]:
        """
        Map each transcription *segment_text* to the most likely character name.

        Strategy:
          1. For each segment, check whether any known character name appears in
             the segment text (case-insensitive word-boundary match).
          2. If a match is found, return that character's name.
          3. Otherwise, fall back to the dominant character in the nearest script
             block by text similarity (word overlap heuristic).
          4. Final fallback: 'narrator'.

        Returns
        -------
        A list of character names, one per segment, same length as segment_texts.
        """
        if not blocks:
            return ["narrator"] * len(segment_texts)

        # Build character → regex lookup (whole-word, case-insensitive)
        char_patterns = {
            char: re.compile(rf"\b{re.escape(char)}\b", re.IGNORECASE)
            for char in self.known_characters
        }

        result: list[str] = []
        for seg_text in segment_texts:
            matched = self._find_character_in_text(seg_text, char_patterns)
            if matched:
                result.append(matched)
            else:
                # Word-overlap fallback against all script blocks
                best_char, best_overlap = "narrator", 0
                seg_words = set(seg_text.lower().split())
                for block in blocks:
                    block_words = set(block.text.lower().split())
                    overlap = len(seg_words & block_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_char = block.character
                result.append(best_char)

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _classify_line(self, line: str, lineno: int) -> Optional[ScriptBlock]:
        """Try each detection pattern and return the best ScriptBlock."""

        # Pattern 1: "Ali: some text"
        m = _RE_COLON_LABEL.match(line)
        if m:
            char_raw, text = m.group(1).strip(), m.group(2).strip()
            char = self._normalise_character(char_raw)
            if char:
                return ScriptBlock(character=char, text=text, line_no=lineno, confidence=1.0)

        # Pattern 2: "[Ali] some text" or "(Ali) some text"
        m = _RE_BRACKET_LABEL.match(line)
        if m:
            char_raw, text = m.group(1).strip(), m.group(2).strip()
            char = self._normalise_character(char_raw)
            if char and text:
                return ScriptBlock(character=char, text=text, line_no=lineno, confidence=0.9)

        # Pattern 3: "HASSAN walks in and says something"
        m = _RE_ALLCAPS_START.match(line)
        if m:
            char_raw, text = m.group(1).strip().title(), m.group(2).strip()
            char = self._normalise_character(char_raw)
            if char:
                return ScriptBlock(character=char, text=text, line_no=lineno, confidence=0.5)

        # Pattern 4: Check if any known character name appears anywhere in the line
        lower_line = line.lower()
        for char in self.known_characters:
            if re.search(rf"\b{re.escape(char.lower())}\b", lower_line):
                return ScriptBlock(character=char, text=line, line_no=lineno, confidence=0.4)

        # Fallback: narrator
        return ScriptBlock(character="narrator", text=line, line_no=lineno, confidence=0.0)

    def _normalise_character(self, raw: str) -> str:
        """
        Return the canonical character name or empty string if unknown.

        If known_characters is set, only returns names in that list (fuzzy match).
        Otherwise returns title-cased *raw*.
        """
        if not self.known_characters:
            return raw.title()

        raw_lower = raw.lower().strip()
        # Exact match
        for char in self.known_characters:
            if char.lower() == raw_lower:
                return char
        # Partial / substring match
        for char in self.known_characters:
            if raw_lower in char.lower() or char.lower() in raw_lower:
                return char
        # No match — return narrator tag
        return "narrator"

    def _find_character_in_text(
        self, text: str, patterns: dict[str, re.Pattern[str]]
    ) -> Optional[str]:
        """Return the first character name whose pattern matches *text*."""
        for char, pattern in patterns.items():
            if pattern.search(text):
                return char
        return None

    @staticmethod
    def _merge_narrator_blocks(blocks: list[ScriptBlock]) -> list[ScriptBlock]:
        """Merge consecutive 'narrator' ScriptBlocks into single blocks."""
        merged: list[ScriptBlock] = []
        for block in blocks:
            if (
                merged
                and merged[-1].character == "narrator"
                and block.character == "narrator"
            ):
                # Merge text, keep first line_no, lowest confidence
                prev = merged[-1]
                merged[-1] = ScriptBlock(
                    character="narrator",
                    text=prev.text + " " + block.text,
                    line_no=prev.line_no,
                    confidence=min(prev.confidence, block.confidence),
                )
            else:
                merged.append(block)
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_script = """
# My Video Script

Ali: In this scene we see the main character entering the building.
Zara: Zara then appears at the window and waves down at him.
The city skyline stretches out behind them both.
[Hassan] Hassan is waiting in the lobby, holding a briefcase.
NARRATOR In the final frame, the camera pulls back to reveal the full scene.
Ali walks towards the exit, while Zara calls out from above.
"""

    parser = ScriptParser(known_characters=["Ali", "Zara", "Hassan"])
    blocks = parser.parse_text(sample_script)

    print(f"\nParsed {len(blocks)} blocks:\n" + "-" * 50)
    for b in blocks:
        print(f"  [{b.confidence:.1f}] {b.character:12s} L{b.line_no:03d}: {b.text[:60]}")

    # Test alignment
    segs = [
        "Ali walks into the room carrying the briefcase",
        "She watches from above as the scene unfolds below",
        "The camera pans slowly across the horizon",
    ]
    assignments = parser.align_to_segments(blocks, segs)
    print(f"\nSegment → Character assignments:")
    for text, char in zip(segs, assignments):
        print(f"  {char:12s} ← '{text[:50]}'")


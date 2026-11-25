"""
services/subtitle_engine.py
============================
Phase 2 & Phase 3 — High-Impact ASS / SRT Subtitle Generator for AI Video Editor Studio.

Converts TranscriptionResult into custom-styled ASS or plain SRT subtitles calibrated
for 2K (2560x1440) and 1080p video outputs.

Features:
  • Accurate RGB Hex to ASS (&HAABBGGRR) color conversion.
  • Outline Mode (BorderStyle=1) with solid black 3.5px stroke.
  • Background Box Mode (BorderStyle=3) with solid or semi-transparent background
    (YouTube / CapCut style) ensuring 100% legibility over any footage.
  • Word-level karaoke highlighting (\\kN tags) when word timestamps are available.
  • Non-overlapping, monotonic dialogue event timestamps.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Any

from services.transcriber import TranscriptionResult, TranscriptionSegment
from utils.logger import get_logger

logger = get_logger(__name__)

# Output resolution canvas map
SUBTITLE_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "2k":    (2560, 1440),
    "4k":    (3840, 2160),
}

_PLAY_RES_X = 2560
_PLAY_RES_Y = 1440


# ─────────────────────────────────────────────────────────────────────────────
# ASS Color Conversion Helpers (&HAABBGGRR byte order)
# ─────────────────────────────────────────────────────────────────────────────

def to_ass_color(color_input: str, alpha: int = 0) -> str:
    """
    Convert a color name or RGB hex string (#RRGGBB) into standard ASS format:
    &HAABBGGRR

    Parameters
    ----------
    color_input : Color name ('white', 'yellow', 'cyan', etc.) or hex string ('#FFFF00', '#FFFFFF')
    alpha       : 0 = completely opaque, 128 (0x80) = 50% semi-transparent, 255 (0xFF) = fully transparent.
    """
    if not color_input:
        return f"&H{alpha:02X}FFFFFF"

    color_str = color_input.strip()

    # Already formatted ASS color code
    if color_str.startswith(("&H", "&h")):
        return color_str

    named_colors: dict[str, tuple[int, int, int]] = {
        "white":        (255, 255, 255),
        "yellow":       (255, 255, 0),
        "cyan":         (0, 255, 255),
        "green":        (74, 222, 128),
        "light_green":  (0, 255, 128),
        "orange":       (255, 128, 0),
        "purple":       (192, 132, 252),
        "amber":        (245, 158, 11),
        "black":        (0, 0, 0),
        "dark":         (15, 23, 42),
    }

    low = color_str.lower()
    if low in named_colors:
        r, g, b = named_colors[low]
        return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

    # Parse hex #RRGGBB or RRGGBB
    clean_hex = color_str.lstrip("#").strip()
    if len(clean_hex) == 3:
        clean_hex = "".join(c * 2 for c in clean_hex)

    if len(clean_hex) >= 6:
        try:
            r = int(clean_hex[0:2], 16)
            g = int(clean_hex[2:4], 16)
            b = int(clean_hex[4:6], 16)
            return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"
        except ValueError:
            pass

    return f"&H{alpha:02X}FFFFFF"


# ─────────────────────────────────────────────────────────────────────────────
# ASS Timestamp Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _to_ass_time(seconds: float) -> str:
    """Convert seconds (float) to ASS timestamp format H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02}:{s:05.2f}"


def _to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle Style Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubtitleStyle:
    """
    ASS style configuration for crisp 2K video rendering.
    """
    primary_colour: str = "&H00FFFFFF"    # Main text color
    secondary_colour: str = "&H0000FFFF"  # Active karaoke word highlight
    outline_colour: str = "&H00000000"    # Stroke or box padding border
    back_colour: str = "&H00000000"       # Background box color or shadow
    bold: bool = True
    italic: bool = False
    font_name: str = "Arial"
    font_size: int = 58                   # Tuned for 2K (2560x1440)
    outline_width: float = 14.0           # Stroke width or Box padding
    shadow_depth: float = 0.0             # 0 for modern clean flat look
    alignment: int = 2                    # 2 = Bottom-Center (numpad layout)
    margin_l: int = 120
    margin_r: int = 120
    margin_v: int = 90                    # 90px above bottom
    borderstyle: int = 1                  # 1 = Outline, 3 = Background Box

    def to_ass_style_line(self, name: str = "Default") -> str:
        """Serialise to a single ASS [V4+ Styles] line."""
        bold_val = "-1" if self.bold else "0"
        italic_val = "-1" if self.italic else "0"
        return (
            f"Style: {name},"
            f"{self.font_name},"
            f"{self.font_size},"
            f"{self.primary_colour},"
            f"{self.secondary_colour},"
            f"{self.outline_colour},"
            f"{self.back_colour},"
            f"{bold_val},"     # Bold
            f"{italic_val},"   # Italic
            f"0,"              # Underline
            f"0,"              # StrikeOut
            f"100,"            # ScaleX
            f"100,"            # ScaleY
            f"0,"              # Spacing
            f"0,"              # Angle
            f"{self.borderstyle},"
            f"{self.outline_width:.1f},"
            f"{self.shadow_depth:.1f},"
            f"{self.alignment},"
            f"{self.margin_l},"
            f"{self.margin_r},"
            f"{self.margin_v},"
            f"1"               # Encoding (1 = Unicode)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle Translation & Sanitization Helpers (Always-English Subtitles)
# ─────────────────────────────────────────────────────────────────────────────

URDU_TO_ENGLISH_FALLBACK: dict[str, str] = {
    "بادشاہ": "The King",
    "ملکہ": "The Queen",
    "شہزادہ": "The Prince",
    "شہزادی": "The Princess",
    "شاہی": "Royal",
    "محل": "palace",
    "سلامت": "Majesty",
    "تشریف": "arrived",
    "لائے": "came",
    "وزیر": "The Minister",
    "اعلان": "announcement",
    "کہانی": "story",
    "ایک": "a",
    "اور": "and",
    "تھا": "was",
    "تھی": "was",
    "تھے": "were",
    "ہے": "is",
    "ہیں": "are",
    "میں": "in",
    "پر": "on",
    "سے": "from",
    "کو": "to",
    "کا": "of",
    "کی": "of",
    "کے": "of",
    "نے": "has",
    "کیا": "made",
    "یہ": "This",
    "وہ": "That",
    "سب": "all",
    "جھک": "bowed",
    "کر": "and",
    "جنگ": "war",
    "خبر": "news",
    "سنائی": "shared",
    "لوگ": "people",
    "وقت": "time",
    "دن": "day",
    "رات": "night",
    "شہر": "city",
    "زندگی": "life",
}


def has_non_latin_script(text: str) -> bool:
    """Detect if string contains Arabic/Urdu, Devanagari, Cyrillic, CJK, etc."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0900-\u097F\u4E00-\u9FFF\u0400-\u04FF]", text))


def has_non_latin_script_in_segments(segments: Sequence[TranscriptionSegment]) -> bool:
    """Check if any segment in the collection contains non-Latin or Arabic/Urdu script."""
    return any(has_non_latin_script(s.text) for s in segments)


def sanitize_english_text(text: str) -> str:
    """Strictly strip any non-Latin or Arabic/Urdu characters and normalize whitespace."""
    # Strip any Urdu/Arabic/non-Latin scripts
    cleaned = re.sub(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u0900-\u097F\u4E00-\u9FFF\u0400-\u04FF]", "", text)
    # Collapse duplicate whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sanitize_segment_list(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    """Sanitize segment text, clamp timestamps, and deduplicate consecutive identical lines."""
    sanitized: list[TranscriptionSegment] = []
    prev_text = ""
    for s in segments:
        clean_text = sanitize_english_text(s.text)
        if not clean_text:
            continue
        # Avoid exact duplicate back-to-back lines
        if clean_text.lower() == prev_text.lower():
            continue
        start_t = max(0.0, s.start)
        end_t = max(start_t + 0.3, s.end)
        sanitized.append(
            TranscriptionSegment(
                start=start_t,
                end=end_t,
                text=clean_text,
                words=s.words,
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
            )
        )
        prev_text = clean_text
    return sanitized


def _translate_segments_with_gemini(
    segments: list[TranscriptionSegment],
    api_key: str,
    source_language: str,
) -> list[TranscriptionSegment]:
    from services.gemini_resolver import generate_content_with_fallback

    batch_payload = [
        {"i": idx, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
        for idx, s in enumerate(segments)
    ]
    prompt = (
        f"You are an expert film translator and subtitle editor. "
        f"Translate the following timestamped subtitle segments from {source_language} into concise, "
        "accurate, natural English subtitles for video overlay.\n\n"
        "Rules:\n"
        "1. Output strictly valid JSON format: a list of objects with keys 'i', 'start', 'end', and 'text'.\n"
        "2. The 'text' MUST be strictly in English only. NEVER include any Urdu or foreign non-Latin characters.\n"
        "3. Preserve the exact 'start' and 'end' timestamps for each segment 'i'.\n"
        "4. Keep each subtitle line concise (5 to 7 words where possible).\n\n"
        f"Input Segments ({source_language}):\n{json.dumps(batch_payload, separators=(',', ':'), ensure_ascii=False)}"
    )

    resp, _ = generate_content_with_fallback(
        api_key=api_key,
        contents=prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        timeout=45.0,
    )

    data = json.loads(resp.text)
    mapping = {item["i"]: item["text"] for item in data if "i" in item and "text" in item}

    translated = []
    for idx, s in enumerate(segments):
        t_text = mapping.get(idx, "")
        if not t_text:
            t_text = sanitize_english_text(s.text)
        translated.append(
            TranscriptionSegment(
                start=s.start,
                end=s.end,
                text=t_text,
                words=[],
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
            )
        )
    return translated


def translate_and_sanitize_to_english(
    segments: list[TranscriptionSegment],
    gemini_api_key: Optional[str] = None,
    source_language: str = "ur",
    audio_path: Optional[Path] = None,
) -> list[TranscriptionSegment]:
    """
    Translates non-English subtitle segments to English and sanitizes the output.
    Ensures timestamps stay strictly aligned and zero non-English characters leak.
    """
    if not segments:
        return []

    # If all segments are already English, sanitize and return
    if not has_non_latin_script_in_segments(segments) and source_language.lower() in ("en", "english"):
        return _sanitize_segment_list(segments)

    # 1. Try Google Gemini batch translation if key is available
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if api_key:
        from services.gemini_matcher import SmartSceneMatcher
        if SmartSceneMatcher.is_api_key_valid(api_key):
            try:
                translated = _translate_segments_with_gemini(segments, api_key, source_language)
                if translated and len(translated) == len(segments):
                    return _sanitize_segment_list(translated)
            except Exception as e:
                logger.warning("Gemini subtitle translation failed (%s), falling back to offline translator", e)

    # 2. Try Whisper translation if audio exists
    if audio_path and Path(audio_path).exists():
        try:
            from services.transcriber import TranscriptionService
            ts = TranscriptionService()
            whisper_res = ts.transcribe(
                Path(audio_path),
                language=source_language,
                task="translate",
                chunk_subtitles=True,
                max_words_per_chunk=6,
                use_cache=True,
            )
            if whisper_res and whisper_res.segments:
                return _sanitize_segment_list(whisper_res.segments)
        except Exception as we:
            logger.debug("Whisper translation fallback skipped or failed: %s", we)

    # 3. Deterministic offline translation dictionary & sanitization fallback
    fallback_segments = []
    for s in segments:
        eng_words = []
        for word in s.text.split():
            clean_w = word.strip("،؛؟۔.,!?:;")
            punct = word[len(clean_w):] if len(word) > len(clean_w) else ""
            if clean_w in URDU_TO_ENGLISH_FALLBACK:
                eng_words.append(URDU_TO_ENGLISH_FALLBACK[clean_w] + punct)
            else:
                latin = sanitize_english_text(clean_w)
                if latin:
                    eng_words.append(latin + punct)

        translated_text = " ".join(eng_words).strip()
        if not translated_text:
            translated_text = f"Scene event at {s.start:.1f}s"

        fallback_segments.append(
            TranscriptionSegment(
                start=s.start,
                end=s.end,
                text=translated_text,
                words=[],
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
            )
        )

    return _sanitize_segment_list(fallback_segments)


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle Engine
# ─────────────────────────────────────────────────────────────────────────────

class SubtitleEngine:
    """
    Converts TranscriptionResult → clean, high-impact 2K ASS or SRT subtitle file.
    Supports dynamic colors, outline mode, and solid/semi-transparent background box.
    Enforces strictly English-only subtitles regardless of narration language.
    """

    def __init__(
        self,
        color: str = "#FFFFFF",
        text_color: Optional[str] = None,
        highlight_color: str = "#FFFF00",
        has_bg_box: bool = False,
        bg_style: str = "outline",
        bg_color: str = "#000000",
        bg_opacity: float = 0.5,
        box_padding: float = 14.0,
        stroke_color: str = "#000000",
        stroke_thickness: float = 3.5,
        word_highlight: bool = False,
        font_size: int = 58,
        font_name: str = "Arial",
        bold: bool = True,
        italic: bool = False,
        alignment: int = 2,
        vertical_margin: int = 90,
        max_chars_per_line: int = 38,
        resolution: str = "2k",
        gemini_api_key: Optional[str] = None,
        ensure_english: bool = True,
        **kwargs: Any,
    ) -> None:
        self.resolution = resolution.lower()
        res_pair = SUBTITLE_RESOLUTIONS.get(self.resolution, (_PLAY_RES_X, _PLAY_RES_Y))
        self.play_res_x, self.play_res_y = res_pair
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.ensure_english = ensure_english

        effective_text_color = text_color if text_color is not None else color
        primary_ass = to_ass_color(effective_text_color, alpha=0)
        secondary_ass = to_ass_color(highlight_color, alpha=0)

        # Check whether background box mode is requested
        use_box = has_bg_box or bg_style in ("solid_black", "semi_transparent_black", "solid_box", "transparent_box")

        if use_box:
            borderstyle = 3  # Opaque/semi-transparent bounding box
            outline_width = max(4.0, min(float(box_padding), 35.0))  # Box padding margin around text
            # Determine alpha for background box
            if bg_style == "solid_black" or bg_opacity >= 0.95:
                alpha_val = 0      # Fully solid
            else:
                alpha_val = int(round((1.0 - bg_opacity) * 255))
            back_ass = to_ass_color(bg_color, alpha=alpha_val)
            outline_ass = back_ass
        else:
            borderstyle = 1  # Standard high-contrast solid outline
            outline_width = stroke_thickness
            outline_ass = to_ass_color(stroke_color, alpha=0)
            back_ass = "&H00000000"

        self.style = SubtitleStyle(
            primary_colour=primary_ass,
            secondary_colour=secondary_ass,
            outline_colour=outline_ass,
            back_colour=back_ass,
            borderstyle=borderstyle,
            outline_width=outline_width,
            shadow_depth=0.0,
            font_name=font_name,
            font_size=font_size,
            bold=bold,
            italic=italic,
            alignment=alignment,
            margin_v=vertical_margin,
        )

        self.word_highlight = word_highlight
        self.max_chars_per_line = max_chars_per_line

        logger.info(
            "SubtitleEngine configured: Font=%s (%dpt), Primary=%s, BorderStyle=%d (Box=%s), Alignment=%d",
            font_name, font_size, primary_ass, borderstyle, use_box, alignment
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_ass(
        self,
        transcription: TranscriptionResult,
        output_path: Path,
        ensure_english: Optional[bool] = None,
    ) -> Path:
        """
        Write an ASS subtitle file from *transcription* to *output_path*.
        If ensure_english is True (default), automatically translates any Urdu or
        non-English script to clean, aligned English subtitles before writing.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        do_english = self.ensure_english if ensure_english is None else ensure_english
        segments = transcription.segments
        if do_english and (has_non_latin_script_in_segments(segments) or (transcription.language and transcription.language.lower() not in ("en", "english"))):
            segments = translate_and_sanitize_to_english(
                segments,
                gemini_api_key=self.gemini_api_key,
                source_language=transcription.language or "ur",
                audio_path=getattr(transcription, "audio_path", None),
            )

        content = self._build_ass(segments)
        output_path.write_text(content, encoding="utf-8-sig")

        logger.info(
            "ASS subtitles written: %d events → %s (English guaranteed)",
            len(segments), output_path.name,
        )
        return output_path

    def generate_srt(
        self,
        transcription: TranscriptionResult,
        output_path: Path,
        ensure_english: Optional[bool] = None,
    ) -> Path:
        """Write a plain SRT subtitle file (strictly English). Returns *output_path*."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        do_english = self.ensure_english if ensure_english is None else ensure_english
        segments = transcription.segments
        if do_english and (has_non_latin_script_in_segments(segments) or (transcription.language and transcription.language.lower() not in ("en", "english"))):
            segments = translate_and_sanitize_to_english(
                segments,
                gemini_api_key=self.gemini_api_key,
                source_language=transcription.language or "ur",
                audio_path=getattr(transcription, "audio_path", None),
            )

        lines: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            text = self._wrap_text(seg.text.strip())
            lines.append(str(idx))
            lines.append(f"{_to_srt_time(seg.start)} --> {_to_srt_time(seg.end)}")
            lines.append(text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("SRT subtitles written: %d events → %s (English guaranteed)", len(segments), output_path.name)
        return output_path

    # ── ASS Document Builder ───────────────────────────────────────────────────

    def _build_ass(self, segments: list[TranscriptionSegment]) -> str:
        """Assemble the full ASS document string."""
        header = self._ass_header()
        styles = self._ass_styles()
        events = self._ass_events(segments)
        return "\n".join([header, styles, events])

    def _ass_header(self) -> str:
        return "\n".join([
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            f"PlayResX: {self.play_res_x}",
            f"PlayResY: {self.play_res_y}",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.601",
        ])

    def _ass_styles(self) -> str:
        lines = [
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            self.style.to_ass_style_line("Default"),
        ]
        return "\n".join(lines)

    def _ass_events(self, segments: list[TranscriptionSegment]) -> str:
        lines = [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        prev_end = 0.0
        for seg in segments:
            # Strictly prevent timestamp overlap across dialogue events
            start_t = max(seg.start, prev_end)
            end_t = max(seg.end, start_t + 0.4)

            text = self._segment_to_ass_text(seg)
            if not text.strip():
                continue

            lines.append(
                f"Dialogue: 0,{_to_ass_time(start_t)},{_to_ass_time(end_t)},Default,,0,0,0,,{text}"
            )
            prev_end = end_t

        return "\n".join(lines)

    # ── Text Processing ────────────────────────────────────────────────────────

    def _segment_to_ass_text(self, seg: TranscriptionSegment) -> str:
        """
        Convert a segment to ASS dialogue text.
        If word_highlight is enabled, wraps each word in {\\kN} karaoke tags.
        """
        if self.word_highlight and seg.words:
            return self._build_karaoke_text(seg)
        return self._wrap_text(seg.text.strip())

    def _build_karaoke_text(self, seg: TranscriptionSegment) -> str:
        """Build ASS karaoke text with {\\kN} tags per word."""
        parts: list[str] = []
        for word_obj in seg.words:
            duration_cs = int(round((word_obj.end - word_obj.start) * 100))
            duration_cs = max(duration_cs, 1)
            word_text = word_obj.word.replace("{", "\\{").replace("}", "\\}")
            parts.append(f"{{\\k{duration_cs}}}{word_text.strip()}")
        return " ".join(parts)

    def _wrap_text(self, text: str) -> str:
        """
        Wrap subtitle text with ASS line break (\\N).
        Balances 2-line wraps so neither line looks awkward or overflows the box margin.
        """
        text = text.strip()
        if not text or len(text) <= self.max_chars_per_line:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        # If it fits within 2 lines, balance split point near the center
        if len(text) <= self.max_chars_per_line * 2:
            best_idx = 1
            min_diff = float("inf")
            for i in range(1, len(words)):
                l1 = " ".join(words[:i])
                l2 = " ".join(words[i:])
                if len(l1) <= self.max_chars_per_line and len(l2) <= self.max_chars_per_line:
                    diff = abs(len(l1) - len(l2))
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = i
            if min_diff != float("inf"):
                return f"{' '.join(words[:best_idx])}\\N{' '.join(words[best_idx:])}"

        # Fallback greedy wrap for 3+ lines
        lines: list[str] = []
        current = ""

        for word in words:
            test = (current + " " + word).strip()
            if len(test) > self.max_chars_per_line and current:
                lines.append(current)
                current = word
            else:
                current = test

        if current:
            lines.append(current)

        return r"\N".join(lines)

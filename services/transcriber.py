"""
services/transcriber.py
=======================
Phase 1 — Audio Transcription Service

Uses faster-whisper to transcribe MP3/WAV audio into timestamped segments.

Device selection strategy
--------------------------
  compute_device=auto (default)
      → Checks for CUDA at runtime.
      → GPU  : device="cuda",  compute_type="float16"
      → CPU  : device="cpu",   compute_type="int8"    (developer laptop)

  compute_device=cuda  → Force GPU (raises if CUDA not available)
  compute_device=cpu   → Force CPU

The result is a TranscriptionResult dataclass containing:
  - language (detected)
  - full_text (all segments joined)
  - segments  (list of TranscriptionSegment with start/end/text/words)

Caching
-------
  Every result is persisted to SQLite via database.db so re-transcribing
  the same audio file (identical SHA-256 hash) is instant on subsequent runs.

Usage
-----
  from services.transcriber import TranscriptionService
  svc = TranscriptionService()
  result = svc.transcribe(Path("voiceover.mp3"), project_id=1)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Any

from utils.config import get_config
from utils.file_helpers import compute_file_hash, is_audio_file
from database.db import (
    init_db,
    insert_transcription,
    get_cached_transcription,
    insert_media_asset,
    asset_hash_exists,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordTimestamp:
    """A single word with its precise start/end time (from faster-whisper)."""
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptionSegment:
    """One logical sentence/phrase segment from Whisper."""
    start: float            # seconds
    end: float              # seconds
    text: str               # clean transcript text
    words: list[WordTimestamp] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_srt_block(self, index: int) -> str:
        """Return this segment formatted as a single SRT subtitle block."""
        def _fmt(secs: float) -> str:
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            ms = int((secs - int(secs)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        return f"{index}\n{_fmt(self.start)} --> {_fmt(self.end)}\n{self.text.strip()}\n"


@dataclass
class TranscriptionResult:
    """Complete output of a single transcription run."""
    audio_path: Path
    audio_hash: str
    language: str
    full_text: str
    segments: list[TranscriptionSegment]
    model_used: str
    elapsed_seconds: float
    from_cache: bool = False

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def to_srt(self) -> str:
        """Export all segments as a complete SRT subtitle string."""
        return "\n".join(seg.to_srt_block(i + 1) for i, seg in enumerate(self.segments))

    def save_srt(self, output_path: Path) -> None:
        """Write SRT file to disk."""
        output_path.write_text(self.to_srt(), encoding="utf-8")
        logger.info("SRT saved → %s", output_path)

    def segments_as_json(self) -> str:
        """Serialise segments to JSON string (for DB storage)."""
        return json.dumps(
            [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ],
                }
                for seg in self.segments
            ],
            ensure_ascii=False,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Device resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device_and_compute_type(compute_device: str) -> tuple[str, str]:
    """
    Return (device, compute_type) based on the config and runtime CUDA check.

    CPU fallback rules (PRD §2 Developer Adaptation Strategy):
      - CPU: compute_type = "int8"   (fastest, smallest memory footprint)
      - GPU: compute_type = "float16" (best quality / speed on RTX 2070)
    """
    if compute_device == "cpu":
        logger.info("Device forced to CPU (config override).")
        return "cpu", "int8"

    # Check CUDA availability
    cuda_available = False
    try:
        import ctranslate2  # noqa: F401 — only checking availability
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        logger.debug("torch/ctranslate2 not fully installed — CUDA detection skipped.")
        cuda_available = False

    if compute_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "compute_device='cuda' was forced but CUDA is not available. "
                "Either install CUDA drivers or set COMPUTE_DEVICE=auto in .env"
            )
        logger.info("Device forced to CUDA.")
        return "cuda", "float16"

    # auto
    if cuda_available:
        logger.info("CUDA detected → using GPU (float16).")
        return "cuda", "float16"
    else:
        logger.info("No CUDA detected → falling back to CPU (int8).")
        return "cpu", "int8"


def chunk_segments(
    segments: list[TranscriptionSegment],
    max_words: int = 6,
    max_duration: float = 3.5,
) -> list[TranscriptionSegment]:
    """
    Break down long segments into bite-sized subtitle chunks (max 5-7 words per event)
    so text never clutters the screen or overlaps.
    """
    chunked: list[TranscriptionSegment] = []

    for seg in segments:
        words = seg.words
        # If we have word timestamps, chunk by words precisely
        if words and len(words) > max_words:
            for i in range(0, len(words), max_words):
                w_slice = words[i : i + max_words]
                chunk_text = " ".join(w.word.strip() for w in w_slice).strip()
                chunk_start = w_slice[0].start
                chunk_end = w_slice[-1].end
                if chunk_end <= chunk_start:
                    chunk_end = chunk_start + 0.5
                chunked.append(
                    TranscriptionSegment(
                        start=chunk_start,
                        end=chunk_end,
                        text=chunk_text,
                        words=w_slice,
                        avg_logprob=seg.avg_logprob,
                        no_speech_prob=seg.no_speech_prob,
                    )
                )
        else:
            text_words = seg.text.strip().split()
            if len(text_words) > max_words:
                total_w = len(text_words)
                dur_per_word = seg.duration / max(total_w, 1)
                for i in range(0, total_w, max_words):
                    w_slice = text_words[i : i + max_words]
                    chunk_text = " ".join(w_slice)
                    c_start = seg.start + (i * dur_per_word)
                    c_end = min(seg.start + ((i + len(w_slice)) * dur_per_word), seg.end)
                    chunked.append(
                        TranscriptionSegment(
                            start=round(c_start, 2),
                            end=round(c_end, 2),
                            text=chunk_text,
                            words=[],
                            avg_logprob=seg.avg_logprob,
                            no_speech_prob=seg.no_speech_prob,
                        )
                    )
            else:
                chunked.append(seg)

    return chunked


# ─────────────────────────────────────────────────────────────────────────────
# Transcription Service
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionService:
    """
    Wraps faster-whisper with automatic device selection and SQLite caching.

    Parameters
    ----------
    model_size : str, optional
        Override the model size from config (tiny/base/small/medium/large-v3).
    compute_device : str, optional
        Override compute device from config (auto/cpu/cuda).
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        compute_device: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self._model_size = model_size or cfg.whisper_model_size
        self._compute_device_pref = compute_device or cfg.compute_device
        self._model = None   # Lazy-load on first transcribe() call

        # Ensure DB is ready
        init_db()

    # ── Lazy model loader ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the faster-whisper model the first time it is needed."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device_and_compute_type(
            self._compute_device_pref
        )
        logger.info(
            "Loading faster-whisper model '%s' on %s (%s) …",
            self._model_size, device, compute_type,
        )
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
            # Use all available CPU threads; on GPU this is ignored
            cpu_threads=0,
        )
        logger.info(
            "Model loaded in %.2f s.", time.perf_counter() - t0
        )
        self._device = device
        self._compute_type = compute_type

    # ── Main API ───────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: Path,
        project_id: Optional[int] = None,
        language: Optional[str] = "en",
        task: str = "transcribe",
        word_timestamps: bool = True,
        use_cache: bool = True,
        force_retranscribe: bool = False,
        chunk_subtitles: bool = True,
        max_words_per_chunk: int = 6,
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_cb: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe *audio_path* and return a TranscriptionResult.

        Parameters
        ----------
        audio_path          : Path to MP3/WAV/M4A/FLAC audio file.
        project_id          : SQLite project id; if provided, result is persisted.
        language            : ISO 639-1 language code (default 'en' for clean English).
        task                : 'transcribe' or 'translate' (translates foreign speech to English).
        word_timestamps     : Include word-level timing.
        use_cache           : Return cached result if audio hash exists in DB.
        force_retranscribe  : Bypass cache and force fresh Whisper inference.
        chunk_subtitles     : Break down long sentences into bite-sized 5-7 word chunks.
        max_words_per_chunk : Maximum words per subtitle event.
        beam_size           : Beam search width (5 = default quality).
        vad_filter          : Strip silent segments with VAD before transcribing.
        """
        audio_path = Path(audio_path).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not is_audio_file(audio_path):
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: .mp3 .wav .aac .flac .ogg .m4a"
            )

        # ── Cache check ────────────────────────────────────────────────────────
        logger.info("Computing hash for %s …", audio_path.name)
        audio_hash = compute_file_hash(audio_path)

        if use_cache and not force_retranscribe:
            cached_row = get_cached_transcription(audio_hash)
            # Only use cache if the cached language matches the requested language
            if cached_row is not None and (language is None or cached_row["language"] == language):
                logger.info("Cache hit for '%s' (language=%s) — skipping Whisper.", audio_path.name, cached_row["language"])
                if progress_cb:
                    progress_cb(1.0)
                raw_segs = _segments_from_json(cached_row["segments_json"])
                segments = chunk_segments(raw_segs, max_words=max_words_per_chunk) if chunk_subtitles else raw_segs
                return TranscriptionResult(
                    audio_path=audio_path,
                    audio_hash=audio_hash,
                    language=cached_row["language"] or "en",
                    full_text=cached_row["full_text"],
                    segments=segments,
                    model_used=cached_row["model_used"] or self._model_size,
                    elapsed_seconds=0.0,
                    from_cache=True,
                )

        # ── Load model ─────────────────────────────────────────────────────────
        if progress_cb:
            progress_cb(0.1)
        self._load_model()

        # ── Run transcription ──────────────────────────────────────────────────
        logger.info("Transcribing '%s' [lang=%s, task=%s] …", audio_path.name, language, task)
        t0 = time.perf_counter()

        raw_segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 400,
            },
        )

        total_dur = getattr(info, "duration", 0.0) or 1.0

        # Materialise generator into list
        segments: list[TranscriptionSegment] = []
        for raw_seg in raw_segments:
            if progress_cb and total_dur > 0:
                pct = min(0.1 + 0.85 * (raw_seg.end / total_dur), 0.95)
                progress_cb(pct)
            words = []
            if word_timestamps and raw_seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in raw_seg.words
                ]
            segments.append(
                TranscriptionSegment(
                    start=raw_seg.start,
                    end=raw_seg.end,
                    text=raw_seg.text,
                    words=words,
                    avg_logprob=raw_seg.avg_logprob,
                    no_speech_prob=raw_seg.no_speech_prob,
                )
            )

        elapsed = time.perf_counter() - t0
        detected_language = info.language or (language or "en")
        full_text = " ".join(seg.text.strip() for seg in segments)

        # Apply bite-sized chunking
        if chunk_subtitles:
            segments = chunk_segments(segments, max_words=max_words_per_chunk)

        logger.info(
            "Transcription complete: %d chunked segments | language=%s | %.1f s",
            len(segments), detected_language, elapsed,
        )

        result = TranscriptionResult(
            audio_path=audio_path,
            audio_hash=audio_hash,
            language=detected_language,
            full_text=full_text,
            segments=segments,
            model_used=self._model_size,
            elapsed_seconds=elapsed,
            from_cache=False,
        )

        # ── Persist to DB ──────────────────────────────────────────────────────
        if project_id is not None:
            self._persist(result, project_id, audio_path)

        if progress_cb:
            progress_cb(1.0)

        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _persist(
        self, result: TranscriptionResult, project_id: int, audio_path: Path
    ) -> None:
        """Save audio asset and transcription to SQLite."""
        try:
            if not asset_hash_exists(result.audio_hash):
                insert_media_asset(
                    project_id=project_id,
                    file_path=str(audio_path),
                    file_hash=result.audio_hash,
                    media_type="audio",
                )
            insert_transcription(
                project_id=project_id,
                audio_hash=result.audio_hash,
                full_text=result.full_text,
                segments_json=result.segments_as_json(),
                language=result.language,
                model_used=result.model_used,
            )
            logger.info("Transcription persisted to database.")
        except Exception as exc:
            # DB errors must never abort the transcription result
            logger.warning("Could not persist transcription to DB: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Internal JSON → dataclass deserialiser
# ─────────────────────────────────────────────────────────────────────────────

def _segments_from_json(segments_json: str) -> list[TranscriptionSegment]:
    """Reconstruct TranscriptionSegment list from stored JSON."""
    raw = json.loads(segments_json)
    segments = []
    for item in raw:
        words = [
            WordTimestamp(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                probability=w.get("probability", 1.0),
            )
            for w in item.get("words", [])
        ]
        segments.append(
            TranscriptionSegment(
                start=item["start"],
                end=item["end"],
                text=item["text"],
                words=words,
                avg_logprob=item.get("avg_logprob", 0.0),
                no_speech_prob=item.get("no_speech_prob", 0.0),
            )
        )
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test block — run:  python services/transcriber.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from utils.config import setup_logging
    from database.db import create_project

    setup_logging()

    # ── Determine test audio path ──────────────────────────────────────────────
    if len(sys.argv) > 1:
        test_audio = Path(sys.argv[1])
    else:
        # Look for any audio file in current directory for a quick test
        here = Path(__file__).parent.parent  # project root
        candidates = list(here.glob("*.mp3")) + list(here.glob("*.wav"))
        if candidates:
            test_audio = candidates[0]
            print(f"[AUTO] Using first audio file found: {test_audio}")
        else:
            print(
                "\n[ERROR] No audio file provided.\n"
                "Usage:  python services/transcriber.py path/to/audio.mp3\n"
                "        (or place any .mp3/.wav in the project root)\n"
            )
            sys.exit(1)

    # ── Initialise a test project ──────────────────────────────────────────────
    from database.db import init_db, create_project
    init_db()
    project_id = create_project("TEST_RUN", template_type="template_1")
    print(f"Test project created → id={project_id}")

    # ── Run transcription ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRANSCRIPTION TEST")
    print(f"  Audio file : {test_audio}")
    print(f"  Model      : {get_config().whisper_model_size}")
    print(f"  Device pref: {get_config().compute_device}")
    print("=" * 60 + "\n")

    svc = TranscriptionService()
    result = svc.transcribe(test_audio, project_id=project_id)

    # ── Display results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULTS")
    print("=" * 60)
    print(f"  Language      : {result.language}")
    print(f"  Segments      : {result.segment_count}")
    print(f"  Elapsed       : {result.elapsed_seconds:.2f} s")
    print(f"  From cache    : {result.from_cache}")
    print(f"  Model used    : {result.model_used}")
    print()
    print("FULL TEXT:")
    print("-" * 60)
    print(result.full_text)
    print()
    print("SEGMENTS (timestamped):")
    print("-" * 60)
    for i, seg in enumerate(result.segments, start=1):
        print(f"  [{i:03}] {seg.start:7.2f}s → {seg.end:7.2f}s | {seg.text.strip()}")

    # ── SRT export ────────────────────────────────────────────────────────────
    srt_path = test_audio.with_suffix(".srt")
    result.save_srt(srt_path)
    print(f"\nSRT written → {srt_path}")

    # ── Cache test: run again, should be instant ───────────────────────────────
    print("\nRunning again to verify cache …")
    result2 = svc.transcribe(test_audio, project_id=project_id)
    assert result2.from_cache, "Expected cached result on second run!"
    print(f"  Cache hit confirmed ✓  (elapsed: {result2.elapsed_seconds:.2f} s)")

    print("\n✓ All transcription tests passed.")


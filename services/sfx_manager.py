"""
services/sfx_manager.py
=======================
Transition Sound Effects (SFX) Library & Audio Mixing Engine for AI Video Editor Studio.

Features:
  1. Curated library of 5–6 high-impact transition sound effects:
     - whoosh_fast.wav    ("Fast Whoosh")
     - whoosh_deep.wav    ("Deep Swoosh")
     - swish_clean.wav    ("Clean Swish")
     - glitch_digital.wav ("Digital Glitch")
     - riser_short.wav    ("Short Riser")
     - pop_smooth.wav     ("Smooth Pop")
  2. Safe fallback synthesis: If any SFX asset is missing on disk, synthesizes
     clean 48kHz 16-bit PCM WAV audio on the fly so rendering NEVER crashes.
  3. Random non-repeating sequence generator: Avoids back-to-back identical SFX
     when 2+ sound effects are selected.
  4. Precise transition timeline mixing: Positions sound effects at clip transition
     boundaries with clamped gain to prevent any audio distortion or clipping.
  5. Clean skip when 0 sound effects are selected.
"""

from __future__ import annotations

import logging
import math
import os
import random
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Standard sample rate for video production
DEFAULT_SAMPLE_RATE = 48000

# Curated catalog metadata
CURATED_SFX_CATALOG: list[dict[str, str]] = [
    {
        "id": "whoosh_fast",
        "name": "Fast Whoosh",
        "filename": "whoosh_fast.wav",
        "description": "Crisp aerodynamic whoosh transition",
    },
    {
        "id": "whoosh_deep",
        "name": "Deep Swoosh",
        "filename": "whoosh_deep.wav",
        "description": "Cinematic low-frequency sub sweep",
    },
    {
        "id": "swish_clean",
        "name": "Clean Swish",
        "filename": "swish_clean.wav",
        "description": "Snappy high-frequency whip cut",
    },
    {
        "id": "glitch_digital",
        "name": "Digital Glitch",
        "filename": "glitch_digital.wav",
        "description": "Modern micro-stutter tech transition",
    },
    {
        "id": "riser_short",
        "name": "Short Riser",
        "filename": "riser_short.wav",
        "description": "Tension ramp-up into scene change",
    },
    {
        "id": "pop_smooth",
        "name": "Smooth Pop",
        "filename": "pop_smooth.wav",
        "description": "Minimalist resonant scene cut impulse",
    },
]


def synthesize_sfx_audio(sfx_id: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> list[int]:
    """
    Synthesize high-quality 16-bit PCM mono samples for a designated SFX ID
    using pure mathematical signal modeling (zero external dependencies).
    """
    samples: list[int] = []

    if sfx_id == "whoosh_fast":
        # Fast aerodynamic whoosh (0.42s): bandpass swept noise + mid sine
        dur = 0.42
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.sin(math.pi * (t / dur)) ** 2
            freq = 250.0 + 1400.0 * math.sin(math.pi * (t / dur))
            # Sine wave + pseudo-white noise
            tone = math.sin(2.0 * math.pi * freq * t) * 0.65
            noise = ((math.sin(i * 12.9898 + 78.233) % 1.0) * 2.0 - 1.0) * 0.35
            val = int((tone + noise) * env * 24000.0)
            samples.append(max(-32767, min(32767, val)))

    elif sfx_id == "whoosh_deep":
        # Deep cinematic sub swoosh (0.55s): low bass pitch drop with warm body
        dur = 0.55
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.sin(math.pi * (t / dur)) ** 1.5
            freq = 140.0 * math.exp(-2.2 * (t / dur)) + 38.0
            tone = math.sin(2.0 * math.pi * freq * t) * 0.8
            sub = math.sin(math.pi * freq * t) * 0.2
            val = int((tone + sub) * env * 26000.0)
            samples.append(max(-32767, min(32767, val)))

    elif sfx_id == "swish_clean":
        # Snappy high-frequency swish (0.32s)
        dur = 0.32
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.sin(math.pi * (t / dur)) ** 3.0
            freq = 700.0 + 2600.0 * math.sin(math.pi * (t / dur))
            tone = math.sin(2.0 * math.pi * freq * t) * 0.7
            airy = ((math.sin(i * 45.123) % 1.0) * 2.0 - 1.0) * 0.3
            val = int((tone + airy) * env * 22000.0)
            samples.append(max(-32767, min(32767, val)))

    elif sfx_id == "glitch_digital":
        # Modern digital stutter / glitch (0.30s)
        dur = 0.30
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.sin(math.pi * (t / dur))
            stutter = 1.0 if (int(t * 40.0) % 2 == 0) else 0.15
            freq = 440.0 if t < 0.1 else (880.0 if t < 0.2 else 220.0)
            # Square-like harmonic
            sig = math.copysign(1.0, math.sin(2.0 * math.pi * freq * t)) * 0.4
            noise = ((math.sin(i * 93.21) % 1.0) * 2.0 - 1.0) * 0.6
            val = int((sig + noise) * env * stutter * 18000.0)
            samples.append(max(-32767, min(32767, val)))

    elif sfx_id == "riser_short":
        # Exponential riser into scene change (0.50s)
        dur = 0.50
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = (t / dur) ** 2.0  # Swells up towards the cut
            freq = 120.0 * math.exp(2.0 * (t / dur))  # Rises from 120Hz to ~880Hz
            tone = math.sin(2.0 * math.pi * freq * t) * 0.75
            harmonic = math.sin(4.0 * math.pi * freq * t) * 0.25
            val = int((tone + harmonic) * env * 25000.0)
            samples.append(max(-32767, min(32767, val)))

    elif sfx_id == "pop_smooth":
        # Smooth resonant impulse pop (0.24s)
        dur = 0.24
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.exp(-14.0 * t)  # Fast decay
            freq = 320.0 * math.exp(-8.0 * t) + 65.0
            tone = math.sin(2.0 * math.pi * freq * t)
            val = int(tone * env * 28000.0)
            samples.append(max(-32767, min(32767, val)))

    else:
        # Generic subtle whoosh fallback
        dur = 0.35
        n = int(dur * sample_rate)
        for i in range(n):
            t = i / sample_rate
            env = math.sin(math.pi * (t / dur)) ** 2
            val = int(math.sin(2.0 * math.pi * 400.0 * t) * env * 20000.0)
            samples.append(max(-32767, min(32767, val)))

    return samples


def write_wav_file(path: Path, samples: list[int], sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Write 16-bit mono PCM samples to disk as a standard WAV file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return path


class SFXManager:
    """
    Manages the transition SFX library, synthesis fallback, random selection,
    and multi-channel audio track placement.
    """

    def __init__(self, sfx_dir: Optional[Path] = None) -> None:
        cfg = get_config()
        self.sfx_dir = Path(sfx_dir) if sfx_dir is not None else (cfg.project_root / "assets" / "sfx")
        self.ensure_default_sfx_library()

    def ensure_default_sfx_library(self) -> dict[str, Path]:
        """
        Ensure all 6 curated transition SFX exist in `assets/sfx`.
        If any file is missing, synthesizes it immediately so the pipeline never crashes.
        """
        self.sfx_dir.mkdir(parents=True, exist_ok=True)
        resolved_map: dict[str, Path] = {}

        for item in CURATED_SFX_CATALOG:
            sfx_id = item["id"]
            wav_path = self.sfx_dir / item["filename"]
            if not wav_path.exists() or wav_path.stat().st_size < 100:
                logger.info("Synthesizing missing transition SFX asset: %s", item["filename"])
                samples = synthesize_sfx_audio(sfx_id)
                write_wav_file(wav_path, samples)
            resolved_map[sfx_id] = wav_path

        return resolved_map

    def get_catalog(self) -> list[dict[str, Any]]:
        """Return the curated SFX catalog with resolved paths and existence flags."""
        catalog = []
        for item in CURATED_SFX_CATALOG:
            sfx_id = item["id"]
            wav_path = self.sfx_dir / item["filename"]
            catalog.append({
                "id": sfx_id,
                "name": item["name"],
                "filename": item["filename"],
                "description": item["description"],
                "path": wav_path,
                "exists": wav_path.exists(),
            })
        return catalog

    def get_sfx_path(self, sfx_id_or_name: str) -> Optional[Path]:
        """Resolve a specific sound effect ID or filename to its Path."""
        clean = sfx_id_or_name.strip().lower()
        for item in CURATED_SFX_CATALOG:
            if clean in (item["id"].lower(), item["name"].lower(), item["filename"].lower()):
                p = self.sfx_dir / item["filename"]
                if not p.exists():
                    self.ensure_default_sfx_library()
                return p
        # Check direct path
        cand = Path(sfx_id_or_name)
        if cand.exists():
            return cand
        return None

    def generate_sfx_sequence(
        self,
        selected_sfx: Sequence[Path | str],
        count: int,
        seed: Optional[int] = None,
    ) -> list[Path]:
        """
        Generate an ordered list of `count` sound effect Paths for scene transitions.
        
        Rules:
          - If selected_sfx is empty or count <= 0: returns [].
          - If len(selected_sfx) == 1: repeats that single sound effect.
          - If len(selected_sfx) >= 2: picks randomly ensuring NO back-to-back
            identical repeats (seq[i] != seq[i-1]).
        """
        if not selected_sfx or count <= 0:
            return []

        resolved_paths: list[Path] = []
        for s in selected_sfx:
            p = self.get_sfx_path(str(s)) if isinstance(s, str) else s
            if p and p.exists():
                resolved_paths.append(p)

        if not resolved_paths:
            return []

        if len(resolved_paths) == 1:
            return [resolved_paths[0]] * count

        rng = random.Random(seed)
        sequence: list[Path] = []
        last_choice: Optional[Path] = None

        for _ in range(count):
            candidates = [p for p in resolved_paths if p != last_choice]
            choice = rng.choice(candidates)
            sequence.append(choice)
            last_choice = choice

        return sequence

    def build_transition_audio_track(
        self,
        clip_durations: list[float],
        selected_sfx: Sequence[Path | str],
        output_path: Path,
        volume: float = 0.35,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        lead_in_sec: float = 0.15,
        seed: Optional[int] = None,
    ) -> Optional[Path]:
        """
        Assemble a single continuous multi-transition WAV audio track aligned with
        the exact video clip cut boundaries.

        Parameters
        ----------
        clip_durations : List of durations in seconds for each clip in the video.
        selected_sfx   : User-selected SFX IDs or Paths.
        output_path    : Destination path for generated continuous sfx_track.wav.
        volume         : Master volume factor for SFX (0.0 to 1.0, clamped).
        sample_rate    : Audio sample rate in Hz (default 48000).
        lead_in_sec    : Time in seconds before cut boundary where SFX begins (e.g. 0.15s).
        seed           : Optional random seed for deterministic reproduction.

        Returns
        -------
        output_path if transitions were placed, or None if skipped.
        """
        if not selected_sfx or len(clip_durations) < 2:
            return None

        n_transitions = len(clip_durations) - 1
        sequence = self.generate_sfx_sequence(selected_sfx, n_transitions, seed=seed)
        if not sequence:
            return None

        total_duration = sum(clip_durations)
        total_samples = int(math.ceil(total_duration * sample_rate))
        master_buffer = [0] * total_samples

        # Clamp volume between 0.0 and 1.0 (safeguard against clipping)
        clamped_vol = max(0.0, min(float(volume), 1.0))
        if clamped_vol <= 0.001:
            return None

        # Calculate transition cut points
        cut_points_sec = []
        accum = 0.0
        for dur in clip_durations[:-1]:
            accum += dur
            cut_points_sec.append(accum)

        # Place each sound effect into the master buffer
        for cut_t, sfx_path in zip(cut_points_sec, sequence):
            sfx_samples = self._load_wav_samples(sfx_path, target_sr=sample_rate)
            if not sfx_samples:
                continue

            start_t = max(0.0, cut_t - lead_in_sec)
            start_idx = int(start_t * sample_rate)

            for offset, sample in enumerate(sfx_samples):
                dest_idx = start_idx + offset
                if dest_idx >= total_samples:
                    break
                # Apply volume scaling & soft mix
                scaled_val = int(sample * clamped_vol)
                mixed = master_buffer[dest_idx] + scaled_val
                # Soft limiter / clamp to 16-bit range
                master_buffer[dest_idx] = max(-32767, min(32767, mixed))

        return write_wav_file(output_path, master_buffer, sample_rate=sample_rate)

    def _load_wav_samples(self, wav_path: Path, target_sr: int = DEFAULT_SAMPLE_RATE) -> list[int]:
        """Read 16-bit PCM samples from a WAV file."""
        if not wav_path.exists():
            return []
        try:
            with wave.open(str(wav_path), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth != 2:
                logger.warning("Unsupported sampwidth %d in %s, expected 16-bit PCM", sampwidth, wav_path.name)
                return []

            unpacked = struct.unpack(f"<{n_frames * channels}h", raw)
            # If stereo or multi-channel, downmix to mono
            if channels == 1:
                return list(unpacked)
            else:
                mono = []
                for i in range(0, len(unpacked), channels):
                    avg = int(sum(unpacked[i : i + channels]) / channels)
                    mono.append(avg)
                return mono
        except Exception as exc:
            logger.warning("Could not read WAV samples from %s: %s", wav_path.name, exc)
            return []

    def mix_voiceover_with_sfx(
        self,
        voiceover_path: Path,
        sfx_track_path: Optional[Path],
        output_path: Path,
    ) -> Path:
        """
        Mix voiceover audio with the transition SFX track into a final audio file using FFmpeg.
        If sfx_track_path is None or missing, copies or links voiceover_path directly.
        """
        voiceover_path = Path(voiceover_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not sfx_track_path or not Path(sfx_track_path).exists():
            return voiceover_path

        from core.ffmpeg_wrapper import resolve_ffmpeg
        ffmpeg_bin, _ = resolve_ffmpeg()

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(voiceover_path),
            "-i", str(Path(sfx_track_path).resolve()),
            "-filter_complex",
            "[0:a]volume=1.0[vo];[1:a]volume=1.0[sfx];[vo][sfx]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            str(output_path),
        ]

        logger.info("Mixing transition SFX track with voiceover → %s", output_path.name)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning("FFmpeg SFX audio mix failed: %s; falling back to clean voiceover.", res.stderr)
            return voiceover_path

        return output_path

"""
core/ffmpeg_wrapper.py
======================
Phase 2 — FFmpeg subprocess wrapper for AI Video Editor Studio.

Responsibilities
----------------
  1. resolve_ffmpeg()   — Find ffmpeg/ffprobe binaries (PATH, common install dirs,
                          or the imageio-ffmpeg bundled binary).
  2. probe_media()      — Return resolution, duration, codec info via ffprobe JSON.
  3. build_ken_burns()  — Generate a 1080p MP4 clip from a still image with a smooth
                          Ken Burns (zoom/pan) effect. CPU / NVENC auto-detected.
  4. concatenate_clips()— Stitch a list of clip files into one video, mix in a
                          voiceover audio track, burn ASS subtitles, and encode the
                          final 1080p MP4.

Design principles
-----------------
  * Every FFmpeg call runs via subprocess.Popen with stderr piped so progress
    (time= lines) can be reported without blocking the main thread.
  * On CPU machines: libx264, preset ultrafast/medium.
  * On GPU (CUDA) machines: h264_nvenc — auto-detected via `ffmpeg -encoders`.
  * No shell=True; arguments always passed as a list to avoid injection and
    path-with-spaces bugs.
  * Windows-safe path handling: forward-slash conversion for filter strings.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional, Any

from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Resolutions
# ─────────────────────────────────────────────────────────────────────────────

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "2k":    (2560, 1440),
    "4k":    (3840, 2160),
}

DEFAULT_RESOLUTION = "1080p"
OUTPUT_WIDTH, OUTPUT_HEIGHT = RESOLUTIONS[DEFAULT_RESOLUTION]
OUTPUT_FPS = 30

# Maximum zoom factor for Ken Burns presets (gentle, cinematic 10% zoom over clip duration)
KB_MAX_ZOOM = 1.10

# Minimum clip duration in seconds (clamp very short segments)
MIN_CLIP_DURATION = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Binary resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_ffmpeg() -> tuple[str, Optional[str]]:
    """
    Return (ffmpeg_path, ffprobe_path) as strings (ffprobe may be None).

    Search order:
      1. Already on PATH (fastest).
      2. FFMPEG_PATH environment variable.
      3. Common Windows install locations.
      4. imageio-ffmpeg bundled binary (installed with scenedetect/opencv).

    Raises RuntimeError if neither ffmpeg binary is found.
    """
    # Prefer explicit env override
    env_path = os.environ.get("FFMPEG_PATH", "").strip()

    def _check_bin(cmd: str) -> bool:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError):
            return False

    def _probe_pair(ff: str) -> tuple[str, Optional[str]] | None:
        """Return (ffmpeg, ffprobe) if ffmpeg works; ffprobe is optional."""
        ffpath = Path(ff)
        ffdir = ffpath.parent if ffpath.is_file() else ffpath
        ffmpeg = str(ffdir / "ffmpeg.exe") if sys.platform == "win32" else str(ffdir / "ffmpeg")
        ffprobe = str(ffdir / "ffprobe.exe") if sys.platform == "win32" else str(ffdir / "ffprobe")
        if ffpath.is_file():
            ffmpeg = str(ffpath)
            ffprobe_candidate = str(ffpath.parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe"))
            ffprobe = ffprobe_candidate if _check_bin(ffprobe_candidate) else None

        if _check_bin(ffmpeg):
            ffprobe_ok = ffprobe if (ffprobe and _check_bin(ffprobe)) else None
            return ffmpeg, ffprobe_ok
        return None

    # 1. PATH
    ffmpeg_on_path = shutil.which("ffmpeg")
    ffprobe_on_path = shutil.which("ffprobe")
    if ffmpeg_on_path and _check_bin(ffmpeg_on_path):
        return ffmpeg_on_path, (ffprobe_on_path if (ffprobe_on_path and _check_bin(ffprobe_on_path)) else None)

    # 2. ENV override
    if env_path:
        pair = _probe_pair(env_path)
        if pair:
            logger.info("Using FFmpeg from FFMPEG_PATH: %s", env_path)
            return pair

    # 3. Common Windows install paths
    common_roots = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        str(Path.home() / "ffmpeg" / "bin"),
    ]
    for root in common_roots:
        pair = _probe_pair(root)
        if pair:
            logger.info("Found FFmpeg at: %s", root)
            return pair

    # 4. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if _check_bin(ff):
            ffprobe_candidate = shutil.which("ffprobe")
            logger.info("Using imageio-ffmpeg bundled binary: %s", ff)
            return ff, (ffprobe_candidate if (ffprobe_candidate and _check_bin(ffprobe_candidate)) else None)
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg not found. Install it and add to PATH, or set FFMPEG_PATH in .env.\n"
        "Windows: https://www.gyan.dev/ffmpeg/builds/  (download ffmpeg-release-essentials.zip)\n"
        "Extract and add the bin/ folder to your system PATH, then restart your terminal.\n"
        "Alternatively: pip install imageio-ffmpeg"
    )


# Cache resolved paths — only find them once per session
_FFMPEG_BIN: Optional[str] = None
_FFPROBE_BIN: Optional[str] = None


def _ff() -> str:
    global _FFMPEG_BIN, _FFPROBE_BIN
    if _FFMPEG_BIN is None:
        _FFMPEG_BIN, _FFPROBE_BIN = resolve_ffmpeg()
    return _FFMPEG_BIN


def _ffprobe() -> Optional[str]:
    global _FFMPEG_BIN, _FFPROBE_BIN
    if _FFMPEG_BIN is None:
        _FFMPEG_BIN, _FFPROBE_BIN = resolve_ffmpeg()
    return _FFPROBE_BIN


# ─────────────────────────────────────────────────────────────────────────────
# Media probe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MediaInfo:
    """Parsed metadata from ffprobe or ffmpeg fallback."""
    path: Path
    width: int = 0
    height: int = 0
    duration: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    fps: float = 0.0
    is_image: bool = False

    @property
    def has_video(self) -> bool:
        return bool(self.video_codec) or (self.width > 0 and not self.is_image)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_codec)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


def _probe_via_ffmpeg_fallback(path: Path) -> MediaInfo:
    """Fallback probe using ffmpeg -i and PIL/cv2 when ffprobe is absent."""
    info = MediaInfo(path=path)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    info.is_image = path.suffix.lower() in image_exts

    if info.is_image:
        try:
            from PIL import Image
            with Image.open(path) as im:
                info.width, info.height = im.size
        except Exception:
            pass
        return info

    cmd = [_ff(), "-nostdin", "-i", str(path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    stderr = result.stderr

    # Duration: 00:01:23.45
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
    if dur_match:
        h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
        info.duration = h * 3600 + m * 60 + s

    # Video stream: e.g. Video: h264, yuv420p, 1920x1080
    v_match = re.search(r"Video:\s*([A-Za-z0-9_]+).*?,\s*(\d+)x(\d+)", stderr)
    if v_match:
        info.video_codec = v_match.group(1)
        info.width = int(v_match.group(2))
        info.height = int(v_match.group(3))

    fps_match = re.search(r"([\d.]+)\s*fps", stderr)
    if fps_match:
        info.fps = float(fps_match.group(1))

    a_match = re.search(r"Audio:\s*([A-Za-z0-9_]+)", stderr)
    if a_match:
        info.audio_codec = a_match.group(1)

    return info


def probe_media(path: Path) -> MediaInfo:
    """
    Run ffprobe (or ffmpeg fallback) on *path* and return a MediaInfo dataclass.

    Supports images (returns duration=0, is_image=True) and video/audio files.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    probe_bin = _ffprobe()
    if not probe_bin:
        return _probe_via_ffmpeg_fallback(path)

    cmd = [
        probe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode != 0:
        return _probe_via_ffmpeg_fallback(path)

    data = json.loads(result.stdout)
    info = MediaInfo(path=path)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Duration from format level (most reliable)
    try:
        info.duration = float(fmt.get("duration", 0))
    except (TypeError, ValueError):
        info.duration = 0.0

    # Parse stream info
    for s in streams:
        codec_type = s.get("codec_type", "")
        if codec_type == "video":
            info.video_codec = s.get("codec_name", "")
            info.width = int(s.get("width", 0))
            info.height = int(s.get("height", 0))
            # FPS from avg_frame_rate e.g. "30000/1001"
            avg_fr = s.get("avg_frame_rate", "0/1")
            try:
                num, den = avg_fr.split("/")
                info.fps = float(num) / float(den) if float(den) > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                info.fps = 0.0
        elif codec_type == "audio":
            info.audio_codec = s.get("codec_name", "")

    # Detect still images: duration==0 and no audio
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    info.is_image = path.suffix.lower() in image_exts

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Ken Burns presets
# ─────────────────────────────────────────────────────────────────────────────

class KenBurnsPreset(Enum):
    ZOOM_IN_CENTER    = auto()   # Smooth zoom in from center (1.0 -> 1.12)
    ZOOM_OUT_CENTER   = auto()   # Smooth zoom out from center (1.12 -> 1.0)
    PAN_LEFT_TO_RIGHT = auto()   # Fixed subtle zoom 1.08, glide L -> R
    PAN_RIGHT_TO_LEFT = auto()   # Fixed subtle zoom 1.08, glide R -> L
    PAN_TOP_TO_BOTTOM = auto()   # Fixed subtle zoom 1.08, glide T -> B
    PAN_BOTTOM_TO_TOP = auto()   # Fixed subtle zoom 1.08, glide B -> T
    ZOOM_IN_LEFT      = auto()   # Backward compatibility
    ZOOM_IN_RIGHT     = auto()   # Backward compatibility


# The 6 balanced, multi-directional Ken Burns presets for static images
KEN_BURNS_PRESETS: list[KenBurnsPreset] = [
    KenBurnsPreset.ZOOM_IN_CENTER,
    KenBurnsPreset.ZOOM_OUT_CENTER,
    KenBurnsPreset.PAN_LEFT_TO_RIGHT,
    KenBurnsPreset.PAN_RIGHT_TO_LEFT,
    KenBurnsPreset.PAN_TOP_TO_BOTTOM,
    KenBurnsPreset.PAN_BOTTOM_TO_TOP,
]

# Backward compatibility alias
_PRESET_CYCLE: list[KenBurnsPreset] = KEN_BURNS_PRESETS


def get_random_ken_burns_preset(
    previous_preset: Optional[KenBurnsPreset] = None,
    rng: Optional[random.Random] = None,
) -> KenBurnsPreset:
    """
    Randomly select one of the 6 cinematic Ken Burns presets, guaranteeing that
    the returned preset is NEVER the same as *previous_preset*.
    """
    r = rng if rng is not None else random
    candidates = [p for p in KEN_BURNS_PRESETS if p != previous_preset]
    if not candidates:
        candidates = KEN_BURNS_PRESETS
    return r.choice(candidates)


def generate_ken_burns_preset_sequence(
    count: int,
    rng: Optional[random.Random] = None,
) -> list[KenBurnsPreset]:
    """
    Generate an evenly balanced, non-repeating sequence of *count* Ken Burns presets.
    Guarantees no two consecutive elements are identical while ensuring equal
    representation across all 6 directions.
    """
    if count <= 0:
        return []
    r = rng if rng is not None else random.Random()
    result: list[KenBurnsPreset] = []
    pool = list(KEN_BURNS_PRESETS)

    while len(result) < count:
        chunk = pool.copy()
        r.shuffle(chunk)
        if result and chunk[0] == result[-1]:
            # Swap chunk[0] with another element in chunk to avoid consecutive duplicate
            swap_idx = r.randint(1, len(chunk) - 1)
            chunk[0], chunk[swap_idx] = chunk[swap_idx], chunk[0]
        result.extend(chunk)

    return result[:count]


def _get_zoompan_params(preset: KenBurnsPreset, total_frames: int) -> tuple[str, str, str]:
    """
    Return (z, x, y) zoompan parameter strings for the specified preset.
    Operates on a linear smooth progression `on/{total_frames}` with gentle zoom range.
    """
    den = max(total_frames, 1)

    if preset == KenBurnsPreset.ZOOM_IN_CENTER:
        # 1. ZOOM_IN_CENTER: Zoom smoothly from 1.0 -> 1.12 centered
        z = f"1.0+0.12*(on/{den})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    elif preset == KenBurnsPreset.ZOOM_OUT_CENTER:
        # 2. ZOOM_OUT_CENTER: Zoom smoothly from 1.12 -> 1.0 centered
        z = f"1.12-0.12*(on/{den})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    elif preset == KenBurnsPreset.PAN_LEFT_TO_RIGHT:
        # 3. PAN_LEFT_TO_RIGHT: Hold subtle zoom 1.08 while gliding smoothly from left to right
        z = "1.08"
        x = f"(iw-iw/zoom)*(on/{den})"
        y = "ih/2-(ih/zoom/2)"

    elif preset == KenBurnsPreset.PAN_RIGHT_TO_LEFT:
        # 4. PAN_RIGHT_TO_LEFT: Hold subtle zoom 1.08 while gliding smoothly from right to left
        z = "1.08"
        x = f"(iw-iw/zoom)*(1-on/{den})"
        y = "ih/2-(ih/zoom/2)"

    elif preset == KenBurnsPreset.PAN_TOP_TO_BOTTOM:
        # 5. PAN_TOP_TO_BOTTOM: Hold subtle zoom 1.08 while gliding smoothly from top to bottom
        z = "1.08"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(on/{den})"

    elif preset == KenBurnsPreset.PAN_BOTTOM_TO_TOP:
        # 6. PAN_BOTTOM_TO_TOP: Hold subtle zoom 1.08 while gliding smoothly from bottom to top
        z = "1.08"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-on/{den})"

    elif preset == KenBurnsPreset.ZOOM_IN_LEFT:
        # Compatibility: Zoom in and pan slightly left
        z = f"1.0+0.12*(on/{den})"
        x = f"(iw-iw/zoom)*(1-on/{den})"
        y = "ih/2-(ih/zoom/2)"

    elif preset == KenBurnsPreset.ZOOM_IN_RIGHT:
        # Compatibility: Zoom in and pan slightly right
        z = f"1.0+0.12*(on/{den})"
        x = f"(iw-iw/zoom)*(on/{den})"
        y = "ih/2-(ih/zoom/2)"

    else:
        # Default fallback
        z = f"1.0+0.12*(on/{den})"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    return z, x, y


def _build_zoompan_filter(
    duration: float,
    preset: KenBurnsPreset = KenBurnsPreset.ZOOM_IN_CENTER,
    resolution: str = DEFAULT_RESOLUTION,
    fps: int = OUTPUT_FPS,
) -> str:
    """
    Build the high-resolution zoompan + scale filtergraph string.

    Butter-Smooth Strategy:
      - Upscale input image by 2.5x with Lanczos algorithm to create a high-density
        pixel grid (e.g. 4800x2700 for 1080p, 5120x2880 for 2K).
      - Zoompan calculates sub-pixel crops without integer rounding stutter.
      - Downscale directly to target resolution (s=ow x oh) for crisp, anti-aliased output.
    """
    ow, oh = RESOLUTIONS.get(resolution.lower(), RESOLUTIONS[DEFAULT_RESOLUTION])
    if resolution.lower() == "2k":
        pw, ph = 5120, 2880
    else:
        pw = max(3840, int(ow * 2.5))
        ph = int(pw * (oh / ow))

    pw = pw if pw % 2 == 0 else pw + 1
    ph = ph if ph % 2 == 0 else ph + 1

    total_frames = max(int(round(duration * fps)), fps)
    z, x, y = _get_zoompan_params(preset, total_frames)

    prescale = (
        f"scale={pw}:{ph}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={pw}:{ph},"
        f"setsar=1"
    )

    zoompan = (
        f"zoompan="
        f"z='{z}':"
        f"x='{x}':"
        f"y='{y}':"
        f"d={total_frames}:"
        f"fps={fps}:"
        f"s={ow}x{oh}"
    )

    pixfmt = "format=yuv420p"
    return f"{prescale},{zoompan},{pixfmt}"


def _build_portrait_blurred_filter(
    duration: float,
    preset: KenBurnsPreset = KenBurnsPreset.ZOOM_IN_CENTER,
    resolution: str = DEFAULT_RESOLUTION,
    fps: int = OUTPUT_FPS,
    src_w: int = 0,
    src_h: int = 0,
) -> str:
    """
    Dual-layer filtergraph for portrait (9:16) or square images on a 16:9 canvas:
      - 100% Static Blurred Background: scaled to fill WxH, heavily boxblurred, slightly darkened,
        and held completely stationary with zero zoompan motion.
      - Smooth Foreground Ken Burns Motion: scaled cleanly with high-res 2.5x Lanczos upscaling,
        applying the selected Ken Burns preset exclusively to the foreground layer [fg],
        overlaid perfectly centered onto the static background.
    """
    ow, oh = RESOLUTIONS.get(resolution.lower(), RESOLUTIONS[DEFAULT_RESOLUTION])

    # Determine fitted foreground dimensions (fw, fh) preserving aspect ratio
    if src_w > 0 and src_h > 0:
        src_ar = src_w / src_h
    else:
        src_ar = 9.0 / 16.0

    target_ar = ow / oh
    if src_ar < target_ar:
        # Height-constrained (portrait)
        fh = oh
        fw = int(round(oh * src_ar))
    else:
        # Width-constrained (square or intermediate)
        fw = ow
        fh = int(round(ow / src_ar))

    fw = fw if fw % 2 == 0 else fw + 1
    fh = fh if fh % 2 == 0 else fh + 1

    # High-density pre-scale (2.5x) for butter-smooth foreground motion
    pw = int(round(fw * 2.5))
    ph = int(round(fh * 2.5))
    pw = pw if pw % 2 == 0 else pw + 1
    ph = ph if ph % 2 == 0 else ph + 1

    total_frames = max(int(round(duration * fps)), fps)
    z, x, y = _get_zoompan_params(preset, total_frames)

    vf = (
        f"split=2[bg_src][fg_src];"
        f"[bg_src]trim=end_frame=1,scale={ow}:{oh}:force_original_aspect_ratio=increase:flags=lanczos,crop={ow}:{oh},"
        f"boxblur=25:10,eq=brightness=-0.08,setsar=1,loop=loop=-1:size=1:start=0[bg];"
        f"[fg_src]scale={pw}:{ph}:flags=lanczos,setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={total_frames}:fps={fps}:s={fw}x{fh}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,format=yuv420p"
    )
    return vf


# ─────────────────────────────────────────────────────────────────────────────
# Hardware encoder detection
# ─────────────────────────────────────────────────────────────────────────────

_NVENC_AVAILABLE: Optional[bool] = None


def _nvenc_available() -> bool:
    """Check (once) whether h264_nvenc is available and physically functional on this hardware."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    try:
        # Check encoder list first
        result = subprocess.run(
            [_ff(), "-encoders", "-v", "quiet"],
            capture_output=True, text=True, check=False
        )
        if "h264_nvenc" not in result.stdout:
            _NVENC_AVAILABLE = False
        else:
            # Active hardware probe: verify that NVIDIA driver & GPU actually accept NVENC
            test_cmd = [
                _ff(), "-v", "quiet",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.04",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ]
            probe = subprocess.run(test_cmd, capture_output=True, check=False)
            _NVENC_AVAILABLE = (probe.returncode == 0)
    except Exception:
        _NVENC_AVAILABLE = False

    logger.info("h264_nvenc hardware functional: %s", _NVENC_AVAILABLE)
    return _NVENC_AVAILABLE


def _get_encoder_args(use_gpu: Optional[bool] = None) -> list[str]:
    """
    Return FFmpeg video encoder arguments calibrated for crisp 2K video fidelity.

    CPU Fallback: -c:v libx264 -crf 19 -preset medium -pix_fmt yuv420p
    NVENC (Production): -c:v h264_nvenc -preset p5 -cq 19 -b:v 16M -maxrate 22M
    """
    cfg = get_config()
    force_cuda = cfg.compute_device == "cuda"
    force_cpu = cfg.compute_device == "cpu"

    if use_gpu is True or (use_gpu is None and not force_cpu and _nvenc_available()):
        logger.debug("Encoder: h264_nvenc (2K High Quality NVENC)")
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",            # High-quality preset
            "-rc", "vbr",
            "-cq", "19",
            "-b:v", "16M",
            "-maxrate", "22M",
            "-bufsize", "32M",
            "-pix_fmt", "yuv420p",
        ]

    logger.debug("Encoder: libx264 (2K High Quality CPU)")
    return [
        "-c:v", "libx264",
        "-preset", "medium",        # Crisp visual encoding
        "-crf", "19",               # High visual fidelity (visually lossless)
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # Web-compatible MP4
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Progress reader utility
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_progress(
    cmd: list[str],
    total_duration: float,
    label: str = "FFmpeg",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> None:
    """
    Run an FFmpeg command and parse stderr for progress updates.

    *total_duration* is used to compute percentage completion from FFmpeg's
    ``time=`` output.  *progress_cb* receives a float 0.0–1.0.
    Raises RuntimeError on non-zero exit code.
    """
    logger.debug("Running: %s", " ".join(cmd))
    t0 = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    time_pattern = re.compile(r"time=(\d+):(\d+):([\d.]+)")
    last_pct = -1.0

    for line in proc.stderr:  # type: ignore[union-attr]
        m = time_pattern.search(line)
        if m and total_duration > 0:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            current = h * 3600 + mi * 60 + s
            pct = min(current / total_duration, 1.0)
            if pct - last_pct >= 0.05:  # report every 5%
                logger.debug("%s: %.0f%%", label, pct * 100)
                if progress_cb:
                    progress_cb(pct)
                last_pct = pct

    proc.wait()
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg exited with code {proc.returncode} during '{label}'. "
            f"Check logs for details."
        )
    logger.info("%s complete in %.1f s.", label, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Ken Burns clip generator
# ─────────────────────────────────────────────────────────────────────────────

def build_ken_burns_clip(
    image_path: Path,
    duration: float,
    output_path: Path,
    preset: KenBurnsPreset = KenBurnsPreset.ZOOM_IN_CENTER,
    resolution: str = DEFAULT_RESOLUTION,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Render a Ken Burns video clip from a still image in 2K (or specified resolution).

    Parameters
    ----------
    image_path  : Source image (.jpg / .png / etc.)
    duration    : Clip length in seconds (clamped to MIN_CLIP_DURATION).
    output_path : Where to save the resulting .mp4 clip.
    preset      : Which Ken Burns motion preset to apply.
    resolution  : Target resolution ('2k' for 2560x1440, '1080p', etc.).
    progress_cb : Optional 0.0→1.0 progress callback.

    Returns
    -------
    output_path after successful render.
    """
    image_path = Path(image_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(duration, MIN_CLIP_DURATION)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Inspect aspect ratio: if portrait (9:16) or square (1:1), use blurred pillarbox
    src_w, src_h = 0, 0
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            src_w, src_h = im.size
    except Exception:
        pass

    if src_w == 0 or src_h == 0:
        info = probe_media(image_path)
        src_w, src_h = info.width, info.height

    ow, oh = RESOLUTIONS.get(resolution.lower(), RESOLUTIONS[DEFAULT_RESOLUTION])
    target_ar = ow / oh  # 16/9 ≈ 1.777...
    src_ar = (src_w / src_h) if src_h > 0 else target_ar

    # Dual-layer blurred background for portrait or square images
    is_mismatched = (src_h > src_w) or (target_ar - src_ar > 0.35)

    if is_mismatched:
        logger.info(
            "Dual-Layer Blurred Pillarbox applied to %s (%dx%d, AR=%.2f vs target AR=%.2f, Preset=%s)",
            image_path.name, src_w, src_h, src_ar, target_ar, preset.name
        )
        vf = _build_portrait_blurred_filter(
            duration, preset=preset, resolution=resolution, fps=OUTPUT_FPS, src_w=src_w, src_h=src_h
        )
    else:
        vf = _build_zoompan_filter(duration, preset=preset, resolution=resolution, fps=OUTPUT_FPS)

    enc = _get_encoder_args()

    cmd = [
        _ff(),
        "-y",                       # overwrite output
        "-loop", "1",               # treat still image as infinite source
        "-framerate", str(OUTPUT_FPS),
        "-i", str(image_path),
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-r", str(OUTPUT_FPS),
        "-an",                       # no audio in this clip
        *enc,
        str(output_path),
    ]

    logger.info(
        "Ken Burns: %s [%.1f s, %s] → %s",
        image_path.name, duration, preset.name, output_path.name
    )
    _run_with_progress(cmd, duration, label=f"KB:{image_path.name}", progress_cb=progress_cb)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Concat list writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_concat_list(clip_paths: list[Path], list_file: Path) -> None:
    """
    Write an FFmpeg concat demuxer input file.

    Path separators are normalised to forward-slashes and paths with spaces are
    wrapped in single-quotes as required by the concat demuxer spec.
    """
    lines = ["ffconcat version 1.0\n"]
    for p in clip_paths:
        # Normalize to forward-slash (Windows compatibility)
        fp = str(p.resolve()).replace("\\", "/")
        lines.append(f"file '{fp}'\n")
    list_file.write_text("".join(lines), encoding="utf-8")
    logger.debug("Concat list → %s (%d clips)", list_file.name, len(clip_paths))


# ─────────────────────────────────────────────────────────────────────────────
# Final video assembler
# ─────────────────────────────────────────────────────────────────────────────

def concatenate_clips(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    subtitle_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    total_duration: Optional[float] = None,
    resolution: str = DEFAULT_RESOLUTION,
    **kwargs: Any,
) -> Path:
    """
    Concatenate video clips, mix in voiceover audio, optionally burn ASS/SRT
    subtitles, and encode the final 1080p MP4.

    Two-pass strategy
    -----------------
    Pass 1: concat demuxer → joins clips with no re-encode of the video stream.
    Pass 2: add audio + burn subtitles via libx264/nvenc (one transcode).

    This avoids a costly multi-input concat filtergraph for large clip counts
    while still giving us the subtitle filter access to the final combined video.

    Parameters
    ----------
    clip_paths      : Ordered list of clip .mp4 files (must all be 1920×1080).
    audio_path      : Full voiceover audio file.
    output_path     : Final output .mp4 path.
    subtitle_path   : Optional .ass or .srt subtitle file to burn in.
    progress_cb     : Optional 0→1 progress callback.
    total_duration  : Total video duration in seconds (auto-summed from probes
                      if not provided, used for progress reporting).

    Returns
    -------
    output_path after successful render.
    """
    if not clip_paths:
        raise ValueError("clip_paths is empty — nothing to concatenate.")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = get_config()

    # ── Estimate total duration for progress bar ───────────────────────────────
    if total_duration is None:
        total_duration = sum(probe_media(p).duration for p in clip_paths if p.exists())

    # ── Write concat list ──────────────────────────────────────────────────────
    concat_list = cfg.temp_cache_dir / "concat_list.txt"
    _write_concat_list(clip_paths, concat_list)

    # ── Build filter_complex / -vf chain ──────────────────────────────────────
    # Subtitle filter needs special Windows path treatment:
    # backslashes → forward slashes, colon in drive letter escaped as \\:
    vf_parts: list[str] = []
    if subtitle_path and subtitle_path.exists():
        sub_fp = str(subtitle_path.resolve()).replace("\\", "/")
        # Escape colon in drive letter for FFmpeg filter syntax: C:/... → C\:/...
        sub_fp = re.sub(r"^([A-Za-z]):/", r"\1\\:/", sub_fp)
        ext = subtitle_path.suffix.lower()
        if ext == ".ass":
            vf_parts.append(f"ass='{sub_fp}'")
        else:
            vf_parts.append(f"subtitles='{sub_fp}'")

    vf_str = ",".join(vf_parts) if vf_parts else None

    # ── Encoder args ───────────────────────────────────────────────────────────
    enc = _get_encoder_args()

    # ── Build final FFmpeg command ─────────────────────────────────────────────
    cmd = [
        _ff(),
        "-y",
        # Video input: concat demuxer
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        # Audio input: voiceover
        "-i", str(audio_path.resolve()),
        # Subtitle / no-op video filter
        *(["-vf", vf_str] if vf_str else []),
        # Encode
        *enc,
        # Audio stream: re-encode to AAC 192k for compatibility
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        # Sync: end at the shorter of video/audio
        "-shortest",
        # Output
        str(output_path),
    ]

    logger.info(
        "Concatenating %d clip(s) + audio [subs=%s] → %s",
        len(clip_paths),
        subtitle_path.name if subtitle_path else "none",
        output_path.name,
    )
    _run_with_progress(
        cmd, total_duration, label="Final render", progress_cb=progress_cb
    )
    return output_path


def extract_subclip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    resolution: str = DEFAULT_RESOLUTION,
    target_duration: Optional[float] = None,
    keep_audio: bool = False,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Extract a normalized scene slice from a source video file (e.g., episode footage).
    Ensures uniform resolution (Lanczos scale + letterbox/pillarbox pad) and framerate (30 fps).

    Parameters
    ----------
    video_path      : Path to source video file
    start_sec       : Start timestamp in seconds
    end_sec         : End timestamp in seconds
    output_path     : Destination MP4 path
    resolution      : '1080p' (1920x1080) or '2k' (2560x1440)
    target_duration : Optional target duration in seconds to trim or pad
    keep_audio      : Whether to preserve original audio track (False = mute for voiceover overlay)
    progress_cb     : Callback receiving progress float (0.0 to 1.0)
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    raw_dur = max(end_sec - start_sec, 0.1)
    effective_dur = target_duration if target_duration is not None and target_duration > 0 else raw_dur

    w, h = RESOLUTIONS.get(resolution, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    # Scale with aspect ratio maintained + pad to exact canvas dimensions
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1,fps={OUTPUT_FPS}"
    )

    enc = _get_encoder_args()

    cmd = [
        _ff(),
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-i", str(video_path),
        "-vf", vf,
        *enc,
        *(["-c:a", "aac", "-b:a", "128k"] if keep_audio else ["-an"]),
        "-t", f"{effective_dur:.3f}",
        str(output_path),
    ]

    logger.info(
        "Extracting subclip [%.2fs - %.2fs (%.2fs)] from %s → %s",
        start_sec,
        end_sec,
        effective_dur,
        video_path.name,
        output_path.name,
    )
    _run_with_progress(cmd, effective_dur, label=f"Extract {output_path.stem}", progress_cb=progress_cb)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test / diagnostic
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.config import setup_logging
    setup_logging()

    print("Resolving FFmpeg …")
    try:
        ff, ffp = resolve_ffmpeg()
        print(f"  ffmpeg  : {ff}")
        print(f"  ffprobe : {ffp}")
        print(f"  NVENC   : {_nvenc_available()}")
    except RuntimeError as e:
        print(f"[ERROR] {e}")


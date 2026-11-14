import time
from pathlib import Path
from services.hybrid_mixer import HybridMediaMixer
from core.ffmpeg_wrapper import build_ken_burns_clip, extract_subclip, concatenate_clips, probe_media
from services.subtitle_engine import SubtitleEngine
from services.transcriber import TranscriptionResult, TranscriptionSegment

assets_dir = Path("assets/sample_footage")
audio_path = assets_dir / "Prince_William_arrived_in_London.mp3"
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
out_file = output_dir / "test_preview_15s.mp4"
temp_dir = Path("temp/preview_chunks")
temp_dir.mkdir(parents=True, exist_ok=True)

print("1. Setting up 15s preview mixer...")
mixer = HybridMediaMixer(
    images_dir=assets_dir,
    videos_path=assets_dir,
    clip_duration=5.0,
    image_ratio=0.5,
)

slots = mixer.create_mix(total_duration=15.0)
print(f"Slots created: {len(slots)}")
for s in slots:
    desc = s.preset.name if s.media_type == "image" else f"Subclip {s.subclip_start:.1f}s"
    print(f"  Slot {s.slot_index}: {s.media_type.upper()} ({s.duration}s) - {desc}")

print("2. Rendering video chunks...")
t0 = time.time()
clip_paths = []
for idx, s in enumerate(slots):
    clip_p = temp_dir / f"clip_{idx:02d}.mp4"
    if s.media_type == "image":
        build_ken_burns_clip(
            image_path=s.source_path,
            duration=s.duration,
            output_path=clip_p,
            preset=s.preset,
            resolution="1080p",
        )
    else:
        extract_subclip(
            video_path=s.source_path,
            start_sec=s.subclip_start,
            end_sec=s.subclip_end,
            output_path=clip_p,
            resolution="1080p",
            target_duration=s.duration,
            keep_audio=False,
        )
    clip_paths.append(clip_p)
    print(f"  ✓ Chunk {idx+1}/{len(slots)} rendered: {clip_p.name}")

print("3. Generating sample subtitles...")
segs = [
    TranscriptionSegment(start=0.5, end=4.5, text="Prince William arrived in London today."),
    TranscriptionSegment(start=5.0, end=9.5, text="Crowds gathered along the Mall to catch a glimpse."),
    TranscriptionSegment(start=10.0, end=14.5, text="The royal motorcade proceeded to the palace."),
]
sub_res = TranscriptionResult(
    audio_path=audio_path,
    audio_hash="preview_hash",
    language="en",
    full_text=" ".join(s.text for s in segs),
    segments=segs,
    model_used="small",
    elapsed_seconds=0.0,
)
engine = SubtitleEngine(
    text_color="#FFFF00",
    highlight_color="#00FFFF",
    has_bg_box=True,
    bg_style="semi_transparent_black",
    bg_color="#000000",
    bg_opacity=0.75,
    box_padding=14.0,
    font_size=56,
    font_name="Montserrat",
    bold=True,
    resolution="1080p",
)
ass_file = temp_dir / "preview.ass"
engine.generate_ass(sub_res, ass_file)

print("4. Concatenating and burning subtitles...")
concatenate_clips(
    clip_paths=clip_paths,
    audio_path=audio_path,
    output_path=out_file,
    subtitle_path=ass_file,
    total_duration=15.0,
    resolution="1080p",
)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s!")
info = probe_media(out_file)
print(f"Output: {out_file} ({info.width}x{info.height}, duration={info.duration:.1f}s, size={out_file.stat().st_size / (1024*1024):.2f} MB)")


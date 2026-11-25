"""
Unit tests for Auto-Detection of Asset Contents and Automatic export/ Subfolder Generation.
"""
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from services.character_matcher import CharacterMatcher
from ui.views.studio_view import StudioView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_template1_asset_auto_detection(tmp_path: Path):
    """
    Template 1 (Hybrid Images & Video Mix):
    Verify scanning counts character subfolders, valid images, and B-roll clips,
    and ignores the export/ subfolder.
    """
    assets_dir = tmp_path / "template1_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Subfolder 1: Harry (3 images)
    harry_dir = assets_dir / "Harry"
    harry_dir.mkdir()
    (harry_dir / "img1.jpg").write_bytes(b"fake_jpg")
    (harry_dir / "img2.png").write_bytes(b"fake_png")
    (harry_dir / "img3.webp").write_bytes(b"fake_webp")

    # Subfolder 2: Catherine (2 images)
    catherine_dir = assets_dir / "Catherine"
    catherine_dir.mkdir()
    (catherine_dir / "img4.jpeg").write_bytes(b"fake_jpeg")
    (catherine_dir / "img5.png").write_bytes(b"fake_png")

    # Root video clip (1 B-roll)
    (assets_dir / "broll.mp4").write_bytes(b"fake_mp4")

    # Export folder with dummy files (must be completely ignored)
    export_dir = assets_dir / "export"
    export_dir.mkdir()
    (export_dir / "ignored_output.mp4").write_bytes(b"fake_mp4")
    (export_dir / "ignored_thumb.jpg").write_bytes(b"fake_jpg")

    result = CharacterMatcher.scan_asset_folder(assets_dir, template_type="template_1")

    assert result["character_folders_count"] == 2
    assert "Harry" in result["character_pools"]
    assert "Catherine" in result["character_pools"]
    assert "export" not in result["character_pools"]
    assert result["total_images"] == 5
    assert len(result["video_files"]) == 1
    assert "📁 Assets Detected: 2 Character Folders, 5 Images, 1 B-Roll clip" in result["badge_text"]


def test_template2_episode_auto_detection(tmp_path: Path):
    """
    Template 2 (Episode Scene Matching):
    Verify scanning counts raw episode movie files across root and subfolders,
    while ignoring the export/ subfolder.
    """
    footage_dir = tmp_path / "season_episodes"
    footage_dir.mkdir(parents=True, exist_ok=True)

    # 3 root video files
    (footage_dir / "episode_01.mp4").write_bytes(b"fake_mp4")
    (footage_dir / "episode_02.mkv").write_bytes(b"fake_mkv")
    (footage_dir / "episode_03.mov").write_bytes(b"fake_mov")

    # 1 subfolder video file
    extras_dir = footage_dir / "extras"
    extras_dir.mkdir()
    (extras_dir / "bonus_clip.mp4").write_bytes(b"fake_mp4")

    # export folder (must be ignored)
    export_dir = footage_dir / "export"
    export_dir.mkdir()
    (export_dir / "rendered_final.mp4").write_bytes(b"fake_mp4")

    result = CharacterMatcher.scan_asset_folder(footage_dir, template_type="template_2")

    assert len(result["video_files"]) == 4
    assert result["badge_text"] == "🎬 Episode Source Detected: 4 Movie Files Ready"


def test_export_directory_auto_created_on_folder_select(qapp, tmp_path: Path):
    """
    Selecting an asset folder in StudioView must automatically create an export/ subfolder.
    """
    assets_dir = tmp_path / "raw_project_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Initially no export subfolder
    assert not (assets_dir / "export").exists()

    studio_view = StudioView()
    studio_view.edit_assets.setText(str(assets_dir))

    # Verify export/ subfolder is created
    export_dir = assets_dir / "export"
    assert export_dir.exists()
    assert export_dir.is_dir()
    assert studio_view._get_default_export_dir() == export_dir


def test_studio_view_dynamic_output_path_routing(qapp, tmp_path: Path):
    """
    Verify that output path routes into export/ and updates cleanly based on:
    1. Folder name default ({folder}_Final.mp4)
    2. Single audio stem ({audio_stem}.mp4)
    3. Multi-part audio stem ({base_stem}_Full.mp4)
    """
    assets_dir = tmp_path / "Mystery_Documentary"
    assets_dir.mkdir(parents=True, exist_ok=True)

    studio_view = StudioView()
    studio_view.edit_assets.setText(str(assets_dir))

    # 1. Output path defaults to export/Mystery_Documentary_Final.mp4
    expected_default = assets_dir / "export" / "Mystery_Documentary_Final.mp4"
    assert Path(studio_view.edit_output.text()).resolve() == expected_default.resolve()

    # 2. Selecting a single audio file updates output to export/voiceover_narration.mp4
    single_audio = tmp_path / "voiceover_narration.mp3"
    single_audio.write_bytes(b"fake_mp3")
    studio_view.edit_audio.setText(str(single_audio))

    expected_single = assets_dir / "export" / "voiceover_narration.mp4"
    assert Path(studio_view.edit_output.text()).resolve() == expected_single.resolve()

    # 3. Selecting multi-part audio updates output to export/{clean_stem}_Full.mp4
    part1 = tmp_path / "Narration_Part_1.mp3"
    part2 = tmp_path / "Narration_Part_2.mp3"
    part1.write_bytes(b"fake_mp3")
    part2.write_bytes(b"fake_mp3")
    studio_view.edit_audio.setText(f"{part1};{part2}")

    expected_multi = assets_dir / "export" / "Narration_Full.mp4"
    assert Path(studio_view.edit_output.text()).resolve() == expected_multi.resolve()


def test_template_switch_updates_asset_badge_and_label(qapp, tmp_path: Path):
    """
    Switching between Template 1 and Template 2 updates the UI section tag label
    and refreshes the scan summary badge accordingly.
    """
    footage_dir = tmp_path / "episodes_library"
    footage_dir.mkdir(parents=True, exist_ok=True)
    (footage_dir / "movie_ep1.mp4").write_bytes(b"fake_mp4")
    (footage_dir / "movie_ep2.mp4").write_bytes(b"fake_mp4")

    studio_view = StudioView()
    studio_view.edit_assets.setText(str(footage_dir))

    # Switch to Template 2 (Index 1)
    studio_view.combo_format.setCurrentIndex(1)
    assert "EPISODE" in studio_view.assets_tag.text().upper()
    assert "🎬 Episode Source Detected: 2 Movie Files Ready" in studio_view.lbl_assets_badge.text()

    # Switch back to Template 1 (Index 0)
    studio_view.combo_format.setCurrentIndex(0)
    assert "FOOTAGE & VISUAL ASSETS" in studio_view.assets_tag.text().upper()
    assert "📁 Assets Detected" in studio_view.lbl_assets_badge.text()

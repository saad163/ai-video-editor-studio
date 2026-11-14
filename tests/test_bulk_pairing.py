"""
tests/test_bulk_pairing.py
==========================
Unit tests for bulk generation smart auto-pairing and natural sorting.
Verifies:
  1. extract_numeric_index(): digits and spelled-out number words (1..20).
  2. natural_sort_key(): audio 1..audio 10 sort naturally (not 1, 10, 2).
  3. CharacterMatcher.find_matching_footage_folder():
     - audio 1.mp3 -> video 1 content/
     - audio 2.mp3 -> video 2 content/
     - audio 10.mp3 -> video 10 content/
     - spelled words: audio one -> video 1 content/
     - fallback to root if no numbered match
"""

from pathlib import Path
from services.character_matcher import (
    extract_numeric_index,
    natural_sort_key,
    CharacterMatcher,
)


def test_extract_numeric_index():
    assert extract_numeric_index("audio 1.mp3") == 1
    assert extract_numeric_index("audio_02.wav") == 2
    assert extract_numeric_index("audio-10.mp3") == 10
    assert extract_numeric_index("15.mp3") == 15
    assert extract_numeric_index("video 1 content") == 1
    assert extract_numeric_index("video_20_content") == 20

    # Spelled out number words
    assert extract_numeric_index("audio one.mp3") == 1
    assert extract_numeric_index("audio three.wav") == 3
    assert extract_numeric_index("video ten content") == 10

    # No number
    assert extract_numeric_index("background_music.mp3") is None


def test_natural_sort_key():
    files = [
        "audio 10.mp3",
        "audio 1.mp3",
        "audio 2.mp3",
        "audio 20.mp3",
        "audio 3.mp3",
    ]
    sorted_files = sorted(files, key=natural_sort_key)
    expected = [
        "audio 1.mp3",
        "audio 2.mp3",
        "audio 3.mp3",
        "audio 10.mp3",
        "audio 20.mp3",
    ]
    assert sorted_files == expected


def test_find_matching_footage_folder(tmp_path):
    root_footage = tmp_path / "footage_root"
    root_footage.mkdir()

    sub1 = root_footage / "video 1 content"
    sub1.mkdir()
    (sub1 / "img1.jpg").write_bytes(b"dummy")

    sub2 = root_footage / "video 2 content"
    sub2.mkdir()
    (sub2 / "img2.jpg").write_bytes(b"dummy")
    (sub2 / "b_roll.mp4").write_bytes(b"dummy")

    sub10 = root_footage / "video 10 content"
    sub10.mkdir()
    (sub10 / "img10.png").write_bytes(b"dummy")

    # Audio 1 pairing
    matched1, summary1 = CharacterMatcher.find_matching_footage_folder("audio 1.mp3", root_footage)
    assert matched1 == sub1
    assert summary1["total_images"] == 1

    # Audio 2 pairing
    matched2, summary2 = CharacterMatcher.find_matching_footage_folder("audio 2.mp3", root_footage)
    assert matched2 == sub2
    assert summary2["total_images"] == 1
    assert len(summary2["video_files"]) == 1

    # Audio 10 pairing
    matched10, summary10 = CharacterMatcher.find_matching_footage_folder("audio 10.wav", root_footage)
    assert matched10 == sub10
    assert summary10["total_images"] == 1

    # Fallback when no match
    matched_unknown, summary_unknown = CharacterMatcher.find_matching_footage_folder("audio 99.mp3", root_footage)
    assert matched_unknown == root_footage


def test_bulk_view_ui_table_columns(qapp, tmp_path):
    from ui.views.bulk_view import BulkView

    view = BulkView()
    assert view.table.columnCount() == 8
    assert view.combo_template.count() == 2
    assert "Template 1" in view.combo_template.itemText(0)
    assert "Template 2" in view.combo_template.itemText(1)

    # Test auto-pairing row population
    root_footage = tmp_path / "footage"
    root_footage.mkdir()
    sub1 = root_footage / "video 1 content"
    sub1.mkdir()
    (sub1 / "scene.jpg").write_bytes(b"dummy")

    view.edit_footage_dir.setText(str(root_footage))

    audio_file = tmp_path / "audio 1.mp3"
    audio_file.write_bytes(b"dummy")

    view._insert_row(audio_file)

    assert view.table.rowCount() == 1
    assert view.table.item(0, 0).text() == "1"
    assert view.table.item(0, 1).text() == "audio 1.mp3"
    assert "video 1 content" in view.table.item(0, 2).text()
    assert "Template 1" in view.table.item(0, 3).text()
    assert "audio_1_1080p.mp4" in view.table.item(0, 4).text()

    view.close()


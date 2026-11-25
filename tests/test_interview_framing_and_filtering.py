"""
tests/test_interview_framing_and_filtering.py
=============================================
Verification for:
1. evaluate_interview_framing in services/scene_detector.py
   - Human/face presence and interview framing detection (1-person, 2-person).
   - Filtering out non-human footage (blank frames, scenery, uniform backgrounds).
2. filter_interview_scenes in SceneDetectorService.
3. InterviewScanner._verify_segments_framing & filter_mode ("strict" vs "all").
4. Safe empty handling (falls back gracefully to all candidate scenes when zero interview clips pass).
5. SQLite persistent candidate caching keyed by (video_hash + filter_mode).
6. UI "Show all scenes" toggle in InterviewSelectorDialog.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from database.db import (
    init_db,
    get_cached_interview_candidates,
    insert_interview_candidates_cache,
)
from services.interview_scanner import InterviewSegment, InterviewScanner
from services.scene_detector import (
    Scene,
    SceneDetectorService,
    evaluate_interview_framing,
)
from ui.dialogs.interview_selector_dialog import InterviewSelectorDialog


@pytest.fixture(autouse=True)
def ensure_db():
    init_db()


def test_evaluate_interview_framing_human_vs_scenery():
    """Verify computer vision evaluator detects human interview shots and rejects scenery/blank frames."""
    # 1. Blank frame -> reject
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    res_blank = evaluate_interview_framing(blank)
    assert res_blank["is_interview"] is False
    assert res_blank["score"] == 0.0
    assert res_blank["face_count"] == 0

    # 2. Outdoor scenery (sky + grass, no human face) -> reject
    scenery = np.zeros((360, 640, 3), dtype=np.uint8)
    scenery[:180] = [235, 180, 100]  # sky blue
    scenery[180:] = [40, 150, 40]    # green grass
    res_scenery = evaluate_interview_framing(scenery)
    assert res_scenery["is_interview"] is False
    assert res_scenery["score"] == 0.0

    # 3. Simulated 1-person interview (talking head with skin tone in upper-middle framing) -> accept
    person = np.zeros((360, 640, 3), dtype=np.uint8)
    person[:] = [30, 30, 35]  # studio backdrop
    cv2.rectangle(person, (220, 190), (420, 360), (180, 80, 40), -1)  # dark suit
    cv2.ellipse(person, (320, 140), (55, 75), 0, 0, 360, (140, 170, 220), -1)  # face skin
    res_person = evaluate_interview_framing(person)
    assert res_person["is_interview"] is True
    assert res_person["score"] >= 0.40
    assert res_person["face_count"] >= 1

    # 4. Simulated 2-person host & guest interview framing -> accept
    interview_2p = np.zeros((360, 640, 3), dtype=np.uint8)
    interview_2p[:] = [30, 30, 35]
    # Host
    cv2.rectangle(interview_2p, (80, 190), (260, 360), (180, 80, 40), -1)
    cv2.ellipse(interview_2p, (170, 140), (50, 70), 0, 0, 360, (140, 170, 220), -1)
    # Guest
    cv2.rectangle(interview_2p, (380, 190), (560, 360), (60, 60, 150), -1)
    cv2.ellipse(interview_2p, (470, 140), (50, 70), 0, 0, 360, (140, 170, 220), -1)
    res_2p = evaluate_interview_framing(interview_2p)
    assert res_2p["is_interview"] is True
    assert res_2p["score"] >= 0.50
    assert res_2p["face_count"] >= 2


def test_scene_detector_filter_interview_scenes(tmp_path):
    """Verify SceneDetectorService filters out scenery/B-roll scenes while keeping interview scenes."""
    detector = SceneDetectorService()

    # Create dummy keyframe images
    kf_human = tmp_path / "kf_human.jpg"
    kf_scenery = tmp_path / "kf_scenery.jpg"

    person = np.zeros((360, 640, 3), dtype=np.uint8)
    person[:] = [30, 30, 35]
    cv2.rectangle(person, (220, 190), (420, 360), (180, 80, 40), -1)
    cv2.ellipse(person, (320, 140), (55, 75), 0, 0, 360, (140, 170, 220), -1)
    cv2.imwrite(str(kf_human), person)

    scenery = np.zeros((360, 640, 3), dtype=np.uint8)
    scenery[:180] = [235, 180, 100]
    scenery[180:] = [40, 150, 40]
    cv2.imwrite(str(kf_scenery), scenery)

    scenes = [
        Scene(1, "h1", Path("v.mp4"), 1, 0.0, 10.0, 10.0, keyframe_path=kf_human),
        Scene(2, "h1", Path("v.mp4"), 2, 10.0, 20.0, 10.0, keyframe_path=kf_scenery),
    ]

    filtered = detector.filter_interview_scenes(scenes, min_score=0.38, relax_if_empty=False)
    assert len(filtered) == 1
    assert filtered[0].scene_index == 1
    assert filtered[0].is_interview is True


def test_scene_detector_filter_safe_empty_handling(tmp_path):
    """Verify safe fallback to all scenes when video has no interview shots."""
    detector = SceneDetectorService()

    kf_scenery = tmp_path / "kf_scenery2.jpg"
    scenery = np.zeros((360, 640, 3), dtype=np.uint8)
    scenery[:] = [0, 120, 0]
    cv2.imwrite(str(kf_scenery), scenery)

    scenes = [
        Scene(1, "h2", Path("v2.mp4"), 1, 0.0, 15.0, 15.0, keyframe_path=kf_scenery),
        Scene(2, "h2", Path("v2.mp4"), 2, 15.0, 30.0, 15.0, keyframe_path=kf_scenery),
    ]

    # With relax_if_empty=True, returns all 2 scenes rather than crashing
    relaxed = detector.filter_interview_scenes(scenes, min_score=0.38, relax_if_empty=True)
    assert len(relaxed) == 2

    # With relax_if_empty=False, returns empty list
    strict = detector.filter_interview_scenes(scenes, min_score=0.38, relax_if_empty=False)
    assert len(strict) == 0


def test_interview_scanner_strict_vs_all_modes():
    """Verify InterviewScanner._verify_segments_framing respects strict vs all filter_mode."""
    scanner = InterviewScanner(api_key="mock_key")

    seg1 = InterviewSegment(1, 0.0, 15.0, 15.0, "00:00", "00:15", "Guest Dialogue", "Hello world")
    seg2 = InterviewSegment(2, 15.0, 30.0, 15.0, "00:15", "00:30", "B-roll Scenery", "Ambient audio")

    # Mock cv2.VideoCapture to return person for seg1, scenery for seg2
    person_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    person_frame[:] = [30, 30, 35]
    cv2.rectangle(person_frame, (220, 190), (420, 360), (180, 80, 40), -1)
    cv2.ellipse(person_frame, (320, 140), (55, 75), 0, 0, 360, (140, 170, 220), -1)

    scenery_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 25.0

    call_count = [0]
    def _read_frame():
        call_count[0] += 1
        # Calls 1 & 2 for seg1, Calls 3 & 4 for seg2
        if call_count[0] <= 2:
            return True, person_frame
        return True, scenery_frame

    mock_cap.read.side_effect = _read_frame

    with patch("cv2.VideoCapture", return_value=mock_cap), patch("pathlib.Path.is_file", return_value=True):
        # 1. Strict mode should filter out seg2
        verified_strict = scanner._verify_segments_framing(
            video_path=Path("dummy_interview.mp4"),
            segments=[seg1, seg2],
            filter_mode="strict",
        )
        assert len(verified_strict) == 1
        assert verified_strict[0].label == "Guest Dialogue"
        assert verified_strict[0].is_human_verified is True

        # Reset call counter
        call_count[0] = 0
        # 2. All mode should preserve both segments
        verified_all = scanner._verify_segments_framing(
            video_path=Path("dummy_interview.mp4"),
            segments=[seg1, seg2],
            filter_mode="all",
        )
        assert len(verified_all) == 2


def test_interview_candidates_sqlite_cache():
    """Verify SQLite persistent caching properly stores and isolates candidates by filter_mode."""
    video_hash = "testhash_12345678"

    payload_strict = [{"segment_id": 1, "start_sec": 5.0, "end_sec": 20.0, "duration_sec": 15.0, "start_time": "00:05", "end_time": "00:20", "label": "Strict Interview"}]
    payload_all = [
        {"segment_id": 1, "start_sec": 5.0, "end_sec": 20.0, "duration_sec": 15.0, "start_time": "00:05", "end_time": "00:20", "label": "Strict Interview"},
        {"segment_id": 2, "start_sec": 25.0, "end_sec": 40.0, "duration_sec": 15.0, "start_time": "00:25", "end_time": "00:40", "label": "B-roll Scene"},
    ]

    insert_interview_candidates_cache(video_hash, json.dumps(payload_strict), filter_mode="strict")
    insert_interview_candidates_cache(video_hash, json.dumps(payload_all), filter_mode="all")

    # Strict query
    cached_strict = get_cached_interview_candidates(video_hash, filter_mode="strict")
    assert cached_strict is not None
    data_strict = json.loads(cached_strict["segments_json"])
    assert len(data_strict) == 1
    assert data_strict[0]["label"] == "Strict Interview"

    # All query
    cached_all = get_cached_interview_candidates(video_hash, filter_mode="all")
    assert cached_all is not None
    data_all = json.loads(cached_all["segments_json"])
    assert len(data_all) == 2


def test_dialog_show_all_scenes_toggle(qtbot):
    """Verify InterviewSelectorDialog 'Show all scenes' toggle switches between strict and all candidate sets."""
    seg_interview = InterviewSegment(
        segment_id=1,
        start_sec=10.0,
        end_sec=30.0,
        duration_sec=20.0,
        start_time="00:10",
        end_time="00:30",
        label="Host introduces guest",
        speaker="Host",
        interview_score=0.95,
        is_human_verified=True,
    )
    seg_broll = InterviewSegment(
        segment_id=2,
        start_sec=35.0,
        end_sec=55.0,
        duration_sec=20.0,
        start_time="00:35",
        end_time="00:55",
        label="Outdoor nature B-roll cut",
        speaker="Speaker",
        interview_score=0.10,
        is_human_verified=False,
    )

    dlg = InterviewSelectorDialog(
        source_video="C:/videos/test.mp4",
        segments=[seg_interview, seg_broll],
    )
    qtbot.addWidget(dlg)
    dlg.show()

    # Initial default: shows both as initialized or filtered
    assert hasattr(dlg, "chk_show_all_scenes")
    assert dlg.chk_show_all_scenes.isChecked() is False

    # Check unchecked state filter
    dlg._apply_scene_filter(show_all=False)
    # Filtered to only interview segment
    assert len(dlg.segments) == 1
    assert dlg.segments[0].label == "Host introduces guest"
    assert dlg.table.rowCount() == 1

    # Check checked state: shows all
    dlg.chk_show_all_scenes.setChecked(True)
    assert len(dlg.segments) == 2
    assert dlg.table.rowCount() == 2

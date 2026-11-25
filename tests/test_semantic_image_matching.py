"""
tests/test_semantic_image_matching.py
======================================
Unit & integration tests for Template 1 Semantic Image Matching:
  - Gemini Flash Vision image indexing & SHA-256 SQLite caching (0 API calls on rerun)
  - Bilingual Roman Urdu / English context matching (e.g. 'taaj' -> crown image)
  - Zero-cost offline heuristic fallback when API key is empty
  - HybridMediaMixer & CharacterMatcher integration
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from PIL import Image

from services.image_semantic_matcher import (
    ImageSemanticIndexer,
    score_image_semantics,
    ImageSemantics,
    BILINGUAL_SYNONYMS,
)
from services.character_matcher import CharacterMatcher
from services.hybrid_mixer import HybridMediaMixer
from services.transcriber import TranscriptionResult, TranscriptionSegment
from database.db import (
    init_db,
    get_cached_image_semantics,
    save_image_semantics,
)


@pytest.fixture
def clean_test_db(tmp_path, monkeypatch):
    """Isolate SQLite database for semantic matching tests."""
    test_db = tmp_path / "test_semantics.db"
    monkeypatch.setattr("database.db._get_db_path", lambda: test_db)
    init_db()
    return test_db


@pytest.fixture
def sample_character_images(tmp_path):
    """Generate mock character image files on disk."""
    harry_dir = tmp_path / "Harry"
    harry_dir.mkdir(parents=True)

    img1 = harry_dir / "harry_coronation.jpg"
    img2 = harry_dir / "harry_car.jpg"

    # Create small valid test JPEG images using PIL
    im = Image.new("RGB", (64, 64), color=(200, 100, 50))
    im.save(img1)
    im2 = Image.new("RGB", (64, 64), color=(50, 100, 200))
    im2.save(img2)

    return img1, img2


def test_gemini_vision_indexing_and_caching(clean_test_db, sample_character_images):
    """Verify Gemini Vision indexes images, caches in SQLite, and performs 0 API calls on second run."""
    img1, img2 = sample_character_images

    indexer = ImageSemanticIndexer(
        gemini_api_key="AIzaSyMockKeyForTestingPurposes12345",
        project_id=None,
        project_title="Test Vision Project",
    )

    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = json.dumps([
        {
            "index": 0,
            "tags": ["crown", "coronation", "royal", "taaj", "throne"],
            "description": "Prince Harry wearing a golden royal crown during ceremony",
        },
        {
            "index": 1,
            "tags": ["car", "black car", "motorcade", "gaari", "travel"],
            "description": "Prince Harry stepping into a luxury black car",
        },
    ])
    mock_meta = MagicMock()
    mock_meta.prompt_token_count = 500
    mock_meta.candidates_token_count = 150
    mock_meta.total_token_count = 650
    mock_gemini_resp.usage_metadata = mock_meta

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model_inst = MagicMock()
        mock_model_inst.generate_content.return_value = mock_gemini_resp
        mock_model_cls.return_value = mock_model_inst

        # Run 1: Should call Gemini Vision once for the batch
        results = indexer.index_images([img1, img2])
        assert len(results) == 2
        assert indexer.api_calls_count == 1
        assert indexer.total_tokens == 650
        assert "crown" in results[img1].tags
        assert "car" in results[img2].tags

    # Run 2: Same images from clean database must load from cache with 0 API calls!
    indexer_cached = ImageSemanticIndexer(
        gemini_api_key="AIzaSyMockKeyForTestingPurposes12345",
        project_id=None,
        project_title="Test Vision Project 2",
    )

    cached_results = indexer_cached.index_images([img1, img2])
    assert len(cached_results) == 2
    assert indexer_cached.api_calls_count == 0  # 0 API calls!
    assert indexer_cached.total_tokens == 0     # 0 tokens!
    assert cached_results[img1].is_cached is True
    assert cached_results[img2].is_cached is True
    assert "crown" in cached_results[img1].tags
    assert "car" in cached_results[img2].tags


def test_bilingual_context_scoring(clean_test_db, sample_character_images):
    """Verify bilingual Roman Urdu & English narration selects the exact matching visual."""
    img1, img2 = sample_character_images

    sem1 = ImageSemantics(
        file_hash="h1",
        file_path=img1,
        character_pool="Harry",
        tags=["crown", "coronation", "royal", "throne"],
        description="Prince Harry wearing a golden royal crown",
    )
    sem2 = ImageSemantics(
        file_hash="h2",
        file_path=img2,
        character_pool="Harry",
        tags=["car", "motorcade", "vehicle"],
        description="Prince Harry stepping into a black car",
    )

    # 1. Roman Urdu "taaj" must score high on crown image
    urdu_crown_text = "Prince Harry ko taaj pehnaya ja raha hai"
    score1 = score_image_semantics(sem1, urdu_crown_text, character_hint="Harry")
    score2 = score_image_semantics(sem2, urdu_crown_text, character_hint="Harry")
    assert score1 > score2
    assert score1 > 20.0

    # 2. English "car" must score high on car image
    eng_car_text = "Prince Harry arrived in his luxury black car"
    c_score1 = score_image_semantics(sem1, eng_car_text, character_hint="Harry")
    c_score2 = score_image_semantics(sem2, eng_car_text, character_hint="Harry")
    assert c_score2 > c_score1
    assert c_score2 > 20.0

    # 3. Test CharacterMatcher.select_best_image_for_text integration
    matcher = CharacterMatcher(img1.parent.parent, image_semantics={img1: sem1, img2: sem2})
    best_img_crown = matcher.select_best_image_for_text("harry", urdu_crown_text)
    assert best_img_crown == img1

    best_img_car = matcher.select_best_image_for_text("harry", "Prince Harry enters the vehicle")
    assert best_img_car == img2


def test_offline_heuristic_fallback(clean_test_db, tmp_path):
    """Verify offline fallback when GEMINI_API_KEY is empty."""
    catherine_dir = tmp_path / "Catherine"
    catherine_dir.mkdir(parents=True)

    img_speech = catherine_dir / "catherine_speech_podium.jpg"
    im = Image.new("RGB", (64, 64), color=(100, 200, 100))
    im.save(img_speech)

    indexer = ImageSemanticIndexer(gemini_api_key="")
    results = indexer.index_images([img_speech])

    assert len(results) == 1
    assert indexer.api_calls_count == 0
    assert "speech" in results[img_speech].tags
    assert "podium" in results[img_speech].tags

    # Match against speech narration
    score = score_image_semantics(results[img_speech], "Princess Catherine delivered a powerful speech")
    assert score > 15.0


def test_hybrid_media_mixer_with_semantics(clean_test_db, sample_character_images):
    """Verify HybridMediaMixer uses semantic image selection during mix creation."""
    img1, img2 = sample_character_images

    sem1 = ImageSemantics(
        file_hash="h1",
        file_path=img1,
        character_pool="Harry",
        tags=["crown", "coronation", "royal"],
        description="Prince Harry wearing a royal crown",
    )
    sem2 = ImageSemantics(
        file_hash="h2",
        file_path=img2,
        character_pool="Harry",
        tags=["car", "vehicle"],
        description="Prince Harry getting into a car",
    )

    mixer = HybridMediaMixer(
        images_dir=img1.parent.parent,
        videos_path=None,
        clip_duration=5.0,
        image_ratio=1.0,  # 100% images
        image_semantics={img1: sem1, img2: sem2},
    )

    transcription = TranscriptionResult(
        audio_path=Path("dummy.mp3"),
        audio_hash="dummyhash",
        language="en",
        full_text="Prince Harry was given the royal crown. Later he traveled by car to the palace.",
        segments=[
            TranscriptionSegment(start=0.0, end=5.0, text="Prince Harry was given the royal crown"),
            TranscriptionSegment(start=5.0, end=10.0, text="Later he traveled by car to the palace"),
        ],
        model_used="small",
        elapsed_seconds=1.0,
    )

    slots = mixer.create_mix(total_duration=10.0, transcription=transcription)
    assert len(slots) == 2
    assert slots[0].source_path == img1  # Crown image for first slot
    assert slots[1].source_path == img2  # Car image for second slot

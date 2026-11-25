"""
tests/test_gemini_resolver.py
=============================
Tests for dynamic Gemini model resolution (2026 standards), TTS filtering, and automatic 404 fallback.
"""

from unittest.mock import MagicMock, patch
import pytest

from services.gemini_resolver import (
    generate_content_with_fallback,
    get_available_candidate_models,
    is_valid_text_vision_model,
    _CACHED_WORKING_MODELS,
    _SESSION_UNAVAILABLE_MODELS,
    clear_unavailable_models,
    get_unavailable_models,
    PREFERRED_FLASH_MODELS,
)


def test_is_valid_text_vision_model_excludes_tts_and_non_gemini():
    """Verify that is_valid_text_vision_model strictly rejects -tts, audio, gemma, and non-gemini models."""
    # TTS models MUST be rejected
    assert is_valid_text_vision_model("gemini-2.5-flash-preview-tts") is False
    assert is_valid_text_vision_model("gemini-2.5-pro-preview-tts") is False
    assert is_valid_text_vision_model("gemini-3.0-flash-preview-tts") is False
    assert is_valid_text_vision_model("tts-gemini-preview") is False

    # Audio-only, embedding, gemma, deprecated models MUST be rejected
    assert is_valid_text_vision_model("gemini-flash-audio") is False
    assert is_valid_text_vision_model("gemma-4-26b") is False
    assert is_valid_text_vision_model("gemma-4-31b-it") is False
    assert is_valid_text_vision_model("text-embedding-004") is False
    assert is_valid_text_vision_model("gemini-deprecated-model") is False

    # 2026 generation models MUST be accepted
    assert is_valid_text_vision_model("gemini-3-flash-preview") is True
    assert is_valid_text_vision_model("gemini-3.8-flash") is True
    assert is_valid_text_vision_model("gemini-3.1-pro-preview") is True
    assert is_valid_text_vision_model("gemini-3.0-flash") is True
    assert is_valid_text_vision_model("gemini-3-pro") is True


def test_preferred_models_prioritizes_gemini_3_standards():
    """Verify PREFERRED_FLASH_MODELS contains only verified active models and excludes deprecated 404 models."""
    assert PREFERRED_FLASH_MODELS == ["gemini-3-flash-preview", "gemini-3.8-flash", "gemini-2.5-flash"]
    deprecated = [
        "gemini-3.8-flash-latest",
        "gemini-3.1-pro",
        "gemini-3.0-flash",
        "gemini-3.0-pro",
        "gemini-3-flash",
        "gemini-3-pro",
    ]
    for d in deprecated:
        assert d not in PREFERRED_FLASH_MODELS, f"Deprecated model '{d}' must not be in PREFERRED_FLASH_MODELS"


def test_get_available_candidate_models_fallback_list():
    """Verify that when list_models fails or returns nothing, active PREFERRED_FLASH_MODELS are returned."""
    clear_unavailable_models()
    with patch("google.generativeai.list_models", side_effect=Exception("Network error")):
        candidates = get_available_candidate_models("dummy_key")
        assert len(candidates) == 3
        assert candidates[0] == "gemini-3-flash-preview"
        assert candidates[1] == "gemini-3.8-flash"
        assert candidates[2] == "gemini-2.5-flash"


def test_deprecated_404_models_removed_from_candidates():
    """Verify deprecated 404 models are completely excluded from default candidate models."""
    clear_unavailable_models()
    candidates = get_available_candidate_models("test_key_clean")
    deprecated = [
        "gemini-3.8-flash-latest",
        "gemini-3.1-pro",
        "gemini-3.0-flash",
        "gemini-3.0-pro",
        "gemini-3-flash",
        "gemini-3-pro",
    ]
    for d in deprecated:
        assert d not in candidates, f"Deprecated 404 model '{d}' should not be in default candidates"


def test_get_available_candidate_models_dynamic_discovery_prioritizes_gemini_3():
    """Verify dynamic discovery prioritizes gemini-3 and flash models while ignoring TTS and Gemma."""
    clear_unavailable_models()
    mock_m1 = MagicMock()
    mock_m1.name = "models/gemini-2.5-flash-preview-tts"
    mock_m1.supported_generation_methods = ["generateContent"]

    mock_m2 = MagicMock()
    mock_m2.name = "models/gemini-3.8-flash"
    mock_m2.supported_generation_methods = ["generateContent"]

    mock_m3 = MagicMock()
    mock_m3.name = "models/gemini-3.1-pro-preview"
    mock_m3.supported_generation_methods = ["generateContent"]

    mock_m4 = MagicMock()
    mock_m4.name = "models/gemma-4-26b"
    mock_m4.supported_generation_methods = ["generateContent"]

    with patch("google.generativeai.list_models", return_value=[mock_m1, mock_m2, mock_m3, mock_m4]):
        candidates = get_available_candidate_models("test_key_dyn", query_api=True)
        assert "gemini-3.8-flash" in candidates
        assert "gemini-3.1-pro-preview" in candidates
        # Must NEVER include TTS or Gemma
        assert "gemini-2.5-flash-preview-tts" not in candidates
        assert "gemma-4-26b" not in candidates
        # gemini-3.8-flash should precede pro
        assert candidates.index("gemini-3.8-flash") < candidates.index("gemini-3.1-pro-preview")


def test_generate_content_fallback_on_404():
    """Verify that if gemini-3-flash-preview fails with 404, the next model is tried, succeeds, and 404 model is blacklisted."""
    _CACHED_WORKING_MODELS.clear()
    clear_unavailable_models()

    mock_resp = MagicMock()
    mock_resp.text = "Hello world"

    call_count = 0
    attempted_models = []

    def mock_generative_model_factory(model_name, **kwargs):
        nonlocal call_count
        call_count += 1
        attempted_models.append(model_name)
        inst = MagicMock()
        if model_name == "gemini-3-flash-preview":
            # First candidate raises 404
            inst.generate_content.side_effect = RuntimeError(
                "404 models/gemini-3-flash-preview is not found for API version v1beta"
            )
        else:
            # Next candidate succeeds
            inst.generate_content.return_value = mock_resp
        return inst

    with patch("google.generativeai.list_models", return_value=[]), \
         patch("google.generativeai.GenerativeModel", side_effect=mock_generative_model_factory):
        resp, model_used = generate_content_with_fallback(
            api_key="test_api_key_404",
            contents="test prompt",
            timeout=5.0,
        )

        assert resp == mock_resp
        assert model_used == "gemini-3.8-flash"
        assert _CACHED_WORKING_MODELS["test_api_key_404"] == "gemini-3.8-flash"
        # Verify 404 model is stored in session blacklist
        assert "gemini-3-flash-preview" in get_unavailable_models()


def test_404_session_blacklist_prevents_repeated_probing():
    """Verify that once a model returns 404, it is NEVER probed again in subsequent calls in the same session."""
    _CACHED_WORKING_MODELS.clear()
    clear_unavailable_models()

    mock_resp = MagicMock()
    mock_resp.text = "Success on working model"

    probed_models = []

    def mock_generative_model_factory(model_name, **kwargs):
        probed_models.append(model_name)
        inst = MagicMock()
        if model_name in ("gemini-3-flash-preview", "gemini-3-pro", "gemini-3.0-flash"):
            inst.generate_content.side_effect = RuntimeError(f"404 models/{model_name} not found")
        else:
            inst.generate_content.return_value = mock_resp
        return inst

    with patch("google.generativeai.list_models", return_value=[]), \
         patch("google.generativeai.GenerativeModel", side_effect=mock_generative_model_factory):

        # Batch 1: will probe gemini-3-flash-preview (fails 404) then gemini-3.8-flash (succeeds)
        resp1, m1 = generate_content_with_fallback("test_key_bl", "prompt 1")
        assert m1 == "gemini-3.8-flash"
        assert "gemini-3-flash-preview" in probed_models

        # Reset tracker and working cache to simulate a second batch checking candidates
        probed_models.clear()
        _CACHED_WORKING_MODELS.clear()

        # Batch 2: must NEVER probe gemini-3-flash-preview again!
        resp2, m2 = generate_content_with_fallback("test_key_bl", "prompt 2")
        assert m2 == "gemini-3.8-flash"
        assert "gemini-3-flash-preview" not in probed_models, "404 model was re-probed on batch 2!"


def test_per_request_timeout_fails_over_to_next_candidate():
    """Verify that a stalled model triggers TimeoutError after timeout seconds and fails over to next candidate."""
    import concurrent.futures
    _CACHED_WORKING_MODELS.clear()
    clear_unavailable_models()

    mock_resp = MagicMock()
    mock_resp.text = "Timeout fallback success"

    attempted_models = []

    def mock_generative_model_factory(model_name, **kwargs):
        attempted_models.append(model_name)
        inst = MagicMock()
        if model_name == "gemini-3-flash-preview":
            # Simulate hung model call raising TimeoutError
            inst.generate_content.side_effect = concurrent.futures.TimeoutError("Stalled request timed out")
        else:
            inst.generate_content.return_value = mock_resp
        return inst

    with patch("google.generativeai.list_models", return_value=[]), \
         patch("google.generativeai.GenerativeModel", side_effect=mock_generative_model_factory):

        resp, working_model = generate_content_with_fallback(
            api_key="test_key_timeout",
            contents="test prompt",
            timeout=2.0,
        )

        assert resp == mock_resp
        assert working_model == "gemini-3.8-flash"
        assert attempted_models[0] == "gemini-3-flash-preview"
        assert attempted_models[1] == "gemini-3.8-flash"


def test_fast_flash_routing_direct_and_no_retired_models():
    """Verify fastest active Flash models are prioritized without wasting time probing retired models."""
    clear_unavailable_models()
    candidates = get_available_candidate_models("test_key_flash")
    assert candidates[0] == "gemini-3-flash-preview"
    assert candidates[1] == "gemini-3.8-flash"
    assert candidates[2] == "gemini-2.5-flash"

    # Strict check: retired 1.5 and 2.0 models must NOT be in candidates
    for c in candidates:
        assert "1.5" not in c, f"Found retired model: {c}"
        assert "2.0" not in c, f"Found retired model: {c}"


def test_gemini_matcher_batching_executes_without_artificial_sleep_delays():
    """
    Verify that multi-batch scene matching on large episodes executes with ZERO artificial
    sleep throttling delays when responses succeed, while maintaining accurate scene alignment.
    """
    from services.gemini_matcher import SmartSceneMatcher, NarrativeChunk
    from services.scene_detector import Scene
    from pathlib import Path
    import json

    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyMockFastBatching12345")

    # Create 300 chunks (which will span 2 batches under batch size 250: 250 + 50)
    chunks = [
        NarrativeChunk(
            chunk_index=i,
            start_sec=i * 5.0,
            end_sec=(i + 1) * 5.0,
            duration_sec=5.0,
            text=f"Narrative scene description chunk number {i}",
        )
        for i in range(300)
    ]

    scenes = [
        Scene(
            scene_id=j,
            video_hash=f"hash_{j}",
            video_path=Path(f"episode_part_{j}.mp4"),
            scene_index=j,
            start_sec=j * 15.0,
            end_sec=(j + 1) * 15.0,
            duration_sec=15.0,
            description=f"Action sequence in scene {j}",
        )
        for j in range(20)
    ]

    batch_calls = 0

    def mock_gen_fallback(*args, **kwargs):
        nonlocal batch_calls
        batch_calls += 1
        prompt = kwargs.get("contents") or args[1]
        resp = MagicMock()
        # Parse chunk indices from prompt to generate matching output
        matches = []
        for ch_idx in range(300):
            if f'"chunk_index":{ch_idx}' in str(prompt) or f'"chunk_index": {ch_idx}' in str(prompt):
                matches.append({"chunk_index": ch_idx, "candidate_index": ch_idx % len(scenes)})
        resp.text = json.dumps(matches)
        resp.usage_metadata = {"prompt_token_count": 800, "candidates_token_count": 200, "total_token_count": 1000}
        return resp, "gemini-3.8-flash"

    with patch("services.gemini_matcher.get_cached_scene_match", return_value=None), \
         patch("services.gemini_resolver.generate_content_with_fallback", side_effect=mock_gen_fallback), \
         patch("time.sleep") as mock_sleep:

        matched = matcher.match_scenes(chunks, scenes, language="en")

        # 300 chunks divided into batches of 250 = 2 batches
        assert len(matched) == 300
        assert batch_calls == 2

        # Critical verification: time.sleep must NEVER be called on successful responses
        mock_sleep.assert_not_called()

        # Scene alignment accuracy: all chunks must be assigned a scene
        for i, ch in enumerate(matched):
            assert ch.matched_scene is not None
            expected_cand = i % len(scenes)
            assert ch.matched_scene.scene_id == scenes[expected_cand].scene_id


def test_single_pass_large_batching_125_to_250_chunks():
    """Verify that a standard 15-30 min script (125-250 chunks) matches in 1 single Gemini API call."""
    from services.gemini_matcher import SmartSceneMatcher, NarrativeChunk
    from services.scene_detector import Scene
    from pathlib import Path
    import json

    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyMockSinglePass12345")

    # 200 chunks (fits entirely inside single 250-chunk window)
    chunks = [
        NarrativeChunk(
            chunk_index=i,
            start_sec=i * 6.0,
            end_sec=(i + 1) * 6.0,
            duration_sec=6.0,
            text=f"Story narrative chunk {i}",
        )
        for i in range(200)
    ]

    scenes = [
        Scene(1, "vid_hash_1", Path("ep1.mp4"), 1, 0.0, 10.0, 10.0, description="Opening scene"),
        Scene(2, "vid_hash_1", Path("ep1.mp4"), 2, 10.0, 20.0, 10.0, description="Climax scene"),
    ]

    api_calls = 0

    def mock_gen(*args, **kwargs):
        nonlocal api_calls
        api_calls += 1
        prompt = kwargs.get("contents") or args[1]
        matches = [
            {"chunk_index": i, "candidate_index": i % len(scenes)}
            for i in range(200)
        ]
        resp = MagicMock()
        resp.text = json.dumps(matches)
        resp.usage_metadata = {"prompt_token_count": 2500, "candidates_token_count": 500, "total_token_count": 3000}
        return resp, "gemini-3-flash-preview"

    with patch("services.gemini_matcher.get_cached_scene_match", return_value=None), \
         patch("services.gemini_resolver.generate_content_with_fallback", side_effect=mock_gen):
        matched = matcher.match_scenes(chunks, scenes, language="en")
        assert len(matched) == 200
        # Must execute in exactly 1 single API call
        assert api_calls == 1
        assert matcher.api_calls_count == 1
        for ch in matched:
            assert ch.matched_scene is not None

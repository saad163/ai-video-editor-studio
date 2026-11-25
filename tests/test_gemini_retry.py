"""
tests/test_gemini_retry.py
==========================
Unit tests verifying smart rate-limit wait & retry (exponential backoff) and chunk batch throttling.
"""

from unittest.mock import MagicMock, patch
import pytest

from services.gemini_resolver import (
    is_rate_limit_error,
    is_daily_quota_exhausted,
    parse_raw_retry_delay,
    extract_retry_delay,
    execute_with_rate_limit_retry,
    generate_content_with_fallback,
    DailyQuotaExhaustedError,
    _CACHED_WORKING_MODELS,
    MAX_RETRY_DELAY,
)
from services.gemini_matcher import SmartSceneMatcher, NarrativeChunk
from services.scene_detector import Scene
from pathlib import Path


def test_is_rate_limit_error_detection():
    """Verify is_rate_limit_error detects various 429 and Free Tier quota messages."""
    assert is_rate_limit_error("429 Resource has been exhausted") is True
    assert is_rate_limit_error("Generate content input tokens per minute limit exceeded on Free Tier") is True
    assert is_rate_limit_error("Please retry in 13.387887577s.") is True
    assert is_rate_limit_error("retry_delay { seconds: 13 }") is True

    class ResourceExhausted(Exception):
        pass

    assert is_rate_limit_error(ResourceExhausted("quota exceeded")) is True

    # 404 is NOT a rate limit error
    assert is_rate_limit_error("404 models/gemini-1.5-flash is not found") is False
    assert is_rate_limit_error("ValueError: Invalid argument") is False


def test_extract_retry_delay_parsing():
    """Verify extraction of exact seconds from Google AI Studio error messages."""
    msg1 = "Generate content input tokens per minute limit exceeded. Please retry in 13.387887577s."
    delay1 = extract_retry_delay(msg1)
    assert 13.0 <= delay1 <= 16.0

    msg2 = "retry_delay { seconds: 13 }"
    delay2 = extract_retry_delay(msg2)
    assert delay2 == 14.0  # 13 + 1 safety margin

    msg3 = "Please retry after 20 seconds."
    delay3 = extract_retry_delay(msg3)
    assert 20.0 <= delay3 <= 22.0

    # Default fallback
    assert extract_retry_delay("Generic rate limit without time", default_delay=15.0) == 15.0


def test_extract_retry_delay_caps_at_30_seconds():
    """Verify that absurdly high wait times (e.g. 57,000s daily quota reset) are capped to 30.0s."""
    absurd_msg = "Quota exceeded. Please retry in 57321.4923s."
    delay = extract_retry_delay(absurd_msg)
    assert delay == MAX_RETRY_DELAY
    assert delay == 30.0

    large_seconds_msg = "retry_delay { seconds: 1200 }"
    delay2 = extract_retry_delay(large_seconds_msg)
    assert delay2 == 30.0


def test_execute_with_rate_limit_retry_success_after_backoff():
    """Verify that execute_with_rate_limit_retry catches 429, sleeps, and succeeds on retry."""
    attempts = 0
    sleeps = []

    def mock_action():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("429: Generate content input tokens per minute limit exceeded. Please retry in 5s.")
        return "SUCCESS"

    def mock_sleeper(seconds: float):
        sleeps.append(seconds)

    result = execute_with_rate_limit_retry(
        mock_action,
        model_name="gemini-3.8-flash",
        max_retries=4,
        sleeper=mock_sleeper,
    )

    assert result == "SUCCESS"
    assert attempts == 3
    assert len(sleeps) == 2
    assert all(s >= 5.0 for s in sleeps)
    assert all(s <= 30.0 for s in sleeps)


def test_generate_content_with_fallback_handles_rate_limit():
    """Verify that generate_content_with_fallback transparently retries on 429 without dropping the model."""
    _CACHED_WORKING_MODELS.clear()
    attempts = 0
    sleeps = []

    mock_resp = MagicMock()
    mock_resp.text = '{"status": "ok"}'

    def mock_generate_content(contents):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("429 ResourceExhausted: Please retry in 2s.")
        return mock_resp

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = mock_generate_content

    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        resp, model_used = generate_content_with_fallback(
            api_key="test_key_retry",
            contents="hello",
            sleeper=lambda s: sleeps.append(s),
        )

        assert resp == mock_resp
        assert model_used == "gemini-3-flash-preview"
        assert len(sleeps) == 1
        assert sleeps[0] >= 2.0
        assert sleeps[0] <= 30.0


def test_is_daily_quota_exhausted_detection():
    """Verify is_daily_quota_exhausted detects GenerateRequestsPerDay, daily quota phrases, and retry_delay > 60s."""
    # 1. Metric: GenerateRequestsPerDay
    msg_metric = (
        "Resource has been exhausted (e.g. check quota). Quota exceeded for quota metric "
        "'Generate Content API requests per day' and limit 'GenerateRequestsPerDay per day' "
        "of service 'generativelanguage.googleapis.com'"
    )
    assert is_daily_quota_exhausted(msg_metric) is True

    # 2. Daily quota phrase
    assert is_daily_quota_exhausted("ResourceExhausted: daily quota exceeded for Free Tier") is True
    assert is_daily_quota_exhausted("429: requests per day limit reached") is True

    # 3. Absurd retry delay > 60s (e.g. daily reset window in 57,321s)
    absurd_msg = "ResourceExhausted: Please retry in 57321.4923s."
    assert is_daily_quota_exhausted(absurd_msg) is True

    large_seconds_msg = "retry_delay { seconds: 86400 }"
    assert is_daily_quota_exhausted(large_seconds_msg) is True

    # 4. Normal per-minute rate limit (delay <= 60s) is NOT daily exhaustion
    normal_msg = "429: Rate limit exceeded. Please retry in 13.38s."
    assert is_daily_quota_exhausted(normal_msg) is False

    normal_sec_msg = "retry_delay { seconds: 15 }"
    assert is_daily_quota_exhausted(normal_sec_msg) is False


def test_daily_quota_fast_exit_without_sleep_or_retry():
    """Verify that when daily quota exhaustion occurs, execute_with_rate_limit_retry fails fast without sleeping."""
    attempts = 0
    sleeps = []

    def mock_action():
        nonlocal attempts
        attempts += 1
        raise RuntimeError(
            "429 Quota exceeded for quota metric 'Generate Content API requests per day' and limit 'GenerateRequestsPerDay per day'. Please retry in 57321s."
        )

    with pytest.raises(DailyQuotaExhaustedError, match="Daily Free Tier quota exhausted"):
        execute_with_rate_limit_retry(
            mock_action,
            model_name="gemini-3.8-flash",
            max_retries=4,
            sleeper=lambda s: sleeps.append(s),
        )

    # Must fail immediately on attempt 1 without sleeping or retrying 30s x 3 times
    assert attempts == 1
    assert len(sleeps) == 0, "Daily quota must fail fast without executing sleeps!"


def test_smart_scene_matcher_chunk_batching_210_chunks():
    """Verify SmartSceneMatcher batches a ~210-chunk script in 1 single API call (batch size 250) without sleep delay."""
    import json
    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyValidDummyKey123456789012345")

    # Create 210 dummy chunks (fits in 1 single batch of 250)
    chunks = [
        NarrativeChunk(i, float(i * 2), float(i * 2 + 2), 2.0, f"Chunk sentence {i}")
        for i in range(210)
    ]
    scenes = [
        Scene(1, "h1", Path("s.mp4"), 1, 0.0, 10.0, 10.0, description="Scene description")
    ]

    mock_resp = MagicMock()
    mock_resp.text = json.dumps([{"chunk_index": i, "candidate_index": 0} for i in range(210)])
    mock_meta = MagicMock()
    mock_meta.prompt_token_count = 100
    mock_meta.candidates_token_count = 20
    mock_meta.total_token_count = 120
    mock_resp.usage_metadata = mock_meta

    with patch("services.gemini_resolver.generate_content_with_fallback", return_value=(mock_resp, "gemini-3-flash-preview")) as mock_gen, \
         patch("time.sleep") as mock_sleep:

        matched = matcher._match_with_gemini(chunks, scenes, language="en")
        assert len(matched) == 210

        # With batch size 250, 210 chunks are processed in 1 single API call
        assert mock_gen.call_count == 1

        # Zero artificial throttling delay on successful responses
        assert mock_sleep.call_count == 0

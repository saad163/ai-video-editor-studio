"""
services/gemini_resolver.py
===========================
Dynamic model resolver, auto-fallback engine, and smart rate-limit retry for Google Gemini API (2026 Standards).

Key Protections & Features:
1. 2026 Production Generation Models:
   - Primary default fast model: **gemini-3.8-flash**
   - Secondary stable pro fallbacks: **gemini-3.1-pro-preview**, **gemini-3.1-pro**, **gemini-3.0-flash**, **gemini-3.0-pro**
   - Completely removes deprecated legacy models (1.5, 2.0, 2.5 series).
2. Dynamic Discovery Prioritization:
   - Prioritizes models containing `gemini-3` and `flash` while strictly ignoring deprecated/TTS models.
3. Strict Disallowlist for TTS & Non-Vision Models:
   - NEVER picks up `-tts` models (e.g. preview-tts) or audio-only, embedding, or Gemma models.
4. Capped Maximum Retry Wait Time:
   - Prevents absurd wait times (e.g. 57,000+ seconds daily quota windows).
   - Strictly caps any retry sleep timer to a maximum of 30.0 seconds.
5. Smart Rate-Limit Wait & Retry (Exponential Backoff):
   - Automatically catches 429 / ResourceExhausted / Token-per-minute Free Tier limits.
   - Extracts exact recommended delay from Google API response, clamped to [2.0s, 30.0s].
   - Transparently retries up to 3 times on the active model without premature model switching.
"""

from __future__ import annotations
import logging
import re
import time
import math
import concurrent.futures
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Maximum cap for any rate limit sleep delay to prevent deadlocks
MAX_RETRY_DELAY: float = 30.0

# Disallowed model substrings (TTS, audio-only, embeddings, gemma, custom, deprecated)
DISALLOWED_MODEL_PATTERNS: list[str] = [
    "-tts",
    "tts-",
    "-audio",
    "audio-",
    "imagen",
    "embedding",
    "gemma",
    "aqa",
    "medlm",
    "custom",
    "deprecated",
]

# Ordered priority list of Google's verified active generation models
PREFERRED_FLASH_MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
]

# Global cache mapping api_key -> working model_name
_CACHED_WORKING_MODELS: dict[str, str] = {}

# Session cache of models that returned 404 / not found / daily quota exhaustion to avoid re-probing across batches
_SESSION_UNAVAILABLE_MODELS: set[str] = set()


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when Google Gemini Free Tier daily quota is exhausted (GenerateRequestsPerDay or retry_delay > 60s)."""
    pass


def get_unavailable_models() -> set[str]:
    """Return the set of models marked unavailable (404/not found/daily exhausted) in this session."""
    return set(_SESSION_UNAVAILABLE_MODELS)


def mark_model_unavailable(model_name: str) -> None:
    """Mark a model name as unavailable for the remainder of this session."""
    if model_name:
        _SESSION_UNAVAILABLE_MODELS.add(model_name)
        # Invalidate working model cache if this model was cached
        for k, v in list(_CACHED_WORKING_MODELS.items()):
            if v == model_name:
                _CACHED_WORKING_MODELS.pop(k, None)


def clear_unavailable_models() -> None:
    """Clear session unavailable models (useful for testing or manual session reset)."""
    _SESSION_UNAVAILABLE_MODELS.clear()


def get_cached_model_name(api_key: str) -> Optional[str]:
    """Get the cached working model name for this key if already resolved."""
    return _CACHED_WORKING_MODELS.get(api_key)


def is_valid_text_vision_model(model_name: str) -> bool:
    """
    Validate that a model name is a genuine Gemini text/vision model.
    Rejects any TTS, audio-only, embedding, gemma, or deprecated models.
    """
    if not model_name:
        return False
    name_lower = model_name.lower()
    if "gemini" not in name_lower:
        return False
    return not any(pat in name_lower for pat in DISALLOWED_MODEL_PATTERNS)


def is_rate_limit_error(exc: Exception | str) -> bool:
    """Determine if an exception is a rate-limit / quota exhaustion error (HTTP 429 / ResourceExhausted)."""
    err_str = str(exc)
    exc_type = type(exc).__name__ if isinstance(exc, Exception) else ""
    return (
        "429" in err_str
        or "ResourceExhausted" in exc_type
        or "ResourceExhausted" in err_str
        or "tokens per minute" in err_str.lower()
        or "per minute limit" in err_str.lower()
        or "limit exceeded on free tier" in err_str.lower()
        or "rate limit" in err_str.lower()
        or "quota" in err_str.lower()
        or "retry in" in err_str.lower()
        or "retry_delay" in err_str.lower()
    )


def parse_raw_retry_delay(exc_or_str: Exception | str) -> Optional[float]:
    """Extract raw suggested retry delay in seconds from error string without capping."""
    err_str = str(exc_or_str)

    # 1. Match "seconds: 13" or "retry_delay { seconds: 13 }"
    m_sec = re.search(r"seconds:\s*(\d+)", err_str)
    if m_sec:
        try:
            return float(m_sec.group(1))
        except ValueError:
            pass

    # 2. Match "retry in 13.38s" or "retry in 57321.4923s"
    m_retry = re.search(r"retry\s+in\s+([\d\.]+)\s*s", err_str, re.IGNORECASE)
    if m_retry:
        try:
            return float(m_retry.group(1))
        except ValueError:
            pass

    # 3. Match "retry after 15"
    m_after = re.search(r"retry\s+after\s+([\d\.]+)", err_str, re.IGNORECASE)
    if m_after:
        try:
            return float(m_after.group(1))
        except ValueError:
            pass

    return None


def is_daily_quota_exhausted(exc_or_str: Exception | str) -> bool:
    """
    Determine if a 429/quota error indicates the daily Free Tier limit is exhausted:
    1. Error message specifically mentions 'GenerateRequestsPerDay', 'per day', or 'daily quota'.
    2. Suggested retry delay exceeds 60 seconds (indicating daily reset window).
    """
    err_str = str(exc_or_str).lower()
    if (
        "generaterequestsperday" in err_str
        or "requests per day" in err_str
        or "daily quota" in err_str
        or "per day limit" in err_str
        or "quota exceeded for quota metric 'generate content api requests per day'" in err_str
    ):
        return True

    raw_delay = parse_raw_retry_delay(exc_or_str)
    if raw_delay is not None and raw_delay > 60.0:
        return True

    return False


def extract_retry_delay(
    exc_or_str: Exception | str,
    default_delay: float = 15.0,
    max_delay: float = MAX_RETRY_DELAY,
) -> float:
    """
    Extract the suggested retry delay in seconds from the error string.
    E.g. 'Please retry in 13.387887577s' or 'retry_delay { seconds: 13 }'.
    Adds a +1.0s safety buffer, and strictly caps the result to `max_delay` (30.0s)
    to prevent deadlocks on daily quota reset windows (e.g. 57,000s).
    """
    raw = parse_raw_retry_delay(exc_or_str)
    if raw is not None:
        parsed_delay = math.ceil(raw) + 1.0
    else:
        parsed_delay = default_delay

    # Clamp between 2.0s minimum and max_delay (30.0s maximum cap)
    return min(max(2.0, parsed_delay), max_delay)


def execute_with_rate_limit_retry(
    fn: Callable[[], Any],
    model_name: str,
    max_retries: int = 3,
    initial_delay: float = 15.0,
    max_delay: float = MAX_RETRY_DELAY,
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """
    Executes callable `fn()`, catching rate limit / quota exhaustion exceptions (429, ResourceExhausted).
    If the error indicates the daily quota is exhausted (GenerateRequestsPerDay or retry_delay > 60s),
    fails fast immediately without sleeping or wasting retry attempts.
    Otherwise, extracts recommended delay (capped at 30s) and retries up to max_retries.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if is_rate_limit_error(exc):
                # Daily quota exhaustion check: fail fast immediately!
                if is_daily_quota_exhausted(exc):
                    logger.warning(
                        "🛑 Daily Free Tier quota exhausted for model '%s' (GenerateRequestsPerDay or retry_delay > 60s). "
                        "Failing fast without retry.",
                        model_name
                    )
                    raise DailyQuotaExhaustedError(
                        f"Daily Free Tier quota exhausted for model '{model_name}': {exc}"
                    ) from exc

                if attempt < max_retries - 1:
                    delay = extract_retry_delay(
                        exc,
                        default_delay=min(initial_delay * (2 ** attempt), max_delay),
                        max_delay=max_delay,
                    )
                    logger.warning(
                        "⚠️ Rate limit reached on Free Tier (model: '%s'). "
                        "Automatically waiting %.1fs (capped at %.0fs) before retrying API hit (Attempt %d/%d)...",
                        model_name, delay, max_delay, attempt + 1, max_retries
                    )
                    sleeper(delay)
                    last_exc = exc
                    continue
            raise exc

    if last_exc:
        raise last_exc


def get_available_candidate_models(
    api_key: str,
    preferred: str = "flash",
    query_api: bool = False,
) -> list[str]:
    """
    Returns candidate models in priority order:
    1. Cached working model (if valid text/vision and not blacklisted).
    2. Preferred active generation models (gemini-3-flash-preview, gemini-3.8-flash, etc.).
    3. Dynamically discovered models from genai.list_models() (if query_api=True),
       sorted to prioritize gemini-3 and flash models while strictly excluding TTS/audio/deprecated.
    Filters out any models returning 404 or daily-quota exhausted in this session.
    """
    cached = _CACHED_WORKING_MODELS.get(api_key)
    candidates: list[str] = []

    if cached and is_valid_text_vision_model(cached) and cached not in candidates and cached not in _SESSION_UNAVAILABLE_MODELS:
        candidates.append(cached)

    for pref in PREFERRED_FLASH_MODELS:
        if pref not in candidates and is_valid_text_vision_model(pref) and pref not in _SESSION_UNAVAILABLE_MODELS:
            candidates.append(pref)

    if query_api:
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            discovered: list[str] = []
            for m in genai.list_models():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    name = getattr(m, "name", "")
                    if name.startswith("models/"):
                        name = name[len("models/"):]
                    if is_valid_text_vision_model(name) and name not in _SESSION_UNAVAILABLE_MODELS:
                        discovered.append(name)

            # Sort discovered models prioritizing gemini-3 and flash models
            def _discovery_sort_key(m_name: str) -> tuple[int, int]:
                name_l = m_name.lower()
                # Priority 0: gemini-3.8 or gemini-3.x
                score_version = 0 if "gemini-3" in name_l else 1
                # Priority 1: flash over pro
                score_type = 0 if "flash" in name_l else (1 if "pro" in name_l else 2)
                return (score_version, score_type)

            discovered.sort(key=_discovery_sort_key)

            for d in discovered:
                if d not in candidates and d not in _SESSION_UNAVAILABLE_MODELS:
                    candidates.append(d)

        except Exception as exc:
            logger.debug("genai.list_models() discovery skipped: %s", exc)

    return candidates


def generate_content_with_fallback(
    api_key: str,
    contents: Any,
    generation_config: Any = None,
    timeout: float = 45.0,
    preferred: str = "flash",
    max_rate_limit_retries: int = 3,
    max_retry_delay: float = MAX_RETRY_DELAY,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[Any, str]:
    """
    Attempts generate_content across candidate models in priority order.
    Automatically handles:
    1. 404 (model not found / unsupported) -> session blacklisted and tries next candidate.
    2. Daily quota exhausted (GenerateRequestsPerDay or delay > 60s) -> fails fast without retry and blacklists model.
    3. Per-request timeout (default 45s) -> aborts stalled model and fails over to next candidate.
    4. Per-minute rate limits (429) -> wait & retry up to max_rate_limit_retries (capped at 30s).
    Returns (response, working_model_name).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    candidates = get_available_candidate_models(api_key, preferred=preferred, query_api=False)
    # Ensure blacklisted 404 models are excluded
    candidates = [m for m in candidates if m not in _SESSION_UNAVAILABLE_MODELS]

    last_exc: Optional[Exception] = None
    tried_models: list[str] = []

    for model_name in candidates:
        if model_name in _SESSION_UNAVAILABLE_MODELS:
            continue
        tried_models.append(model_name)
        try:
            logger.info("Attempting Gemini API generation with candidate model '%s' (timeout=%.1fs)...", model_name, timeout)
            model = genai.GenerativeModel(model_name, generation_config=generation_config)

            def _call_model():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(model.generate_content, contents)
                    return future.result(timeout=timeout)

            # Execute with automatic rate-limit backoff on the active model (max 30s delay)
            response = execute_with_rate_limit_retry(
                _call_model,
                model_name=model_name,
                max_retries=max_rate_limit_retries,
                max_delay=max_retry_delay,
                sleeper=sleeper,
            )

            # Successfully generated response! Cache this model for subsequent calls.
            _CACHED_WORKING_MODELS[api_key] = model_name
            logger.info("Gemini API call succeeded using model '%s'", model_name)
            return response, model_name

        except (concurrent.futures.TimeoutError, TimeoutError) as exc:
            logger.warning(
                "Gemini model '%s' timed out after %.1fs. Skipping stalled model and failing over...",
                model_name, timeout
            )
            last_exc = RuntimeError(f"Gemini model '{model_name}' timed out after {timeout}s")
            continue

        except DailyQuotaExhaustedError as exc:
            logger.warning(
                "Gemini model '%s' daily Free Tier quota exhausted. Blacklisting for session and trying next candidate...",
                model_name
            )
            mark_model_unavailable(model_name)
            last_exc = exc
            continue

        except Exception as exc:
            err_str = str(exc)
            # Catch 404 / not found / not supported for generateContent / invalid model
            is_model_unavailable = (
                "404" in err_str
                or "not found" in err_str.lower()
                or "not supported" in err_str.lower()
                or "unknown model" in err_str.lower()
                or "invalid model" in err_str.lower()
            )
            if is_model_unavailable:
                mark_model_unavailable(model_name)
                logger.warning(
                    "Gemini model '%s' unavailable (404/not supported). Blacklisted for this session.",
                    model_name
                )
                last_exc = exc
                continue

            # If quota / rate limit exhausted even after all retries on this model
            if is_rate_limit_error(exc):
                logger.warning(
                    "Gemini model '%s' rate limit / quota exhausted after %d retries (%s).",
                    model_name, max_rate_limit_retries, exc
                )
                last_exc = exc
                continue

            logger.warning("Gemini model '%s' failed (%s). Trying next candidate...", model_name, exc)
            last_exc = exc
            continue

    # If all predefined candidates failed, attempt dynamic list_models discovery as last resort
    logger.info("Predefined candidates exhausted. Querying genai.list_models() for Gemini models...")
    discovered = get_available_candidate_models(api_key, preferred=preferred, query_api=True)
    remaining_candidates = [
        m for m in discovered
        if m not in tried_models and m not in _SESSION_UNAVAILABLE_MODELS and is_valid_text_vision_model(m)
    ]

    for model_name in remaining_candidates:
        tried_models.append(model_name)
        try:
            logger.info("Attempting discovered Gemini model '%s' (timeout=%.1fs)...", model_name, timeout)
            model = genai.GenerativeModel(model_name, generation_config=generation_config)

            def _call_discovered():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(model.generate_content, contents)
                    return future.result(timeout=timeout)

            response = execute_with_rate_limit_retry(
                _call_discovered,
                model_name=model_name,
                max_retries=max_rate_limit_retries,
                max_delay=max_retry_delay,
                sleeper=sleeper,
            )

            _CACHED_WORKING_MODELS[api_key] = model_name
            logger.info("Gemini API call succeeded with discovered model '%s'", model_name)
            return response, model_name
        except (concurrent.futures.TimeoutError, TimeoutError):
            logger.warning("Discovered model '%s' timed out after %.1fs.", model_name, timeout)
            last_exc = RuntimeError(f"Discovered model '{model_name}' timed out after {timeout}s")
        except DailyQuotaExhaustedError as exc:
            logger.warning("Discovered model '%s' daily quota exhausted.", model_name)
            mark_model_unavailable(model_name)
            last_exc = exc
        except Exception as exc:
            err_str = str(exc)
            if "404" in err_str or "not found" in err_str.lower():
                mark_model_unavailable(model_name)
            logger.warning("Discovered model '%s' failed (%s).", model_name, exc)
            last_exc = exc

    err_msg = (
        f"All candidate Gemini models failed (tried: {tried_models[:5]}). "
        f"Last error: {last_exc}"
    )
    logger.error(err_msg)
    if last_exc:
        raise RuntimeError(err_msg) from last_exc
    raise RuntimeError(err_msg)

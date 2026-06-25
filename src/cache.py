"""SHA256-keyed file cache for all LLM calls. All anthropic.Anthropic() usage goes here.

Design:
- cache_key combines model + prompt + input_data into a single SHA256 hex digest.
- Cached responses are stored as JSON files under data/cache/<key>.json.
- Every call (hit or miss) is logged to the agent_calls table when an engine is supplied.
- Cache hits have tokens_in=0, tokens_out=0, cost_usd=0.0 — no API tokens consumed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from db.models import AgentCall
from pricing import cost_usd as _cost_usd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cache_key(model: str, prompt: str, input_data: dict, tool: dict | None = None) -> str:
    """Compute a deterministic SHA256 key for an LLM call.

    Args:
        model: Model string (e.g. 'claude-haiku-4-5-20251001').
        prompt: System or user prompt text.
        input_data: Dict of structured inputs passed alongside the prompt.
        tool: Anthropic tool definition dict, or None for plain-text calls.
              Changing the tool schema invalidates existing cache entries.

    Returns:
        64-character lowercase hex digest.
    """
    payload = json.dumps(
        {"model": model, "prompt": prompt, "input_data": input_data, "tool": tool},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(key: str) -> dict | None:
    """Read a cached response from disk.

    Args:
        key: Output of cache_key().

    Returns:
        Parsed JSON dict, or None if the file does not exist or is corrupt.
    """
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt cache file %s (%s) — treating as miss", path.name, exc)
        return None


def save_cached(key: str, response: dict) -> None:
    """Write a response dict to disk under data/cache/<key>.json.

    Args:
        key: Output of cache_key().
        response: JSON-serialisable dict returned by the LLM call.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")


def cached_call(
    model: str,
    prompt: str,
    input_data: dict,
    call_fn: Callable[[], dict],
    *,
    agent_name: str = "unknown",
    engine: Engine | None = None,
    tool: dict | None = None,
) -> dict:
    """Wrap an Anthropic API call with cache check and audit logging.

    On the first call with a given (model, prompt, input_data) triple the
    callable call_fn() is invoked, its response persisted to disk, and a row
    written to agent_calls. Subsequent identical calls return the cached dict
    without touching the API; a zero-cost row is still written to agent_calls
    so analysts can see cache hit rates.

    Args:
        model: Anthropic model string.
        prompt: System prompt (used in cache key and prompt_hash).
        input_data: Structured inputs dict (used in cache key and input_hash).
        call_fn: Zero-argument callable that performs the API call and returns
            a JSON-serialisable dict. Must include a 'usage' key with
            'input_tokens' and 'output_tokens' for cost tracking.
        agent_name: Identifies the calling agent in agent_calls logs.
        engine: SQLAlchemy engine for logging. If None, skips DB write.

    Returns:
        Response dict — from cache or directly from call_fn.
    """
    key = cache_key(model, prompt, input_data, tool=tool)

    # ── Cache hit path ────────────────────────────────────────────────────
    t0 = time.monotonic()
    hit = get_cached(key)
    hit_latency_ms = (time.monotonic() - t0) * 1000

    if hit is not None:
        logger.debug("CACHE HIT  key=%.12s  agent=%s", key, agent_name)
        _log_call(
            model=model,
            prompt=prompt,
            input_data=input_data,
            response=hit,
            agent_name=agent_name,
            engine=engine,
            from_cache=True,
            latency_ms=hit_latency_ms,
        )
        return hit

    # ── Cache miss path ───────────────────────────────────────────────────
    t0 = time.monotonic()
    response = call_fn()
    api_latency_ms = (time.monotonic() - t0) * 1000

    save_cached(key, response)
    logger.debug("CACHE MISS key=%.12s  agent=%s  latency=%.0fms", key, agent_name, api_latency_ms)
    _log_call(
        model=model,
        prompt=prompt,
        input_data=input_data,
        response=response,
        agent_name=agent_name,
        engine=engine,
        from_cache=False,
        latency_ms=api_latency_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _log_call(
    *,
    model: str,
    prompt: str,
    input_data: dict,
    response: dict,
    agent_name: str,
    engine: Engine | None,
    from_cache: bool,
    latency_ms: float,
) -> None:
    usage = response.get("usage", {})
    # Cache hits consume no tokens — log zeros to keep cost accurate.
    if from_cache:
        tokens_in = 0
        tokens_out = 0
    else:
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))

    cost = _cost_usd(model, tokens_in, tokens_out)

    logger.info(
        "agent=%-20s model=%-30s cached=%-5s  in=%5d out=%5d  cost=$%.5f  latency=%.0fms",
        agent_name,
        model,
        from_cache,
        tokens_in,
        tokens_out,
        cost,
        latency_ms,
    )

    if engine is None:
        return

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    row = AgentCall(
        timestamp=datetime.datetime.utcnow(),
        agent_name=agent_name,
        model_string=model,
        prompt_hash=prompt_hash,
        input_hash=input_hash,
        response_json=json.dumps(response, ensure_ascii=False),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        latency_ms=latency_ms,
        cached=from_cache,
    )
    with Session(engine) as session:
        session.add(row)
        session.commit()
